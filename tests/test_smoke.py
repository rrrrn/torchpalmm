"""Smoke tests: fit on tiny synthetic data and check the public API behaves."""

import numpy as np
import torch

from torchpalmm import LinearMixedModel, SmoothingSplineMixed
from torchpalmm.linear_mixed_model_smoothing import LinearMixedModelSmoothing

torch.set_default_dtype(torch.float64)


def _p_ok(d):
    assert "p_value" in d and 0.0 <= float(d["p_value"]) <= 1.0


def test_linear_mixed_model():
    rng = np.random.default_rng(0)
    G, ni = 20, 6
    beta = np.array([1.0, 0.5])
    X, Z, y, ids = [], [], [], []
    for i in range(G):
        Xi = np.column_stack([np.ones(ni), rng.normal(size=ni)])
        yi = Xi @ beta + rng.normal(scale=0.6) + rng.normal(scale=1.0, size=ni)
        X.append(Xi); Z.append(np.ones((ni, 1))); y.append(yi); ids += [i] * ni
    X = np.vstack(X); Z = np.vstack(Z); y = np.concatenate(y)[:, None]; ids = np.array(ids)

    m = LinearMixedModel(y=y, X=X, Z=Z, id=ids, corr_structure="iid", rand_D_type="diagonal")
    m.fit_likelihood(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200,
                     fit_type="REML", verbose=False)

    beta_hat = np.asarray(m.beta).ravel()
    assert beta_hat.shape == (2,)
    assert np.allclose(beta_hat, beta, atol=0.5)        # rough recovery
    _p_ok(m.fixed_t_test(1))                             # slope t-test


def test_smoothing_spline_mixed():
    rng = np.random.default_rng(1)
    G, GRID = 20, 25
    y, X, Z, sp, ids = [], [], [], [], []
    for i in range(G):
        n = int(rng.integers(8, 12))
        t = np.sort(rng.integers(0, GRID, size=n)) / GRID
        x = rng.normal(size=n)
        yi = 0.8 * x + np.sin(2 * np.pi * t) + rng.normal(scale=0.6) + rng.normal(scale=0.5, size=n)
        y.append(yi); X.append(x); Z.append(np.ones(n)); sp.append(t); ids += [i] * n
    y = np.concatenate(y); X = np.concatenate(X)[:, None]; Z = np.concatenate(Z)[:, None]
    sp = np.concatenate(sp); ids = np.array(ids)

    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=X, Z=Z,
                              spline_inputs=sp, obs_id=ids,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=150, fit_type="reml", verbose=False)

    fitted, _ = ss.get_blup()
    assert "theta_fitted" in fitted.columns and len(fitted) > 0
    _p_ok(ss.fixed_t_test(0))                            # covariate test
    res = ss.linearity_test(s=0)                         # H0: smooth is linear
    _p_ok(res)
    assert "reject" in res


def test_diagonal_residual_variance_in_both_trees():
    """Diagonal residuals use nmax centered log-variances; sigma2 is their geometric mean."""
    G, ni = 4, 3
    ids = np.repeat(np.arange(G), ni)
    X = np.ones((G * ni, 1))
    Z = np.ones((G * ni, 1))
    y = np.arange(G * ni, dtype=float)[:, None] / 10.0
    NB = torch.arange(G * ni, dtype=torch.float64).reshape(G, ni, 1) / 10.0

    model = LinearMixedModelSmoothing(
        y=y, X=X, Z=Z, id=ids, spline_inputs=[NB],
        corr_structure="diagonal", rand_D_type="diagonal")
    expected_init = model._diagonal_rho_init()
    model.reset_params(OLS=False)
    assert model.rho_raw.numel() == ni
    assert torch.allclose(model.rho_raw, expected_init)

    raw = torch.log(torch.tensor([1.0, 2.0, 0.5]))
    with torch.no_grad():
        model.rho_raw.copy_(raw)
    R = model.resid_corr()
    expected_diag = torch.exp(raw - raw.mean())
    assert torch.allclose(torch.diagonal(R), expected_diag)
    assert torch.count_nonzero(R - torch.diag(torch.diagonal(R))) == 0

    D = model.param_to_sqmatrix(model.D_params, n=1,
                                correlation_structure="diagonal")
    loss, sigma2 = model.loss_fn_reml(D, model.spline_params)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(sigma2)
    assert model.rho_raw.grad is not None
    assert torch.isfinite(model.rho_raw.grad).all()


def test_spline_variance_gradient_below_old_floor():
    """torch.clamp zeroes gradients outside its range; a min=0.0 forward clamp on
    log(tau^2) previously trapped spline_params at tau^2=1 once it dropped below 0."""
    G, ni = 4, 3
    ids = np.repeat(np.arange(G), ni)
    X = np.ones((G * ni, 1))
    Z = np.ones((G * ni, 1))
    y = np.arange(G * ni, dtype=float)[:, None] / 10.0
    NB = torch.arange(G * ni, dtype=torch.float64).reshape(G, ni, 1) / 10.0

    model = LinearMixedModelSmoothing(
        y=y, X=X, Z=Z, id=ids, spline_inputs=[NB],
        corr_structure="iid", rand_D_type="diagonal")
    model.reset_params(OLS=False)

    with torch.no_grad():
        model.spline_params.copy_(torch.tensor([-5.0], dtype=model.spline_params.dtype))

    D = model.param_to_sqmatrix(model.D_params, n=1, correlation_structure="diagonal")
    loss, _ = model.loss_fn_reml(D, model.spline_params)
    loss.backward()
    assert torch.isfinite(model.spline_params.grad).all()
    assert model.spline_params.grad.abs().item() > 0


def test_summary_and_spline_effect_test():
    rng = np.random.default_rng(2)
    G, GRID = 25, 25
    y, X, Z, sp, ids = [], [], [], [], []
    for i in range(G):
        n = int(rng.integers(8, 12))
        t = np.sort(rng.integers(0, GRID, size=n)) / GRID
        x = rng.normal(size=n)
        yi = 0.8 * x + np.sin(2 * np.pi * t) + rng.normal(scale=0.6) + rng.normal(scale=0.5, size=n)
        y.append(yi); X.append(x); Z.append(np.ones(n)); sp.append(t); ids += [i] * n
    y = np.concatenate(y); X = np.concatenate(X)[:, None]; Z = np.concatenate(Z)[:, None]
    sp = np.concatenate(sp); ids = np.array(ids)

    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=X, Z=Z,
                              spline_inputs=sp, obs_id=ids,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=150, fit_type="reml", verbose=False)

    res = ss.spline_effect_test(s=0)
    for col in ("p_linear", "p_nonlinear", "p_fisher", "p_wald"):
        assert 0.0 <= float(res[col].iloc[0]) <= 1.0
    assert bool(res["reject_fisher"].iloc[0]) and bool(res["reject_wald"].iloc[0])  # sin(2 pi t) is strong

    out = ss.summary(names=["x"], spline_names=["t"], verbose=False)
    assert list(out["coefficients"]["term"]) == ["x", "(Intercept)", "s(t) linear"]
    assert {"coefficients", "variance", "splines"} <= set(out)


def _cumulative_data(G=40, T=3, seed=4):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, size=(G, T))
    t = np.tile(np.arange(1, T + 1), G)
    ids = np.repeat(np.arange(G), T)
    xr = np.repeat(xs, T, axis=0)                                  # row r: subject's x_1..x_T
    m = (t[:, None] >= np.arange(1, T + 1)[None, :]).astype(float)
    f = lambda l, x: np.sin(2 * np.pi * x) * (l == 1) + (x - 0.5) * (l == 2) + 0 * x
    y = sum(m[:, l] * f(l + 1, xr[:, l]) for l in range(T)) + 0.3 * t
    y = y + np.repeat(rng.normal(scale=0.5, size=G), T) + rng.normal(scale=0.3, size=G * T)
    return y, xr, m, t, ids


def test_smooth_mask_levels_and_nb():
    import warnings
    y, xr, m, t, ids = _cumulative_data()
    Z = np.ones((len(y), 1))
    yt = torch.as_tensor(y).view(-1, 1)
    kw = dict(y=yt, Z=Z, spline_inputs=xr, obs_id=ids, smooth_mask=m, rand_D_type="diagonal")

    X_nolevel = np.zeros((len(y), 1))                              # no time effects in X
    import pytest
    with pytest.warns(UserWarning, match="time-specific intercepts"):
        ss = SmoothingSplineMixed(X=X_nolevel, **kw)
    assert ss.level_names == ["level(s2)", "level(s3)"]

    X_time = np.column_stack([(t == k).astype(float) for k in (2, 3)])   # day dummies
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ss = SmoothingSplineMixed(X=X_time, **kw)
    assert ss.level_names == []

    NB1 = ss.NB_list[1]                                            # spline 2: inactive on day 1
    assert torch.all(NB1[:, 0, :] == 0) and not torch.all(NB1[:, 1, :] == 0)
    assert ss.NB_list[0].abs().sum() > 0

    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    out = ss.summary(names=["day2", "day3"], spline_names=["x1", "x2", "x3"], verbose=False)
    assert len(out["splines"]) == 3


def _reference_t_test(m, l):
    """Uncached Satterthwaite test: recomputes the Hessian and C_theta for every contrast."""
    from scipy import stats
    from torch.autograd.functional import hessian
    th = m._theta_hat().requires_grad_(True)
    H = hessian(lambda t: m.nll(t), th)
    A = torch.linalg.inv(0.5 * (H + H.T) + 1e-8 * torch.eye(H.shape[0]))
    px = m.X_.shape[1]
    v = torch.zeros(px, 1); v[l] = 1.0
    f, _ = m.f_theta(th, v)
    g = torch.autograd.grad(f, th)[0]
    df = 2.0 * f ** 2 / (g @ A @ g)
    t = (m.beta.reshape(-1)[l] / torch.sqrt(f)).item()
    return t, df.item(), 2.0 * (1.0 - stats.t.cdf(abs(t), df.item()))


def test_fixed_t_test_cache_matches_reference():
    y, xr, m, t, ids = _cumulative_data(G=30)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=np.column_stack([t == 2, t == 3]).astype(float),
                              Z=np.ones((len(y), 1)), spline_inputs=xr, obs_id=ids, smooth_mask=m,
                              rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    lmm = ss.lmm

    table = lmm.fixed_t_test()                                  # fills the cache
    state = lmm._tt_cache
    for l in range(len(table)):
        t_ref, df_ref, p_ref = _reference_t_test(lmm, l)
        assert np.isclose(table[l]["t"], t_ref, rtol=1e-8)
        assert np.isclose(table[l]["df"], df_ref, rtol=1e-6)
        assert np.isclose(table[l]["p_value"], p_ref, rtol=1e-6, atol=1e-12)

    assert lmm.fixed_t_test(l=1) == {k: v for k, v in table[1].items() if k != "coef"}
    assert lmm._tt_cache is state                               # Hessian reused, not recomputed
    v = lmm.fixed_t_test(l=[1.0, 0.0, -1.0, 0.0, 0.0, 0.0][:len(table)])
    assert 0.0 <= v["p_value"] <= 1.0

    with torch.no_grad():
        lmm.sigma2 = lmm.sigma2 * 1.05
        lmm.spline_params.add_(0.05)
    lmm.fixed_t_test(l=0)
    assert lmm._tt_cache is not state                           # refit parameters invalidate the cache


def test_masked_basis_uses_active_x_only():
    rng = np.random.default_rng(5)
    G, ni = 60, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    we = (np.tile(np.arange(ni), G) >= 4).astype(float)             # last 2 rows of each subject
    x = np.round(rng.uniform(0, 1, N), 2)
    x[we == 1] = np.round(rng.uniform(0.3, 0.7, int(we.sum())), 2)  # weekend x covers a sub-range only
    y = np.sin(2 * np.pi * x) + we * 0.5 * (x - 0.5) + rng.normal(scale=0.3, size=N) \
        + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=we[:, None], Z=np.ones((N, 1)),
                              spline_inputs=np.column_stack([x, x]), obs_id=ids,
                              smooth_mask=np.column_stack([np.ones(N), we]), rand_D_type="diagonal")
    n_active = np.unique(x[we == 1]).size
    assert ss.NB_list[1].shape[2] == n_active - 2 < ss.NB_list[0].shape[2]
    assert ss.level_names == []
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    fdf = ss.smooth_confint()
    assert set(fdf.loc[fdf.spline_id == 1, "spline_var"].round(2)) <= set(x[we == 1].round(2))


def test_chol_jitter_recovers_slightly_indefinite_matrix():
    v = torch.randn(30, 1)
    M = v @ v.T * 1e4                                            # rank 1, PSD
    M = M - 1e-6 * torch.eye(30)                                 # rounding-sized negative eigenvalues
    with np.testing.assert_raises(Exception):
        torch.linalg.cholesky(M + 1e-10 * torch.eye(30))
    L = LinearMixedModelSmoothing._chol_jitter(M)
    assert torch.isfinite(L).all()


def test_zero_spline_variance_stays_finite():
    rng = np.random.default_rng(1)
    G, ni = 60, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    x = np.round(rng.uniform(0, 1, N), 2)
    y = 0.8 * x + rng.normal(scale=0.4, size=N) + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=rng.normal(size=(N, 1)), Z=np.ones((N, 1)),
                              spline_inputs=x, obs_id=ids, rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    ss.tau_sq_list = torch.zeros_like(ss.tau_sq_list)            # exactly linear smooth
    res = ss.spline_effect_test()
    assert np.isfinite(res[["edf", "wald_stat", "p_wald"]].to_numpy()).all()
    assert abs(float(res["edf"].iloc[0]) - 1.0) < 1e-3
    assert np.isfinite(ss.smooth_confint()[["lower", "upper"]].to_numpy()).all()
