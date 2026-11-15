"""Smoke tests: fit on tiny synthetic data and check the public API behaves."""

import numpy as np
import pytest
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
    expected_init = model._init_rho_raw("moment")
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

    out = ss.summary(names=["x"], spline_names=["t"])
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
    out = ss.summary(names=["day2", "day3"], spline_names=["x1", "x2", "x3"])
    assert len(out["splines"]) == 3


def _reference_t_test(m, l):
    """Uncached Satterthwaite test: recomputes the Hessian and C_theta for every contrast."""
    from scipy import stats
    from torch.autograd.functional import hessian
    th = m._theta_hat().requires_grad_(True)
    H = hessian(lambda t: m.nll(t), th)
    A = LinearMixedModel._projected_inverse(0.5 * (H + H.T))[0]
    px = m.X_.shape[1]
    v = torch.zeros(px, 1); v[l] = 1.0
    f = (v.T @ m.C_theta(th) @ v).squeeze()
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


def test_robust_cholesky_recovers_slightly_indefinite_matrix():
    v = torch.randn(30, 1)
    M = v @ v.T * 1e4                                            # rank 1, PSD
    M = M - 1e-6 * torch.eye(30)                                 # rounding-sized negative eigenvalues
    with np.testing.assert_raises(Exception):
        torch.linalg.cholesky(M + 1e-10 * torch.eye(30))
    L, _ = LinearMixedModelSmoothing._robust_cholesky(M)
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


def test_smooth_confint_with_mean_option():
    y, xr, m, t, ids = _cumulative_data(G=40)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=np.column_stack([t == 2, t == 3]).astype(float),
                              Z=np.ones((len(y), 1)), spline_inputs=xr, obs_id=ids, smooth_mask=m,
                              rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    a = ss.smooth_confint(with_mean=True)
    b = ss.smooth_confint()
    for c in range(3):
        ac, bc = a[a.spline_id == c], b[b.spline_id == c]
        assert abs(bc["theta_fitted"].mean()) < 1e-10                       # centered over the grid
        assert np.allclose(ac["theta_fitted"] - ac["theta_fitted"].mean(), bc["theta_fitted"])
        assert (bc["se"].to_numpy() <= ac["se"].to_numpy() + 1e-10).all()   # shape-only band is not wider
    assert (b[b.spline_id == 0]["se"].to_numpy() < a[a.spline_id == 0]["se"].to_numpy()).all()


def test_summary_prints_only_when_displayed(capsys):
    from torchpalmm.linear_mixed_model import SummaryResult
    out = SummaryResult(coefficients=__import__("pandas").DataFrame({"term": ["a"], "estimate": [1.0]}),
                        variance=__import__("pandas").DataFrame({"component": ["residual"], "variance": [2.0]}))
    assert capsys.readouterr().out == ""                       # building / returning it prints nothing
    text = repr(out)
    assert "Fixed effects" in text and "Variance components" in text and isinstance(out, dict)


def test_lbfgs_survives_nonfinite_trial_points(monkeypatch):
    rng = np.random.default_rng(0)
    G, ni = 40, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    x = np.round(rng.uniform(0, 1, N), 2)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=N) + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=rng.normal(size=(N, 1)), Z=np.ones((N, 1)),
                              spline_inputs=x, obs_id=ids, rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)

    orig = LinearMixedModelSmoothing.loss_fn_reml

    def bad(self, D, spline_param):
        loss, s2 = orig(self, D, spline_param)
        if spline_param.detach().abs().max() > 4.0:             # objective is not finite here
            loss = loss * float("nan")
        return loss, s2

    monkeypatch.setattr(LinearMixedModelSmoothing, "loss_fn_reml", bad)
    ss.lmm.fit_likelihood(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="REML", verbose=False)
    assert torch.isfinite(torch.as_tensor(ss.lmm.final_loss)) and ss.lmm.convg == 1


def test_lbfgs_survives_cholesky_failure_at_trial_points(monkeypatch):
    rng = np.random.default_rng(0)
    G, ni = 40, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    x = np.round(rng.uniform(0, 1, N), 2)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=N) + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=rng.normal(size=(N, 1)), Z=np.ones((N, 1)),
                              spline_inputs=x, obs_id=ids, rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)

    orig = LinearMixedModelSmoothing.loss_fn_reml

    def bad(self, D, spline_param):
        if spline_param.detach().abs().max() > 4.0:
            torch.linalg.cholesky(-torch.eye(3))                 # raises LinAlgError like a non-PD B
        return orig(self, D, spline_param)

    monkeypatch.setattr(LinearMixedModelSmoothing, "loss_fn_reml", bad)
    ss.lmm.fit_likelihood(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="REML", verbose=False)
    assert ss.lmm.convg == 1


def test_level_columns_skip_duplicate_masks():
    torch.manual_seed(0)
    n = 60
    t = torch.arange(n).float() % 6
    m = (t >= 2).double()
    ss = SmoothingSplineMixed.__new__(SmoothingSplineMixed)
    ss.X = torch.randn(n, 2).double()
    X_fit = ss.X.clone()
    with pytest.warns(UserWarning):
        X2, names = ss._add_level_columns(X_fit, torch.stack([m, m, 1 - m], 1))
    assert names == ["level(s1)"] and X2.shape[1] == 3


def test_null_refit_warm_start_matches_cold():
    torch.manual_seed(1)
    G, m = 20, 8
    ids = np.repeat(np.arange(G), m)
    x = np.random.rand(G * m, 2)
    y = torch.as_tensor(np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] + np.repeat(np.random.randn(G), m) + 0.3 * np.random.randn(G * m)).view(-1, 1)
    ss = SmoothingSplineMixed(y=y, X=np.random.randn(G * m, 1), Z=np.ones((G * m, 1)), spline_inputs=x, obs_id=ids,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=150, fit_type="reml", verbose=False)
    warm = ss.lmm.linearity_test(s=0)["p_value"]
    cold = ss.lmm.linearity_test(s=0, null_fit_kwargs=dict(init=None))["p_value"]
    assert abs(warm - cold) < 1e-3


def test_blup_exposes_intercept_separately():
    torch.manual_seed(2)
    G, m = 15, 8
    ids = np.repeat(np.arange(G), m)
    x = np.random.rand(G * m, 1)
    y = torch.as_tensor(5 + np.sin(3 * x[:, 0]) + np.random.randn(G * m) * 0.3).view(-1, 1)
    ss = SmoothingSplineMixed(y=y, X=np.random.randn(G * m, 1), Z=np.ones((G * m, 1)), spline_inputs=x, obs_id=ids,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    f = ss.get_blup(random_effects=False)[0]
    assert np.allclose(f["theta_fitted"] - f["theta_no_intercept"], ss.intercept)


def test_fitted_values_masked_smooth_and_conditional():
    torch.manual_seed(3)
    G, m = 20, 10
    ids = np.repeat(np.arange(G), m)
    t = np.tile(np.arange(m), G)
    x = np.round(np.random.rand(G * m, 1), 1)
    mask = (t >= 3).astype(float)[:, None]
    days = (t[:, None] == np.arange(4, m)).astype(float)
    y = torch.as_tensor(50 + np.sin(3 * x[:, 0]) * mask[:, 0] + np.repeat(np.random.randn(G), m) + 0.3 * np.random.randn(G * m)).view(-1, 1)
    ss = SmoothingSplineMixed(y=y, X=days[:, :3], Z=np.ones((G * m, 1)), spline_inputs=x, obs_id=ids, smooth_mask=mask,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    marg, cond = ss.fitted_values(), ss.fitted_values(conditional=True)
    assert marg.shape == (G * m,) and abs(ss.residuals(conditional=False).mean()) < 5
    assert np.sum((y.numpy().ravel() - cond) ** 2) < np.sum((y.numpy().ravel() - marg) ** 2)


def test_smooth_sum_confint_matches_single_smooth():
    torch.manual_seed(4)
    G, m = 15, 8
    ids = np.repeat(np.arange(G), m)
    x = np.round(np.random.rand(G * m, 2), 1)
    y = torch.as_tensor(np.sin(3 * x[:, 0]) + np.random.randn(G * m) * 0.3).view(-1, 1)
    ss = SmoothingSplineMixed(y=y, X=np.random.randn(G * m, 1), Z=np.ones((G * m, 1)), spline_inputs=x, obs_id=ids,
                              corr_structure="iid", rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    one = ss.smooth_confint(col_smooth=1)
    sm = ss.smooth_sum_confint([1])
    assert np.allclose(one["upper"].to_numpy(), sm["upper"].to_numpy(), atol=1e-6)
    assert (ss.smooth_sum_confint([0, 1])["se"] >= 0).all()


def test_smoothing_ml_fit_matches_reml_shape():
    rng = np.random.default_rng(0)
    G, ni = 40, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    x = np.round(rng.uniform(0, 1, N), 2)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=N) + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=rng.normal(size=(N, 1)), Z=np.ones((N, 1)),
                              spline_inputs=x, obs_id=ids, rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=150, fit_type="ml", verbose=False)
    assert ss.lmm.convg == 1 and ss.lmm.fit_type == "ml"
    assert torch.isfinite(torch.as_tensor(ss.lmm.final_loss))
    assert ss.lmm.fixed_t_test(l=0)["p_value"] >= 0.0


def test_smoothing_spline_single_curve():
    from torchpalmm import SmoothingSpline
    t = np.repeat(np.arange(30) / 30, 4)
    y = (np.sin(2 * np.pi * t) + np.random.default_rng(0).normal(scale=0.3, size=t.size))[:, None]
    sp = SmoothingSpline(y=y, spline_inputs=t, device="cpu")
    sp.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=150, fit_type="reml", verbose=False)
    curve = sp.get_fitted_theta()
    assert len(curve) == 30 and np.isfinite(curve["theta_fitted"]).all()


def test_random_restarts_keep_best_fit():
    rng = np.random.default_rng(0)
    G, ni = 30, 6
    N = G * ni
    ids = np.repeat(np.arange(G), ni)
    x = np.round(rng.uniform(0, 1, N), 2)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=N) + np.repeat(rng.normal(scale=0.5, size=G), ni)
    ss = SmoothingSplineMixed(y=torch.as_tensor(y).view(-1, 1), X=rng.normal(size=(N, 1)), Z=np.ones((N, 1)),
                              spline_inputs=x, obs_id=ids, rand_D_type="diagonal")
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False)
    base = ss.lmm.final_loss
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=100, fit_type="reml", verbose=False, n_restarts=3)
    assert ss.lmm.final_loss <= base + 1e-8


def test_satterthwaite_falls_back_to_z():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(20), 6)
    X = np.column_stack([np.ones(120), rng.normal(size=120)])
    y = (X @ [1.0, 0.5] + np.repeat(rng.normal(scale=0.6, size=20), 6) + rng.normal(size=120))[:, None]
    m = LinearMixedModel(y=y, X=X, Z=np.ones((120, 1)), id=ids, corr_structure="iid", rand_D_type="diagonal")
    m.fit_likelihood(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="REML")
    state = m._ttest_state()
    state["A"].zero_(); state["unit"].clear()                    # var_f = 0: df undefined
    with pytest.warns(UserWarning, match="z reference"):
        r = m.fixed_t_test(1)
    assert r["df"] == float("inf") and 0.0 <= r["p_value"] <= 1.0
    assert np.isfinite(m.coef_table()[["ci_lower", "ci_upper"]].to_numpy()).all()


def test_dataframe_labels_knots_and_helpers():
    import pandas as pd
    rng = np.random.default_rng(6)
    G, ni = 30, 6
    df = pd.DataFrame({"subj": np.repeat([f"s{i:02d}" for i in range(G)], ni),
                       "t": rng.uniform(0, 1, G * ni), "grp": rng.choice(["a", "b", "c"], G * ni)})
    df["y"] = (np.sin(2 * np.pi * df["t"]) + (df["grp"] == "b") + np.repeat(rng.normal(scale=0.5, size=G), ni)
               + rng.normal(scale=0.3, size=G * ni))
    ss = SmoothingSplineMixed.from_dataframe(df, y="y", smooth="t", group="subj", x=["grp"], n_knots=10, device="cpu")
    assert ss.NB_list[0].shape[2] == 8 and ss.x_names == ["grp_b", "grp_c"]
    ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="reml")
    assert list(ss.summary()["coefficients"]["term"][:3]) == ["grp_b", "grp_c", "(Intercept)"]
    assert list(ss.random_effects().index[:2]) == ["s00", "s01"]
    aug = ss.augment()
    assert len(aug) == G * ni and np.isfinite(aug["resid_std"]).all()
    r2 = ss.r_squared()
    assert 0.0 <= r2["marginal_R2"] <= r2["conditional_R2"] <= 1.0
    lrt = ss.linearity_test(s=0, test="lrt")
    assert lrt["test"] == "lrt" and 0.0 <= lrt["p_value"] <= 1.0

    nox = SmoothingSplineMixed(y=df["y"], Z=np.ones((G * ni, 1)), spline_inputs=df["t"], obs_id=df["subj"], device="cpu")
    nox.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="reml")
    assert nox.summary()["coefficients"]["term"].tolist() == ["(Intercept)", "s(t) linear"]
