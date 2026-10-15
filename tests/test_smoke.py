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
