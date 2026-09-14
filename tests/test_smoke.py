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
    """Diagonal residuals use one reference variance and nmax-1 log ratios."""
    G, ni = 4, 3
    ids = np.repeat(np.arange(G), ni)
    X = np.ones((G * ni, 1))
    Z = np.ones((G * ni, 1))
    y = np.arange(G * ni, dtype=float)[:, None] / 10.0
    NB = torch.arange(G * ni, dtype=torch.float64).reshape(G, ni, 1) / 10.0

    model = LinearMixedModelSmoothing(
        y=y, X=X, Z=Z, id=ids, spline_inputs=[NB],
        corr_structure="diagonal", rand_D_type="diagonal")
    model.reset_params(OLS=False)
    assert model.rho_raw.numel() == ni - 1

    with torch.no_grad():
        model.rho_raw.copy_(torch.log(torch.tensor([2.0, 0.5])))
    R = model.resid_corr()
    assert torch.allclose(torch.diagonal(R), torch.tensor([1.0, 2.0, 0.5]))
    assert torch.count_nonzero(R - torch.diag(torch.diagonal(R))) == 0

    D = model.param_to_sqmatrix(model.D_params, n=1,
                                correlation_structure="diagonal")
    loss, sigma2 = model.loss_fn_reml(D, model.spline_params)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(sigma2)
    assert model.rho_raw.grad is not None
    assert torch.isfinite(model.rho_raw.grad).all()
