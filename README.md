# torchamm

Scalable **REML** estimation for linear mixed models with multiple **additive smooth
(penalized smoothing-spline) components**, implemented in PyTorch. Computation is batched
over groups (GPU-capable), so it scales where `lme4` and `mgcv::gamm` do not. It also
provides a fixed-effect Satterthwaite *t*-test and a score test for **linearity** of a
smooth component (H0: the curve is linear).

## Install

```bash
pip install -e .            # from this folder
pip install -e ".[examples]"   # + matplotlib/jupyter for the vignette
```

## Quickstart

```python
import torch, numpy as np
from torchamm import LinearMixedModel, SmoothingSplineMixed

# --- linear mixed model: y = X beta + Z b + eps ---
m = LinearMixedModel(y=y, X=X, Z=Z, id=subject_id,
                     corr_structure="iid", rand_D_type="diagonal")
m.fit_likelihood(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=300, fit_type="REML")
m.beta                       # fixed-effect estimates
m.fixed_t_test(1)            # Satterthwaite t-test, H0: beta_1 = 0

# --- smoothing-spline mixed model: y = X beta + f(t) + Z b + eps ---

ss = SmoothingSplineMixed(y=y, X=X, Z=Z,
                          spline_inputs=t, obs_id=subject_id,
                          corr_structure="iid", rand_D_type="diagonal")
ss.fit(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200, fit_type="reml")
ss.get_blup()                # fitted smooth (DataFrame)
ss.fixed_t_test(0)           # test a fixed covariate
ss.linearity_test(s=0)       # score test, H0: component 0 is linear
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for an end-to-end vignette
with plots, and the methodology notes in [`docs/`](docs/):
[`methods.md`](docs/methods.md) (model + scalable REML),
[`linearity_test.md`](docs/linearity_test.md) (the linearity score test), and
[`optimization_scheme.md`](docs/optimization_scheme.md).

## Public API

| object | purpose |
|---|---|
| `LinearMixedModel` | (generalized) linear mixed model, REML/ML |
| `LinearMixedModelSmoothing` | LMM with global additive smooth components |
| `SmoothingSpline` | a single penalized smoothing spline |
| `SmoothingSplineMixed` | smoothing-spline mixed model (high-level API) |
| `lrt_test` | likelihood-ratio test between two fitted models |

## License

See [LICENSE](LICENSE).
