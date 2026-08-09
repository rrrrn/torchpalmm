# torchamm

Scalable **REML** estimation for linear mixed models with multiple **additive smooth
(penalized smoothing-spline) components**, implemented in PyTorch. Computation is batched
over groups (GPU-capable), so it scales where `lme4` and `mgcv::gamm` do not. It also
provides a fixed-effect Satterthwaite *t*-test and a score test for **linearity** of a
smooth component (H0: the curve is linear).

## Install

**From GitHub, without cloning:**

```bash
pip install git+https://github.com/rrrrn/torchamm.git

# with the example deps (matplotlib/jupyter for the vignette):
pip install "torchamm[examples] @ git+https://github.com/rrrrn/torchamm.git"
```

**From a local clone (editable, for development):**

```bash
git clone https://github.com/rrrrn/torchamm.git
cd torchamm
pip install -e .                 # core
pip install -e ".[examples,test]"   # + matplotlib/jupyter + pytest
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

Two notebooks in [`examples/`](examples):

- [`quickstart.ipynb`](examples/quickstart.ipynb) — the two-minute tour: one LMM, one
  smoothing-spline mixed model, with plots.
- [`vignette.ipynb`](examples/vignette.ipynb) — the full walkthrough: unstructured random
  effects and the Satterthwaite coefficient table, model selection with `lrt_test`,
  the `iid`/`cs`/`ar1`/`exp` residual correlation structures, subject-level BLUPs,
  several additive smooths in one model, the linearity score test, and scaling behaviour.

For the methodology (model, scalable REML, the linearity score test), see:

<!-- TODO: link your paper / project page here -->
- Paper / project: **TODO**

## Public API

| object | purpose |
|---|---|
| `LinearMixedModel` | (generalized) linear mixed model, REML/ML |
| `LinearMixedModelSmoothing` | LMM with global additive smooth components |
| `SmoothingSpline` | a single penalized smoothing spline |
| `SmoothingSplineMixed` | smoothing-spline mixed model (high-level API) |
| `lrt_test` | likelihood-ratio test between two fitted models |

## Citation

If you use `torchamm`, please cite: **TODO**

```bibtex
```

## License

See [LICENSE](LICENSE).
