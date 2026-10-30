# torchpalmm API reference

## Classes

| object | purpose |
|---|---|
| `SmoothingSplineMixed` | smoothing-spline mixed model [3-6] (high-level API) |
| `SmoothingSpline` | one penalized curve: `SmoothingSplineMixed` with a single group and no random effects |
| `LinearMixedModel` | linear mixed model, REML or ML |
| `LinearMixedModelSmoothing` | LMM with additive smooth components (the engine behind `SmoothingSplineMixed`) |
| `lrt_test(m1, m2, boundary=False)` | likelihood-ratio test between two fitted models; `boundary=True` applies the chi-bar-square correction [9, 10] |

## `SmoothingSplineMixed(y, X, Z, spline_inputs, obs_id, ...)`

| argument | meaning | default |
|---|---|---|
| `y` | response, length N | |
| `X` | fixed-effect design, N x px (no intercept column; one is added through the splines) | |
| `Z` | random-effect design, N x pz | |
| `spline_inputs` | covariate of each smooth, N or N x S | |
| `obs_id` | subject id of each row, length N | |
| `device` | `"cpu"` or `"cuda"` | CUDA if available |
| `corr_structure` | residual correlation: `iid`, `cs`, `ar1`, `exp` | `iid` |
| `rand_D_type` | random-effect covariance: `diagonal` or `unstructured` | `diagonal` |
| `rand_D_estimate` | initial estimate of `D` | `sample` |
| `smooth_mask` | N x S array of 0/1 turning each smooth on for some rows | `None` |

## `fit(...)`

| argument | meaning | default |
|---|---|---|
| `optimizer` | `torch.optim.LBFGS` [11] is recommended | `Adam` |
| `lr`, `max_iter`, `tol` | step size, iteration cap, convergence tolerance | `1e-3`, `1000`, `1e-6` |
| `fit_type` | `"reml"` [1] or `"ml"` | `"reml"` |
| `warm_start` | initialize from an OLS fit | `True` |
| `n_restarts` | extra random restarts, best loss is kept | `1` |
| `verbose` | print the optimizer log | `False` |

`fit` returns the underlying `LinearMixedModelSmoothing` and sets `beta`, `D`, `sigma2`, `tau_sq_list` (one smoothing variance per smooth) and `rho` on the model.

## After fitting

| call | returns |
|---|---|
| `summary(names, spline_names, re_names, alpha)` | coefficient, variance and smooth-term tables (DataFrames) |
| `get_blup()` | fitted smooth components (DataFrame) |
| `smooth_confint(alpha)`, `smooth_sum_confint(ids, weights)` | pointwise bands for one smooth or a weighted sum of smooths |
| `fitted_values(conditional)`, `residuals(conditional)` | fitted values and residuals |
| `fixed_t_test(l)` | Satterthwaite t-test [2] for a fixed-effect contrast `l` |
| `linearity_test(s)` | score test, H0: smooth `s` is linear (variance-component score test [6, 7]) |
| `spline_effect_test(s)` | H0: smooth `s` is zero (Fisher-combined and Wald p-values) |

`LinearMixedModel(y, X, Z, id, corr_structure, rand_D_type)` has the same `fit_likelihood(optimizer, lr, max_iter, fit_type, ...)`, `fixed_t_test`, `coef_table`, `variance_table` and `summary`. Use `help(torchpalmm.SmoothingSplineMixed)` for the docstrings.

## References

Background for the methods behind the API; numbers match the bracketed tags above.

1. Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block information when block sizes are unequal. *Biometrika*, 58(3), 545-554. (REML)
2. Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance components. *Biometrics Bulletin*, 2(6), 110-114.
3. Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
4. Green, P. J. and Silverman, B. W. (1994). *Nonparametric Regression and Generalized Linear Models*. Chapman & Hall. (penalized smoothing splines)
5. Zhang, D., Lin, X., Raz, J. and Sowers, M. (1998). Semiparametric stochastic mixed models for longitudinal data. *Journal of the American Statistical Association*, 93(442), 710-719.
6. Lin, X. and Zhang, D. (1999). Inference in generalized additive mixed models by using smoothing splines. *Journal of the Royal Statistical Society B*, 61(2), 381-400.
7. Zhang, D. and Lin, X. (2003). Hypothesis testing in semiparametric additive mixed models. *Biostatistics*, 4(1), 57-74.
8. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press. (`mgcv`, the main baseline)
9. Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association*, 82(398), 605-610.
10. Stram, D. O. and Lee, J. W. (1994). Variance components testing in the longitudinal mixed effects model. *Biometrics*, 50(4), 1171-1177.
11. Liu, D. C. and Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45, 503-528. (L-BFGS)
12. Paszke, A. et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
