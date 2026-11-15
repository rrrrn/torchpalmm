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
| `X` | fixed-effect design, N x px (no intercept column; one is added through the splines); `None` for no columns; DataFrame columns name the summary rows | `None` |
| `Z` | random-effect design, N x pz; `None` for no random effects | `None` |
| `spline_inputs` | covariate of each smooth, N or N x S | |
| `obs_id` | subject label of each row (any type), length N | |
| `device` | `"cpu"` or `"cuda"` | CUDA if available |
| `corr_structure` | residual correlation: `iid`, `cs`, `ar1`, `exp` | `iid` |
| `rand_D_type` | random-effect covariance: `diagonal` or `unstructured` | `diagonal` |
| `rand_D_estimate` | initial estimate of `D` | `sample` |
| `smooth_mask` | N x S array of 0/1 turning each smooth on for some rows | `None` |
| `n_knots` | reduced-rank basis: k knots per smooth (int or list), placed as `mgcv`'s `bs="cr"`, so memory and time do not grow with the number of distinct x; `None` puts a knot at every distinct x | `None` |
| `standardize` | center and scale the non-binary X columns internally; `summary()` reports their coefficients per original unit and the intercept at mean X | `False` |

`SmoothingSplineMixed.from_dataframe(data, y, smooth, group, x=(), random_slopes=(), random_intercept=True, **kwargs)` builds the model from a DataFrame. `x` is a list of columns (categorical ones are dummy-coded with the first level as reference) or a [formulaic](https://github.com/matthewwardrop/formulaic) right-hand side such as `"age + C(cls) * sex"` (`pip install "torchpalmm[formula]"`). Rows with missing values are dropped.

## `fit(...)`

| argument | meaning | default |
|---|---|---|
| `optimizer` | `torch.optim.LBFGS` [11] is recommended | `Adam` |
| `lr`, `max_iter`, `tol` | step size, iteration cap, convergence tolerance | `1e-3`, `1000`, `1e-6` |
| `fit_type` | `"reml"` [1] or `"ml"`; only the loss changes, β is profiled out under both | `"reml"` |
| `warm_start` | initialize from an OLS fit | `True` |
| `n_restarts` | extra random restarts, best loss is kept | `1` |
| `verbose` | print the optimizer log | `False` |
| `convg_check` | `"loss"`, or `"all"` / `"params"`, which also require a gradient norm below `grad_control_tol` | `"loss"` |
| `check_hessian` | also require a positive definite Hessian; a non-PD Hessian away from the boundary marks the fit as not converged | `False` |

`fit` returns the underlying `LinearMixedModelSmoothing` and sets `beta`, `D`, `sigma2`, `tau_sq_list` (one smoothing variance per smooth), `rho` and `boundary` (variance components estimated at ~0) on the model. With L-BFGS, `max_iter` is split into `max_iter // max_iter_inner` outer steps; the log and warnings report the actual number of L-BFGS iterations. Fits warn when subjects have rank(Z_i) < q (e.g. fewer rows than random effects), when sigma^2 falls below 0.1% of var(y), and when a variance component sits on the boundary.

- The gradient norm in the log and in the stopping rules is evaluated at each accepted iterate. An outer step with no progress and a gradient norm of at least `grad_control_tol` is a stalled line search: L-BFGS restarts once from steepest descent, and a second stall in a row ends the fit as not converged (`lmm.stalled`). Stalled steps never count toward `plateau_patience`.
- log(tau2 / sigma2) has a floor of -20, so a smooth on the boundary reports tau2 = sigma2 * exp(-20) rather than 0, and is listed in `boundary`.
- On the log scale the gradient vanishes near tau2 = 0, so a smooth can get stuck there. After fitting, every smooth on the boundary is checked with its REML score in tau2. If the score is positive (the criterion rises away from 0), the model is refit once with those smooths restarted at tau2 = sigma2, the lower criterion is kept, and a warning reports the outcome (`lmm.stuck_smooths()`).
- Missing or non-finite values in `y`, `X`, `Z`, `spline_inputs` or `obs_id` raise an error that names the column.
- Labels: the variance table, `boundary` and the Hessian warnings use the same names (`Var(<re name>)`, `tau2 s(<spline name>)`). Warnings about a smooth also give its 0-based index `s` as used by `linearity_test`.

## After fitting

| call | returns |
|---|---|
| `summary(names, spline_names, re_names, alpha, nonlinear_test)` | coefficient, variance and smooth-term tables (DataFrames); names default to the DataFrame columns, and short lists are padded with `X1`, `spline1`, `b1`, ... |
| `get_blup()` | fitted smooth components (DataFrame) |
| `smooth_confint(alpha, with_mean, transform)`, `smooth_sum_confint(ids, weights, transform)` | pointwise bands for one smooth or a weighted sum of smooths; `transform` (increasing, e.g. `np.exp` for ratios of a centered log-scale curve, `np.expm1` with `with_mean=True` to undo `log1p`) maps the curve and band to the original scale |
| `fitted_values(conditional)`, `residuals(conditional)` | fitted values and residuals |
| `augment()` | per-row DataFrame: group, y, marginal and conditional fitted values and residuals, standardized conditional residual |
| `random_effects()` | BLUPs as a DataFrame indexed by subject label |
| `r_squared()` | marginal and conditional R^2 (Nakagawa and Schielzeth) and the ICC |
| `convergence_diagnostics()` | gradient norm, Hessian eigenvalues, flat directions and boundary components |
| `AIC()` | AIC from the ML log-likelihood at the fitted estimates (use an ML fit) |
| `fixed_t_test(l)` | Satterthwaite t-test [2] for a fixed-effect contrast `l` |
| `linearity_test(s, test)` | H0: smooth `s` is linear; `test="score"` (variance-component score test [6, 7]) or `"lrt"` (likelihood ratio with the 50:50 chi2_0 / chi2_1 boundary mixture [9, 10]; p is NaN, with a warning, when the null refit beats the full fit, which shows the full fit is not at its optimum) |
| `spline_effect_test(s, nonlinear_test)` | H0: smooth `s` is zero (Fisher-combined and Wald p-values); reports the nonlinearity test used, whether its null refit converged, and `edf_conflict` when linearity is rejected while the main fit's EDF is ~1 |

The score tests (`linearity_test`, `spline_effect_test`, `random_effect_test`) are REML-based: their null model is always refit by REML, and they warn when called on an ML fit. Use ML fits for `lrt_test` on fixed effects. Null refits inherit the main fit's optimizer settings, start from its estimates, take overrides through `null_fit_kwargs`, and warn if they do not converge.

The Satterthwaite df uses the Hessian of the REML (or ML) criterion. If the Hessian is not positive definite, the warning names the flat or boundary parameters and those directions are dropped (pseudo-inverse on the positive eigenspace). If the df is still undefined, the test falls back to a z reference (df = inf) with a warning.

### Smoothing variance and penalty scaling

Each smooth is f(x) = c0 + c1 x + B a with a ~ N(0, tau2 I), the mixed-model form of a natural cubic spline with penalty (1/tau2) * integral f''(x)^2 dx. It matches `mgcv`'s `bs="cr"` with the same knots and smoothing parameter lambda = sigma2 / tau2. `mgcv` reports `sp` for its rescaled penalty `S / S.scale`, so tau2 = sigma2 * S.scale / sp, which is the variance that `gam.vcomp()` reports for the smooth. `tests/test_mgcv_parity.py` checks the estimates against `mgcv::gam(..., method = "REML")` with a `bs = "re"` random intercept.

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
