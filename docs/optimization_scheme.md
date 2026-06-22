# Optimization Scheme for the Smoothing Spline LMM

This document describes the optimization strategy used in `LinearMixedModelSmoothing` and provides references.

---

## Problem structure

The REML objective is

$$\ell_{\text{REML}}(\boldsymbol{\tau}, D) = \tfrac{1}{2}\bigl[\log|V| + \log|X^\top V^{-1} X| + (n-p)\log\hat\sigma^2\bigr]$$

where $V = \sigma^2 I + ZDZ^\top + \sum_{c=1}^S \tau_c^2 N_c N_c^\top$ and $\hat\sigma^2, \hat\beta$ are profiled out analytically. The free parameters are:

| Group | Symbol | Dimension | Nature |
|---|---|---|---|
| Spline variances | $\tau_1^2,\ldots,\tau_S^2$ | $S$ (one per spline component) | Positive scalars, largely independent |
| Random-effect covariance | $\text{chol}(D)$ | $p_z(p_z+1)/2$ | Dense, strongly mutually coupled |

---

## Fix 2: Log parameterisation of spline variances

### Motivation

The original formulation parameterised the smoothing variances as $\gamma_c^2 \equiv \omega_c^2$, optimising over $\omega_c \in \mathbb{R}$ (square-root scale). Two problems arise. First, the REML gradient $\partial\ell/\partial\omega_c = 2\omega_c\,(\partial\ell/\partial\gamma_c^2)$ vanishes as $\omega_c \to 0$, creating a spurious flat region at the boundary. Second, gradient magnitudes across components can differ by orders of magnitude when the optimal $\gamma_c^2$ values span a wide range, producing a poorly conditioned optimisation landscape for the joint L-BFGS step.

### Strategy: optimise over $\rho_c = \log\gamma_c^2$

Replacing the square-root parameterisation with $\gamma_c^2 \equiv \exp(\rho_c)$ removes the vanishing-gradient boundary and makes the optimisation scale-free: the gradient $\partial\ell/\partial\rho_c = \gamma_c^2\,(\partial\ell/\partial\gamma_c^2)$ is on a comparable scale across components regardless of the absolute magnitude of $\gamma_c^2$. This mirrors the standard practice in **mgcv** (Wood, 2011) and in Gaussian process marginal likelihood optimisation of working in log-variance space. Initialisation is set to $\rho_c^{(0)} = \max(0,\,\log\overline{\|\mathbf{N}_c\|^2})$, providing a component-specific scale while ensuring $\gamma_c^2 \geq 1$ at the start (under-smoothed initialisation that allows the optimizer to smooth down rather than up).

### Known limitation with large $S$

The log parameterisation substantially improves convergence at $S = 1$ but leaves a residual MISE gap at $S \geq 5$ relative to **mgcv**. The root cause, detailed in Section 3 of `methods.md`, is not the parameterisation itself but the interaction between $\sigma^2$ profiling and the multi-component additive landscape: some components are driven to $\gamma_c^2 \to 0$ at genuine stationary points of the profiled criterion (confirmed by gradient-norm diagnostics). The log parameterisation prevents the Cholesky breakdown that occurred at $\gamma_c^2 \to \infty$ (ceiling clamp at $\rho_c \leq 12$) and partially mitigates the collapse at $\gamma_c^2 \to 0$ (floor clamp at $\rho_c \geq 0$), but does not eliminate the degenerate optima.

---

## Fix 3: Decoupled parameter groups

### Motivation

When $\boldsymbol{\tau}$ and $\text{chol}(D)$ are concatenated into a single parameter vector, joint optimizers face two problems:

1. **Scale mismatch.** $\partial\ell/\partial\tau_c$ and $\partial\ell/\partial D_{ij}$ differ by orders of magnitude, so a single global step-size is suboptimal for both.
2. **Spurious coupling.** L-BFGS builds a Hessian approximation that mixes $\tau$–$\tau$, $D$–$D$, and $\tau$–$D$ curvature. The $\tau$–$D$ cross-terms are noise (they are small and change sign across iterations), inflating the effective condition number and increasing the number of outer epochs needed.

### Strategy: separate optimizers per parameter group

**For $D$ — L-BFGS.**
$D$ lives in a low-dimensional space ($p_z(p_z+1)/2 \leq 55$ for $p_z \leq 10$). L-BFGS stores a rank-$h$ ($h=10$) approximation to the inverse Hessian and achieves superlinear convergence near the optimum. This is the right tool for a small, well-coupled parameter block.

**For $\boldsymbol{\tau}$ — Adam with $\text{lr} \times \tau_{\text{mult}}$, $\beta_2 = 0.99$.**
Adam maintains a per-parameter diagonal preconditioner $1/\sqrt{v_t}$ where $v_t$ is the running second moment of the gradient. This independently adapts the effective step size for each $\tau_c$, decoupling them without requiring off-diagonal Hessian information. A lower $\beta_2 = 0.99$ (vs.\ the default $0.999$) causes the running estimate to forget stale curvature faster, which helps when $\tau_c$ values change rapidly early in training.

**Gradient clipping for $\boldsymbol{\tau}$** (max-norm = 1.0) prevents a single large-gradient τ component from destabilising the others during the first few epochs.

### References

- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) — AdamW used here.
- Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. *Mathematical Programming*, 45(1–3), 503–528. — L-BFGS.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Chapters 6–7. Springer. — L-BFGS convergence theory.
- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL 2018*. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146) — discriminative fine-tuning: layer-wise learning rates in deep networks (same principle as separate param groups here).
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML 2013*. — gradient clipping to avoid exploding updates.

**Block-coordinate descent convergence** (the alternating D / $\tau$ update is a two-block BCD step):

- Tseng, P. (2001). Convergence of a block coordinate descent method for nondifferentiable minimization. *Journal of Optimization Theory and Applications*, 109(3), 475–494.
- Beck, A., & Tetruashvili, L. (2013). On the convergence of block coordinate descent type methods. *SIAM Journal on Optimization*, 23(4), 2037–2060.

---

## Fix 4: Woodbury identity for component BLUP

### Motivation

Computing the BLUP for component $c$ requires solving systems involving

$$V_{-c} = V_{\text{full}} - \tau_c^2 N_c N_c^\top$$

A naive Cholesky of $V_{-c}$ costs $O(G \cdot n_{\max}^3)$ per component — $O(S \cdot G \cdot n_{\max}^3)$ total. Since $L_{\text{full}} = \text{chol}(V_{\text{full}})$ is already computed (for the random-effects BLUP), we can exploit it.

### Method: Sherman–Morrison–Woodbury

For a rank-$K_c$ update $A - \tau_c^2 U U^\top$ (where $A = V_{\text{full}}$, $U = N_c$):

$$(A - \tau_c^2 U U^\top)^{-1} = A^{-1} + A^{-1} U \underbrace{\left(\tau_c^{-2} I_{K_c} - U^\top A^{-1} U\right)^{-1}}_{K_c \times K_c} U^\top A^{-1}$$

Per-component cost drops from $O(G n_{\max}^3)$ (Cholesky) to $O(G n_{\max}^2 K_c)$ (triangular solve with the cached $L_{\text{full}}$) plus $O(G K_c^3)$ (small $K_c \times K_c$ solve). Since $K_c \ll n_{\max}$, this is a significant saving.

### References

- Woodbury, M. A. (1950). *Inverting Modified Matrices*. Memorandum Report 42, Statistical Research Group, Princeton University.
- Henderson, H. V., & Searle, S. R. (1981). On deriving the inverse of a sum of matrices. *SIAM Review*, 23(1), 53–60.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.), Section 2.1.4. Johns Hopkins University Press.

---

## Fix 5: Chunked two-pass Cholesky

### Motivation

For large $G$ (e.g., $N = 5000$ subjects), tensors of shape $[G, n_{\max}, n_{\max}]$ may not fit comfortably in CPU L3 cache. Processing groups in chunks of size $C \ll G$ keeps the active working set at $O(C \cdot n_{\max}^2)$, improving cache utilisation.

### Method: manual gradient checkpointing

The REML objective requires both $X^\top V^{-1} X$ (needed for $\hat\beta$) and $r^\top V^{-1} r$ (needed for $\hat\sigma^2$). Because $\hat\beta$ depends on all groups, a single-pass chunk cannot be fully self-contained. The two-pass approach:

1. **Pass 1** — iterate over chunks, accumulate $\sum_g X_g^\top V_g^{-1} X_g$ and $\sum_g X_g^\top V_g^{-1} y_g$; compute $\hat\beta$ globally.
2. **Pass 2** — iterate over chunks again with $\hat\beta$ fixed, accumulate $\sum_g r_g^\top V_g^{-1} r_g$.

$V_g$ and $L_g$ are recomputed in pass 2 rather than stored, trading 2× Cholesky compute for $O(C \cdot n_{\max}^2)$ peak memory. PyTorch autograd handles gradients correctly across both passes via the computational graph.

This is equivalent to the gradient checkpointing technique in deep learning, where activations are recomputed during the backward pass to reduce memory.

### References

- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training deep networks with limited memory. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174) — gradient checkpointing (recompute-to-save-memory).
- Griewank, A., & Walther, A. (2000). Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. *ACM TOMS*, 26(1), 19–45. — theoretical foundation.
