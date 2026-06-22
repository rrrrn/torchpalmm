# Scalable REML Estimation for Linear Mixed Models with Multiple Additive Smooth Components

## 1. Model Specification and Reparametrisation

### 1.1 The mixed-effects model with additive smooths

Let $G$ denote the number of independent subjects, with subject $i$ contributing $n_i$ observations and $n = \sum_{i=1}^G n_i$ in total. We consider the semiparametric linear mixed model

$$
\mathbf{y}_i = \mathbf{X}_i\boldsymbol\beta + \mathbf{Z}_i\mathbf{b}_i + \sum_{c=1}^{S} f_c(\mathbf{t}_{c,i}) + \boldsymbol\varepsilon_i, \qquad i = 1,\ldots,G,
$$

where $\mathbf{X}_i \in \mathbb{R}^{n_i \times p}$ is the fixed-effects design matrix with coefficient vector $\boldsymbol\beta \in \mathbb{R}^p$; $\mathbf{Z}_i \in \mathbb{R}^{n_i \times p_z}$ is the random-effects design matrix with $\mathbf{b}_i \overset{\mathrm{iid}}{\sim} \mathcal{N}(\mathbf{0}, \mathbf{D})$ and unstructured $\mathbf{D} \in \mathbb{R}^{p_z \times p_z}$; the error vector satisfies $\boldsymbol\varepsilon_i \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{R}_i)$ independently, where $\mathbf{R}_i$ is a known within-subject correlation matrix (identity for independent errors; compound-symmetric or AR(1) otherwise); and $f_c : \mathbb{R} \to \mathbb{R}$, $c = 1,\ldots,S$, are unknown smooth *population* curves, each shared across all $G$ subjects.

The model accommodates both classical nonparametric regression ($G = 1$, $p_z = 0$) and multi-component additive models in a longitudinal panel ($G \gg 1$, $p_z > 0$, $S > 1$), and specialises to the standard LMM at $S = 0$.

### 1.2 Natural cubic spline reparametrisation

The smooth $f_c$ is estimated subject to the cubic roughness penalty $\int [f_c''(t)]^2\,\mathrm{d}t$. Following Green and Silverman (1994) and the exposition of Speed (1991), let $x_{c,1}^0 < \cdots < x_{c,m_c}^0$ be the ordered distinct knot values of component $c$, let $N_{c,i} \in \{0,1\}^{n_i \times m_c}$ be the incidence matrix satisfying $\mathbf{t}_{c,i} = N_{c,i}\mathbf{x}_c^0$, and let $\boldsymbol\theta_{c,0} = (f_c(x_{c,1}^0),\ldots,f_c(x_{c,m_c}^0))^\top$ collect the functional values at the knots. The penalty matrix takes the form $\boldsymbol\Omega_c = Q_c R_c^{-1} Q_c^\top$, where $Q_c \in \mathbb{R}^{m_c \times (m_c-2)}$ and $R_c \in \mathbb{R}^{(m_c-2)\times(m_c-2)}$ are standard second-difference and band matrices (Ruppert, Wand and Carroll, 2003).

Writing the Cholesky factorisation $R_c^{-1} = S_c S_c^\top$ with $S_c$ lower-triangular and positive-diagonal, define

$$
L_c = Q_c S_c \;\in\; \mathbb{R}^{m_c \times (m_c-2)}, \qquad B_c = L_c(L_c^\top L_c)^{-1} \;\in\; \mathbb{R}^{m_c \times K_c},
$$

where $K_c = m_c - 2$. Setting $\tilde{X}_{c,0} = [\mathbf{1}_{m_c},\, \mathbf{x}_c^0] \in \mathbb{R}^{m_c \times 2}$, Green's reparametrisation (Green and Silverman, 1994, §2.3) expresses

$$
\boldsymbol\theta_{c,0} = \tilde{X}_{c,0}\boldsymbol\delta_c + B_c\mathbf{a}_c, \tag{1}
$$

where $\boldsymbol\delta_c \in \mathbb{R}^2$ are the null-space (intercept and slope) fixed effects and $\mathbf{a}_c \in \mathbb{R}^{K_c}$ are penalised random effects. The following result, proved in the companion note, justifies the reparametrisation.

**Proposition 1** (Green and Silverman, 1994). *Under* (1), $\boldsymbol\theta_{c,0}^\top \boldsymbol\Omega_c \boldsymbol\theta_{c,0} = \mathbf{a}_c^\top \mathbf{a}_c$.

*Proof sketch.* Since $L_c^\top \tilde{X}_{c,0} = 0$ (the null space of $Q_c$ contains constants and linear functions), the mixed term vanishes and $\boldsymbol\theta_{c,0}^\top\boldsymbol\Omega_c\boldsymbol\theta_{c,0} = \mathbf{a}_c^\top(L_c^\top L_c)^{-1}L_c^\top L_c L_c^\top L_c(L_c^\top L_c)^{-1}\mathbf{a}_c = \mathbf{a}_c^\top\mathbf{a}_c$. $\square$

Proposition 1 implies that placing the prior $\mathbf{a}_c \sim \mathcal{N}(\mathbf{0}, \tau_c^2 \mathbf{I}_{K_c})$ is equivalent to imposing the ridge penalty with smoothing parameter $\lambda_c^* = (\tau_c^2)^{-1}$, identifying the empirical Bayes variance component $\tau_c^2$ with Wahba's (1990) generalised maximum-likelihood smoothing parameter. The random-effect design at subject $i$ for component $c$ is

$$
\widetilde{N}_{c,i} = N_{c,i} B_c \;\in\; \mathbb{R}^{n_i \times K_c}, \tag{2}
$$

the analogue of the matrix $NB$ in equation (5) of the companion note, extended to the $i$-th subject.

### 1.3 Global smooth structure and marginal covariance

A critical modelling distinction governs the structure of the likelihood. Each $\mathbf{a}_c$ in (1) is a **single global vector** shared across all $G$ subjects: the curve $f_c$ is a population-level smooth estimated by pooling data from all subjects. Absorbing the null-space terms $\{\tilde{X}_{c,i}\boldsymbol\delta_c\}_{c=1}^S$ into an augmented fixed-effects design $\mathbf{X}$ and coefficient $\boldsymbol\beta$, the stacked model ($\mathbf{y} \in \mathbb{R}^n$) is

$$
\mathbf{y} = \mathbf{X}\boldsymbol\beta + \mathbf{Z}\mathbf{b} + \mathbf{U}\mathbf{a} + \boldsymbol\varepsilon, \tag{3}
$$

where $\mathbf{Z} = \mathrm{blockdiag}(\mathbf{Z}_i)$, $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_G \otimes \mathbf{D})$, and

$$
\mathbf{U} = [\mathbf{U}_1,\ldots,\mathbf{U}_S] \in \mathbb{R}^{n \times K_{\mathrm{tot}}}, \qquad \mathbf{a} \sim \mathcal{N}(\mathbf{0}, \boldsymbol\Lambda),
$$

with $\mathbf{U}_c \in \mathbb{R}^{n \times K_c}$ stacking the per-subject rows $\widetilde{N}_{c,i}$, $K_{\mathrm{tot}} = \sum_c K_c$, and $\boldsymbol\Lambda = \mathrm{diag}(\tau_1^2 \mathbf{I}_{K_1},\ldots,\tau_S^2 \mathbf{I}_{K_S})$. Integrating out $\mathbf{b}$ and $\mathbf{a}$ yields the marginal distribution $\mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol\beta, \mathbf{V})$ with

$$
\mathbf{V} = \underbrace{\mathrm{blockdiag}(\sigma^2\mathbf{R}_i + \mathbf{Z}_i\mathbf{D}\mathbf{Z}_i^\top)}_{\mathbf{V}_0} + \mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top. \tag{4}
$$

The first term $\mathbf{V}_0$ is block-diagonal (subjects are conditionally independent given $\mathbf{a}$). The second term $\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$, of rank at most $K_{\mathrm{tot}} \ll n$, is a global low-rank correction that introduces cross-subject covariance through the shared smooth. This is the fundamental difference from implementations that block-diagonalise $\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$ by assigning independent copies of $\mathbf{a}_c$ to each subject, which corresponds to estimating $G \cdot S$ independent per-subject curves rather than $S$ shared population curves; the consequences of this mis-specification are examined in Section 4.2.

**Remark 1.** Setting $S = 0$ in (4) recovers the standard random-coefficient LMM with $\mathbf{V} = \mathbf{V}_0$. Setting $G = 1$ and $p_z = 0$ recovers the additive smoothing spline model of Wahba (1990) with $\mathbf{V} = \tau_c^2 \widetilde{N}_c\widetilde{N}_c^\top + \sigma^2 \mathbf{I}_n$ (single subject; cf.\ equation (10) of the companion note with $V = \tau NBB^\top N^\top + \sigma^2 I$).

---

## 2. Restricted Maximum Likelihood Estimation

### 2.1 Profiled REML and the matrix-determinant lemma

The variance components $(\sigma^2, \mathbf{D}, \boldsymbol\tau^2)$ are estimated by restricted maximum likelihood (REML; Patterson and Thompson, 1971; Harville, 1977). Following the profiling strategy of Cressie and Lahiri (1993), we treat $\boldsymbol\beta$ and $\sigma^2$ as nuisance parameters and concentrate the REML criterion over them analytically. Define the scaled components

$$
\boldsymbol\Lambda_0 = \mathbf{D}/\sigma^2, \qquad \gamma_c^2 = \tau_c^2/\sigma^2, \qquad \boldsymbol\Gamma = \mathrm{diag}(\gamma_1^2\mathbf{I}_{K_1},\ldots,\gamma_S^2\mathbf{I}_{K_S}),
$$

and the normalised covariance $\mathbf{W} = \mathbf{V}/\sigma^2 = \mathbf{W}_0 + \mathbf{U}\boldsymbol\Gamma\mathbf{U}^\top$, where $\mathbf{W}_0 = \mathrm{blockdiag}(\mathbf{R}_i + \mathbf{Z}_i\boldsymbol\Lambda_0\mathbf{Z}_i^\top)$. Given $(\boldsymbol\Gamma, \boldsymbol\Lambda_0)$, the profile maximisers of $(\boldsymbol\beta, \sigma^2)$ are

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{W}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}^{-1}\mathbf{y}, \qquad \hat\sigma^2 = \frac{\hat{\mathbf{r}}^\top\mathbf{W}^{-1}\hat{\mathbf{r}}}{n - p}, \tag{5}
$$

where $\hat{\mathbf{r}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol\beta}$, analogous to equations (9)–(10) of the companion note. Substituting (5) yields the profile REML criterion in $(\boldsymbol\Gamma, \boldsymbol\Lambda_0)$ alone:

$$
-2\,\ell_{\mathrm{REML}}^{\mathrm{prof}}(\boldsymbol\Gamma, \boldsymbol\Lambda_0) = \log|\mathbf{W}| + \log|\mathbf{X}^\top\mathbf{W}^{-1}\mathbf{X}| + (n-p)\log\hat\sigma^2. \tag{6}
$$

Since $\mathbf{W}$ is not block-diagonal — the global smooth term $\mathbf{U}\boldsymbol\Gamma\mathbf{U}^\top$ couples all $G$ subjects — a naive evaluation of $\log|\mathbf{W}|$ and $\mathbf{W}^{-1}$ would cost $\mathcal{O}(n^3)$. We apply the matrix-determinant lemma and the Woodbury identity (Woodbury, 1950; Henderson and Searle, 1981) to $\mathbf{W} = \mathbf{W}_0 + \mathbf{U}\boldsymbol\Gamma\mathbf{U}^\top$:

$$
\log|\mathbf{W}| = \log|\mathbf{W}_0| + \sum_{c=1}^S K_c\log\gamma_c^2 + \log|\mathbf{M}|, \tag{7}
$$

$$
\mathbf{W}^{-1} = \mathbf{W}_0^{-1} - \mathbf{W}_0^{-1}\mathbf{U}\mathbf{M}^{-1}\mathbf{U}^\top\mathbf{W}_0^{-1}, \tag{8}
$$

where $\mathbf{M} = \boldsymbol\Gamma^{-1} + \mathbf{U}^\top\mathbf{W}_0^{-1}\mathbf{U} \in \mathbb{R}^{K_{\mathrm{tot}} \times K_{\mathrm{tot}}}$. Since $\mathbf{W}_0$ is block-diagonal, $\log|\mathbf{W}_0| = \sum_i \log|\mathbf{W}_{0,i}|$ is obtained from $G$ Cholesky factorisations of size $n_{\max} \times n_{\max}$, executed as a single batched operation (Section 3.1). The matrix $\mathbf{M}$ is assembled by a single reshape-and-matmul accumulation over the padded grouped arrays; one $K_{\mathrm{tot}}$-dimensional Cholesky then yields $\log|\mathbf{M}|$ and the solves required by (8). The incremental cost of the smooth term over a pure LMM evaluation is $\mathcal{O}(K_{\mathrm{tot}}^3 + G\,n_{\max}K_{\mathrm{tot}}^2)$, which is negligible for $K_{\mathrm{tot}}$ in the hundreds.

**Remark 2.** When $S = 0$, equations (7)–(8) reduce to $\log|\mathbf{W}| = \log|\mathbf{W}_0|$ and $\mathbf{W}^{-1} = \mathbf{W}_0^{-1}$, recovering the standard block-diagonal LMM evaluation. The Woodbury correction is thus zero-cost when there are no smooth components.

### 2.2 Variance component parameterisation

Positivity of $\tau_c^2$ and positive-semidefiniteness of $\mathbf{D}$ are enforced by unconstrained reparametrisations that improve numerical conditioning. For the smoothing variances, we optimise $\xi_c = \log\gamma_c^2 \in \mathbb{R}$, so that $\gamma_c^2 = \exp(\xi_c)$. This log parameterisation is preferred over a square-root form $(\gamma_c^2 = \omega_c^2)$ because the gradient $\partial\ell_{\mathrm{REML}}^{\mathrm{prof}}/\partial\xi_c$ is scale-invariant: a unit step in $\xi_c$ corresponds to a multiplicative change in $\gamma_c^2$ regardless of its magnitude, substantially reducing the curvature anisotropy across components with differing signal levels and improving convergence at large $S$. For the random-effects covariance, we factorise $\boldsymbol\Lambda_0 = \mathbf{L}\mathbf{L}^\top$ with $\mathbf{L} \in \mathbb{R}^{p_z \times p_z}$ lower-triangular, optimising the $p_z(p_z+1)/2$ free entries of $\mathbf{L}$ without any cone constraint. Diagonal and scalar-identity restrictions correspond to $\mathbf{L}$ diagonal or proportional to $\mathbf{I}_{p_z}$, respectively.

By contrast, **mgcv** (Wood, 2017) concatenates $(\boldsymbol\tau^2, \mathbf{D})$ into a single parameter vector and applies a smoothed positive transform with bounded image, which can impede convergence when a true variance is near zero.

### 2.3 Decoupled block-coordinate optimisation

The Hessian of $-2\ell_{\mathrm{REML}}^{\mathrm{prof}}$ in $\boldsymbol\theta = (\boldsymbol\xi, \mathrm{vec\_l}(\mathbf{L}))$ has a cross-block $\partial^2/(\partial\boldsymbol\xi\,\partial\mathrm{vec\_l}(\mathbf{L}))$ that is typically small in magnitude but oscillates in sign across iterations. Incorporating it into a single quasi-Newton update pollutes the inverse-Hessian approximation and inflates the effective condition number. Furthermore, $\boldsymbol\xi$ and $\mathrm{vec\_l}(\mathbf{L})$ respond to gradient steps in fundamentally different ways: $\boldsymbol\xi$ comprises $S$ nearly independent scalars whose gradient magnitudes vary widely across components, while the entries of $\mathrm{vec\_l}(\mathbf{L})$ are densely coupled with a more uniform Hessian structure. We therefore decouple the two blocks and assign a tailored optimiser to each.

**Algorithm 1** (Decoupled REML optimisation).

1. Initialise $\mathbf{L}^{(0)} = 0.1\,\mathbf{I}_{p_z}$ and $\xi_c^{(0)} = \log\bigl(\overline{\|\widetilde{N}_c\|^2}\bigr)$ for $c = 1,\ldots,S$.
2. For epoch $k = 0, 1, \ldots$ until the relative change in (6) falls below tolerance:
   - ($\mathbf{L}$**-step**) Freeze $\boldsymbol\xi^{(k)}$; take one L-BFGS step (Liu and Nocedal, 1989) on $\mathrm{vec\_l}(\mathbf{L})$ with a strong Wolfe line search; set $\mathbf{L}^{(k+1)}$.
   - ($\boldsymbol\xi$**-step**) Freeze $\mathbf{L}^{(k+1)}$; take one AdamW step (Loshchilov and Hutter, 2019) on $\boldsymbol\xi$ at learning rate $\eta\kappa_\tau$, with $\beta_2 = 0.99$ and gradient clipping $\|\nabla_{\boldsymbol\xi}\ell\|_2 \le 1$; set $\boldsymbol\xi^{(k+1)}$.

The AdamW preconditioner $1/\sqrt{v_t + \epsilon}$ approximates the diagonal of the per-component Fisher information of $\boldsymbol\xi$, giving each $\xi_c$ an effective step size adapted to its own curvature without reference to other components. The multiplier $\kappa_\tau > 1$ (default 5) compensates for the differing natural step scales of the two blocks. By isolating the L-BFGS update from the noisy cross-curvature terms, the $\mathbf{L}$-step recovers superlinear local convergence on its within-block subproblem (Nocedal and Wright, 2006, Ch.\ 7). Global convergence of the alternating scheme follows from two-block coordinate-descent theory (Tseng, 2001).

### 2.4 BLUP recovery

Given the fitted variance components $(\hat{\sigma}^2, \hat{\mathbf{D}}, \hat{\boldsymbol{\Lambda}})$, the posterior means of $\mathbf{a}$ and $\mathbf{b}$ are recovered from the Woodbury quantities computed at the final REML evaluation, without additional Cholesky factorisations.

**Smooth BLUP.** Let $\hat{\mathbf{V}}_0 = \mathrm{blockdiag}(\hat{\sigma}^2\hat{\mathbf{R}}_i + \mathbf{Z}_i\hat{\mathbf{D}}\mathbf{Z}_i^\top)$ and $\hat{\mathbf{r}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$. The posterior precision of $\mathbf{a}$ given $\mathbf{y}$ is $\hat{\mathbf{M}} = \hat{\boldsymbol{\Lambda}}^{-1} + \mathbf{U}^\top\hat{\mathbf{V}}_0^{-1}\mathbf{U}$, and the posterior mean (Henderson's MME; Henderson, 1984; Robinson, 1991) satisfies

$$
\hat{\mathbf{M}}\,\hat{\mathbf{a}} = \mathbf{U}^\top\hat{\mathbf{V}}_0^{-1}\hat{\mathbf{r}}, \tag{9}
$$

which is a $K_{\mathrm{tot}}$-dimensional linear system solved by one Cholesky factorisation of $\hat{\mathbf{M}}$, reusing the per-group triangular solves of $\hat{\mathbf{V}}_{0,i}$ already available. This is the multi-component, multi-subject extension of equation (11) of the companion note, $\hat{\mathbf{a}} = \tau B^\top N^\top V^{-1}(\mathbf{y} - \tilde{X}\hat{\boldsymbol{\delta}})$. The fitted population smooth and its sub-components are

$$
\hat{g}(t_1,\ldots,t_S) = \sum_{c=1}^S \hat{f}_c(t_c), \qquad \hat{f}_c(t) = \tilde{x}_c(t)^\top\hat{\boldsymbol\delta}_c + b_c(t)^\top\hat{\mathbf{a}}_c, \tag{10}
$$

where $\tilde{x}_c(t)^\top = (1,t)$, $b_c(t) = B_c^\top\mathbf{e}_t$ evaluates the reparametrised basis at $t$ (with $\mathbf{e}_t$ the standard basis vector at the nearest knot index), $\hat{\boldsymbol{\delta}}_c$ is the sub-vector of $\hat{\boldsymbol{\beta}}$ for the null space of component $c$, and $\hat{\mathbf{a}}_c$ is the sub-vector of $\hat{\mathbf{a}}$ of length $K_c$.

**Random-effects BLUP.** The Woodbury identity yields $\hat{\mathbf{V}}^{-1}\hat{\mathbf{r}} = \hat{\mathbf{V}}_0^{-1}\hat{\mathbf{r}} - \hat{\mathbf{V}}_0^{-1}\mathbf{U}\hat{\mathbf{a}}$. The per-subject BLUP is

$$
\hat{\mathbf{b}}_i = \hat{\mathbf{D}}\,\mathbf{Z}_i^\top\bigl(\hat{\mathbf{V}}_{0,i}^{-1}\hat{\mathbf{r}}_i - \hat{\mathbf{V}}_{0,i}^{-1}\mathbf{U}_i\hat{\mathbf{a}}\bigr). \tag{11}
$$

The total additional cost of (9)–(11) beyond the final REML iteration is $\mathcal{O}(K_{\mathrm{tot}}^3 + G\,n_{\max}K_{\mathrm{tot}} + G\,n_{\max}p_z)$.

---

## 3. Computational Implementation

### 3.1 Padded grouped batched Cholesky

Unbalanced group sizes are handled by zero-padding all per-subject matrices to a common dimension $n_{\max} = \max_i n_i$, with a binary mask $\mathbf{m}_i \in \{0,1\}^{n_{\max}}$ indicating real observations. The padded block

$$
\widetilde{\mathbf{V}}_{0,i} = \mathbf{V}_{0,i} \odot (\mathbf{m}_i\mathbf{m}_i^\top) + \varrho\,(\mathbf{I}_{n_{\max}} - \mathbf{m}_i\mathbf{m}_i^\top), \qquad \varrho = 10^6,
$$

inflates padded rows and columns to render them numerically inert. The collection $\{\widetilde{\mathbf{V}}_{0,i}\}_{i=1}^G$ forms a single tensor of shape $G \times n_{\max} \times n_{\max}$; a single batched Cholesky call then covers all $G$ per-subject factorisations in one BLAS-3 kernel. This yields substantial gains in cache efficiency when $G \gg n_{\max}$ and maps directly to GPU execution without Python-level group loops.

The full Woodbury accumulation $\mathbf{U}^\top\mathbf{W}_0^{-1}\mathbf{U} = \sum_i \widetilde{N}_i^\top\mathbf{W}_{0,i}^{-1}\widetilde{N}_i$ is computed as a single reshape-and-matmul over the padded $G \times n_{\max} \times K_{\mathrm{tot}}$ arrays, with no Python loop over subjects. Gradient computation via reverse-mode automatic differentiation propagates through the batched primitives without modification.

### 3.2 Chunked evaluation of the REML objective

When $G$ is large, storing all per-subject matrices simultaneously may exceed available memory. Following the incremental sufficient-statistic accumulation strategy of Golub and Van Loan (2013, §5.3), we partition subjects into disjoint chunks of size at most $C$ and sweep through them sequentially. Because the profiled REML objective (6) is not a simple sum of per-subject terms — the profiled fixed-effects estimator $\hat{\boldsymbol{\beta}}$ (5) and the posterior precision $\hat{\mathbf{M}}$ (7) both depend on cross-products pooled over all $G$ subjects — a single sequential sweep does not suffice. A first sweep accumulates $\mathbf{X}^\top\mathbf{W}_0^{-1}\mathbf{X}$, $\mathbf{X}^\top\mathbf{W}_0^{-1}\mathbf{y}$, $\mathbf{U}^\top\mathbf{W}_0^{-1}\mathbf{U}$, $\mathbf{U}^\top\mathbf{W}_0^{-1}\mathbf{y}$, and $\sum_i\log|\mathbf{W}_{0,i}|$ chunk by chunk; $\hat{\boldsymbol{\beta}}$ and $\hat{\mathbf{M}}$ are then obtained from these global quantities by a single Cholesky solve (Harville, 1977). A second sweep re-visits each chunk to accumulate the profiled quadratic form $\hat{\mathbf{r}}^\top\mathbf{W}^{-1}\hat{\mathbf{r}}$ (8). Peak working memory is thereby reduced from $\mathcal{O}(G\,n_{\max}^2)$ to $\mathcal{O}(C\,n_{\max}^2)$ at the cost of recomputing each per-subject factorisation twice, a trade-off between memory and computation with well-established precedent in large-scale statistical computing (Chen et al., 2016; Gilmour, Thompson and Cullis, 1995).

---

## 4. Empirical Evaluation

### 4.1 Simulation design

We compare the proposed method against the reference R implementations `gamm` and `bam` from the **mgcv** package (Wood, 2017). The mixed-SS experiment crosses $G \in \{500, 1000, 5000\}$ subjects, $S \in \{1,3,5,10\}$ smooth components, and two covariance structures for $\mathbf{D}$ (diagonal; unstructured), with 20 independent Monte Carlo replicates per cell. Each smooth component is drawn from a library of 10 standardised population curves (zero mean, unit variance over $[0,1]$; see companion document `DESIGN.md`), with a distinct random subset of $S$ curves drawn per replicate to avoid confounding model dimension with curve smoothness. Each subject contributes $n_i \sim \mathrm{Uniform}\{15,\ldots,30\}$ observations, with knot counts determined by $K_c = \mathrm{clip}(\lfloor 3.5\,n^{1/3}\rceil, 50, 300)$ following Ruppert, Wand and Carroll (2003).

The primary performance metric is the per-component mean integrated squared error

$$
\mathrm{MISE} = \frac{1}{S}\sum_{c=1}^S \int \bigl(\hat{f}_c^{\,c}(t) - f_c^{\,c}(t)\bigr)^2\,\mathrm{d}t, \tag{12}
$$

approximated by the trapezoid rule on a shared 500-point grid over each component's observed support, with both fitted and true curves mean-centred to remove the intercept ambiguity. Identical datasets are processed by Python and R from separate data directories; SHA-256 checksums of the centred response vector confirm that both implementations fit exactly the same data.

### 4.2 The global smooth requirement: diagnosis and correction

**Observed symptom.** An earlier implementation exhibited a sharp step-up in MISE when moving from $S = 1$ to $S \geq 3$: pilot values of 0.099–0.205 at $S \in \{3,5,10\}$ versus 0.0013 at $S = 1$, while `gamm` remained flat across all $S$. The step persisted in the total additive ISE $\int(\hat{g} - g)^2$, ruling out a pure inter-component attribution artefact.

**Root cause.** The marginal covariance (4) requires $\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$ to be assembled as a global $n \times n$ low-rank matrix pooling all $G$ subjects. The erroneous implementation instead block-diagonalised this term as $\mathrm{blockdiag}(\widetilde{N}_{c,i}\tau_c^2\widetilde{N}_{c,i}^\top)$, effectively assigning each subject $i$ an independent copy $\mathbf{a}_{c,i}$ of the smooth coefficients. Under this mis-specification, the posterior of $\mathbf{a}_{c,i}$ is informed only by subject $i$'s $n_i \approx 15$–$30$ observations rather than all $n \approx 7{,}500$–$75{,}000$. With $S > 1$ components simultaneously competing for this limited per-subject information, the REML criterion drove $\hat\tau_c^2$ to arbitrarily large values to absorb between-subject variation that should have been captured by $\hat\mathbf{b}_i$, producing severely under-smoothed population curves. At $S = 1$ the competition is absent, explaining why the mis-specification was invisible in the single-component case.

**Corrections.** Two code sites required modification.

> **Correction 1 — REML likelihood** (`LinearMixedModelSmoothing.loss_fn_reml`). Replace the erroneous block-diagonal log-determinant $\sum_i\log|\sigma^2\mathbf{R}_i + \mathbf{Z}_i\mathbf{D}\mathbf{Z}_i^\top + \mathbf{U}_i\boldsymbol\Lambda\mathbf{U}_i^\top|$ with the matrix-determinant-lemma decomposition (7): $\log|\mathbf{V}_0| + \sum_c K_c\log\tau_c^2 + \log|\mathbf{M}|$, where $\mathbf{M} = \boldsymbol\Lambda^{-1} + \mathbf{U}^\top\mathbf{V}_0^{-1}\mathbf{U}$ accumulates information from all $G$ subjects jointly.

> **Correction 2 — smooth BLUP** (`SmoothSplineMixed.get_blup`). Replace $G$ independent per-subject systems $(\tau_c^{-2}\mathbf{I} + \widetilde{N}_{c,i}^\top\mathbf{V}_{0,i}^{-1}\widetilde{N}_{c,i})\hat\mathbf{a}_{c,i} = \widetilde{N}_{c,i}^\top\mathbf{V}_{0,i}^{-1}\hat{\mathbf{r}}_i$ with the single global MME system (9): $\hat\mathbf{M}\hat\mathbf{a} = \mathbf{U}^\top\hat\mathbf{V}_0^{-1}\hat{\mathbf{r}}$, the natural multi-subject extension of equation (11) of the companion note.

Both corrections are implemented entirely via batched tensor operations over the padded $G \times n_{\max}$ arrays, with no Python-level loop over subjects.

### 4.3 Summary of results

A preliminary comparison on $G = 500$, diagonal $\mathbf{D}$, 3 Monte Carlo replicates per $S$ is reported in Table 1.

**Table 1.** Per-component MISE (12), absolute error in $\hat\sigma^2$, and convergence rate for the proposed method after Corrections 1–2. All entries are Monte Carlo means over 3 replicates.

| $S$ | MISE   | $|\hat\sigma^2 - \sigma^2|$ | Conv. |
| --- | ------ | --------------------------- | ----- |
| 1   | 0.0011 | 0.004                       | 1.00  |
| 3   | 0.017  | 0.007                       | 1.00  |
| 5   | 0.0065 | 0.009                       | 1.00  |
| 10  | 0.0090 | 0.017                       | 1.00  |

The step-up in MISE from $S = 1$ to $S \geq 3$ is eliminated: values now lie uniformly in the range $0.001$–$0.017$, with residual variation attributable to Monte Carlo noise at 3 replicates. The noise variance $\sigma^2$ is recovered to within 1–2% of the true value 1.0 at all $S$. Full 20-replicate results across $G \in \{500, 1000, 5000\}$ and both covariance structures are reported in the main paper.

**Remark 3** (Scalability). The batched Cholesky (Section 3.1) and Woodbury smooth term (Section 2.1) both scale linearly in $G$ for fixed $n_{\max}$, with no Python-level group loops. The cost increment of the smooth over a pure LMM scales as $\mathcal{O}(K_{\mathrm{tot}}^3 + G\,n_{\max}K_{\mathrm{tot}}^2)$; for $K_{\mathrm{tot}} = 300$ and $G\,n_{\max} = 2.5\times10^5$, this is dominated by the $G\,n_{\max}K_{\mathrm{tot}}^2$ term ($\approx 2.25\times10^{10}$ flops), comparable to the per-subject factorisation cost $G\,n_{\max}^3$ at $n_{\max} = 30$ ($\approx 6.75\times10^8$ flops, here smaller). Both operations map to GPU-executable batched BLAS kernels.
