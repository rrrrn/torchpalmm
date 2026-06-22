# Is the Curve Really Curved? A Tutorial on the Linearity Test

This is a tutorial for someone who has had a first course in regression and
mathematical statistics (you know least squares, the normal distribution, the
chi-square distribution, and the idea of a hypothesis test). By the end you will
understand exactly what `linearity_test` computes and why. It documents the test of
**linearity** of a smooth component in the spline model of `methods.md`.

## 1. The question, in plain words

We fit a flexible curve $f(t)$ to data: think of $t$ as time and $f(t)$ as some
mean response that bends and wiggles. A natural question is:

> Is the curve genuinely curved, or could a straight line have done just as well?

If a straight line is enough, the "wiggles" we see are just noise, and reporting a
fancy curve would be over-interpreting the data. The linearity test answers this
question with a p-value.

## 2. The model, and where "linearity" lives

Our curve is built so that it splits cleanly into two pieces:

$$
f(t) = \underbrace{\delta_0 + \delta_1\,t}_{\text{straight line}}
       \;+\; \underbrace{\widetilde N\,\mathbf a}_{\text{wiggles}} .
$$

- The **straight line** part $\delta_0 + \delta_1 t$ is an ordinary intercept and
  slope. These are regular regression coefficients (in the code they sit inside the
  fixed-effects design $\mathbf X$).
- The **wiggle** part $\widetilde N\,\mathbf a$ is a sum of bumpy basis functions
  (columns of $\widetilde N$) with coefficients $\mathbf a$. The key modelling trick
  is that we treat these coefficients as **random**:

$$
\mathbf a \sim \mathcal N(\mathbf 0,\ \tau^2 \mathbf I).
$$

Here $\tau^2$ is a single number, the **variance of the wiggles**. This is the heart
of the whole test:

- If $\tau^2$ is **large**, the coefficients $\mathbf a$ can be big, so the curve
  wiggles a lot.
- If $\tau^2 = 0$, then $\mathbf a = \mathbf 0$ with probability one: every wiggle is
  switched off and the curve is **exactly the straight line**.

So "the curve is linear" is the same statement as "$\tau^2 = 0$". A roughness penalty
on the curve corresponds to shrinking $\mathbf a$, and $\tau^2$ is the inverse of the
smoothing parameter (small $\tau^2$ = heavy smoothing toward a line). The variable
$t$ here is a single smooth component; the full model can have several, plus ordinary
fixed effects $\boldsymbol\beta$ and subject-level random effects $\mathbf b$.

## 3. The hypothesis (and why it is tricky)

We want to test

$$
H_0:\ \tau^2 = 0 \quad(\text{straight line})
\qquad\text{versus}\qquad
H_1:\ \tau^2 > 0 \quad(\text{genuinely curved}).
$$

Notice the alternative is **one-sided**: a variance cannot be negative, so $\tau^2$
lives in $[0,\infty)$ and under $H_0$ it sits **on the edge** (the boundary) of that
range.

**Why the boundary matters.** Most of the tests you have seen (the Wald test, the
likelihood-ratio test) rely on the true parameter being in the *interior* of the
parameter space, so the estimate can wiggle a little to either side. Here it cannot:
the estimate $\hat\tau^2$ can never go below $0$. A useful mental picture: imagine
estimating how unfair a coin is, when you know in advance it is either fair or biased
toward heads (never toward tails). Even with a perfectly fair coin, your estimate of
the bias will land exactly at "fair" about half the time and "somewhat biased toward
heads" the other half. That asymmetry changes the reference distribution you compare
your statistic against. The same thing happens for $\hat\tau^2$ at $0$, and it is why
the usual "compare to $\chi^2_1$" rule is **wrong** here (Self and Liang, 1987). We
will deal with this properly in §6.

## 4. Three classic tests, and why we pick the score test

You have likely met three ways to test a parameter:

1. **Wald test:** fit the big model, look at $\hat\tau^2$ and its standard error. Bad
   idea here, because $\hat\tau^2$ at the boundary has a non-normal sampling
   distribution and its standard error is unreliable.
2. **Likelihood-ratio test (LRT):** fit both the small (linear) and big (curved)
   models, compare their fits. Works, but you must fit the curved model, which means
   estimating a variance that is genuinely near zero, and that optimisation is slow
   and finicky.
3. **Score test (also called Lagrange-multiplier test):** fit **only the small
   (null) model**, then ask "if I nudged $\tau^2$ up from $0$, would the likelihood
   want to increase?" The "score" is the slope of the log-likelihood at $\tau^2=0$. A
   large positive slope is evidence against $H_0$.

We use the **score test**. It only needs the cheap null fit, and it avoids estimating
a near-zero variance. This is the approach of Lin (1997) and, for spline models
specifically, Zhang and Lin (2003).

**One trap to flag.** You might think we can just take the code's objective `nll`
(negative log-likelihood) and differentiate it with respect to the smoothing
parameter. Internally, though, the code stores the wiggle variance as
$\log\tau^2$, not $\tau^2$. The boundary $\tau^2=0$ becomes $\log\tau^2=-\infty$, and
by the chain rule the slope in that coordinate is
$\partial/\partial(\log\tau^2)=\tau^2\,\partial/\partial\tau^2$, which is **zero** at
the boundary. So a naive automatic derivative would hand back $0$ and tell us
nothing. We instead compute the slope by hand, in the natural $\tau^2$ coordinate.
That hand calculation is the rest of this document.

## 5. Deriving the score statistic

We now build the statistic step by step. Take it slowly; each step is small.

### 5.1 The model as one big normal

Stack all observations into a vector $\mathbf y$. Because every random piece (the
subject effects $\mathbf b$, the wiggles $\mathbf a$, the noise) is normal, $\mathbf y$
is one big multivariate normal,

$$
\mathbf y \sim \mathcal N(\mathbf X\boldsymbol\beta,\ \mathbf V(\gamma)),
\qquad \gamma \equiv \tau^2,
$$

where $\mathbf X\boldsymbol\beta$ holds all the straight-line and fixed parts, and
$\mathbf V(\gamma)$ is the covariance matrix. Switching on the wiggles adds a term to
the covariance:

$$
\mathbf V(\gamma) = \mathbf V_0 + \gamma\,\mathbf K,\qquad \mathbf K = \mathbf U\mathbf U^\top,
\qquad \frac{\partial \mathbf V}{\partial \gamma} = \mathbf K .
$$

- $\mathbf U$ is the matrix of wiggle-basis columns for the component being tested
  (called $\widetilde N$ above, stacked over subjects).
- $\mathbf V_0$ is the covariance **under $H_0$**: it contains the noise, the subject
  random effects, and any *other* smooth components, but **not** the wiggles of the
  component we are testing.
- $\mathbf K = \mathbf U\mathbf U^\top$ is how one unit of wiggle-variance enters the
  covariance. It is "low rank": $\mathbf U$ has only $K$ columns ($K\ll n$), which is
  what makes everything cheap later.

### 5.2 REML and the projection matrix $\mathbf P$

To estimate variances we use **REML** (restricted maximum likelihood). The one-line
reason REML exists: ordinary maximum likelihood underestimates variances because it
does not "pay" for having estimated the regression coefficients $\boldsymbol\beta$
(the same reason we divide by $n-1$, not $n$, for a sample variance). REML corrects
this. Its log-likelihood, up to constants, is

$$
\ell_R(\gamma)=-\tfrac12\log|\mathbf V|-\tfrac12\log|\mathbf X^\top\mathbf V^{-1}\mathbf X|-\tfrac12\,\mathbf y^\top\mathbf P\mathbf y,
$$

where the star of the show is the **projection matrix**

$$
\mathbf P=\mathbf V^{-1}-\mathbf V^{-1}\mathbf X\mathbf M^{-1}\mathbf X^\top\mathbf V^{-1},
\qquad \mathbf M=\mathbf X^\top\mathbf V^{-1}\mathbf X .
$$

**What $\mathbf P$ does, intuitively.** $\mathbf P\mathbf y$ is the generalised-least-squares
residual (the data with the fitted fixed-effects part removed), pre-weighted by
$\mathbf V^{-1}$. The crucial fact is that $\mathbf P$ "deletes" anything in the direction
of the fixed effects. We will use these standard properties (you can take them as
given, or verify them as an exercise):

- (i) $\mathbf P\mathbf X=\mathbf 0$  (it kills the fixed-effect directions),
- (ii) $\mathbf P\mathbf V\mathbf P=\mathbf P$  (it is a projection, idempotent in the
  $\mathbf V$ metric),
- (iii) $\mathbf V^{-1}-\mathbf P=\mathbf V^{-1}\mathbf X\mathbf M^{-1}\mathbf X^\top\mathbf V^{-1}$,
- (iv) $\partial\mathbf V^{-1}/\partial\gamma=-\mathbf V^{-1}\mathbf K\mathbf V^{-1}$  (the
  derivative of a matrix inverse),
- (v) $\partial\mathbf P/\partial\gamma=-\mathbf P\mathbf K\mathbf P$.

### 5.3 The slope of the log-likelihood (the score)

The score is $\partial\ell_R/\partial\gamma$ at $\gamma=0$. We differentiate the three
terms of $\ell_R$ one at a time. Two small calculus facts are all we need:
$\partial\log|\mathbf V|/\partial\gamma=\operatorname{tr}(\mathbf V^{-1}\partial\mathbf V/\partial\gamma)$,
and the trace is cyclic ($\operatorname{tr}(ABC)=\operatorname{tr}(BCA)$).

**Term 1, $-\tfrac12\log|\mathbf V|$:**
$$
\frac{\partial}{\partial\gamma}\Big(-\tfrac12\log|\mathbf V|\Big)=-\tfrac12\operatorname{tr}(\mathbf V^{-1}\mathbf K).
$$

**Term 2, $-\tfrac12\log|\mathbf M|$:** with $\partial\mathbf M/\partial\gamma=-\mathbf X^\top\mathbf V^{-1}\mathbf K\mathbf V^{-1}\mathbf X$ (from (iv)),
$$
\frac{\partial}{\partial\gamma}\Big(-\tfrac12\log|\mathbf M|\Big)
=\tfrac12\operatorname{tr}\!\big(\mathbf V^{-1}\mathbf X\mathbf M^{-1}\mathbf X^\top\mathbf V^{-1}\mathbf K\big)
\overset{\text{(iii)}}{=}\tfrac12\operatorname{tr}\!\big((\mathbf V^{-1}-\mathbf P)\mathbf K\big).
$$

**Terms 1 and 2 together.** The two $\mathbf V^{-1}$ traces cancel, and what is left
is the same trace but with $\mathbf P$ in place of $\mathbf V^{-1}$. This swap is exactly
REML's bias correction showing up:
$$
-\tfrac12\operatorname{tr}(\mathbf V^{-1}\mathbf K)+\tfrac12\operatorname{tr}\!\big((\mathbf V^{-1}-\mathbf P)\mathbf K\big)
=-\tfrac12\operatorname{tr}(\mathbf P\mathbf K).
$$

**Term 3, $-\tfrac12\mathbf y^\top\mathbf P\mathbf y$:** using (v),
$$
\frac{\partial}{\partial\gamma}\Big(-\tfrac12\mathbf y^\top\mathbf P\mathbf y\Big)
=\tfrac12\,\mathbf y^\top\mathbf P\mathbf K\mathbf P\mathbf y.
$$

**Add them up** and set $\gamma=0$ (so $\mathbf P\to\mathbf P_0$, the projection built
from the null covariance $\mathbf V_0$). The score is

$$
\boxed{\;U=\tfrac12\Big(\mathbf y^\top\mathbf P_0\mathbf K\mathbf P_0\mathbf y-\operatorname{tr}(\mathbf P_0\mathbf K)\Big)\;}
$$

Read this as: a **quadratic form in the data** (the first term, which is large when
the residuals line up with the wiggle directions) minus a **bookkeeping constant**
(the trace, which is its expected size under $H_0$). Big positive $U$ = the wiggles
explain more than chance = evidence the curve is really curved.

### 5.4 Simplifying with the low-rank structure

Because $\mathbf K=\mathbf U\mathbf U^\top$, define the small $K\times K$ matrix
$\mathbf W=\mathbf U^\top\mathbf P_0\mathbf U$ and the residual direction
$\mathbf w=\mathbf P_0\mathbf y$. Then

$$
S^+\equiv\tfrac12\,\mathbf y^\top\mathbf P_0\mathbf K\mathbf P_0\mathbf y=\tfrac12\lVert\mathbf U^\top\mathbf w\rVert^2,
\qquad
e\equiv\tfrac12\operatorname{tr}(\mathbf P_0\mathbf K)=\tfrac12\operatorname{tr}(\mathbf W),
\qquad U=S^+-e.
$$

So $S^+$ is just the squared length of a $K$-vector, and $e$ is a trace of a small
matrix. Everything is computed through $\mathbf U$ (which has few columns), never the
full $n\times n$ matrices.

### 5.5 What distribution does $S^+$ follow under $H_0$?

This is the elegant part. Let $\mathbf g=\mathbf U^\top\mathbf P_0\mathbf y$, so
$S^+=\tfrac12\lVert\mathbf g\rVert^2$. Since $\mathbf y$ is normal, $\mathbf g$ (a linear
function of $\mathbf y$) is normal too. Its mean and covariance, using properties (i)
and (ii):

$$
\mathbb E[\mathbf g]=\mathbf U^\top\mathbf P_0\,\mathbf X\boldsymbol\beta=\mathbf 0
\quad(\text{by } \mathbf P_0\mathbf X=0),
\qquad
\operatorname{Cov}(\mathbf g)=\mathbf U^\top\mathbf P_0\mathbf V_0\mathbf P_0\mathbf U=\mathbf U^\top\mathbf P_0\mathbf U=\mathbf W .
$$

So $\mathbf g\sim\mathcal N(\mathbf 0,\mathbf W)$. Now rotate to the eigenvectors of
$\mathbf W$ (write $\mathbf W=\sum_j\lambda_j\mathbf v_j\mathbf v_j^\top$). In those
coordinates $\mathbf g$ has independent components with variances $\lambda_j$, so

$$
\boxed{\;S^+=\tfrac12\lVert\mathbf g\rVert^2=\tfrac12\sum_j\lambda_j\,\chi^2_{1,j}\;}
$$

a **weighted sum of independent $\chi^2_1$ variables**, with weights equal to the
eigenvalues of $\mathbf W$. Intuitively, each eigenvalue is one "independent direction
of wiggle" the data could move in, and $\lambda_j$ measures how much room that
direction has under the null.

From this one line, the mean and variance drop out for free:

$$
\mathbb E[S^+]=\tfrac12\sum_j\lambda_j=\tfrac12\operatorname{tr}\mathbf W=e,
\qquad
\operatorname{Var}(S^+)=\tfrac14\sum_j\lambda_j^2\cdot 2=\tfrac12\lVert\mathbf W\rVert_F^2\equiv\mathcal I,
$$

(using $\operatorname{Var}(\chi^2_1)=2$). The quantity $\mathcal I$ is the **information**:
it is the variance of the score under the null, and it is what we divide by to
standardise. Note $S^+$ and $e$ have the same mean, so the score $U=S^+-e$ is centred
at $0$ under $H_0$, exactly as a score should be.

## 6. Turning the statistic into a p-value

We have a statistic $S^+=\tfrac12\sum_j\lambda_j\chi^2_{1,j}$, a weighted sum of
chi-squares, which has no simple closed form. We approximate it by a single **scaled
chi-square** $a\,\chi^2_b$ (Zhang and Lin, 2003), choosing $a$ and $b$ so the mean and
variance match:

$$
\underbrace{a\,b}_{\text{mean}}=e,\qquad \underbrace{2a^2b}_{\text{variance}}=\mathcal I
\quad\Longrightarrow\quad
a=\frac{\mathcal I}{2e}=\frac{\lVert\mathbf W\rVert_F^2}{2\operatorname{tr}\mathbf W},\qquad
b=\frac{2e^2}{\mathcal I}=\frac{(\operatorname{tr}\mathbf W)^2}{\lVert\mathbf W\rVert_F^2}.
$$

Then $p=\Pr\{\chi^2_b > S^+/a\}$. The number $b$ is the **effective degrees of
freedom** of the wiggle: if all eigenvalues are equal it is the number of basis
functions, and if one eigenvalue dominates it is close to $1$. This is the
Satterthwaite idea you may have seen for unequal-variance t-tests, reused here; it
adapts to how flexible the smooth really is.

A cruder alternative is the one-sided normal / Self and Liang (1987) boundary mixture
$p=\tfrac12\Pr\{\chi^2_1\ge\max(U,0)^2/\mathcal I\}$ (Lin's 1997 individual test). It
collapses the wiggle to a single degree of freedom, so it is poorly calibrated at small
samples (it over-rejects when the score is skewed); we do not use it.

**External check.** The classic R package `RLRsim` (`exactRLRT`) implements the exact
restricted likelihood-ratio version of this test (Crainiceanu and Ruppert, 2004;
Crainiceanu, Ruppert, Claeskens and Wand, 2005). We use it as the independent baseline
when validating.

## 7. How the computation actually runs

Two practical points make this fast and stable.

**Subjects are independent**, so the big covariance $\mathbf V_0$ is block-diagonal,
one block $\mathbf V_{0,i}$ per subject. Every inverse becomes a small per-subject
Cholesky factorisation (size $n_{\max}\times n_{\max}$), never an $n\times n$ solve.
The pieces are assembled by summing over subjects $i$:

$$
\mathbf M=\sum_i\mathbf X_i^\top\mathbf V_{0,i}^{-1}\mathbf X_i,\quad
\hat{\boldsymbol\beta}_0=\mathbf M^{-1}\!\sum_i\mathbf X_i^\top\mathbf V_{0,i}^{-1}\mathbf y_i,\quad
\mathbf w_i=\mathbf V_{0,i}^{-1}(\mathbf y_i-\mathbf X_i\hat{\boldsymbol\beta}_0),
$$
$$
\mathbf g=\sum_i\mathbf U_i^\top\mathbf w_i,\quad S^+=\tfrac12\lVert\mathbf g\rVert^2,\qquad
\mathbf W=\sum_i\mathbf U_i^\top(\mathbf P_0\mathbf U)_i,\quad e=\tfrac12\operatorname{tr}\mathbf W,\ \ \mathcal I=\tfrac12\lVert\mathbf W\rVert_F^2 .
$$

These sums are the `einsum` lines in `_linearity_score`.

**Fitting the null.** The null model keeps everything except the wiggles of the
tested component: all fixed effects (including that component's straight-line slope),
the subject random effects, and any other smooth components. In code:

- If there are other smooth components, refit a `LinearMixedModelSmoothing` with the
  tested component's basis removed.
- If it was the only smooth, the null is just an ordinary linear mixed model, so we
  refit `LinearMixedModel`.

We then read off $\mathbf D$ (subject-effect covariance), $\sigma^2\mathbf R$ (noise),
and the other components' variances, and assemble $\mathbf V_0$ using the **tested**
model's own grouped designs so the subject and padding bookkeeping line up exactly.

## 8. Using it

```python
ss = SmoothingSplineMixed(...).fit(...)

ss.linearity_test()        # test every smooth component
ss.linearity_test(s=0)     # just component 0
```

Each call returns a dictionary (or a list of them, one per component) with the
component id (`spline_id`), the `statistic`, the `df` and `scale_a` of the scaled
chi-square reference, the `p_value`, and a `reject` flag. A small p-value means: a
straight line is **not** enough, the curve is genuinely curved.

## 9. How we know it works

The right way to trust a test is to check its **size** and **power**:

- **Size (type I error):** simulate data where the curve really is a straight line.
  A good test rejects $H_0$ only about $\alpha$ (say 5%) of the time, no more.
- **Power:** simulate data with more and more curvature. A good test rejects more and
  more often as the curvature grows.

Plotting the rejection rate against the truth, and comparing our p-values to
`exactRLRT`, is how we validate the implementation (this mirrors the existing
`hyptest_size_power` experiment).

## 10. References and further reading

- Self, S.G. and Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood
  estimators and likelihood ratio tests under nonstandard conditions. *JASA* 82,
  605-610. (The boundary problem and the 50:50 mixture.)
- Lin, X. (1997). Variance component testing in generalised linear models with random
  effects. *Biometrika* 84, 309-326. (The score test for a variance component.)
- Zhang, D. and Lin, X. (2003). Hypothesis testing in semiparametric additive mixed
  models. *Biostatistics* 4, 57-74. (The score test specialised to splines, with the
  Satterthwaite reference.)
- Crainiceanu, C. and Ruppert, D. (2004). Likelihood ratio tests in linear mixed
  models with one variance component. *JRSS-B* 66, 165-185.
- Crainiceanu, C., Ruppert, D., Claeskens, G. and Wand, M.P. (2005). Exact likelihood
  ratio tests for penalised splines. *Biometrika* 92, 91-103.
- For the underlying spline-as-mixed-model representation, see `methods.md` in this
  folder.
