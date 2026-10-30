import math
import warnings
import torch
from .linear_mixed_model import FitLog, LinearMixedModel
def _checktensor(x, dtype = torch.float64):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)

class LinearMixedModelSmoothing(LinearMixedModel):
    def __init__(self, device = 'cpu', y=None, X=None, Z=None, id=None, 
                 spline_inputs=None,
                 corr_structure=None, rand_D_type="unstructured", rand_D_estimate="sample"):
        if spline_inputs is None:
            raise ValueError("spline_inputs must be provided for LinearMixedModelSmoothing.")
        super().__init__(device=device, y=_checktensor(y), X=_checktensor(X), Z=_checktensor(Z), id=id,
                         corr_structure=corr_structure, rand_D_type=rand_D_type, rand_D_estimate=rand_D_estimate)
        self.NB_list = [_checktensor(spline).to(self.device) for spline in spline_inputs] 
        self.len_spline = len(self.NB_list)
        self.rank_X = torch.linalg.matrix_rank(self.X_)

        mask_3d = self.mask.unsqueeze(-1)

        # Global smooth design U = [NB_1|…|NB_S]; the smooth is one shared
        # random vector per component, handled via Woodbury in the REML loss.
        self.K_list    = [int(NB.shape[2]) for NB in self.NB_list]
        self.K_tot     = int(sum(self.K_list))
        self._K_repeats = torch.tensor(self.K_list, device=self.device)
        self.U_grouped = torch.cat([NB * mask_3d for NB in self.NB_list],
                                   dim=2)

        self._woodbury = self._corr_kind() == "iid" and self.pz < self.nmax
        if self._woodbury:
            A = torch.cat([self.masked_X, self.masked_y.unsqueeze(-1), self.U_grouped], dim=2)
            if self.pz:
                Q, self._wb_R = torch.linalg.qr(self.masked_Z)
                self._wb_W = Q.transpose(-1, -2) @ A
                A -= Q @ self._wb_W
            A = A.reshape(-1, A.shape[-1])
            self._wb_base = A.transpose(0, 1) @ A
            self._wb_pad_logdet = (self.n_groups * self.nmax - self.mask.sum().item()) * math.log(1e6)

    # Relative residual structures. Diagonal uses an identified variance-ratio
    # representation because the common residual scale σ² is profiled out.
    def _corr_kind(self):
        kind = "iid" if self.corr_structure is None else self._structure_kind(self.corr_structure)
        if kind == "unstructured":
            raise ValueError(f"Unsupported correlation structure: {self.corr_structure}")
        return kind

    def _has_corr_params(self):
        kind = self._corr_kind()
        return kind in ("cs", "ar1", "exp") or (kind == "diag" and self.nmax > 1)

    def _project_params(self):
        with torch.no_grad():
            self.spline_params.data.clamp_(-20.0, 20.0)   # loss clamps log tau^2 here; beyond it the gradient is 0
            if self._has_corr_params():
                self.rho_raw.data.clamp_(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

    def resid_corr(self):
        """Relative residual correlation R; σ² is profiled separately. For kind="diag",
        R's diagonal has unit geometric mean, making σ² a pooled scale across positions."""
        return self._resid_corr_from_raw(getattr(self, "rho_raw", None))

    def _resid_corr_from_raw(self, rho_raw):
        """Relative residual covariance R built from its unconstrained parameters."""
        kind = self._corr_kind()
        if kind == "iid":
            return self.eye_nmax
        if kind == "diag":
            # nmax free log-variances, centered so no within-group position is a
            # privileged reference; the profiled scalar sigma² is their geometric mean.
            if self.nmax == 1:
                return self.eye_nmax
            raw = rho_raw.reshape(-1)
            log_var = raw - raw.mean()
            return torch.diag(torch.exp(log_var.clamp(-self.RHO_RAW_MAX,
                                                       self.RHO_RAW_MAX)))
        rho_raw = rho_raw.clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)
        if kind == "cs":
            # full admissible range (-1/(nmax-1), 1); a plain sigmoid reaches only rho > 0
            rho = self.cs_rho(rho_raw, self.nmax)
            J = torch.ones((self.nmax, self.nmax), device=self.device, dtype=self.y_.dtype)
            return rho * J + (1.0 - rho) * self.eye_nmax
        lag = self._lag(self.nmax, self.device, self.y_.dtype)
        if kind == "ar1":
            rho = torch.tanh(rho_raw)                       # (-1, 1); AR1 is PD throughout
            # integer exponent: exact for rho<0, and unlike |rho|^lag * sign(rho) it keeps
            # the right derivative at rho=0, which is the "fixed" init.
            return rho ** lag.long()
        # exponential: R[i,j] = exp(-φ|i-j|), φ>0 (AR1 with ρ=exp(-φ)∈(0,1))
        phi = torch.nn.functional.softplus(rho_raw)
        return torch.exp(-phi * lag)

    def _ols_residuals(self):
        """OLS residuals and per-group layout; shared by moment-rho and sample-D init."""
        dtype = self.y_.dtype
        beta_ols = torch.linalg.lstsq(self.X_, self.y_).solution
        r_ = (self.y_ - self.X_ @ beta_ols).reshape(-1)
        r_g = torch.zeros(self.n_groups, self.nmax, device=self.device, dtype=dtype)
        for i, idx in enumerate(self._group_idx):
            l = idx.numel()
            r_g[i, :l] = r_[idx]
        r_g = r_g * self.mask
        return r_, r_g

    def _sample_D_init(self):
        """Method-of-moments estimate of D_param = D_true / sigma2 from OLS residuals.

        E[Z_i^T r_i r_i^T Z_i / n_i] ≈ D_true + sigma2 * Z_i^T Z_i / n_i, so dividing
        the group-averaged outer product by sigma2_ols gives a noisy but well-scaled
        start, orders of magnitude closer than a flat 0.1*I init at large pz.
        """
        r_, r_g = self._ols_residuals()
        n_obs = self.mask.sum().item()
        sigma2_ols = ((r_ ** 2).sum() / max(n_obs - self.X_.shape[1], 1)).clamp(min=1e-6)

        n_i = self.mask.sum(1)
        ZTr = (self.masked_Z.transpose(-1, -2) @
               r_g.unsqueeze(-1)).squeeze(-1)
        ZTr_scaled = ZTr / n_i.clamp(min=1).sqrt().unsqueeze(-1)
        D_acc = ZTr_scaled.T @ ZTr_scaled
        D_raw = D_acc / (self.n_groups * sigma2_ols)

        D_sym = 0.5 * (D_raw + D_raw.T)
        eigvals, eigvecs = torch.linalg.eigh(D_sym)
        eigvals = eigvals.clamp(min=1e-4)
        return eigvecs @ torch.diag(eigvals) @ eigvecs.T

    def _init_rho_raw(self, corr_init_method):
        """Starting values for rho_raw.

        diag: nmax centered log-variances from pooled OLS residuals by within-subject position
        (no privileged reference position; the profiled scalar sigma² is their geometric mean).

        Otherwise "moment": for compound symmetry, E[r_j r_k] / E[r_j^2] within groups (j≠k) equals
        sigma2*rho / (sigma2 + tr(D)), which is attenuated by the random-effects variance but always
        in (0, rho_true). Starting rho_raw below the true value is far safer than starting at
        rho_raw = 0, which tends to drift into the spurious rho→1 local minimum for large pz.
        The target rho is mapped through the inverse of each structure's own link: a shared logit
        is right only for cs (tanh(logit(p)) flips sign for p < 0.5, softplus(logit(p)) gives 1-p).
        """
        kind = self._corr_kind()
        if kind != "diag" and corr_init_method != "moment":
            return torch.zeros(1, device=self.device, dtype=self.y_.dtype)
        _, r_g = self._ols_residuals()
        if kind == "diag":
            var_by_position = (r_g.pow(2) * self.mask).sum(0) / self.mask.sum(0).clamp_min(1.0)
            log_var = torch.log(var_by_position.clamp_min(1e-8))
            return (log_var - log_var.mean()).clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

        sum_r  = r_g.sum(1)
        sum_r2 = (r_g ** 2).sum(1)
        n_i    = self.mask.sum(1)
        n_pairs = (n_i * (n_i - 1)).sum()
        n_obs   = self.mask.sum()
        if n_pairs < 1 or n_obs < 1:
            return torch.zeros(1, device=self.device, dtype=self.y_.dtype)

        mean_cross = ((sum_r ** 2 - sum_r2)[n_i > 1]).sum() / n_pairs
        mean_sq    = sum_r2.sum() / n_obs
        rho = (mean_cross / (mean_sq + 1e-12)).clamp(0.005, 0.95)
        if kind == "cs":
            raw = self.cs_rho_inverse(rho, self.nmax)
        elif kind == "ar1":
            raw = torch.atanh(rho.clamp(-0.999, 0.999))
        else:                                       # exp: rho = exp(-softplus(raw))
            phi = -torch.log(rho.clamp(1e-6, 1.0 - 1e-6))
            raw = torch.log(torch.expm1(phi))       # inverse softplus
        return raw.clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX).reshape(1)

    def initialize_rand_D(self, type="unstructured", estimate="sample",
                          corr_init_method="moment"):
        spline_param = torch.tensor([
            self.NB_list[i].pow(2).mean().sqrt().item()
            for i in range(self.len_spline)
        ], device=self.device, dtype=self.y_.dtype)
        rho_init = torch.log(spline_param ** 2).clamp(min=0.0)  # log(τ²), clamped so τ²≥1 (under-smoothed start)

        self.spline_params = torch.nn.Parameter(rho_init)  # log(τ²) per component

        if self._has_corr_params():
            self.rho_raw = torch.nn.Parameter(self._init_rho_raw(corr_init_method))

        kind = self._structure_kind(type)
        if self.pz == 0:
            self.D_params = torch.nn.Parameter(torch.zeros(0, device=self.device, dtype=self.y_.dtype))

        elif kind == "unstructured":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                L_hat = torch.linalg.cholesky(D_psd)
            else:
                L_hat = 0.1 * torch.eye(self.pz, device=self.device, dtype=self.y_.dtype)
            self.D_params = torch.nn.Parameter(
                L_hat[torch.tril_indices(self.pz, self.pz, 0)[0],
                      torch.tril_indices(self.pz, self.pz, 0)[1]]
            )
        elif kind == "diag":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                diag_init = torch.diag(D_psd).clamp(min=1e-4).sqrt()
                self.D_params = torch.nn.Parameter(diag_init)
            else:
                self.D_params = torch.nn.Parameter(
                    torch.ones(self.pz, device=self.device, dtype=self.y_.dtype)
                )
        elif kind == "iid":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                iid_init = torch.diag(D_psd).mean().clamp(min=1e-4).sqrt().reshape(1)
                self.D_params = torch.nn.Parameter(iid_init)
            else:
                self.D_params = torch.nn.Parameter(
                    torch.ones(1, device=self.device, dtype=self.y_.dtype)
                )

    def _woodbury_chol(self, D):
        """Cholesky of T_i = I + R_i D R_i' (pz x pz per group)."""
        R = self._wb_R
        T = torch.eye(self.pz, device=self.device, dtype=R.dtype) + R @ D @ R.transpose(-1, -2)
        return self._robust_cholesky(T)[0]

    def _woodbury_forms(self, D):
        """(log|B|, [X y U]' B^-1 [X y U]) for iid residuals. With Z_i = Q_i R_i,
        B_i^-1 = I - Q_i Q_i' + Q_i T_i^-1 Q_i' and |B_i| = |T_i|, T_i = I + R_i D R_i'."""
        if not self.pz:
            return self._wb_pad_logdet, self._wb_base
        L_T = self._woodbury_chol(D)
        logdetB = 2.0 * torch.log(torch.diagonal(L_T, dim1=-2, dim2=-1)).sum() + self._wb_pad_logdet
        W = self._wb_W
        return logdetB, self._wb_base + torch.einsum("gzi,gzj->ij", W, torch.cholesky_solve(W, L_T))

    def _reml_relative(self, D, spline_param, R):
        """Relative-scale REML pieces (σ² profiled out). R is the unit-diagonal residual
        correlation. Returns (logdetV, logdet_XtVinvX, rVinv_r, XtVinvX, beta_hat), all in
        relative units (true V = σ²·this). Shared by loss_fn_reml and the fixed-effect test."""
        p        = self.X_.shape[1]

        # symmetric bound is exp() overflow safety only; a min>0 would trap the gradient at 0.
        tau_sq  = torch.exp(spline_param.clamp(min=-20.0, max=20.0))
        lam     = torch.repeat_interleave(tau_sq, self._K_repeats)
        lam_inv = 1.0 / lam

        if self._woodbury:
            logdetB, A = self._woodbury_forms(D)
            A_XX, A_Xy, A_yy = A[:p, :p], A[:p, p:p + 1], A[p, p]
            A_UX, A_Uy, A_UU = A[p + 1:, :p], A[p + 1:, p:p + 1], A[p + 1:, p + 1:]
        else:
            G        = self.n_groups
            nm       = self.nmax
            Ktot     = self.K_tot
            mask_vec = self.mask.unsqueeze(-1)
            X_eff    = (self.X_grouped * mask_vec)
            y_eff    = (self.y_grouped * mask_vec.squeeze(-1))
            U_all    = self.U_grouped

            L, _ = self._robust_cholesky(self._group_cov(R, D))
            logdetB = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)))

            BiX = torch.cholesky_solve(X_eff, L)
            Biy = torch.cholesky_solve(y_eff.unsqueeze(-1), L)
            BiU = torch.cholesky_solve(U_all, L)

            # Per-group quadratic forms via one matmul over stacked [G*nm, .].
            Xm, ym, Um    = X_eff.reshape(G * nm, p), y_eff.reshape(G * nm, 1), U_all.reshape(G * nm, Ktot)
            BiXm, Biym, BiUm = BiX.reshape(G * nm, p), Biy.reshape(G * nm, 1), BiU.reshape(G * nm, Ktot)
            A_XX = Xm.transpose(0, 1) @ BiXm
            A_Xy = Xm.transpose(0, 1) @ Biym
            A_yy = (ym * Biym).sum()
            A_UX = Um.transpose(0, 1) @ BiXm
            A_Uy = Um.transpose(0, 1) @ Biym
            A_UU = Um.transpose(0, 1) @ BiUm

        M   = A_UU + torch.diag(lam_inv)
        M   = 0.5 * (M + M.transpose(0, 1))
        L_M, _ = self._robust_cholesky(M)
        logdetM = 2.0 * torch.sum(torch.log(torch.diagonal(L_M)))
        Minv_UX = torch.cholesky_solve(A_UX, L_M)
        Minv_Uy = torch.cholesky_solve(A_Uy, L_M)
        logdetV = logdetB + torch.sum(torch.log(lam)) + logdetM
        XtVinvX = A_XX - A_UX.transpose(0, 1) @ Minv_UX
        XtVinvy = A_Xy - A_UX.transpose(0, 1) @ Minv_Uy
        ytVinvy = A_yy - (A_Uy.transpose(0, 1) @ Minv_Uy).squeeze()

        beta_hat       = torch.linalg.solve(XtVinvX, XtVinvy)
        rVinv_r        = ytVinvy - (XtVinvy.transpose(0, 1) @ beta_hat).squeeze()
        logdet_XtVinvX = self._logdet_spd(XtVinvX)
        return logdetV, logdet_XtVinvX, rVinv_r, XtVinvX, beta_hat

    def loss_fn_reml(self, D, spline_param):
        """Profiled REML for V = B + UΛUᵀ: B = blockdiag(R + ZDZᵀ) per group,
        UΛUᵀ the global smooth (U = [NB_1|…|NB_S], Λ = diag(τ_d² over its coeffs)).
        Returns (neg_reml_loglik, sigma2_scalar)."""
        n_eff = int(self.mask.sum().item())
        p     = self.X_.shape[1]
        logdetV, logdet_XtVinvX, rVinv_r, _, beta_hat = self._reml_relative(
            D, spline_param, self.resid_corr())
        sigma2_scalar = rVinv_r / (n_eff - self.rank_X)
        neg_loglik = 0.5 * (logdetV + logdet_XtVinvX + (n_eff - p) * torch.log(sigma2_scalar))
        self._beta_cache   = beta_hat.detach()
        self._sigma2_cache = sigma2_scalar.detach()
        return neg_loglik, sigma2_scalar

    def loss_fn(self, D, spline_param):
        """Profiled ML (beta and σ² profiled out). Returns (neg_ml_loglik, sigma2_scalar)."""
        n_eff = int(self.mask.sum().item())
        logdetV, _, rVinv_r, _, beta_hat = self._reml_relative(D, spline_param, self.resid_corr())
        sigma2_scalar = rVinv_r / n_eff
        neg_loglik = 0.5 * (logdetV + n_eff * torch.log(sigma2_scalar))
        self._beta_cache   = beta_hat.detach()
        self._sigma2_cache = sigma2_scalar.detach()
        return neg_loglik, sigma2_scalar

    def reset_params(self, OLS=False, estimate="sample", corr_init_method="moment"):
        if OLS & (len(self.X_) > 0) and estimate == "sample":
            beta = torch.linalg.lstsq(self.X_, self.y_).solution.clone()
            residuals = self.y_ - self.X_ @ beta
            sigma2 = (residuals.T @ residuals) / (len(self.y_) - self.X_.shape[1])
        else:
            beta = torch.ones((self.X_.shape[1], 1), device=self.device)
            sigma2 = torch.tensor(.1, device=self.device)

        self.initialize_rand_D(type=self.rand_D_type, estimate=self.rand_D_estimate,
                               corr_init_method=corr_init_method)

        return beta, sigma2

    def update_iter(self, loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss,
                    grad_control_tol=1e-2):
        D = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
        opt.zero_grad()
        total_loss, _ = loss_function(D=D, spline_param=self.spline_params)
        total_loss.backward()
        conv_params = [self.spline_params, self.D_params]
        if self._has_corr_params():
            conv_params.append(self.rho_raw)
        grad_norm = sum(
            p.grad.norm().item() ** 2 for p in conv_params
            if p.grad is not None
        ) ** 0.5
        opt.step()
        self._project_params()

        with torch.no_grad():
            D = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
            current_spline_param = torch.exp(self.spline_params)
            current_D = D

            # Reuse the forward-pass cache; no second Cholesky needed.
            beta_  = self._beta_cache
            sigma2 = self._sigma2_cache

            current_sigma2 = sigma2.clone().detach()
            beta_ = beta_.clone().detach()

            eps = 1e-8
            beta_change  = (torch.max(torch.abs((beta_ - prev_beta) / (prev_beta.abs() + eps))).item()
                            if beta_.numel() > 0 else 0.0)
            D_change     = (torch.max(torch.abs((current_D.flatten() - prev_D.flatten()) / (prev_D.flatten().abs() + eps))).item()
                            if current_D.numel() > 0 else 0.0)
            sigma2_change = torch.max(torch.abs((current_sigma2 - prev_sigma2) / (prev_sigma2.abs() + eps))).item()

            current_loss = total_loss.item()
            rel_change   = abs((current_loss - prev_loss) / (prev_loss + 1e-12))

        if current_loss > prev_loss:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

        convg_criteria = 0
        if convg_check == "loss":
            if rel_change < tol and grad_norm < grad_control_tol:
                convg_criteria = 1
        elif convg_check == "all":
            if (rel_change < tol) and (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                convg_criteria = 1
        elif convg_check == "params":
            if (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                convg_criteria = 1
        else:
            raise ValueError("Invalid convergence check method.")
        return convg_criteria, current_loss, current_sigma2, current_D, current_spline_param, beta_, rel_change, beta_change, D_change, sigma2_change

    def _header_rows(self):
        rows = super()._header_rows()
        rows[-1] = ("residual", self._corr_kind())
        return rows + [("smooths", f"S = {self.len_spline}, K = {', '.join(map(str, self.K_list))}")]

    def fit_likelihood(self, optimizer=torch.optim.Adam, lr=1e-3, max_iter=1000, fit_type="REML",
                       tol=1e-6, warm_start=True, verbose=False, convg_check="loss",
                       max_iter_inner=10, tau_lr_mult=5.0, grad_control_tol=1e-2,
                       corr_init_method="moment", plateau_patience=5, init=None):
        """
        fit_type         : "REML" (default) or "ML"; both profile beta and sigma^2 out analytically.
        init             : dict with D_params, spline_params (log tau^2) and optionally rho_raw,
                           used as the starting values in place of the default initialization.
        tau_lr_mult      : lr multiplier for spline_params vs D_params in Adam.
        corr_init_method : "fixed" (rho_raw=0) or "moment" (default; data-driven
                           init via within-group OLS-residual correlations; avoids the
                           rho→1 / sigma2→large spurious local minimum for large pz).
        plateau_patience : (LBFGS, convg_check="loss") also converge once rel_loss < tol
                           holds for this many consecutive epochs, regardless of grad_norm.
                           Catches the case where the loss has genuinely converged but
                           gradient noise keeps grad_norm from ever clearing grad_control_tol.
        """
        log = FitLog(verbose)
        beta, sigma2 = self.reset_params(OLS=warm_start, estimate=self.rand_D_estimate,
                                         corr_init_method=corr_init_method)
        if init is not None:
            with torch.no_grad():
                for name, val in init.items():
                    getattr(self, name).copy_(val)

        self.fit_type = fit_type
        if log.on:
            self._log_header(log, fit_type, beta, warm_start)

        if not (len(self.y_) == len(self.X_) == len(self.Z_)):
            raise ValueError("Incompatible shapes: y, X, and Z must have the same length.")

        loss_function = self.loss_fn if fit_type.lower() == "ml" else self.loss_fn_reml

        prev_beta         = beta.clone().detach()
        D                 = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
        prev_D            = D.detach()
        prev_spline_param = self.spline_params.detach() ** 2
        prev_sigma2       = sigma2.clone().detach()
        prev_loss         = float("inf")

        if optimizer != torch.optim.LBFGS:
            log.table(f"AdamW: lr={lr:g} (x{tau_lr_mult:g} for tau), {max_iter} steps, "
                      f"tol={tol:g} on {convg_check}", "step", "rel_loss")
            # τ gets a higher lr (faster convergence) and lower β₂ (quicker forgetting).
            opt_groups = [
                {'params': [self.spline_params], 'lr': lr * tau_lr_mult, 'betas': (0.9, 0.99)},
                {'params': [self.D_params],      'lr': lr,               'betas': (0.9, 0.999)},
            ]
            if self._has_corr_params():
                opt_groups.append({'params': [self.rho_raw], 'lr': lr * 0.1, 'betas': (0.9, 0.999)})
            opt = torch.optim.AdamW(opt_groups, weight_decay=0.0)

            step = 0
            convg = 0
            for step in range(max_iter):
                convg_criteria, current_loss, current_sigma2, current_D, current_spline_param, current_beta, rel_change, _, _, _ = self.update_iter(
                    loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss,
                    grad_control_tol=grad_control_tol,
                )
                log.row(step, current_loss, rel_change)
                if convg_criteria == 1:
                    convg = 1
                    break
                prev_spline_param = current_spline_param
                prev_loss         = current_loss
                prev_beta         = current_beta.clone().detach()
                prev_D            = current_D.detach()
                prev_sigma2       = current_sigma2.detach()

            # The Adam path profiles beta out analytically (REML and ML); expose the cached value.
            self.beta_ = current_beta

            log.done(convg, step, current_loss)
            if convg == 0:
                warnings.warn("LinearMixedModelSmoothing: Adam did not converge in "
                              f"{max_iter} steps.")

        else:
            if max_iter_inner is None:
                max_iter_inner = 10
            max_iter_outer = max_iter // max_iter_inner
            log.table(f"LBFGS: lr={lr:g}, {max_iter_outer} epochs x {max_iter_inner} inner steps, "
                      f"tol={tol:g} on {convg_check}", "epoch", "rel_loss", "rel_param", "grad_norm")

            # Joint LBFGS over D_params + spline_params: τ sees the same second-order
            # curvature as D, not just 1 Adam step per outer epoch.
            lbfgs_params = []
            if len(self.D_params) > 0:
                lbfgs_params.append(self.D_params)
            lbfgs_params.append(self.spline_params)
            if self._has_corr_params():
                lbfgs_params.append(self.rho_raw)

            opt_joint = torch.optim.LBFGS(
                lbfgs_params, lr=lr, max_iter=max_iter_inner,
                history_size=10, line_search_fn="strong_wolfe"
            )

            evaluated = [False]

            def closure_joint():
                opt_joint.zero_grad()
                D_ = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
                try:
                    loss, _ = loss_function(D=D_, spline_param=self.spline_params)
                except torch.linalg.LinAlgError:
                    if not evaluated[0]:
                        raise                       # the starting point itself is invalid
                    loss = torch.tensor(float("nan"), dtype=torch.float64, device=self.device)
                if torch.isfinite(loss):
                    loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(q.grad).all() for q in lbfgs_params):
                    # failed or non-finite trial point: report a very bad finite value so the line search backs off
                    for q in lbfgs_params:
                        q.grad = torch.zeros_like(q)
                    return loss.detach().new_tensor(1e12)
                evaluated[0] = True
                return loss

            convg = 0
            epoch = 0
            plateau = 0
            for epoch in range(max_iter_outer):
                opt_joint.step(closure_joint)
                self._project_params()

                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in lbfgs_params
                    if p.grad is not None
                ) ** 0.5

                with torch.no_grad():
                    # LBFGS.step() returns the closure's pre-step loss; recompute here
                    # so current_loss/beta_/sigma2 reflect the post-step, post-projection state.
                    D_           = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
                    loss_post, _ = loss_function(D=D_, spline_param=self.spline_params)
                    self.beta_   = self._beta_cache
                    self.sigma2  = self._sigma2_cache
                    current_loss          = loss_post.item()
                    current_sigma2        = self.sigma2.clone().detach()
                    current_D             = D_.clone().detach()
                    current_spline_param  = torch.exp(self.spline_params.clone().detach())

                    rel_loss         = abs((current_loss - prev_loss) / (prev_loss + 1e-12))
                    rel_param_beta   = (torch.max(torch.abs((self.beta_ - prev_beta) / (prev_beta.abs() + 1e-12))).item()
                                        if self.beta_.numel() > 0 else 0.0)
                    rel_param_sigma2 = torch.max(torch.abs((current_sigma2.flatten() - prev_sigma2.flatten()) / (prev_sigma2.flatten().abs() + 1e-12))).item()
                    rel_param_spline = torch.max(torch.abs((current_spline_param.flatten() - prev_spline_param.flatten()) / (prev_spline_param.flatten().abs() + 1e-12))).item()
                    rel_param        = max(rel_param_beta, rel_param_sigma2, rel_param_spline)

                prev_loss         = current_loss
                prev_beta         = self.beta_.detach()
                prev_D            = current_D.detach()
                prev_sigma2       = current_sigma2.detach()
                prev_spline_param = current_spline_param.detach()
                log.row(epoch, current_loss, rel_loss, rel_param, grad_norm)

                if convg_check == "loss":
                    plateau = plateau + 1 if rel_loss < tol else 0
                    if (rel_loss < tol and grad_norm < grad_control_tol) or plateau >= plateau_patience:
                        convg = 1
                elif convg_check == "all":
                    if rel_loss < tol and rel_param < tol:
                        convg = 1
                elif convg_check == "params":
                    if rel_param < tol:
                        convg = 1
                elif convg_check == "grad":
                    if grad_norm < tol:
                        convg = 1
                else:
                    raise ValueError("Invalid convergence check method.")

                if convg == 1:
                    break

            st = opt_joint.state[opt_joint._params[0]]               # PyTorch's cumulative LBFGS counters
            self.n_epochs, self.n_lbfgs_steps, self.n_func_evals = epoch + 1, int(st.get('n_iter', 0)), int(st.get('func_evals', 0))
            log.done(convg, epoch, current_loss)
            if convg == 0:
                warnings.warn(f"LinearMixedModelSmoothing: LBFGS did not converge in "
                              f"{max_iter_outer} epochs (rel_loss={rel_loss:.2e}, grad_norm={grad_norm:.2e}).")

        self.final_loss   = current_loss
        self.convg        = convg
        self.sigma2       = current_sigma2.clone().detach()
        self.resid_cov    = self.sigma2 * self.resid_corr().detach()
        self.beta         = self.beta_.clone().detach()
        self.D            = current_D.clone().detach() * self.sigma2
        self.tuning_params = current_spline_param.clone().detach() * self.sigma2
        self.n_iter       = (epoch + 1) * max_iter_inner if optimizer == torch.optim.LBFGS else step
        return self

    # --- Fixed-effect hypothesis testing (Satterthwaite; σ² profiled, in θ as log σ²) ---
    # Reuses the inherited fixed_t_test / _contrast_test; only the
    # parametrization hooks below are smoothing-specific. theta stacks
    # [D_params, spline_params, (rho_raw), log σ²]; C = σ²(X'V⁻¹X)⁻¹ in relative units.

    def _theta_hat(self):
        parts = [self.D_params.detach().reshape(-1), self.spline_params.detach().reshape(-1)]
        if self._has_corr_params():
            parts.append(self.rho_raw.detach().reshape(-1))
        parts.append(torch.log(self.sigma2.to(self.device).reshape(1)))
        return torch.cat(parts)

    def _split_theta(self, theta):
        n_d, n_s = self.D_params.numel(), self.spline_params.numel()
        n_r = self.rho_raw.numel() if self._has_corr_params() else 0
        d_params = theta[:n_d]
        s_params = theta[n_d:n_d + n_s]
        rho_raw  = theta[n_d + n_s:n_d + n_s + n_r] if n_r else None
        log_s2   = theta[-1]
        D = self.param_to_sqmatrix(d_params, n=self.pz, correlation_structure=self.rand_D_type)
        R = self._resid_corr_from_raw(rho_raw) if n_r else self.eye_nmax
        return D, s_params, R, log_s2

    def nll(self, theta):
        n_eff = int(self.mask.sum().item())
        D, s_params, R, log_s2 = self._split_theta(theta)
        logdetV, logdet_XtVinvX, rVinv_r, _, _ = self._reml_relative(D, s_params, R)
        s2 = torch.exp(log_s2)
        if getattr(self, "fit_type", "REML").lower() == "ml":
            return 0.5 * (n_eff * log_s2 + rVinv_r / s2 + logdetV)
        return 0.5 * ((n_eff - self.rank_X) * log_s2 + rVinv_r / s2 + logdetV + logdet_XtVinvX)

    def C_theta(self, theta):
        D, s_params, R, log_s2 = self._split_theta(theta)
        _, _, _, XtVinvX, _ = self._reml_relative(D, s_params, R)
        return torch.exp(log_s2) * torch.linalg.inv(XtVinvX)

    # --- Linearity test for a smooth component (variance-component score test) ---
    # H0: tau_c^2 = 0 (component c is linear) vs H1: tau_c^2 > 0 (nonlinear).
    # Boundary score test (Lin 1997; Zhang & Lin 2003), in closed form in the natural
    # tau^2 scale: the internal log(tau^2) score is identically zero at the boundary.
    # Only the null fit is needed. See docs/linearity_test.md.

    def _assemble_null_chol(self, s, y=None, null_fit_kwargs=None):
        """Fit the H0 (tau_s^2 = 0) model and return the per-group Cholesky of V0, the absolute
        marginal covariance with component s's smooth dropped but its linear column retained
        (it lives in X_). V0 is built from the tested model's own grouped Z / NB so group/mask
        alignment is exact."""
        y = self.y_ if y is None else _checktensor(y).to(self.device).view(-1, 1)
        nfk = dict(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200,
                   verbose=False, convg_check="loss")
        others = [j for j in range(self.len_spline) if j != s]
        if others and hasattr(self, "D_params"):
            nfk["init"] = dict(D_params=self.D_params.detach(),
                               spline_params=self.spline_params.detach()[others])
            if self._has_corr_params():
                nfk["init"]["rho_raw"] = self.rho_raw.detach()
        if null_fit_kwargs:
            nfk.update(null_fit_kwargs)

        if others:
            null = LinearMixedModelSmoothing(
                device=self.device, y=y, X=self.X_, Z=self.Z_, id=self.id_,
                spline_inputs=[self.NB_list[j] for j in others],
                corr_structure=self.corr_structure, rand_D_type=self.rand_D_type,
                rand_D_estimate=self.rand_D_estimate)
            null.fit_likelihood(**nfk)
            D_abs    = null.D
            resid    = (null.sigma2 * null.resid_corr()).expand(self.n_groups, -1, -1).clone()
            for j, tau2 in zip(others, null.tuning_params.tolist()):
                k0  = sum(self.K_list[:j])
                U_j = self.U_grouped[:, :, k0:k0 + self.K_list[j]]
                resid.baddbmm_(U_j, U_j.transpose(-1, -2), alpha=tau2)      # += tau^2 NB NB'
        else:
            null = LinearMixedModel(
                y=y, X=self.X_, Z=self.Z_, id=self.id_,
                corr_structure=self.corr_structure, rand_D_type=self.rand_D_type)
            null.fit_likelihood(optimizer=nfk["optimizer"], lr=nfk["lr"],
                                max_iter=nfk["max_iter"], fit_type="REML",
                                convg_check=nfk["convg_check"], verbose=nfk["verbose"])
            D_abs  = null.D
            resid  = null.sigma2

        return self._robust_cholesky(self._group_cov(resid, D_abs))[0]

    def linearity_test(self, s=None, alpha=0.05, null_fit_kwargs=None):
        """Score test of H0: smooth component s is linear (tau_s^2 = 0) vs nonlinear,
        with the Zhang & Lin (2003) scaled chi-square reference.

        s=None tests every component and returns a list of result dicts.
        Returns a dict with spline_id, statistic, df, scale_a, p_value, reject.
        """
        if s is None:
            return [self.linearity_test(j, alpha=alpha, null_fit_kwargs=null_fit_kwargs)
                    for j in range(self.len_spline)]
        L_V0 = self._assemble_null_chol(s, null_fit_kwargs=null_fit_kwargs)
        S_plus, e, info = self._boundary_score(self.NB_list[s] * self.mask.unsqueeze(-1), L_V0)
        out = self._boundary_pvalue(S_plus, e, info)
        out.update(spline_id=s, reject=bool(out["p_value"] < alpha))
        return out
