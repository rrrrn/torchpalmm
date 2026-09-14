import warnings
import torch
from .linear_mixed_model import LinearMixedModel, _checktensor
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
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.y_ = _checktensor(y).to(self.device).view(-1, 1)  # N x 1
        self.X_ = _checktensor(X).to(self.device)              # N x px
        self.Z_ = _checktensor(Z).to(self.device)              # N x pz
        self.NB_list = [_checktensor(spline).to(self.device) for spline in spline_inputs] 
        self.len_spline = len(self.NB_list)  # S

        self.id_ = torch.as_tensor(id, device=self.device).view(-1)
        unique_ids = torch.unique(self.id_)
        self._group_idx = [(self.id_ == uid).nonzero(as_tuple=True)[0] for uid in unique_ids]
        self.lengths_   = [(self.id_ == uid).sum().item() for uid in unique_ids]
        
        self.n_groups = len(self.lengths_)
        self.nmax = max(self.lengths_)
        px = self.X_.shape[1]
        self.rank_X = torch.linalg.matrix_rank(self.X_)
        pz = self.Z_.shape[1]

        self.X_grouped = torch.zeros((self.n_groups, self.nmax, px), device=self.device, dtype=self.X_.dtype)
        self.Z_grouped = torch.zeros((self.n_groups, self.nmax, pz), device=self.device, dtype=self.Z_.dtype)
        self.y_grouped = torch.zeros((self.n_groups, self.nmax), device=self.device, dtype=self.y_.dtype)
        self.corr_structure = corr_structure
        self.rand_D_type = rand_D_type
        self.rand_D_estimate = rand_D_estimate

        for i, idx in enumerate(self._group_idx):
            l = idx.numel()
            self.X_grouped[i, :l] = self.X_[idx]
            self.Z_grouped[i, :l] = self.Z_[idx]
            self.y_grouped[i, :l] = self.y_[idx].squeeze()

        self.mask = torch.zeros((self.n_groups, self.nmax), device=self.device)
        for i, l in enumerate(self.lengths_):
            self.mask[i, :l] = 1.0

        mask = self.mask.unsqueeze(-1)
        self.masked_X = self.X_grouped * mask
        self.masked_Z = self.Z_grouped * mask
        self.masked_y = self.y_grouped * self.mask

        self.eye_nmax = torch.eye(self.nmax, device=self.device, dtype=self.y_grouped.dtype)
        self.pz = pz

        # NB_list[s] is [G, nmax, K_s] with K_s varying; NB @ NB.T is [G, nmax, nmax] either way.
        mask_3d = self.mask.unsqueeze(-1)                   # [G, nmax, 1]
        self.NB_NB = torch.stack([
            (NB * mask_3d) @ (NB * mask_3d).transpose(-1, -2)
            for NB in self.NB_list
        ], dim=0)                                           # [S, G, nmax, nmax]

        # Global smooth design U = [NB_1|…|NB_S]; the smooth is one shared
        # random vector per component, handled via Woodbury in the REML loss.
        self.K_list    = [int(NB.shape[2]) for NB in self.NB_list]
        self.K_tot     = int(sum(self.K_list))
        self._K_repeats = torch.tensor(self.K_list, device=self.device)
        self.U_grouped = torch.cat([NB * mask_3d for NB in self.NB_list],
                                   dim=2)                   # [G, nmax, K_tot]
    
    # Relative residual structures. Diagonal uses an identified variance-ratio
    # representation because the common residual scale σ² is profiled out.
    _CORR_ALIASES = {
        "iid": "iid", "identity": "iid",
        "independent": "diag", "diagonal": "diag",
        "compoundsymmetry": "cs", "cs": "cs", "compound_symmetry": "cs",
        "ar1": "ar1", "autoregressive": "ar1",
        "exponential": "exp", "exp": "exp",
    }

    def _corr_kind(self):
        if self.corr_structure is None:
            return "iid"
        k = self.corr_structure.lower()
        if k not in self._CORR_ALIASES:
            raise ValueError(f"Unsupported correlation structure: {self.corr_structure}")
        return self._CORR_ALIASES[k]

    def _has_corr_params(self):
        kind = self._corr_kind()
        return kind in ("cs", "ar1", "exp") or (kind == "diag" and self.nmax > 1)

    def _project_params(self):
        if self._has_corr_params():
            with torch.no_grad():
                self.rho_raw.data.clamp_(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

    def _lag_matrix(self):
        idx = torch.arange(self.nmax, device=self.device, dtype=self.y_.dtype)
        return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    def resid_corr(self):
        """Relative residual correlation R (unit diagonal); σ² is profiled separately."""
        return self._resid_corr_from_raw(getattr(self, "rho_raw", None))

    def _resid_corr_from_raw(self, rho_raw):
        """Relative residual covariance R built from its unconstrained parameters."""
        kind = self._corr_kind()
        if kind == "iid":
            return self.eye_nmax
        if kind == "diag":
            # The profiled scalar sigma² is the first-position residual variance;
            # nmax-1 log-variance ratios identify the remaining diagonal entries.
            if self.nmax == 1:
                return self.eye_nmax
            log_var = torch.cat((rho_raw.new_zeros(1), rho_raw.reshape(-1)))
            return torch.diag(torch.exp(log_var.clamp(-self.RHO_RAW_MAX,
                                                       self.RHO_RAW_MAX)))
        rho_raw = rho_raw.clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)
        if kind == "cs":
            # full admissible range (-1/(nmax-1), 1); a plain sigmoid reaches only rho > 0
            rho = self.cs_rho(rho_raw, self.nmax)
            J = torch.ones((self.nmax, self.nmax), device=self.device, dtype=self.y_.dtype)
            return rho * J + (1.0 - rho) * self.eye_nmax
        lag = self._lag_matrix()
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

        n_i = self.mask.sum(1)                               # [G]
        # ZTr[g, :] = Z_g^T r_g  (batched over groups)
        ZTr = (self.masked_Z.transpose(-1, -2) @
               r_g.unsqueeze(-1)).squeeze(-1)                # [G, pz]
        ZTr_scaled = ZTr / n_i.clamp(min=1).sqrt().unsqueeze(-1)
        D_acc = ZTr_scaled.T @ ZTr_scaled                   # [pz, pz]
        D_raw = D_acc / (self.n_groups * sigma2_ols)         # approx D_true / sigma2

        D_sym = 0.5 * (D_raw + D_raw.T)
        eigvals, eigvecs = torch.linalg.eigh(D_sym)
        eigvals = eigvals.clamp(min=1e-4)
        return eigvecs @ torch.diag(eigvals) @ eigvecs.T

    def _moment_rho_init(self):
        """Moment estimate of rho from OLS residuals.

        For compound symmetry, E[r_j r_k] / E[r_j^2] within groups (j≠k) equals
        sigma2*rho / (sigma2 + tr(D)), which is attenuated by the random-effects
        variance but always in (0, rho_true).  Starting rho_raw below the true
        value is far safer than starting at rho_raw = 0, which tends to
        drift into the spurious rho→1 local minimum for large pz.
        """
        dtype = self.y_.dtype
        _, r_g = self._ols_residuals()

        sum_r  = r_g.sum(1)
        sum_r2 = (r_g ** 2).sum(1)
        n_i    = self.mask.sum(1)

        n_pairs = (n_i * (n_i - 1)).sum()
        n_obs   = self.mask.sum()
        if n_pairs < 1 or n_obs < 1:
            return torch.zeros(1, device=self.device, dtype=dtype)

        mean_cross = ((sum_r ** 2 - sum_r2)[n_i > 1]).sum() / n_pairs
        mean_sq    = sum_r2.sum() / n_obs
        rho_mom    = (mean_cross / (mean_sq + 1e-12)).clamp(0.005, 0.95)
        return self._rho_raw_from_rho(rho_mom).reshape(1)

    def _diagonal_rho_init(self):
        """Initialize diagonal residual variance ratios from pooled OLS residuals.

        The diagonal residual structure is indexed by within-subject position.  The
        common profiled scale is fixed by the first position, so only the log ratios
        of the remaining position-wise mean squared residuals are returned.
        """
        dtype = self.y_.dtype
        _, r_g = self._ols_residuals()
        mask = self.mask.to(dtype=dtype)
        counts = mask.sum(dim=0).clamp_min(1.0)
        var_by_position = (r_g.pow(2) * mask).sum(dim=0) / counts
        var_by_position = var_by_position.clamp_min(1e-8)
        ratios = var_by_position[1:] / var_by_position[0]
        return torch.log(ratios).clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

    def _rho_raw_from_rho(self, rho):
        """Invert the link used by _resid_corr_from_raw so a target rho starts the optimizer
        at that rho under every structure. A shared logit is right only for cs:
        tanh(logit(p)) flips sign for p < 0.5, and softplus(logit(p)) gives 1-p."""
        kind = self._corr_kind()
        if kind == "cs":
            raw = self.cs_rho_inverse(rho, self.nmax)
        elif kind == "ar1":
            raw = torch.atanh(rho.clamp(-0.999, 0.999))
        else:                                       # exp: rho = exp(-softplus(raw))
            phi = -torch.log(rho.clamp(1e-6, 1.0 - 1e-6))
            raw = torch.log(torch.expm1(phi))       # inverse softplus
        return raw.clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

    def initialize_rand_D(self, type="unstructured", estimate="sample",
                          corr_init_method="moment"):
        spline_param = torch.tensor([
            self.NB_list[i].pow(2).mean().sqrt().item()
            for i in range(self.len_spline)
        ], device=self.device, dtype=self.y_.dtype)
        spline_param = spline_param.clamp(min=1e-3)
        rho_init = torch.log(spline_param ** 2).clamp(min=0.0)  # log(τ²), clamped so τ²≥1 (under-smoothed start)

        self.spline_params = torch.nn.Parameter(rho_init)  # log(τ²) per component

        if self._has_corr_params():
            if self._corr_kind() == "diag":
                rho_raw_init = self._diagonal_rho_init()
            elif corr_init_method == "moment":
                rho_raw_init = self._moment_rho_init()
            else:
                rho_raw_init = torch.zeros(1, device=self.device, dtype=self.y_.dtype)
            self.rho_raw = torch.nn.Parameter(rho_raw_init)

        if type == "unstructured":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                L_hat = torch.linalg.cholesky(
                    D_psd + 1e-6 * torch.eye(self.pz, device=self.device, dtype=self.y_.dtype)
                )
            else:
                L_hat = 0.1 * torch.eye(self.pz, device=self.device, dtype=self.y_.dtype)
            self.D_params = torch.nn.Parameter(
                L_hat[torch.tril_indices(self.pz, self.pz, 0)[0],
                      torch.tril_indices(self.pz, self.pz, 0)[1]]
            )
        elif type == "diagonal":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                diag_init = torch.diag(D_psd).clamp(min=1e-4).sqrt()
                self.D_params = torch.nn.Parameter(diag_init)
            else:
                self.D_params = torch.nn.Parameter(
                    torch.ones(self.pz, device=self.device, dtype=self.y_.dtype)
                )
        elif type == "iid":
            if estimate == "sample":
                D_psd = self._sample_D_init()
                iid_init = torch.diag(D_psd).mean().clamp(min=1e-4).sqrt().reshape(1)
                self.D_params = torch.nn.Parameter(iid_init)
            else:
                self.D_params = torch.nn.Parameter(
                    torch.ones(1, device=self.device, dtype=self.y_.dtype)
                )

    def _reml_relative(self, D, spline_param, R):
        """Relative-scale REML pieces (σ² profiled out). R is the unit-diagonal residual
        correlation. Returns (logdetV, logdet_XtVinvX, rVinv_r, XtVinvX, beta_hat), all in
        relative units (true V = σ²·this). Shared by loss_fn_reml and the fixed-effect test."""
        G        = self.n_groups
        nm       = self.nmax
        p        = self.X_.shape[1]
        Ktot     = self.K_tot
        f64      = torch.float64
        mask_vec = self.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        X_eff    = (self.X_grouped * mask_vec).to(f64)               # [G, nm, p]
        y_eff    = (self.y_grouped * mask_vec.squeeze(-1)).to(f64)   # [G, nm]
        U_all    = self.U_grouped.to(f64)                            # [G, nm, Ktot]

        ZDZ     = torch.matmul(self.masked_Z @ D, self.masked_Z.transpose(-1, -2)).to(f64)
        R       = R.to(f64)
        tau_sq  = torch.exp(spline_param.clamp(min=0.0, max=20.0))   # [S]
        lam     = torch.repeat_interleave(tau_sq, self._K_repeats)   # [Ktot]
        lam_inv = 1.0 / lam

        B = (R + ZDZ) * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6
        B = self.stablize_tensor(B)
        L = torch.linalg.cholesky(B)                                 # [G, nm, nm]
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
        L_M = torch.linalg.cholesky(M + 1e-10 * torch.eye(Ktot, device=self.device, dtype=f64))
        logdetM = 2.0 * torch.sum(torch.log(torch.diagonal(L_M)))
        Minv_UX = torch.cholesky_solve(A_UX, L_M)
        Minv_Uy = torch.cholesky_solve(A_Uy, L_M)
        logdetV = logdetB + torch.sum(torch.log(lam)) + logdetM
        XtVinvX = A_XX - A_UX.transpose(0, 1) @ Minv_UX
        XtVinvy = A_Xy - A_UX.transpose(0, 1) @ Minv_Uy
        ytVinvy = A_yy - (A_Uy.transpose(0, 1) @ Minv_Uy).squeeze()

        beta_hat       = torch.linalg.solve(XtVinvX, XtVinvy)
        rVinv_r        = ytVinvy - (XtVinvy.transpose(0, 1) @ beta_hat).squeeze()
        logdet_XtVinvX = torch.logdet(XtVinvX + 1e-6 * torch.eye(p, device=self.device, dtype=f64))
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
        neg_loglik = 0.5 * (logdetV + logdet_XtVinvX + (n_eff - p) * torch.log(sigma2_scalar + 1e-12))
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
    
    def beta_hat(self, D, spline_param=None):

        if self.mask.dim() == 2:
            mask_vec = self.mask.unsqueeze(-1)
        else:
            mask_vec = self.mask

        X_eff = self.X_grouped * mask_vec.to(dtype=self.X_grouped.dtype)        # [G, nmax, px]
        y_eff = self.y_grouped * mask_vec.squeeze(-1).to(dtype=self.y_grouped.dtype)  # [G, nmax]

        ZDZ   = torch.matmul(self.masked_Z @ D, self.masked_Z.transpose(-1, -2))
        I = torch.eye(self.nmax, device=self.device, dtype=torch.float64)
        R = self.resid_corr()
        V_full = R + ZDZ + (self.NB_NB * torch.exp(spline_param).view(-1, 1, 1, 1)).sum(dim=0)  # [G, nmax, nmax]

        # apply mask: keep only observed blocks; huge variance on padded diag
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)      
        Vi = V_full * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6

        Vi = 0.5 * (Vi + Vi.transpose(-1, -2))

        jitter = 1e-10
        for _ in range(8):
            LVi, info = torch.linalg.cholesky_ex(Vi + jitter * I)
            if (info == 0).all():
                Vi = Vi + jitter * I
                break
            jitter *= 10.0

        Vinv_X = torch.cholesky_solve(X_eff, LVi)                # [G, nmax, px]
        Vinv_y = torch.cholesky_solve(y_eff.unsqueeze(-1), LVi)  # [G, nmax, 1]

        XtVinvX = torch.sum(X_eff.transpose(1, 2) @ Vinv_X, dim=0)  # [px, px]
        XtVinvy = torch.sum(X_eff.transpose(1, 2) @ Vinv_y, dim=0)  # [px, 1]

        betahat = torch.linalg.solve(XtVinvX, XtVinvy)
        r = y_eff - (X_eff @ betahat).squeeze(-1)
        r_exp = r.unsqueeze(-1)
        Vinv_r = torch.cholesky_solve(r_exp, LVi)
        sigma2_hat = torch.sum(r_exp.squeeze(-1) * Vinv_r.squeeze(-1)) / (mask_vec.sum() - self.rank_X)
        return betahat, sigma2_hat
    
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
    
    def fit_likelihood(self, optimizer=torch.optim.Adam, lr=1e-3, max_iter=1000, fit_type="REML",
                       tol=1e-6, warm_start=True, verbose=False, convg_check="loss",
                       max_iter_inner=10, tau_lr_mult=5.0, group_chunk=None, grad_control_tol=1e-2,
                       corr_init_method="moment", plateau_patience=5):
        """
        tau_lr_mult      : lr multiplier for spline_params vs D_params in Adam.
        group_chunk      : process groups in chunks; reduces peak memory at 2x Cholesky cost.
        corr_init_method : "fixed" (rho_raw=0) or "moment" (default; data-driven
                           init via within-group OLS-residual correlations; avoids the
                           rho→1 / sigma2→large spurious local minimum for large pz).
        plateau_patience : (LBFGS, convg_check="loss") also converge once rel_loss < tol
                           holds for this many consecutive epochs, regardless of grad_norm.
                           Catches the case where the loss has genuinely converged but
                           gradient noise keeps grad_norm from ever clearing grad_control_tol.
        """
        beta, sigma2 = self.reset_params(OLS=warm_start, estimate=self.rand_D_estimate,
                                         corr_init_method=corr_init_method)

        self._group_chunk = group_chunk

        if verbose:
            print("OLS beta:", beta.T)

        if not (len(self.y_) == len(self.X_) == len(self.Z_)):
            raise ValueError("Incompatible shapes: y, X, and Z must have the same length.")

        loss_function = self.loss_fn_reml

        prev_beta         = beta.clone().detach()
        D                 = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
        prev_D            = D.detach()
        prev_spline_param = self.spline_params.detach() ** 2
        prev_sigma2       = sigma2.clone().detach()
        prev_loss         = float("inf")

        # Adam path
        if optimizer != torch.optim.LBFGS:
            if verbose:
                print(f"Using {optimizer.__name__} with tau_lr_mult={tau_lr_mult}")
            # τ gets a higher lr (faster convergence) and lower β₂ (quicker forgetting).
            if fit_type.lower() == "ml":
                self.beta_ = torch.nn.Parameter(beta)
                opt_groups = [
                    {'params': [self.spline_params], 'lr': lr * tau_lr_mult, 'betas': (0.9, 0.99)},
                    {'params': [self.D_params],      'lr': lr,               'betas': (0.9, 0.999)},
                    {'params': [self.beta_],          'lr': lr,               'betas': (0.9, 0.999)},
                ]
            else:
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
                if convg_criteria == 1:
                    convg = 1
                    break
                prev_spline_param = current_spline_param
                prev_loss         = current_loss
                prev_beta         = current_beta.clone().detach()
                prev_D            = current_D.detach()
                prev_sigma2       = current_sigma2.detach()
                if step % 10 == 0 and verbose:
                    print(f"Step {step}, loss = {current_loss:.6f}, rel_change = {rel_change:.6f}")

            # The Adam REML path profiles beta out analytically; expose the cached value.
            self.beta_ = current_beta

            if convg == 0:
                warnings.warn("LinearMixedModelSmoothing: Adam did not converge in "
                              f"{max_iter} steps.")

        # LBFGS path
        else:
            if max_iter_inner is None:
                max_iter_inner = 10
            max_iter_outer = max_iter // max_iter_inner
            if verbose:
                print(f"LBFGS: {max_iter_outer} outer epochs, {max_iter_inner} inner steps, tau_lr_mult={tau_lr_mult}")

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

            def closure_joint():
                opt_joint.zero_grad()
                D_ = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
                loss, _ = self.loss_fn_reml(D=D_, spline_param=self.spline_params)
                loss.backward()
                return loss

            convg = 0
            epoch = 0
            plateau = 0
            for epoch in range(max_iter_outer):
                total_loss = opt_joint.step(closure_joint)
                self._project_params()

                loss_check = total_loss.detach()

                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in lbfgs_params
                    if p.grad is not None
                ) ** 0.5

                with torch.no_grad():
                    D_           = self.param_to_sqmatrix(self.D_params, n=self.pz, correlation_structure=self.rand_D_type)
                    self.beta_   = self._beta_cache
                    self.sigma2  = self._sigma2_cache
                    current_loss          = loss_check.item()
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

                if verbose and (epoch % 10 == 0 or convg == 1):
                    print(f"LBFGS: Epoch {epoch}, loss = {current_loss:.6f}, grad_norm = {grad_norm:.3e}, rel_loss = {rel_loss:.6f}, rel_param = {rel_param:.6f}")

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
                    if verbose:
                        print(f"LBFGS converged at epoch {epoch}, loss = {current_loss:.6f}")
                    break

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
    # Reuses the inherited fixed_t_test / _contrast_test / f_theta; only the
    # parametrization hooks below are smoothing-specific. theta stacks
    # [D_params, spline_params, (rho_raw), log σ²]; C = σ²(X'V⁻¹X)⁻¹ in relative units.

    def _theta_parts(self):
        n_d = self.D_params.numel()
        n_s = self.spline_params.numel()
        n_r = self.rho_raw.numel() if self._has_corr_params() else 0
        return n_d, n_s, n_r

    def _theta_hat(self):
        parts = [self.D_params.detach().reshape(-1), self.spline_params.detach().reshape(-1)]
        if self._has_corr_params():
            parts.append(self.rho_raw.detach().reshape(-1))
        parts.append(torch.log(self.sigma2.to(self.device).reshape(1)))
        return torch.cat(parts)

    def _split_theta(self, theta):
        n_d, n_s, n_r = self._theta_parts()
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
        return 0.5 * ((n_eff - self.rank_X) * log_s2 + rVinv_r / s2 + logdetV + logdet_XtVinvX)

    def C_theta(self, theta):
        D, s_params, R, log_s2 = self._split_theta(theta)
        _, _, _, XtVinvX, _ = self._reml_relative(D, s_params, R)
        p = XtVinvX.shape[0]
        XtVinvX = XtVinvX + 1e-10 * torch.eye(p, device=XtVinvX.device, dtype=XtVinvX.dtype)
        return torch.exp(log_s2) * torch.linalg.inv(XtVinvX)

    # --- Linearity test for a smooth component (variance-component score test) ---
    # H0: tau_c^2 = 0 (component c is linear) vs H1: tau_c^2 > 0 (nonlinear).
    # Boundary score test (Lin 1997; Zhang & Lin 2003), in closed form in the natural
    # tau^2 scale: the internal log(tau^2) score is identically zero at the boundary.
    # Only the null fit is needed. See docs/linearity_test.md.

    def _chol_group_cov(self, V):
        """Mask padded rows and return a jittered per-group Cholesky of V [G, nm, nm]."""
        mask_vec = self.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        V = V * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6
        V = 0.5 * (V + V.transpose(-1, -2))
        I = torch.eye(self.nmax, device=self.device, dtype=self.y_.dtype)
        jitter = 1e-10
        for _ in range(8):
            L, info = torch.linalg.cholesky_ex(V + jitter * I)
            if (info == 0).all():
                return L
            jitter *= 10.0
        return L

    def _assemble_null_cov(self, s, y=None, null_fit_kwargs=None):
        """Fit the H0 (tau_s^2 = 0) model and return (null_model, V0, L_V0) where V0 is
        the absolute marginal covariance [G, nm, nm] with component s's smooth dropped
        but its linear column retained (it lives in X_). V0 is built from the tested
        model's own grouped Z / NB so group/mask alignment is exact."""
        y = self.y_ if y is None else _checktensor(y).to(self.device).view(-1, 1)
        nfk = dict(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200,
                   verbose=False, convg_check="loss")
        if null_fit_kwargs:
            nfk.update(null_fit_kwargs)
        others = [j for j in range(self.len_spline) if j != s]

        if others:
            null = LinearMixedModelSmoothing(
                device=self.device, y=y, X=self.X_, Z=self.Z_, id=self.id_,
                spline_inputs=[self.NB_list[j] for j in others],
                corr_structure=self.corr_structure, rand_D_type=self.rand_D_type,
                rand_D_estimate=self.rand_D_estimate)
            null.fit_likelihood(**nfk)
            D_abs    = null.D
            resid    = null.sigma2 * null.resid_corr()                       # sigma^2 R
            smooth   = (self.NB_NB[others]
                        * null.tuning_params.view(-1, 1, 1, 1)).sum(0)       # sum tau^2 NBNB'
        else: 
            null = LinearMixedModel(
                y=y, X=self.X_, Z=self.Z_, id=self.id_,
                corr_structure=self.corr_structure, rand_D_type=self.rand_D_type)
            null.fit_likelihood(optimizer=nfk["optimizer"], lr=nfk["lr"],
                                max_iter=nfk["max_iter"], fit_type="REML",
                                convg_check=nfk["convg_check"], verbose=nfk["verbose"])
            D_abs  = null.D
            resid  = null.sigma2                                             # [nm, nm] absolute
            smooth = torch.zeros((self.n_groups, self.nmax, self.nmax),
                                 device=self.device, dtype=self.y_.dtype)

        ZDZ = self.masked_Z @ D_abs @ self.masked_Z.transpose(-1, -2)
        V0  = resid.unsqueeze(0) + ZDZ + smooth # [G, nm, nm] absolute covariance under H0: tau_s^2 = 0
        return null, V0, self._chol_group_cov(V0)

    def _linearity_score(self, s, L_V0):
        """Closed-form REML score pieces for H0: tau_s^2 = 0 given the null Cholesky L_V0.
        Returns (S_plus, e, info) with S_plus = 1/2 y'P0 K_s P0 y, e = 1/2 tr(P0 K_s),
        info = 1/2 ||U_s' P0 U_s||_F^2 ; the score is U = S_plus - e."""
        mask_vec = self.mask.unsqueeze(-1)
        X_eff = self.masked_X
        y_eff = self.masked_y.unsqueeze(-1)
        U_s   = self.NB_list[s] * mask_vec                                   # [G, nm, Ks]

        ViX = torch.cholesky_solve(X_eff, L_V0)
        Viy = torch.cholesky_solve(y_eff, L_V0)
        ViU = torch.cholesky_solve(U_s,   L_V0)
        XtViX = torch.einsum('gip,giq->pq', X_eff, ViX)
        XtViy = torch.einsum('gip,gio->po', X_eff, Viy)
        XtViU = torch.einsum('gip,gik->pk', X_eff, ViU)

        beta0 = torch.linalg.solve(XtViX, XtViy)                            # [p, 1]
        w     = Viy - ViX @ beta0                                           # P0 y  [G, nm, 1]
        Usw   = torch.einsum('gik,gio->k', U_s, w)                          # [Ks]
        S_plus = 0.5 * (Usw ** 2).sum()

        P0U = ViU - ViX @ torch.linalg.solve(XtViX, XtViU)                  # [G, nm, Ks]
        W   = torch.einsum('gik,gil->kl', U_s, P0U)                         # [Ks, Ks]
        e    = 0.5 * torch.diagonal(W).sum()
        info = 0.5 * (W ** 2).sum()
        return S_plus, e, info

    @staticmethod
    def _linearity_pvalue(S_plus, e, info):
        """Zhang & Lin (2003) scaled chi-square reference: S+ ~ a * chi2_b matched by
        mean (e) and variance (info)."""
        from scipy import stats
        Sp, e, I = float(S_plus), float(e), float(info)
        a = I / (2.0 * max(e, 1e-12))
        b = 2.0 * e ** 2 / max(I, 1e-12)
        return {"statistic": Sp, "scale_a": a, "df": b,
                "p_value": float(stats.chi2.sf(Sp / max(a, 1e-12), b))}

    def linearity_test(self, s=None, alpha=0.05, null_fit_kwargs=None):
        """Score test of H0: smooth component s is linear (tau_s^2 = 0) vs nonlinear,
        with the Zhang & Lin (2003) scaled chi-square reference.

        s=None tests every component and returns a list of result dicts.
        Returns a dict with spline_id, statistic, df, scale_a, p_value, reject.
        """
        if s is None:
            return [self.linearity_test(j, alpha=alpha, null_fit_kwargs=null_fit_kwargs)
                    for j in range(self.len_spline)]
        _, _, L_V0 = self._assemble_null_cov(s, null_fit_kwargs=null_fit_kwargs)
        S_plus, e, info = self._linearity_score(s, L_V0)
        out = self._linearity_pvalue(S_plus, e, info)
        out.update(spline_id=s, reject=bool(out["p_value"] < alpha))
        return out
