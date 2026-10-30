import math
import time
import warnings
import pandas as pd
import torch
def _checktensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x

class SummaryResult(dict):
    """Dict of summary DataFrames; printing it (or leaving it as a cell's last expression) shows the tables."""

    def _text(self):
        with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
            parts = ["Fixed effects (Satterthwaite t-tests)", self["coefficients"].to_string(index=False),
                     "\nVariance components", self["variance"].to_string(index=False)]
            if "splines" in self:
                parts += ["\nSmooth terms (H0: f = 0)", self["splines"].to_string(index=False)]
        return "\n".join(parts + ["\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1"])

    __repr__ = __str__ = _text

    def _repr_pretty_(self, p, cycle):
        p.text(self._text())


class FitLog:
    """verbose=True fit output: a header, an iteration table (every `every` iterations plus
    the last one) and a status line. Every method is a no-op when verbose is False."""

    def __init__(self, verbose, every=10):
        self.on, self.every = bool(verbose), every
        self._t0 = time.perf_counter()
        self._last = self._shown = None

    @staticmethod
    def vec(v, k=6):
        v = torch.as_tensor(v).reshape(-1).tolist()
        return "[" + ", ".join(f"{x:.4g}" for x in v[:k]) + (", ..." if len(v) > k else "") + "]"

    def header(self, title, rows):
        if self.on:
            w = max(len(k) for k, _ in rows)
            print("\n".join([title] + [f"  {k:<{w}}  {v}" for k, v in rows]))

    def table(self, optimizer, unit, *cols):
        self.unit = unit
        if self.on:
            print(f"\n  {optimizer}\n  {unit:>5}  {'loss':>15}" + "".join(f"  {c:>10}" for c in cols))

    def row(self, it, loss, *vals):
        self._last = (it, loss, vals)
        if self.on and it % self.every == 0:
            self._print(*self._last)

    def _print(self, it, loss, vals):
        cells = "".join(f"  {v:10.2e}" if math.isfinite(v) else f"  {'-':>10}" for v in vals)
        print(f"  {it:5d}  {loss:15.6f}{cells}")
        self._shown = it

    def done(self, converged, it, loss):
        if not self.on:
            return
        if self._last is not None and self._shown != self._last[0]:
            self._print(*self._last)
        status = f"converged at {self.unit} {it}" if converged else f"not converged, stopped at {self.unit} {it}"
        print(f"  {status}: loss = {loss:.6f} ({time.perf_counter() - self._t0:.2f} s)")


class LinearMixedModel:
    # Raw correlation parameters are held in [-RHO_RAW_MAX, RHO_RAW_MAX]. At the bound the
    # implied rho is already 0.99995, whereas past |raw| ~ 19 the links saturate to exactly
    # +-1 in float64: V goes singular and drho/draw underflows to 0, so the optimizer can
    # never come back. _project_params puts the parameter back after every step.
    RHO_RAW_MAX = 10.0

    _STRUCTURE_ALIASES = {
        "unstructured": "unstructured", "full": "unstructured", "unspecified": "unstructured",
        "iid": "iid", "identity": "iid",
        "independent": "diag", "diagonal": "diag",
        "compoundsymmetry": "cs", "cs": "cs", "compound_symmetry": "cs",
        "ar1": "ar1", "autoregressive": "ar1",
        "exponential": "exp", "exp": "exp",
    }

    @classmethod
    def _structure_kind(cls, name):
        if name is None:
            return "unstructured"
        try:
            return cls._STRUCTURE_ALIASES[name.lower()]
        except KeyError:
            raise ValueError(f"Unsupported correlation structure: {name}") from None

    @staticmethod
    def cs_rho_lower(n):
        """Lower end of the admissible compound-symmetry correlation range for an n-block."""
        return -1.0 / (n - 1) if n > 1 else 0.0

    @classmethod
    def cs_rho(cls, raw, n):
        """raw -> rho over the full admissible CS range (-1/(n-1), 1)."""
        lo = cls.cs_rho_lower(n)
        return lo + (1.0 - lo) * torch.sigmoid(raw.clamp(-cls.RHO_RAW_MAX, cls.RHO_RAW_MAX))

    @classmethod
    def cs_rho_inverse(cls, rho, n):
        """Inverse of cs_rho."""
        lo = cls.cs_rho_lower(n)
        u = ((rho - lo) / (1.0 - lo)).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(u / (1.0 - u)).clamp(-cls.RHO_RAW_MAX, cls.RHO_RAW_MAX)

    @staticmethod
    def _lag(n, device, dtype):
        idx = torch.arange(n, device=device, dtype=dtype)
        return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    @staticmethod
    def construct_lower_triangular_matrix(elements, n):
        sigma_L = torch.zeros((n, n), device=elements.device, dtype=elements.dtype)
        tril_indices = torch.tril_indices(row=n, col=n, offset=0, device=elements.device)
        sigma_L[tril_indices[0], tril_indices[1]] = elements
        return sigma_L

    def initialize_correlation(self, elements=None):
        n_params = self._n_resid_params()
        if elements is None or elements.shape[0] != n_params:
            elements = torch.ones(n_params, device=self.device)
        self.sigma_L = torch.nn.Parameter(elements)


    @staticmethod
    def ols_matrix_elements(elements, correlation, nmax):
        kind = LinearMixedModel._structure_kind(correlation)
        if kind == "unstructured":
            # scalar -> lower-triangular elements of length nmax*(nmax+1)/2
            rows, cols = torch.tril_indices(row=nmax, col=nmax, offset=0)
            return (torch.eye(nmax, device=elements.device) * elements)[rows, cols]
        if kind == "iid":
            return torch.tensor([elements], device=elements.device, dtype=elements.dtype)
        if kind == "diag":
            return elements * torch.ones(nmax, device=elements.device, dtype=elements.dtype)
        if kind == "cs":
            # param[0] = sigma, param[1] = raw with rho = cs_rho(raw, nmax); start at rho = 0.25.
            raw = LinearMixedModel.cs_rho_inverse(
                torch.tensor(0.25, device=elements.device, dtype=elements.dtype), nmax)
            return torch.stack([elements.reshape(()), raw.reshape(())])
        return torch.tensor([elements, 0], device=elements.device, dtype=elements.dtype)

    def param_to_sqmatrix(self, param, n, correlation_structure=None):
        dtype, device = self.y_.dtype, param.device
        if len(param)==0:
            return torch.zeros((n, n), device=device, dtype=dtype)
        kind = self._structure_kind(correlation_structure)
        if kind == "unstructured":
            L = self.construct_lower_triangular_matrix(param, n)
            return (L @ L.T).to(device=device, dtype=dtype)
        if kind == "iid":
            return (param**2) * torch.eye(n, device=device, dtype=dtype)
        if kind == "diag":
            return torch.diag(param**2).to(device=device, dtype=dtype)
        if kind == "cs":
            # sigma2 = param[0]^2, rho = cs_rho(param[1], n) over the full admissible range;
            # a*I + b*J with a,b >= 0 could only reach rho >= 0.
            rho = self.cs_rho(param[1], n).to(device=device, dtype=dtype)
            eye = torch.eye(n, device=device, dtype=dtype)
            J = torch.ones((n, n), device=device, dtype=dtype)
            return param[0]**2 * ((1.0 - rho) * eye + rho * J)
        lag = self._lag(n, device, dtype)
        if kind == "ar1":
            rho = torch.tanh(param[1].clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX))
            # integer exponent: exact for rho<0, and unlike |rho|^lag * sign(rho) it keeps
            # the right derivative at rho=0, which is where the OLS warm start begins.
            return param[0]**2 * rho ** lag.long()
        # softplus, not exp: exp(param) overflows to inf and -inf*0 makes the diagonal NaN.
        phi = torch.nn.functional.softplus(param[1].clamp(-self.RHO_RAW_MAX, self.RHO_RAW_MAX))
        return param[0]**2 * torch.exp(-phi * lag)

    def initialize_rand_D(self, pz, type = "unstructured", estimate = "sample"):
        kind = self._structure_kind(type)
        if kind == "unstructured":
            if estimate == "sample":
                # init from the sample second moment of Z
                D_hat = torch.zeros((pz, pz), device=self.device, dtype=self.y_.dtype)
                total_count = 0
                for i, l in enumerate(self.lengths_):
                    Zi = self.Z_grouped[i, :l, :]
                    D_hat += Zi.T @ Zi
                    total_count += l
                D_hat /= total_count
                L_hat = torch.linalg.cholesky(D_hat + 1e-5 * torch.eye(pz, device=self.device, dtype=D_hat.dtype))
                self.L_elements = L_hat[torch.tril_indices(row=pz, col=pz, offset=0)[0], torch.tril_indices(row=pz, col=pz, offset=0)[1]]
                self.L_elements = torch.nn.Parameter(self.L_elements)
            else:
                self.L_elements = torch.ones(pz * (pz + 1) // 2, requires_grad=True, device=self.device)
                self.L_elements = torch.nn.Parameter(self.L_elements)
        elif kind == "diag":
            self.L_elements = torch.ones(pz, requires_grad=True, device=self.device)
            self.L_elements = torch.nn.Parameter(self.L_elements)
        elif kind == "iid":
            self.L_elements = torch.ones(1, requires_grad=True, device=self.device)
            self.L_elements = torch.nn.Parameter(self.L_elements)

    def __init__(self, device = 'cpu', y=None, X=None, Z=None, id=None, corr_structure=None, rand_D_type="unstructured", rand_D_estimate="sample"):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.y_ = _checktensor(y).to(self.device).view(-1, 1)
        self.X_ = _checktensor(X).to(self.device)
        self.Z_ = _checktensor(Z).to(self.device)
        self.id_ = torch.as_tensor(id, device=self.device).view(-1)
        unique_ids = torch.unique(self.id_)
        self._group_idx = [(self.id_ == uid).nonzero(as_tuple=True)[0] for uid in unique_ids]
        self.lengths_   = [(self.id_ == uid).sum().item() for uid in unique_ids]

        self.n_groups = len(self.lengths_)
        self.nmax = max(self.lengths_)
        px = self.X_.shape[1]
        pz = self.Z_.shape[1]

        self.X_grouped = torch.zeros((self.n_groups, self.nmax, px), device=self.device, dtype=self.X_.dtype)
        self.Z_grouped = torch.zeros((self.n_groups, self.nmax, pz), device=self.device, dtype=self.Z_.dtype)
        self.y_grouped = torch.zeros((self.n_groups, self.nmax), device=self.device, dtype=self.y_.dtype)

        self.corr_structure = corr_structure
        self.rand_D_type = rand_D_type
        self.rand_D_estimate = rand_D_estimate

        lengths = torch.tensor(self.lengths_, device=self.device)
        valid = torch.arange(self.nmax, device=self.device)[None, :] < lengths[:, None]
        order = torch.cat(self._group_idx)
        self.X_grouped[valid] = self.X_[order]
        self.Z_grouped[valid] = self.Z_[order]
        self.y_grouped[valid] = self.y_[order].squeeze(-1)

        self.mask = valid.to(self.y_.dtype)

        mask = self.mask.unsqueeze(-1)  
        self.masked_X = self.X_grouped * mask
        self.masked_Z = self.Z_grouped * mask
        self.masked_y = self.y_grouped * self.mask

        self.eye_nmax = torch.eye(self.nmax, device=self.device, dtype=self.y_grouped.dtype)
        self.pz = pz

    @staticmethod
    def _robust_cholesky(M, n_tries=10):
        """Plain Cholesky; only if it fails, retry with a growing diagonal jitter. Returns (L, M_used)."""
        L, info = torch.linalg.cholesky_ex(M)
        if not info.any():
            return L, M
        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        jitter = 1e-8 * torch.diagonal(M, dim1=-2, dim2=-1).mean().detach().clamp_min(1.0)
        for _ in range(n_tries):
            Mj = M + jitter * eye
            L, info = torch.linalg.cholesky_ex(Mj)
            if not info.any():
                return L, Mj
            jitter = jitter * 10.0
        return torch.linalg.cholesky(Mj), Mj

    @classmethod
    def _logdet_spd(cls, M):
        L, _ = cls._robust_cholesky(M)
        return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)


    def _group_cov(self, R, D, Z=None):
        """Per-group R + Z D Z' with padded rows replaced by a variance of 1e6, [G, nmax, nmax]."""
        Z = self.masked_Z if Z is None else Z
        mask_mat = self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)
        return (R + Z @ D @ Z.transpose(-1, -2)) * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6

    def _has_resid_corr_param(self):
        """True when sigma_L[1] is a raw correlation parameter rather than a scale."""
        return self._structure_kind(self.corr_structure) in ("cs", "ar1", "exp")

    def _project_params(self):
        """Put raw correlation parameters back into the range where the link has gradient."""
        if self._has_resid_corr_param():
            with torch.no_grad():
                self.sigma_L.data[1].clamp_(-self.RHO_RAW_MAX, self.RHO_RAW_MAX)

    def beta_hat(self, sigma2, D):
        mask_vec = self.mask.unsqueeze(-1)
        X_eff = self.X_grouped * mask_vec
        y_eff = self.y_grouped * self.mask
        if self.corr_structure == "iid" and self.rand_D_type == "iid":
            if sigma2.ndim == 2 and sigma2.shape[0] == self.nmax:
                sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).mean(dim=-1)
            if D.ndim == 2 and D.shape[0] == self.Z_.shape[1]:
                tau2 = torch.diagonal(D, dim1=-2, dim2=-1).mean(dim=-1)

            Z = self.masked_Z * mask_vec
            ZtZ = Z.transpose(-1,-2) @ Z

            I_q = torch.eye(self.pz, device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            tau2   = tau2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.clamp_min(1e-8)
            tau2   = tau2.clamp_min(1e-8)

            M = (1.0/tau2)[..., None, None] * I_q + (1.0/sigma2)[..., None, None] * ZtZ
            L_M = torch.linalg.cholesky(M)
            Zt_y = Z.transpose(-1,-2) @ self.masked_y.unsqueeze(-1)
            tmp = torch.cholesky_solve(Zt_y, L_M)

            Vinv_y = (1.0/sigma2)*self.masked_y.unsqueeze(-1) - (1.0/sigma2**2)*(Z @ tmp)

            Zt_X = Z.transpose(-1,-2) @ self.masked_X
            tmpX = torch.cholesky_solve(Zt_X, L_M)
            Vinv_X = (1.0/sigma2)*self.masked_X - (1.0/sigma2**2)*(Z @ tmpX)

            XtVinvX = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_X, dim=0)
            XtVinvy = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_y, dim=0)

        else:
            LVi, _ = self._robust_cholesky(self._group_cov(sigma2, D))

            Vinv_X = torch.cholesky_solve(X_eff, LVi)
            Vinv_y = torch.cholesky_solve(y_eff.unsqueeze(-1), LVi)

            XtVinvX = torch.sum(X_eff.transpose(1, 2) @ Vinv_X, dim=0)
            XtVinvy = torch.sum(X_eff.transpose(1, 2) @ Vinv_y, dim=0)

        betahat = torch.linalg.solve(XtVinvX, XtVinvy)
        return betahat

    def reset_params(self, OLS=False, estimate="sample"):
        if OLS & (len(self.X_) > 0) and estimate == "sample":
            beta = torch.linalg.lstsq(self.X_, self.y_).solution.clone()
            residuals = self.y_ - self.X_ @ beta
            sigma2 = (residuals.T @ residuals) / (len(self.y_) - self.X_.shape[1])
            self.sigma_L = self.ols_matrix_elements(torch.sqrt(sigma2).squeeze(), self.corr_structure, self.nmax)
            self.sigma_L = torch.nn.Parameter(self.sigma_L)
        else:
            sigma_L = torch.ones(self._n_resid_params(), device=self.device, dtype=self.y_.dtype)
            if self._has_resid_corr_param():
                sigma_L[1] = 0.0          # neutral starting correlation, not a scale of 1
            self.sigma_L = torch.nn.Parameter(sigma_L)
            beta = torch.ones((self.X_.shape[1], 1), device=self.device, dtype=self.y_.dtype)
            sigma2 = torch.ones((1, 1), device=self.device, dtype=self.y_.dtype)

        self.initialize_correlation(elements=self.sigma_L)
        self.initialize_rand_D(pz=self.Z_.shape[1], type=self.rand_D_type, estimate=self.rand_D_estimate)

        return beta, sigma2

    def loss_fn(self, sigma2, D, beta = None):
        if self.beta_ is None and beta is None:
            raise ValueError("Beta coefficients must be provided either during initialization or as an argument.")
        if beta is None:
            beta = self.beta_

        e = self.y_grouped - (self.X_grouped @ beta.to( device=self.device)).squeeze(-1)

        LVi, _ = self._robust_cholesky(self._group_cov(sigma2, D))

        e_exp = e.unsqueeze(-1)
        Vinv_e = torch.cholesky_solve(e_exp, LVi)

        rVinvr = torch.sum(e_exp.squeeze(-1) * Vinv_e.squeeze(-1), dim=1)
        rVinvr = torch.sum(rVinvr)

        logdetVi = 2 * torch.sum(torch.log(torch.diagonal(LVi, dim1=1, dim2=2)), dim=1)
        logdetVsum = torch.sum(logdetVi)

        n_eff = int(self.mask.sum().item())
        neg_loglik = 0.5 * (logdetVsum + rVinvr + n_eff * torch.log(torch.tensor(2.0 * torch.pi, device=self.device)))
        return neg_loglik

    def loss_fn_reml(self, sigma2, D):
        """
        Returns negative REML log-likelihood.
        """
        mask_vec = self.mask.unsqueeze(-1)
        n_eff = int(self.mask.sum().item())

        if self.corr_structure == "iid" and self.rand_D_type == "iid":
            if sigma2.ndim == 2 and sigma2.shape[0] == self.nmax:
                sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).mean(dim=-1)
            if D.ndim == 2 and D.shape[0] == self.Z_.shape[1]:
                tau2 = torch.diagonal(D, dim1=-2, dim2=-1).mean(dim=-1)

            Z = self.masked_Z * mask_vec
            ZtZ = Z.transpose(-1,-2) @ Z

            I_q = torch.eye(self.pz, device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            tau2   = tau2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.clamp_min(1e-8)
            tau2   = tau2.clamp_min(1e-8)

            M = (1.0/tau2)[..., None, None] * I_q + (1.0/sigma2)[..., None, None] * ZtZ

            L_M = torch.linalg.cholesky(M)
            Zt_y = Z.transpose(-1,-2) @ self.masked_y.unsqueeze(-1)
            tmp = torch.cholesky_solve(Zt_y, L_M)

            Vinv_y = (1.0/sigma2)*self.masked_y.unsqueeze(-1) - (1.0/sigma2**2)*(Z @ tmp)

            Zt_X = Z.transpose(-1,-2) @ self.masked_X
            tmpX = torch.cholesky_solve(Zt_X, L_M)
            Vinv_X = (1.0/sigma2)*self.masked_X - (1.0/sigma2**2)*(Z @ tmpX)

            XtVinvX = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_X, dim=0)
            XtVinvy = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_y, dim=0)
            beta_hat = torch.linalg.solve(XtVinvX, XtVinvy)
            r = self.masked_y - (self.masked_X @ beta_hat).squeeze(-1)
            r_exp = r.unsqueeze(-1)
            Zt_r  = Z.transpose(-1,-2) @ r_exp
            tmp_r = torch.cholesky_solve(Zt_r, L_M)
            Vinv_r = (1.0/sigma2)*r_exp - (1.0/sigma2**2)*(Z @ tmp_r)

            logdetM = 2.0 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), dim=-1)
            n_pad = self.n_groups * self.nmax - n_eff
            logdetV_sum = (n_eff * torch.log(sigma2) + self.n_groups * self.pz * torch.log(tau2)
                           + logdetM.sum() + n_pad * math.log(1e6))

        else:
            L, _ = self._robust_cholesky(self._group_cov(sigma2, D))

            logdetV = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)
            logdetV_sum = torch.sum(logdetV)

            X_eff = (self.X_grouped * mask_vec).to(dtype=L.dtype)
            y_eff = (self.y_grouped * mask_vec.squeeze(-1)).to(dtype=L.dtype)

            Vinv_y = torch.cholesky_solve(y_eff.unsqueeze(-1), L)
            Vinv_X = torch.cholesky_solve(X_eff, L)

            XtVinvX = torch.sum(X_eff.transpose(1, 2) @ Vinv_X, dim=0)
            XtVinvy = torch.sum(X_eff.transpose(1, 2) @ Vinv_y, dim=0)

            beta_hat = torch.linalg.solve(XtVinvX, XtVinvy)
            r = y_eff - (X_eff @ beta_hat).squeeze(-1)
            r_exp = r.unsqueeze(-1)
            Vinv_r = torch.cholesky_solve(r_exp, L)

        rVinv_r = torch.sum(r_exp.squeeze(-1) * Vinv_r.squeeze(-1), dim=1)
        rVinv_r = torch.sum(rVinv_r)

        logdet_XtVinvX = self._logdet_spd(XtVinvX)
        p = self.X_.shape[1]

        neg_loglik = 0.5 * (logdetV_sum + rVinv_r + logdet_XtVinvX + (n_eff - p) * torch.log(torch.tensor(2.0 * torch.pi)))
        return neg_loglik

    def update_iter(self, loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss):
        sigma2 = self.param_to_sqmatrix(self.sigma_L, n=max(self.lengths_), correlation_structure=self.corr_structure)
        D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type)
        opt.zero_grad()
        total_loss = loss_function(sigma2=sigma2, D=D)
        total_loss.backward()
        opt.step()
        self._project_params()

        with torch.no_grad():
            D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type)
            sigma2 = self.param_to_sqmatrix(self.sigma_L, n=max(self.lengths_), correlation_structure=self.corr_structure)
            current_D = D
            current_sigma2 = sigma2

            if fit_type.lower() == "ml":
                beta_ = self.beta_.detach()   # ML optimizes beta directly
            else:
                beta_ = self.beta_hat(sigma2=sigma2, D=D)

            eps = 1e-8
            beta_change = torch.max(torch.abs((beta_ - prev_beta) / (prev_beta.abs() + eps))).item()
            D_change = torch.max(torch.abs((current_D.flatten() - prev_D.flatten()) / (prev_D.flatten().abs() + eps))).item()
            sigma2_change = torch.max(torch.abs((current_sigma2.flatten() - prev_sigma2.flatten()) / (prev_sigma2.flatten().abs() + eps))).item()

            current_loss = total_loss.item()
            rel_change = abs((current_loss - prev_loss) / (prev_loss + 1e-12))

        if current_loss > prev_loss:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

        convg_criteria = 0
        if convg_check == "loss":
            if rel_change < tol:
                convg_criteria = 1
        elif convg_check == "all":
            if (rel_change < tol) and (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                convg_criteria = 1
        elif convg_check == "params":
            if (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                convg_criteria = 1
        else:
            raise ValueError("Invalid convergence check method.")
        return convg_criteria, current_loss, current_sigma2, current_D, beta_, rel_change, beta_change, D_change, sigma2_change

    def _header_rows(self):
        """Data and model rows of the verbose=True header."""
        D = f"q = {self.pz}, D {self._structure_kind(self.rand_D_type)}" if self.pz else "none"
        return [("data", f"{len(self.y_)} obs in {self.n_groups} groups (max {self.nmax} per group)"),
                ("fixed", f"p = {self.X_.shape[1]}"),
                ("random", D),
                ("residual", self._structure_kind(self.corr_structure))]

    def _log_header(self, log, fit_type, beta, warm_start):
        start = (f"OLS beta = {FitLog.vec(beta)}" if warm_start and self.rand_D_estimate == "sample"
                 else "default (no warm start)")
        log.header(f"{type(self).__name__}: {fit_type.upper()} fit",
                   self._header_rows() + [("start", start)])

    def fit_likelihood(self, optimizer = torch.optim.Adam, lr=1e-3, max_iter=1000, fit_type = "REML",
            tol=1e-6, warm_start=True, verbose=False, convg_check = "loss", max_iter_inner=10):
        if not (len(self.y_) == len(self.X_) == len(self.Z_)):
            raise ValueError("Incompatible shapes: y, X, and Z must have the same length.")

        self.fit_type = fit_type
        ml = fit_type.lower() == "ml"
        log = FitLog(verbose)
        beta, _ = self.reset_params(OLS=warm_start, estimate=self.rand_D_estimate)
        if log.on:
            self._log_header(log, fit_type, beta, warm_start)

        loss_function = self.loss_fn if ml else self.loss_fn_reml
        params = [self.L_elements, self.sigma_L]
        if ml:
            self.beta_ = torch.nn.Parameter(beta)
            params.append(self.beta_)

        if optimizer == torch.optim.LBFGS:
            fitted = self._fit_lbfgs(loss_function, params, beta, lr, max_iter, max_iter_inner, tol, convg_check, log)
        else:
            fitted = self._fit_adam(loss_function, params, beta, lr, max_iter, fit_type, tol, convg_check, log)
        self.convg, self.final_loss, self.sigma2, self.D, self.n_iter = fitted

        self.sigma2 = self.sigma2.clone().detach()
        self.D = self.D.clone().detach()
        self.beta = self.beta_.clone().detach()
        return self

    def _fit_adam(self, loss_function, params, beta, lr, max_iter, fit_type, tol, convg_check, log):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        log.table(f"AdamW: lr={lr:g}, {max_iter} steps, tol={tol:g} on {convg_check}", "step", "rel_loss")
        prev_beta = beta.clone().detach()
        prev_D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type).detach()
        prev_sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure).detach()
        prev_loss = float("inf")

        convg = 0
        for step in range(max_iter):
            convg_criteria, current_loss, current_sigma2, current_D, current_beta, rel_change, _, _, _ = self.update_iter(
                loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss
            )
            log.row(step, current_loss, rel_change)
            if convg_criteria == 1:
                convg = 1
                break

            prev_loss = current_loss
            prev_beta = current_beta.clone().detach()
            prev_D = current_D.detach()
            prev_sigma2 = current_sigma2.detach()

        if fit_type.lower() != "ml":
            self.beta_ = current_beta
        log.done(convg, step, current_loss)
        if not convg:
            warnings.warn(f"LinearMixedModel: Adam did not converge in {max_iter} steps "
                          f"(rel_change={rel_change:.2e}).")
        return convg, current_loss, current_sigma2, current_D, step

    def _fit_lbfgs(self, loss_function, params, beta, lr, max_iter, max_iter_inner, tol, convg_check, log):
        ml = self.fit_type.lower() == "ml"
        max_iter_inner = max_iter_inner or 10
        max_iter_outer = max_iter // max_iter_inner
        log.table(f"LBFGS: lr={lr:g}, {max_iter_outer} epochs x {max_iter_inner} inner steps, "
                  f"tol={tol:g} on {convg_check}", "epoch", "rel_loss", "rel_param", "grad_norm")
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter_inner,
                                history_size=10, line_search_fn="strong_wolfe")
        prev_beta = beta.clone().detach()
        prev_D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type).detach()
        prev_sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure).detach()
        prev_loss = float("inf")

        def closure():
            opt.zero_grad()
            sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure)
            D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type)
            loss = loss_function(sigma2=sigma2, D=D)
            loss.backward()
            return loss

        convg = 0
        for epoch in range(max_iter_outer):
            total_loss = opt.step(closure)
            self._project_params()
            grad_norm = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5

            with torch.no_grad():
                current_sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure)
                current_D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type)
                if not ml:
                    self.beta_ = self.beta_hat(sigma2=current_sigma2, D=current_D).clone().detach()
                current_loss = total_loss.item()

                eps = 1e-12
                rel_loss = abs((current_loss - prev_loss) / (prev_loss + eps))
                rel_param_beta = torch.max(torch.abs((self.beta_ - prev_beta) / (prev_beta.abs() + eps))).item()
                rel_param_D = torch.max(torch.abs((current_D.flatten() - prev_D.flatten()) / (prev_D.flatten().abs() + eps))).item()
                rel_param_sigma2 = torch.max(torch.abs((current_sigma2.flatten() - prev_sigma2.flatten()) / (prev_sigma2.flatten().abs() + eps))).item()
                rel_param = max(rel_param_beta, rel_param_D, rel_param_sigma2)

            prev_loss = current_loss
            prev_beta = self.beta_.detach()
            prev_D = current_D.detach()
            prev_sigma2 = current_sigma2.detach()
            log.row(epoch, current_loss, rel_loss, rel_param, grad_norm)

            if (rel_loss < tol) or grad_norm < 1e-2:
                if convg_check not in ["params", "all"] or rel_param < tol:
                    convg = 1
                break

        log.done(convg, epoch, current_loss)
        if convg == 0:
            warnings.warn(f"LinearMixedModel: LBFGS did not converge in {max_iter_outer} epochs "
                          f"(rel_loss={rel_loss:.2e}, grad_norm={grad_norm:.2e}).")
        return convg, current_loss, current_sigma2, current_D, epoch

    # --- Hypothesis testing for fixed effects ---
    # Satterthwaite t-test (lmerTest): t = l'b / sqrt(l'C l),
    # nu = 2 f^2 / (grad_f' A grad_f) with f(theta)=l'C(theta)l,
    # theta the variance params, A = [d2 nll/d theta2]^-1.

    def _n_resid_params(self):
        kind = self._structure_kind(self.corr_structure)
        if kind == "unstructured":
            return (self.nmax * (self.nmax + 1)) // 2
        if kind == "iid":
            return 1
        if kind == "diag":
            return self.nmax
        return 2

    def _theta_hat(self):
        return torch.cat([self.sigma_L.detach().reshape(-1),
                          self.L_elements.detach().reshape(-1)], dim=0)

    def unpack_theta(self, theta, q):
        n_resid = self._n_resid_params()
        theta_sigma, theta_L = theta[:n_resid], theta[n_resid:]
        kind = self._structure_kind(self.rand_D_type)
        if (kind == "diag" and theta_L.shape[0] != q) or (kind == "unstructured" and theta_L.shape[0] != (q * (q + 1)) // 2):
            raise ValueError("theta_L length does not match the number of random effects.")
        sigma2 = self.param_to_sqmatrix(theta_sigma, n=self.nmax, correlation_structure=self.corr_structure)
        D = self.param_to_sqmatrix(theta_L, n=q, correlation_structure=self.rand_D_type)
        return sigma2, D

    def nll(self, theta):
        sigma2, D = self.unpack_theta(theta, self.D.shape[0])
        if getattr(self, "fit_type", "REML").lower() == "ml":
            return self.loss_fn(sigma2, D, beta=self.beta)
        return self.loss_fn_reml(sigma2, D)

    def C_theta(self, theta):
        sigma2, D = self.unpack_theta(theta, self.D.shape[0])
        VinvX = torch.linalg.solve(self._group_cov(sigma2, D), self.masked_X)
        XtVinvX = (self.masked_X.transpose(1, 2) @ VinvX).sum(dim=0)
        return torch.linalg.inv(XtVinvX)

    def _contrast_test(self, l_vec, theta, C, A):
        """Satterthwaite t-test of l'beta; C = C_theta(theta) with its graph still attached."""
        l_vec = l_vec.to(device=self.device, dtype=self.y_.dtype)
        f = (l_vec.transpose(0, 1) @ C @ l_vec).squeeze()
        grad_f = torch.autograd.grad(f, theta, retain_graph=True)[0]
        var_f = grad_f @ A @ grad_f
        se = torch.sqrt(f)
        est = (l_vec.transpose(0, 1) @ self.beta.to(self.device)).squeeze()
        t_hat = est / se
        df = (2.0 * f ** 2) / var_f
        from scipy import stats
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_hat.item()), df.item()))
        return {"estimate": est.item(), "se": se.item(), "t": t_hat.item(),
                "df": df.item(), "p_value": float(p_value)}

    def _ttest_state(self):
        """theta_hat and A = H^{-1}, H the Hessian of the nll at theta_hat. Cached until the
        fitted parameters change, so repeated tests do not recompute the Hessian."""
        from torch.autograd.functional import hessian
        theta = self._theta_hat()
        key = getattr(self, "fit_type", None)
        c = getattr(self, "_tt_cache", None)
        if c is not None and c["key"] == key and c["theta"].shape == theta.shape \
                and torch.equal(c["theta"], theta):
            return c
        H = hessian(lambda th: self.nll(th), theta.clone().requires_grad_(True))
        H = 0.5 * (H + H.transpose(0, 1))
        A = torch.linalg.inv(H + 1e-8 * torch.eye(H.shape[0], device=H.device, dtype=H.dtype))
        self._tt_cache = dict(theta=theta, key=key, A=A, unit={})
        return self._tt_cache

    def fixed_t_test(self, l=None):
        """Satterthwaite t-test for fixed effects (ref: lmerTest, Kuznetsova et al. 2017).

        l=None   -> per-coefficient table (a list of dicts, one unit contrast each).
        l=int    -> unit contrast e_l for a single coefficient.
        l=vector -> custom contrast of length px.
        Returns a dict with estimate, se, t, df, p_value (or a list of them).
        The Hessian and the unit-contrast results are cached across calls.
        """
        px = self.X_.shape[1]
        if isinstance(l, int) and l >= px:
            raise ValueError("l must be less than the number of fixed effects.")
        state = self._ttest_state()
        unit = state["unit"]
        todo = list(range(px)) if l is None else [l] if isinstance(l, int) else []
        todo = [i for i in todo if i not in unit]

        if todo or (l is not None and not isinstance(l, int)):
            theta = state["theta"].clone().requires_grad_(True)
            C = self.C_theta(theta)

            def e(i):
                v = torch.zeros((px, 1), device=self.device, dtype=self.y_.dtype)
                v[i, 0] = 1.0
                return v

            for i in todo:
                unit[i] = self._contrast_test(e(i), theta, C, state["A"])
            if l is not None and not isinstance(l, int):
                l_vec = torch.as_tensor(l, device=self.device, dtype=self.y_.dtype).reshape(px, 1)
                return self._contrast_test(l_vec, theta, C, state["A"])

        if l is None:
            return [dict(coef=i, **unit[i]) for i in range(px)]
        return dict(unit[l])

    def coef_table(self, names=None, alpha=0.05):
        """Fixed effects: estimate, se, Satterthwaite df, t, p, (1 - alpha) CI."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        from scipy import stats
        rows = self.fixed_t_test(l=None)
        names = list(names) if names is not None else [f"X{i + 1}" for i in range(len(rows))]
        if len(names) != len(rows):
            raise ValueError(f"names has {len(names)} entries, expected {len(rows)}.")
        tab = pd.DataFrame(rows).drop(columns="coef")
        tab.insert(0, "term", names)
        q = stats.t.ppf(1.0 - alpha / 2.0, tab["df"].to_numpy())
        tab["ci_lower"] = tab["estimate"] - q * tab["se"]
        tab["ci_upper"] = tab["estimate"] + q * tab["se"]
        tab["sig"] = ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "" for p in tab["p_value"]]
        return tab

    def variance_table(self, re_names=None):
        """sigma^2 and the nonzero entries of D."""
        s2 = self.sigma2.detach().cpu()
        D = self.D.detach().cpu()
        q = D.shape[0]
        re_names = list(re_names) if re_names is not None else [f"b{i + 1}" for i in range(q)]
        rows = [dict(component="residual", variance=s2.diagonal().mean().item() if s2.dim() == 2 else float(s2))]
        rows += [dict(component=f"Var({re_names[i]})", variance=D[i, i].item()) for i in range(q)]
        rows += [dict(component=f"Cov({re_names[j]}, {re_names[i]})", variance=D[i, j].item())
                 for i in range(q) for j in range(i) if D[i, j].item() != 0.0]
        return pd.DataFrame(rows)

    def summary(self, names=None, re_names=None, alpha=0.05):
        """{"coefficients", "variance"} DataFrames; displays as formatted tables when printed."""
        return SummaryResult(coefficients=self.coef_table(names=names, alpha=alpha),
                             variance=self.variance_table(re_names=re_names))

    # --- Hypothesis testing for random effects ---
    # Boundary score test (Lin 1997; Zhang & Lin 2003): U = S+ - e at H0,
    # S+ = 1/2 ||Z_idx' P0 y||^2, matched to a scaled chi-square a*chi2_b.
    # See docs/linearity_test.md.

    def _random_effect_null_chol(self, idx, null_fit_kwargs=None):
        if type(self) is not LinearMixedModel:
            raise NotImplementedError(
                "random_effect_test assumes V = R + Z D Z^T with no other "
                "covariance terms; it is not valid on subclasses that add their "
                "own (e.g. LinearMixedModelSmoothing -- use linearity_test there)."
            )
        keep = [j for j in range(self.pz) if j not in idx]
        if not keep:
            raise ValueError("random_effect_test needs at least one other random "
                              "effect left under the null; compare a no-random-"
                              "effects fit with lrt_test instead.")
        nfk = dict(optimizer=torch.optim.LBFGS, lr=1.0, max_iter=200,
                   verbose=False, convg_check="loss")
        if null_fit_kwargs:
            nfk.update(null_fit_kwargs)

        null = LinearMixedModel(device=self.device, y=self.y_, X=self.X_, Z=self.Z_[:, keep],
                                id=self.id_, corr_structure=self.corr_structure,
                                rand_D_type=self.rand_D_type, rand_D_estimate=self.rand_D_estimate)
        null.fit_likelihood(optimizer=nfk["optimizer"], lr=nfk["lr"], max_iter=nfk["max_iter"],
                            fit_type="REML", convg_check=nfk["convg_check"], verbose=nfk["verbose"])

        return self._robust_cholesky(self._group_cov(null.sigma2, null.D, self.masked_Z[:, :, keep]))[0]

    def _boundary_score(self, Z_q, L_V0):
        """Score pieces (S_plus, e, info) for H0: the variance of the random block with design Z_q is 0,
        given the null per-group Cholesky L_V0."""
        X_eff = self.masked_X
        y_eff = self.masked_y.unsqueeze(-1)

        ViX = torch.cholesky_solve(X_eff, L_V0)
        Viy = torch.cholesky_solve(y_eff, L_V0)
        ViZ = torch.cholesky_solve(Z_q,   L_V0)
        XtViX = torch.einsum('gip,giq->pq', X_eff, ViX)
        XtViy = torch.einsum('gip,gio->po', X_eff, Viy)
        XtViZ = torch.einsum('gip,gik->pk', X_eff, ViZ)

        beta0 = torch.linalg.solve(XtViX, XtViy)
        w     = Viy - ViX @ beta0
        Zqw   = torch.einsum('gik,gio->k', Z_q, w)
        S_plus = 0.5 * (Zqw ** 2).sum()

        P0Z  = ViZ - ViX @ torch.linalg.solve(XtViX, XtViZ)
        W    = torch.einsum('gik,gil->kl', Z_q, P0Z)
        e    = 0.5 * torch.diagonal(W).sum()
        info = 0.5 * (W ** 2).sum()
        return S_plus, e, info

    @staticmethod
    def _boundary_pvalue(S_plus, e, info):
        """Zhang & Lin (2003) scaled chi-square reference: S+ ~ a * chi2_b matched by mean (e) and variance (info)."""
        from scipy import stats
        Sp, e, I = float(S_plus), float(e), float(info)
        a = I / (2.0 * max(e, 1e-12))
        b = 2.0 * e ** 2 / max(I, 1e-12)
        return {"statistic": Sp, "scale_a": a, "df": b,
                "p_value": float(stats.chi2.sf(Sp / max(a, 1e-12), b))}

    def random_effect_test(self, q=None, alpha=0.05, null_fit_kwargs=None):
        """q=None tests every random effect; q=int tests one; q=list[int] tests
        that block jointly. Returns a dict (or list of dicts) with re_idx,
        statistic, df, scale_a, p_value, reject."""
        if q is None:
            return [self.random_effect_test(j, alpha=alpha, null_fit_kwargs=null_fit_kwargs)
                    for j in range(self.pz)]
        idx = [q] if isinstance(q, int) else list(q)
        L_V0 = self._random_effect_null_chol(idx, null_fit_kwargs=null_fit_kwargs)
        S_plus, e, info = self._boundary_score(self.masked_Z[:, :, idx], L_V0)
        out = self._boundary_pvalue(S_plus, e, info)
        out.update(re_idx=idx if len(idx) > 1 else idx[0], reject=bool(out["p_value"] < alpha))
        return out

    def AIC(self):
        """AIC from the ML log-likelihood at the fitted estimates (use an ML fit for a proper AIC)."""
        nll_value = self.loss_fn(self.sigma2, self.D, beta=self.beta)
        k = len(self.beta) + self._theta_hat().numel()
        return 2.0 * nll_value.item() + 2.0 * k


# --- LRT for model selection ---
def lrt_test(lmm1, lmm2, boundary=False):
    """Likelihood-ratio test between two fitted models on the same data.

    A valid fixed-effects LRT needs both models fit by ML (REML deviances are not
    comparable across different X). For a nested variance component on the boundary
    of its space, set boundary=True to halve the p-value (chi-bar-square correction).
    """
    from scipy import stats
    lr_stat = 2.0 * abs(float(lmm1.final_loss) - float(lmm2.final_loss))
    k1 = len(lmm1.beta) + lmm1._theta_hat().numel()
    k2 = len(lmm2.beta) + lmm2._theta_hat().numel()
    df = abs(k1 - k2)
    p_value = 1.0 - stats.chi2.cdf(lr_stat, df=df) if df > 0 else 1.0
    if boundary:
        p_value *= 0.5
    return lr_stat, float(p_value)