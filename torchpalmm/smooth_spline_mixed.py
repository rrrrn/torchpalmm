
import torch
from .smooth_spline import SmoothingSpline
from .linear_mixed_model_smoothing import LinearMixedModelSmoothing, _checktensor
import pandas as pd
from tqdm import tqdm

class SmoothingSplineMixed(SmoothingSpline):
    @staticmethod
    def construct_incidence_matrix(origin_vec, unique_vec, ids):
        # returns (incidence [G, nmax, m], group indices)
        device = origin_vec.device
        dtype = torch.float64

        unique_ids = torch.unique(ids)
        G = len(unique_ids)
        m = len(unique_vec)

        group_indices = [(ids == g).nonzero(as_tuple=True)[0] for g in unique_ids]
        group_sizes = [len(idx) for idx in group_indices]
        nmax = max(group_sizes)

        incidence_matrices = torch.zeros((G, nmax, m),
                                        dtype=dtype,
                                        device=device)

        for g, g_idx in enumerate(group_indices):
            group_size = len(g_idx)
            matches = (origin_vec[g_idx][:, None] == unique_vec[None, :])
            incidence_matrices[g, :group_size, :] = matches.float()

        return incidence_matrices, group_indices
    
    @staticmethod
    def prepare_design_matrix(original_vec, unique_vec, ids):
        unique_vec = torch.sort(unique_vec, descending=False)[0]
        Q, R = SmoothingSplineMixed.construct_Omega(unique_vec)

        C = torch.linalg.cholesky(R)
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        S = torch.linalg.solve(C, I)  # C^{-1}
        L = Q @ S  # m x m-2

        Nmtx, group_indices = SmoothingSplineMixed.construct_incidence_matrix(original_vec, unique_vec, ids)

        # random-effect design for the spline
        LtL = L.T @ L
        I_m2 = torch.eye(LtL.shape[0], device=L.device, dtype=L.dtype)
        jitter = 1e-10
        for _ in range(10):
            chol, info = torch.linalg.cholesky_ex(LtL + jitter * I_m2)
            if int(info) == 0:
                break
            jitter *= 10.0
        B = torch.cholesky_solve(L.T, chol).T  # m x m-2
        NB = Nmtx @ B  # [G, nmax, m-2]

        # [1, x] parametric part
        input_tilde = torch.stack((torch.ones_like(original_vec), original_vec), dim=1)
        input_tilde_unique = torch.stack((torch.ones_like(unique_vec), unique_vec), dim=1)
        return NB, B, input_tilde, input_tilde_unique, group_indices
    
    def __init__(self, y, X, Z, spline_inputs, obs_id, device=None, corr_structure = 'iid', rand_D_type = 'diagonal',
                 rand_D_estimate = 'sample'):
        self.corr_structure = corr_structure; self.rand_D_type = rand_D_type; self.rand_D_estimate = rand_D_estimate
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.X = _checktensor(X).to(device=self.device)
        self.y = _checktensor(y).to(device=self.device)
        self.Z = _checktensor(Z).to(device=self.device)
        self.pz = Z.shape[1]; self.px = X.shape[1]
        obs_id = _checktensor(obs_id).to(device=self.device)
        self.nmax = max([(obs_id == g).sum().item() for g in torch.unique(obs_id)])
        self.spline_inputs = _checktensor(spline_inputs).to(device=self.device)
        if len(self.spline_inputs.shape) == 1:
            self.spline_inputs = self.spline_inputs.unsqueeze(1)
        self.unique_spline_vec = [torch.sort(torch.unique(self.spline_inputs[:, i]))[0] for i in range(self.spline_inputs.shape[1])]
        self.obs_id = obs_id

        X_fit = self.X; NB_list = []; B_list = []; self.spline_linear_col_idx = []
        self.spline_linear_center = []
        for i in range(self.spline_inputs.shape[1]):
            sp = self.spline_inputs[:, i]
            NB, B, input_tilde, _, group_indices = self.prepare_design_matrix(sp, torch.unique(sp), obs_id)
            NB_list.append(NB); B_list.append(B)

            # Center the slope column; stacked trends are badly conditioned otherwise.
            slope_mean = input_tilde[:, 1].mean()
            self.spline_linear_center.append(slope_mean)
            input_tilde = input_tilde.clone()
            input_tilde[:, 1] = input_tilde[:, 1] - slope_mean

            if i == 0:
                X_fit = torch.hstack((X_fit, input_tilde))
            else:
                X_fit = torch.hstack((X_fit, input_tilde[:, 1:]))
            self.spline_linear_col_idx.append(X_fit.shape[1] - 1)
        self.NB_list = NB_list; self.B_list = B_list; self.group_indices = group_indices

        _G = len(self.group_indices)
        obs_flat_idx = torch.zeros(_G, self.nmax, dtype=torch.long, device=self.device)
        for g, idx in enumerate(self.group_indices):
            obs_flat_idx[g, :idx.numel()] = idx
        self.obs_flat_idx = obs_flat_idx

        self.X_fit = X_fit.to(device=self.device)

    def fit(self, warm_start=True, lr=1e-3, max_iter=1000, optimizer=torch.optim.Adam,
        tol=1e-6, max_iter_inner=10, convg_check="loss", fit_type="reml", grad_control_tol=1e-2,
        tau_lr_mult=5.0, verbose=True, backfit=False, n_restarts=1, rho_init="moment"):
        if backfit:
            raise NotImplementedError("backfit fitting has been removed.")
        self.lmm = LinearMixedModelSmoothing(device=self.device, y=self.y, X=self.X_fit, Z=self.Z,
                                            spline_inputs=self.NB_list,
                                            id=self.obs_id, corr_structure=self.corr_structure, rand_D_type=self.rand_D_type)
        self.lmm.fit_likelihood(lr=lr, max_iter=max_iter, tol=tol, verbose=verbose, optimizer=optimizer, fit_type=fit_type, warm_start=warm_start,
                                    max_iter_inner=max_iter_inner, convg_check=convg_check, grad_control_tol=grad_control_tol, tau_lr_mult=tau_lr_mult,
                                    corr_init_method=rho_init)

        # Random restarts: refit from different log-τ² and rho inits, keep lowest REML.
        if n_restarts > 1:
            def _save():
                s = dict(
                    sigma2=self.lmm.sigma2.clone(), beta=self.lmm.beta.clone(),
                    D=self.lmm.D.clone(), tuning_params=self.lmm.tuning_params.clone(),
                    final_loss=self.lmm.final_loss, convg=self.lmm.convg,
                    n_iter=self.lmm.n_iter,
                    spline_params=self.lmm.spline_params.detach().clone(),
                    D_params=self.lmm.D_params.detach().clone(),
                )
                if hasattr(self.lmm, 'rho_raw'):
                    s['rho_raw'] = self.lmm.rho_raw.detach().clone()
                return s
            def _restore(s):
                self.lmm.sigma2 = s['sigma2']; self.lmm.beta = s['beta']
                self.lmm.D = s['D']; self.lmm.tuning_params = s['tuning_params']
                self.lmm.final_loss = s['final_loss']; self.lmm.convg = s['convg']
                self.lmm.n_iter = s['n_iter']
                with torch.no_grad():
                    self.lmm.spline_params.copy_(s['spline_params'])
                    self.lmm.D_params.copy_(s['D_params'])
                    if 'rho_raw' in s and hasattr(self.lmm, 'rho_raw'):
                        self.lmm.rho_raw.copy_(s['rho_raw'])

            best = _save()
            for k in range(1, n_restarts):
                # log-τ² ~ U(-2, 4) covers τ² from ~0.1 to ~55; rho_raw ~ U(-3, 3)
                # covers rho ~0.05 to ~0.95, breaking the rho→1 saturation trap.
                self.lmm.reset_params(OLS=False)
                with torch.no_grad():
                    self.lmm.spline_params.data.uniform_(-2.0, 4.0)
                    if hasattr(self.lmm, 'rho_raw'):
                        self.lmm.rho_raw.data.uniform_(-3.0, 3.0)
                self.lmm.fit_likelihood(
                    lr=lr, max_iter=max_iter, tol=tol, verbose=False,
                    optimizer=optimizer, fit_type=fit_type, warm_start=False,
                    max_iter_inner=max_iter_inner, convg_check=convg_check,
                    grad_control_tol=grad_control_tol, tau_lr_mult=tau_lr_mult,
                )
                if self.lmm.final_loss < best['final_loss']:
                    if verbose:
                        print(f"  restart {k}: new best REML = {self.lmm.final_loss:.4f} "
                              f"(prev {best['final_loss']:.4f})")
                    best = _save()
            _restore(best)

        self.sigma2 = self.lmm.sigma2.detach()
        self.resid_cov = self.sigma2 * self.lmm.resid_corr().detach()
        self.lmm.resid_cov = self.resid_cov
        self.beta = self.lmm.beta.detach()
        self.D = self.lmm.D.detach()
        self.tau_sq_list = self.lmm.tuning_params.detach()
        if self.lmm._has_corr_params() and self.lmm.nmax > 1:
            # lag-1 entry of the relative correlation (ρ for cs/ar1, exp(-φ) for exp)
            self.rho = float(self.lmm.resid_corr()[0, 1].detach())
        else:
            self.rho = 0.0
        return self.lmm

    def grouped_nb_to_NxK(self, NB_grouped):
        # [G, nmax, K] -> [N, K]
        N = self.obs_id.shape[0]
        K = NB_grouped.shape[2]

        NB_final = torch.zeros((N, K), device=NB_grouped.device, dtype=NB_grouped.dtype)

        for g, idx in enumerate(self.group_indices):
            l = idx.numel()
            NB_final[idx] = NB_grouped[g, :l]

        return NB_final

    def get_blup(self, col_smooth=None, random_effects=True):

        if not hasattr(self, 'beta'):
            raise ValueError("Model not fitted yet.")

        device = self.device
        ids    = self.obs_id.to(device)

        if col_smooth is None:
            colset = set(range(self.spline_inputs.shape[1]))
        else:
            colset = {col_smooth}

        unique_ids, inv = torch.unique(ids, sorted=True, return_inverse=True)
        G = unique_ids.numel()
        N = self.y.shape[0]

        lmm  = self.lmm
        fidx = self.obs_flat_idx.to(device)                               # [G, nmax]
        f64  = torch.float64
        nm, Ktot = lmm.nmax, lmm.K_tot

        r_flat    = self.y.reshape(-1) - (self.X_fit @ self.beta).reshape(-1)  # (N,)
        r_grouped = (r_flat[fidx] * lmm.mask).to(f64)                     # [G, nmax]

        mask_vec = lmm.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        ZDZ_g    = (lmm.masked_Z @ self.D @ lmm.masked_Z.transpose(-1, -2)).to(f64)

        # B = σ²R + ZDZ (no smooth); the smooth is a global low-rank term, so the
        # BLUP â = M⁻¹ UᵀB⁻¹r with M = Λ⁻¹ + UᵀB⁻¹U (Woodbury), shared across subjects.
        B = (self.sigma2 * lmm.resid_corr().unsqueeze(0) + ZDZ_g) * mask_mat \
            + (1.0 - mask_mat) * lmm.eye_nmax * 1e6
        B = 0.5 * (B + B.transpose(-1, -2))
        L_B = torch.linalg.cholesky(B)

        U     = lmm.U_grouped.to(f64)                                     # [G, nmax, Ktot]
        BiU   = torch.cholesky_solve(U, L_B)
        Bir   = torch.cholesky_solve(r_grouped.unsqueeze(-1), L_B)        # [G, nmax, 1]
        Um    = U.reshape(G * nm, Ktot)
        A_UU  = Um.transpose(0, 1) @ BiU.reshape(G * nm, Ktot)
        UtBir = Um.transpose(0, 1) @ Bir.reshape(G * nm, 1)

        lam   = torch.repeat_interleave(self.tau_sq_list.to(f64), lmm._K_repeats)
        M     = A_UU + torch.diag(1.0 / lam)
        M     = 0.5 * (M + M.transpose(0, 1))
        L_M   = torch.linalg.cholesky(M + 1e-10 * torch.eye(Ktot, device=device, dtype=f64))
        a_hat = torch.cholesky_solve(UtBir, L_M).squeeze(-1)              # [Ktot] = E[a|y]

        offsets = [0]
        for k in lmm.K_list:
            offsets.append(offsets[-1] + k)

        fitted_list = []
        for c in colset:
            a_c    = a_hat[offsets[c]:offsets[c + 1]]
            Xeval  = torch.sort(self.unique_spline_vec[c].to(device))[0]
            smooth = self.B_list[c].to(device).to(f64) @ a_c
            slope_idx = self.spline_linear_col_idx[c]
            center_c  = self.spline_linear_center[c].to(device=device, dtype=f64)
            if c == 0:
                linear = self.beta[int(slope_idx)] * (Xeval - center_c) + self.beta[self.px].squeeze()
            else:
                linear = self.beta[slope_idx] * (Xeval - center_c)
            theta_fitted = linear + smooth
            fitted_list.append(pd.DataFrame({
                "spline_id": c,
                "spline_var": Xeval.detach().cpu().numpy(),
                "theta_fitted": theta_fitted.detach().cpu().numpy(),
            }))

        self.fitted_df = pd.concat(fitted_list, ignore_index=True)

        if random_effects:
            # V⁻¹r = B⁻¹r − B⁻¹U â
            Vinv_r = Bir.squeeze(-1) - (BiU @ a_hat.unsqueeze(-1)).squeeze(-1)  # [G, nmax]
            w = torch.zeros(N, device=device, dtype=f64)
            w.index_put_((fidx[lmm.mask.bool()],), Vinv_r[lmm.mask.bool()])
            Zw    = self.Z.to(f64) * w.unsqueeze(1)
            b_hat = torch.zeros((G, self.pz), device=device, dtype=f64)
            b_hat.index_add_(0, inv, Zw)
            b_hat = (self.D.to(f64) @ b_hat.T).T
            self.rand_effects = b_hat.detach()
        else:
            self.rand_effects = None

        return self.fitted_df, self.rand_effects

    def smooth_confint(self, alpha=0.05, col_smooth=None, z=None):
        """Pointwise confidence band for each smooth, conditional on the REML variance
        components (mgcv's default). Refs: Wahba (1983); Nychka (1988)."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        from scipy.stats import norm
        device, f64 = self.device, torch.float64
        z = float(norm.ppf(1.0 - alpha / 2.0)) if z is None else z

        lmm      = self.lmm
        G, nm    = lmm.n_groups, lmm.nmax
        Ktot, p  = lmm.K_tot, self.X_fit.shape[1]
        mask_vec = lmm.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)

        ZDZ  = (lmm.masked_Z @ self.D @ lmm.masked_Z.transpose(-1, -2)).to(f64)
        V0   = (self.sigma2 * lmm.resid_corr().unsqueeze(0) + ZDZ) * mask_mat \
               + (1.0 - mask_mat) * lmm.eye_nmax * 1e6
        L_V0 = torch.linalg.cholesky(0.5 * (V0 + V0.transpose(-1, -2)))

        X   = (lmm.X_grouped * mask_vec).to(f64)
        U   = lmm.U_grouped.to(f64)
        ViX = torch.cholesky_solve(X, L_V0).reshape(G * nm, p)
        ViU = torch.cholesky_solve(U, L_V0).reshape(G * nm, Ktot)
        Xm, Um = X.reshape(G * nm, p), U.reshape(G * nm, Ktot)

        lam = torch.repeat_interleave(self.tau_sq_list.to(f64), lmm._K_repeats)
        H   = torch.zeros(p + Ktot, p + Ktot, dtype=f64, device=device)
        H[:p, :p] = Xm.T @ ViX
        H[p:, :p] = Um.T @ ViX
        H[:p, p:] = H[p:, :p].T
        H[p:, p:] = Um.T @ ViU + torch.diag(1.0 / lam)
        L_H = torch.linalg.cholesky(0.5 * (H + H.T)
                                    + 1e-10 * torch.eye(p + Ktot, dtype=f64, device=device))

        self.get_blup(col_smooth=col_smooth, random_effects=False)
        fdf = self.fitted_df

        offsets = [0]
        for k in lmm.K_list:
            offsets.append(offsets[-1] + k)
        colset = [col_smooth] if col_smooth is not None else range(self.spline_inputs.shape[1])

        out = []
        for c in colset:
            Xeval = torch.sort(self.unique_spline_vec[c].to(device))[0].to(f64)
            E = torch.zeros(Xeval.numel(), p + Ktot, dtype=f64, device=device)
            E[:, int(self.spline_linear_col_idx[c])] = \
                Xeval - self.spline_linear_center[c].to(device=device, dtype=f64)
            if c == 0:
                E[:, self.px] = 1.0
            E[:, p + offsets[c]: p + offsets[c + 1]] = self.B_list[c].to(device).to(f64)
            var = (E * torch.cholesky_solve(E.T, L_H).T).sum(1).clamp(min=0.0)
            se  = torch.sqrt(var).detach().cpu().numpy()
            sub = fdf[fdf["spline_id"] == c].copy()
            sub["se"]    = se
            sub["lower"] = sub["theta_fitted"] - z * se
            sub["upper"] = sub["theta_fitted"] + z * se
            out.append(sub)
        self.confint_df = pd.concat(out, ignore_index=True)
        return self.confint_df

    def fixed_t_test(self, l=None, alpha=0.05, fixed_only=True):
        """Satterthwaite t-test for fixed effects, delegated to the fitted self.lmm.

        fixed_only (default True): the l=None table covers only the first px columns
        (your X); the spline parametric intercept/slope columns stay reachable via an
        explicit l index. Returns a dict (single contrast) or list of dicts (table).
        """
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        res = self.lmm.fixed_t_test(l=l, alpha=alpha)
        if l is None and fixed_only:
            res = [row for row in res if row["coef"] < self.px]
        return res

    def linearity_test(self, s=None, alpha=0.05, null_fit_kwargs=None):
        """Score test of H0: smooth component s is linear (tau_s^2 = 0), delegated to
        the fitted self.lmm. See docs/linearity_test.md. s=None tests all components."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        return self.lmm.linearity_test(s=s, alpha=alpha, null_fit_kwargs=null_fit_kwargs)
