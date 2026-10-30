
import copy
import warnings
import torch
from .linear_mixed_model import SummaryResult
from .linear_mixed_model_smoothing import LinearMixedModelSmoothing, _checktensor
import numpy as np
import pandas as pd
from tqdm import tqdm

class SmoothingSplineMixed:
    @staticmethod
    def construct_Q(x):
        x = x.flatten()
        n = x.shape[0]
        h = x[1:] - x[:-1]

        Q = torch.zeros((n, n-2), dtype=x.dtype, device=x.device)
        cols = torch.arange(n-2)
        Q[cols, cols] = 1.0 / h[cols]
        Q[cols + 1, cols] = -1.0 / h[cols] - 1.0 / h[cols + 1]
        Q[cols + 2, cols] = 1.0 / h[cols + 1]
        return Q

    @staticmethod
    def construct_R(x):
        x = x.flatten()
        n = x.shape[0]
        h = x[1:] - x[:-1]

        R = torch.zeros((n-2, n-2), dtype=x.dtype, device=x.device)
        cols = torch.arange(n-2)
        R[cols, cols] = (h[cols] + h[cols + 1]) / 3.0
        cols = torch.arange(n-3)
        R[cols + 1, cols] = h[cols+1] / 6.0
        R[cols, cols + 1] = h[cols+1] / 6.0
        return R

    @staticmethod
    def construct_Omega(x):
        return SmoothingSplineMixed.construct_Q(x), SmoothingSplineMixed.construct_R(x)

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
        S = torch.linalg.solve_triangular(C, I, upper=False)
        L = Q @ S

        Nmtx, group_indices = SmoothingSplineMixed.construct_incidence_matrix(original_vec, unique_vec, ids)

        # random-effect design for the spline: B = L (L'L)^{-1}
        B = torch.linalg.pinv(L).T
        NB = Nmtx @ B

        # [1, x] parametric part
        input_tilde = torch.stack((torch.ones_like(original_vec), original_vec), dim=1)
        input_tilde_unique = torch.stack((torch.ones_like(unique_vec), unique_vec), dim=1)
        return NB, B, input_tilde, input_tilde_unique, group_indices

    def __init__(self, y, X, Z, spline_inputs, obs_id, device=None, corr_structure = 'iid', rand_D_type = 'diagonal',
                 rand_D_estimate = 'sample', smooth_mask=None):
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
        self.obs_id = obs_id

        S = self.spline_inputs.shape[1]
        m = self._check_mask(smooth_mask, S)
        self.smooth_mask = m
        # knots: distinct x on the rows where the spline is active (all rows when unmasked)
        self.unique_spline_vec = []
        for i in range(S):
            sp = self.spline_inputs[:, i]
            u = torch.sort(torch.unique(sp if m is None else sp[m[:, i] != 0]))[0]
            if u.numel() < 3:
                raise ValueError(f"spline {i} has {u.numel()} distinct active x-values; at least 3 are needed.")
            self.unique_spline_vec.append(u)
        X_fit = self.X; NB_list = []; B_list = []; self.spline_linear_col_idx = []
        self.spline_linear_center = []
        for i in range(S):
            sp = self.spline_inputs[:, i]
            NB, B, input_tilde, _, group_indices = self.prepare_design_matrix(sp, self.unique_spline_vec[i], obs_id)
            if m is not None:
                pad = torch.nn.utils.rnn.pad_sequence(group_indices, batch_first=True)
                valid = torch.arange(pad.shape[1], device=pad.device)[None, :] < torch.tensor(
                    [g.numel() for g in group_indices], device=pad.device)[:, None]
                NB = NB * (m[:, i][pad] * valid).unsqueeze(-1)
            NB_list.append(NB); B_list.append(B)

            # Center the slope column; stacked trends are badly conditioned otherwise.
            slope_mean = input_tilde[:, 1].mean()
            self.spline_linear_center.append(slope_mean)
            input_tilde = input_tilde.clone()
            input_tilde[:, 1] = input_tilde[:, 1] - slope_mean
            if m is not None:
                input_tilde[:, 1] = m[:, i] * input_tilde[:, 1]

            if i == 0:
                X_fit = torch.hstack((X_fit, input_tilde))
            else:
                X_fit = torch.hstack((X_fit, input_tilde[:, 1:]))
            self.spline_linear_col_idx.append(X_fit.shape[1] - 1)
        self.level_names = []
        if m is not None:
            X_fit, self.level_names = self._add_level_columns(X_fit, m)
        self.NB_list = NB_list; self.B_list = B_list; self.group_indices = group_indices

        _G = len(self.group_indices)
        obs_flat_idx = torch.zeros(_G, self.nmax, dtype=torch.long, device=self.device)
        for g, idx in enumerate(self.group_indices):
            obs_flat_idx[g, :idx.numel()] = idx
        self.obs_flat_idx = obs_flat_idx

        self.X_fit = X_fit.to(device=self.device)

    def _check_mask(self, smooth_mask, S):
        """Validate the N x S activation mask (None = every spline active on every row)."""
        if smooth_mask is None:
            return None
        m = _checktensor(smooth_mask).to(device=self.device)
        if m.dim() == 1:
            m = m.unsqueeze(1)
        if m.shape != (self.y.shape[0], S):
            raise ValueError(f"smooth_mask must be N x S = {(self.y.shape[0], S)}, got {tuple(m.shape)}.")
        if not torch.isfinite(m).all() or (m.abs().sum(0) == 0).any():
            raise ValueError("smooth_mask must be finite with no all-zero column.")
        return m

    def _add_level_columns(self, X_fit, m, tol=1e-8):
        """Masked splines are identified only up to the level m_s * c_s, which the global intercept
        absorbs only if m_s is constant. Check that each m_s lies in span([X, 1]) (e.g. time dummies
        in X); append only the columns that add new information (one per distinct mask). Returns (X_fit, names of appended columns)."""
        A = torch.cat([self.X.cpu(), torch.ones(self.X.shape[0], 1, dtype=torch.float64)], dim=1)
        mc = m.cpu()
        added = []
        for s in range(m.shape[1]):
            resid = mc[:, s] - A @ torch.linalg.lstsq(A, mc[:, s:s + 1], driver="gelsd").solution.squeeze(-1)
            if resid.norm() > tol * max(mc[:, s].norm().item(), 1.0):
                added.append(s)
                A = torch.cat([A, mc[:, s:s + 1]], dim=1)
        if added:
            warnings.warn(f"smooth_mask: no time-specific intercepts for splines {added} in X; "
                          f"adding the masked level columns. Pass time dummies in X to control them.")
            X_fit = torch.cat([X_fit, m[:, added].to(X_fit.dtype)], dim=1)
        return X_fit, [f"level(s{s + 1})" for s in added]

    def fit(self, warm_start=True, lr=1e-3, max_iter=1000, optimizer=torch.optim.Adam,
        tol=1e-6, max_iter_inner=10, convg_check="loss", fit_type="reml", grad_control_tol=1e-2,
        tau_lr_mult=5.0, verbose=False, n_restarts=1, rho_init="moment"):
        self.lmm = LinearMixedModelSmoothing(device=self.device, y=self.y, X=self.X_fit, Z=self.Z,
                                            spline_inputs=self.NB_list,
                                            id=self.obs_id, corr_structure=self.corr_structure, rand_D_type=self.rand_D_type, rand_D_estimate=self.rand_D_estimate)
        self.lmm.fit_likelihood(lr=lr, max_iter=max_iter, tol=tol, verbose=verbose, optimizer=optimizer, fit_type=fit_type, warm_start=warm_start,
                                    max_iter_inner=max_iter_inner, convg_check=convg_check, grad_control_tol=grad_control_tol, tau_lr_mult=tau_lr_mult,
                                    corr_init_method=rho_init)

        # Random restarts: refit from different log-τ² and rho inits, keep the lowest loss.
        # log-τ² ~ U(-2, 4) covers τ² from ~0.1 to ~55; rho_raw ~ U(-3, 3) covers rho ~0.05 to ~0.95,
        # breaking the rho→1 saturation trap.
        if n_restarts > 1:
            if verbose:
                print(f"\n  random restarts ({n_restarts - 1})\n  {'restart':>7}  {'loss':>15}  converged")
            best, best_k = copy.deepcopy(self.lmm), 0
            for k in range(1, n_restarts):
                init = dict(spline_params=torch.empty_like(best.spline_params).uniform_(-2.0, 4.0))
                if hasattr(best, 'rho_raw'):
                    init['rho_raw'] = torch.empty_like(best.rho_raw).uniform_(-3.0, 3.0)
                self.lmm.fit_likelihood(
                    lr=lr, max_iter=max_iter, tol=tol, verbose=False,
                    optimizer=optimizer, fit_type=fit_type, warm_start=False,
                    max_iter_inner=max_iter_inner, convg_check=convg_check,
                    grad_control_tol=grad_control_tol, tau_lr_mult=tau_lr_mult, init=init,
                )
                better = self.lmm.final_loss < best.final_loss
                if verbose:
                    print(f"  {k:7d}  {self.lmm.final_loss:15.6f}  {'yes' if self.lmm.convg else 'no':>9}"
                          + ("  new best" if better else ""))
                if better:
                    best, best_k = copy.deepcopy(self.lmm), k
            self.lmm = best
            if verbose:
                print(f"  kept {'the initial fit' if best_k == 0 else f'restart {best_k}'}: "
                      f"loss = {best.final_loss:.6f}")

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
        N = self.obs_id.shape[0]
        K = NB_grouped.shape[2]

        NB_final = torch.zeros((N, K), device=NB_grouped.device, dtype=NB_grouped.dtype)

        for g, idx in enumerate(self.group_indices):
            l = idx.numel()
            NB_final[idx] = NB_grouped[g, :l]

        return NB_final

    def get_blup(self, col_smooth=None, random_effects=True):
        """Fitted curves in self.fitted_df (one row per distinct x of each smooth).
        theta_fitted for smooth 0 includes the global intercept beta[px]; with smooth_mask that
        smooth is active on a subset of rows, so do not rebuild fitted values as mask * theta_fitted.
        theta_no_intercept is theta_fitted without the intercept (all smooths), and the intercept is
        stored in self.intercept: add it to every row once, e.g. X_fit @ beta + sum_c mask_c * f_c(x)."""
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
        fidx = self.obs_flat_idx.to(device)
        f64  = torch.float64
        nm, Ktot = lmm.nmax, lmm.K_tot

        r_flat    = self.y.reshape(-1) - (self.X_fit @ self.beta).reshape(-1)
        r_grouped = (r_flat[fidx] * lmm.mask)

        # B = σ²R + ZDZ (no smooth); the smooth is a global low-rank term, so the
        # BLUP â = M⁻¹ UᵀB⁻¹r with M = Λ⁻¹ + UᵀB⁻¹U (Woodbury), shared across subjects.
        L_B = torch.linalg.cholesky(lmm._group_cov(self.sigma2 * lmm.resid_corr(), self.D))

        U     = lmm.U_grouped
        BiU   = torch.cholesky_solve(U, L_B)
        Bir   = torch.cholesky_solve(r_grouped.unsqueeze(-1), L_B)
        Um    = U.reshape(G * nm, Ktot)
        A_UU  = Um.transpose(0, 1) @ BiU.reshape(G * nm, Ktot)
        UtBir = Um.transpose(0, 1) @ Bir.reshape(G * nm, 1)

        lam   = self._lam()
        M     = A_UU + torch.diag(1.0 / lam)
        M     = 0.5 * (M + M.transpose(0, 1))
        L_M, _ = self.lmm._robust_cholesky(M)
        a_hat = torch.cholesky_solve(UtBir, L_M).squeeze(-1)              # E[a|y]
        self._a_hat = a_hat.detach()

        offsets = [0]
        for k in lmm.K_list:
            offsets.append(offsets[-1] + k)

        fitted_list = []
        for c in colset:
            a_c    = a_hat[offsets[c]:offsets[c + 1]]
            Xeval  = torch.sort(self.unique_spline_vec[c].to(device))[0]
            smooth = self.B_list[c].to(device) @ a_c
            slope_idx = self.spline_linear_col_idx[c]
            center_c  = self.spline_linear_center[c]
            if c == 0:
                linear = self.beta[int(slope_idx)] * (Xeval - center_c) + self.beta[self.px].squeeze()
            else:
                linear = self.beta[slope_idx] * (Xeval - center_c)
            theta_fitted = linear + smooth
            fitted_list.append(pd.DataFrame({
                "spline_id": c,
                "spline_var": Xeval.detach().cpu().numpy(),
                "theta_fitted": theta_fitted.detach().cpu().numpy(),
                "theta_no_intercept": (theta_fitted - (float(self.beta[self.px]) if c == 0 else 0.0)).detach().cpu().numpy(),
            }))

        self.fitted_df = pd.concat(fitted_list, ignore_index=True)
        self.intercept = float(self.beta[self.px])

        if random_effects:
            # V⁻¹r = B⁻¹r − B⁻¹U â
            Vinv_r = Bir.squeeze(-1) - (BiU @ a_hat.unsqueeze(-1)).squeeze(-1)
            w = torch.zeros(N, device=device, dtype=f64)
            w.index_put_((fidx[lmm.mask.bool()],), Vinv_r[lmm.mask.bool()])
            Zw    = self.Z * w.unsqueeze(1)
            b_hat = torch.zeros((G, self.pz), device=device, dtype=f64)
            b_hat.index_add_(0, inv, Zw)
            b_hat = (self.D @ b_hat.T).T
            self.rand_effects = b_hat.detach()
        else:
            self.rand_effects = None

        return self.fitted_df, self.rand_effects

    def _lam(self):
        """Prior variances of the spline coefficients, floored at sigma^2 * exp(-20), the lower
        bound the fit uses, so 1/lam stays finite when a smooth is exactly linear."""
        floor = float(self.sigma2) * float(np.exp(-20.0))
        tau = self.tau_sq_list.clamp(min=floor)
        return torch.repeat_interleave(tau, self.lmm._K_repeats)

    def _posterior_precision(self):
        """Posterior precision H of (beta, a) given the REML variance components, and the
        prior variances lam of a. Cov(beta, a | y) = H^{-1}."""
        f64 = torch.float64
        lmm = self.lmm
        G, nm, K, p = lmm.n_groups, lmm.nmax, lmm.K_tot, self.X_fit.shape[1]
        L0 = torch.linalg.cholesky(lmm._group_cov(self.sigma2 * lmm.resid_corr(), self.D))

        X = lmm.X_grouped * lmm.mask.unsqueeze(-1)
        U = lmm.U_grouped
        ViX = torch.cholesky_solve(X, L0).reshape(G * nm, p)
        ViU = torch.cholesky_solve(U, L0).reshape(G * nm, K)
        X, U = X.reshape(G * nm, p), U.reshape(G * nm, K)

        lam = self._lam()
        H = torch.zeros(p + K, p + K, dtype=f64, device=self.device)
        H[:p, :p] = X.T @ ViX
        H[p:, :p] = U.T @ ViX
        H[:p, p:] = H[p:, :p].T
        H[p:, p:] = U.T @ ViU + torch.diag(1.0 / lam)
        return 0.5 * (H + H.T), lam

    def _smooth_design(self, c, idx=None, intercept=True):
        """Rows mapping the stacked coefficients [X_fit columns; a] to f_c at the sorted knots
        unique_spline_vec[c][idx] (all knots by default). Smooth 0 carries the intercept."""
        p, off = self.X_fit.shape[1], np.cumsum([0] + list(self.lmm.K_list))
        u = self.unique_spline_vec[c]
        idx = torch.arange(u.numel(), device=u.device) if idx is None else idx
        E = torch.zeros(idx.numel(), p + self.lmm.K_tot, dtype=u.dtype, device=self.device)
        E[:, int(self.spline_linear_col_idx[c])] = u[idx] - self.spline_linear_center[c]
        if intercept and c == 0:
            E[:, self.px] = 1.0
        E[:, p + off[c]: p + off[c + 1]] = self.B_list[c][idx]
        return E

    def posterior_moments(self):
        """(mu, H): posterior mean [beta; a] of length p + K_tot and its precision H, so
        Cov = H^{-1}, conditional on the REML variance components."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        self.get_blup(random_effects=False)
        H, _ = self._posterior_precision()
        return torch.cat([self.beta.reshape(-1), self._a_hat]), H

    def smooth_sum_confint(self, ids, weights=None, alpha=0.05, with_mean=False):
        """Pointwise band for sum_c w_c f_c(x) on the x values shared by the smooths in ids,
        using their joint posterior covariance (e.g. ids=[0, 1] for f_ref + delta). Smooth 0
        carries the intercept; with_mean=False centers the curve over the grid."""
        from scipy.stats import norm
        ids = list(ids)
        w = np.ones(len(ids)) if weights is None else np.asarray(weights, float)
        grid = self.unique_spline_vec[ids[0]]
        for c in ids[1:]:
            grid = grid[torch.isin(grid, self.unique_spline_vec[c])]
        if grid.numel() == 0:
            raise ValueError("The smooths share no x values.")
        mu, H = self.posterior_moments()
        L_H, _ = self.lmm._robust_cholesky(H)
        E = sum(wc * self._smooth_design(c, torch.searchsorted(self.unique_spline_vec[c], grid))
                for c, wc in zip(ids, w))
        if not with_mean:
            E = E - E.mean(0, keepdim=True)
        fit = (E @ mu).detach().cpu().numpy()
        se = torch.sqrt((E * torch.cholesky_solve(E.T, L_H).T).sum(1).clamp(min=0.0)).detach().cpu().numpy()
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return pd.DataFrame({"spline_var": grid.detach().cpu().numpy(), "fitted": fit, "se": se,
                             "lower": fit - z * se, "upper": fit + z * se})

    def fitted_values(self, conditional=False):
        """Per-row fitted values (N,), in input order. Marginal: X_fit beta + sum_c mask_c f_c^smooth(x),
        with the intercept counted once on every row. conditional=True adds Z b_hat."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        self.get_blup(random_effects=conditional)
        dev = self.device
        off = np.cumsum([0] + list(self.lmm.K_list))
        fit = (self.X_fit @ self.beta).reshape(-1)
        for c in range(self.spline_inputs.shape[1]):
            u = self.unique_spline_vec[c]
            idx = torch.searchsorted(u, self.spline_inputs[:, c].contiguous()).clamp(max=u.numel() - 1)
            sm = self.B_list[c].to(dev)[idx] @ self._a_hat[off[c]:off[c + 1]]
            fit = fit + (sm if self.smooth_mask is None else self.smooth_mask[:, c] * sm)
        if conditional:
            _, inv = torch.unique(self.obs_id.to(dev), sorted=True, return_inverse=True)
            fit = fit + (self.Z * self.rand_effects[inv]).sum(1)
        return fit.detach().cpu().numpy()

    def residuals(self, conditional=True):
        """y minus fitted_values(conditional), one per row, in input order."""
        return self.y.reshape(-1).detach().cpu().numpy() - self.fitted_values(conditional=conditional)

    def smooth_confint(self, alpha=0.05, col_smooth=None, z=None, with_mean=False):
        """Pointwise confidence band for each smooth, conditional on the REML variance
        components. with_mean=False (mgcv's seWithMean=FALSE) returns the smooth centered over
        its x-grid, with the uncertainty of its shape only; with_mean=True keeps the level (the
        intercept for the first smooth) and its uncertainty, which Marra & Wood (2012) report
        gives better coverage. Refs: Wahba (1983); Nychka (1988)."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0)) if z is None else z

        H, _ = self._posterior_precision()
        L_H, _ = self.lmm._robust_cholesky(H)

        self.get_blup(col_smooth=col_smooth, random_effects=False)
        fdf = self.fitted_df

        colset = [col_smooth] if col_smooth is not None else range(self.spline_inputs.shape[1])

        out = []
        for c in colset:
            E = self._smooth_design(c)
            sub = fdf[fdf["spline_id"] == c].copy()
            if not with_mean:
                E = E - E.mean(0, keepdim=True)
                sub["theta_fitted"] = sub["theta_fitted"] - sub["theta_fitted"].mean()
            var = (E * torch.cholesky_solve(E.T, L_H).T).sum(1).clamp(min=0.0)
            se  = torch.sqrt(var).detach().cpu().numpy()
            sub["se"]    = se
            sub["lower"] = sub["theta_fitted"] - z * se
            sub["upper"] = sub["theta_fitted"] + z * se
            out.append(sub)
        self.confint_df = pd.concat(out, ignore_index=True)
        return self.confint_df

    def fixed_t_test(self, l=None, fixed_only=True):
        """Satterthwaite t-test for fixed effects, delegated to the fitted self.lmm.

        fixed_only (default True): the l=None table covers only the first px columns
        (your X); the spline parametric intercept/slope columns stay reachable via an
        explicit l index. Returns a dict (single contrast) or list of dicts (table).
        """
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        res = self.lmm.fixed_t_test(l=l)
        if l is None and fixed_only:
            res = [row for row in res if row["coef"] < self.px]
        return res

    def linearity_test(self, s=None, alpha=0.05, null_fit_kwargs=None):
        """Score test of H0: smooth component s is linear (tau_s^2 = 0), delegated to
        the fitted self.lmm. See docs/linearity_test.md. s=None tests all components."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        return self.lmm.linearity_test(s=s, alpha=alpha, null_fit_kwargs=null_fit_kwargs)

    def _wald_smooth(self, c, fitted, H, L_H, lam):
        """Wood (2013) Wald test of H0: f_c = 0, f_c = slope * (x - center) + B_c a_c, at the
        observed x. Statistic f' V_f^- f with V_f = Cov(f | y) truncated to rank
        r = round(edf). Returns (edf, stat, r, p)."""
        from scipy import stats
        f64, dev = torch.float64, self.device
        lmm = self.lmm
        p, K = self.X_fit.shape[1], lmm.K_tot
        off = np.concatenate([[0], np.cumsum(lmm.K_list)])
        cols = torch.cat([torch.tensor([int(self.spline_linear_col_idx[c])], device=dev),
                          torch.arange(p + off[c], p + off[c + 1], device=dev)])

        E = self._smooth_design(c, intercept=False)[:, cols]

        Hinv = torch.cholesky_solve(torch.eye(p + K, dtype=f64, device=dev), L_H)
        pen = torch.cat([torch.zeros(p, dtype=f64, device=dev), 1.0 / lam])
        edf = torch.diagonal(Hinv @ (H - torch.diag(pen)))[cols].sum().item()

        Q, R = torch.linalg.qr(E)
        M = R @ Hinv[cols][:, cols] @ R.T
        f = torch.as_tensor(fitted["theta_fitted"].to_numpy().copy(), dtype=f64, device=dev)
        if c == 0:
            f = f - self.beta[self.px].reshape(())
        w = Q.T @ f

        ev, V = torch.linalg.eigh(0.5 * (M + M.T))
        ev, V = ev.flip(0), V.flip(1)
        r = int(min(max(round(edf), 1), ev.numel()))
        stat = ((V[:, :r].T @ w) ** 2 / ev[:r]).sum().item()
        return edf, stat, r, float(stats.chi2.sf(stat, r))

    def spline_effect_test(self, s=None, alpha=0.05, null_fit_kwargs=None):
        """H0: f_s = 0 for spline s (None = all). One row per spline, two tests:
        Fisher: -2 (log p_linear + log p_nonlinear) ~ chi2_4, combining the t-test of the
          linear trend with the boundary score test of tau_s^2 = 0 (asymptotically independent).
        Wald: Wood (2013)-style Wald test of the fitted smooth, chi2_r."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        from scipy import stats
        H, lam = self._posterior_precision()
        L_H, _ = self.lmm._robust_cholesky(H)
        fdf = self.get_blup(random_effects=False)[0]

        rows = []
        for c in (range(self.spline_inputs.shape[1]) if s is None else [s]):
            lin = self.lmm.fixed_t_test(l=int(self.spline_linear_col_idx[c]))
            nonlin = self.lmm.linearity_test(s=c, alpha=alpha, null_fit_kwargs=null_fit_kwargs)
            stat = -2.0 * np.log(np.maximum([lin["p_value"], nonlin["p_value"]], 1e-300)).sum()
            p_fisher = float(stats.chi2.sf(stat, 4))
            edf, w_stat, w_df, p_wald = self._wald_smooth(c, fdf[fdf["spline_id"] == c], H, L_H, lam)
            rows.append(dict(spline_id=c, t_linear=lin["t"], p_linear=lin["p_value"],
                             score_stat=nonlin["statistic"], p_nonlinear=nonlin["p_value"],
                             fisher_stat=stat, p_fisher=p_fisher,
                             edf=edf, wald_stat=w_stat, wald_df=w_df, p_wald=p_wald,
                             reject_fisher=bool(p_fisher < alpha), reject_wald=bool(p_wald < alpha)))
        return pd.DataFrame(rows)

    def summary(self, names=None, spline_names=None, re_names=None, alpha=0.05, null_fit_kwargs=None):
        """{"coefficients", "variance", "splines"} DataFrames; displays as formatted tables when printed.
        Coefficient rows: X, spline intercept, linear slope of each spline."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        S = self.spline_inputs.shape[1]
        names = list(names) if names is not None else [f"X{i + 1}" for i in range(self.px)]
        sp = list(spline_names) if spline_names is not None else [f"spline{j + 1}" for j in range(S)]
        if len(names) != self.px or len(sp) != S:
            raise ValueError(f"names needs {self.px} entries and spline_names {S}.")
        coefs = self.lmm.coef_table(names=names + ["(Intercept)"] + [f"s({n}) linear" for n in sp] + self.level_names,
                                        alpha=alpha)

        tau = pd.DataFrame(dict(component=[f"tau2 s({n})" for n in sp],
                                variance=self.tau_sq_list.detach().cpu().numpy()))
        var = pd.concat([self.lmm.variance_table(re_names=re_names), tau], ignore_index=True)

        tests = self.spline_effect_test(alpha=alpha, null_fit_kwargs=null_fit_kwargs)
        tests.insert(1, "term", [f"s({sp[c]})" for c in tests["spline_id"]])
        return SummaryResult(coefficients=coefs, variance=var, splines=tests)


class SmoothingSpline(SmoothingSplineMixed):
    """One penalized curve, no grouping: SmoothingSplineMixed with a single group and no random effects.
    X and Z default to zero columns, obs_id to one group."""

    def __init__(self, y, X=None, Z=None, spline_inputs=None, obs_id=None, device=None, corr_structure='iid',
                 rand_D_type='diagonal', rand_D_estimate='sample'):
        N = len(y)
        super().__init__(y=y, X=np.zeros((N, 0)) if X is None else X, Z=np.zeros((N, 0)) if Z is None else Z,
                         spline_inputs=spline_inputs, obs_id=np.zeros(N) if obs_id is None else obs_id,
                         device=device, corr_structure=corr_structure, rand_D_type=rand_D_type,
                         rand_D_estimate=rand_D_estimate)

    def get_fitted_theta(self, col=0):
        """Fitted curve at the distinct x values: DataFrame with spline_var and theta_fitted."""
        return self.get_blup(col_smooth=col, random_effects=False)[0]
