
import copy
import warnings
import torch
from .linear_mixed_model import SummaryResult, _fill_names
from .linear_mixed_model_smoothing import LinearMixedModelSmoothing
import numpy as np
import pandas as pd
from tqdm import tqdm


def _checktensor(x, dtype=torch.float64):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype)


def _colnames(v):
    if isinstance(v, pd.DataFrame):
        return [str(c) for c in v.columns]
    if isinstance(v, pd.Series) and v.name is not None:
        return [str(v.name)]
    return None


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
    def place_knots(u, k):
        """k knots spread evenly through the sorted distinct values u, as mgcv's place.knots (bs = "cr")."""
        if k < 3:
            raise ValueError("n_knots must be at least 3.")
        pos = torch.linspace(0, u.numel() - 1, k, dtype=u.dtype, device=u.device)
        lo = pos.floor().long().clamp(max=u.numel() - 2)
        return u[lo] + (pos - lo) * (u[lo + 1] - u[lo])

    @staticmethod
    def basis_matrix(x, knots):
        """N x k natural cubic spline evaluation f(x) = F g from knot values g (Green & Silverman 1994,
        sec. 2.4); the 0/1 incidence matrix when every x is a knot."""
        x, k = x.reshape(-1).contiguous(), knots.numel()
        j = (torch.searchsorted(knots, x, right=True) - 1).clamp(0, k - 2)
        h = (knots[1:] - knots[:-1])[j]
        dl, dr = x - knots[j], knots[j + 1] - x
        rows = torch.arange(x.numel(), device=x.device)
        A = torch.zeros(x.numel(), k, dtype=x.dtype, device=x.device)
        C = torch.zeros_like(A)
        A[rows, j], A[rows, j + 1] = dr / h, dl / h
        C[rows, j], C[rows, j + 1] = -dl * dr / 6 * (1 + dr / h), -dl * dr / 6 * (1 + dl / h)
        Q, R = SmoothingSplineMixed.construct_Omega(knots)
        return A + C[:, 1:-1] @ torch.linalg.solve(R, Q.T)       # second derivatives gamma = R^-1 Q' g

    @staticmethod
    def construct_incidence_matrix(origin_vec, unique_vec, ids):
        """Basis evaluation at origin_vec for knots unique_vec, grouped to [G, nmax, k]; and the group indices."""
        group_indices = [(ids == g).nonzero(as_tuple=True)[0] for g in torch.unique(ids)]
        pad = torch.nn.utils.rnn.pad_sequence(group_indices, batch_first=True)
        valid = torch.arange(pad.shape[1], device=pad.device)[None, :] < torch.tensor(
            [g.numel() for g in group_indices], device=pad.device)[:, None]
        F = SmoothingSplineMixed.basis_matrix(origin_vec.to(torch.float64), unique_vec)
        return F[pad] * valid.unsqueeze(-1), group_indices

    @staticmethod
    def prepare_design_matrix(original_vec, unique_vec, ids, flat=False):
        """flat=True returns NB per row, [N, K], instead of padded per subject, [G, nmax, K]."""
        unique_vec = torch.sort(unique_vec, descending=False)[0]
        Q, R = SmoothingSplineMixed.construct_Omega(unique_vec)

        C = torch.linalg.cholesky(R)
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        S = torch.linalg.solve_triangular(C, I, upper=False)
        L = Q @ S

        # random-effect design for the spline: B = L (L'L)^{-1}
        B = torch.linalg.pinv(L).T
        if flat:
            NB = SmoothingSplineMixed.basis_matrix(original_vec.to(torch.float64), unique_vec) @ B
            group_indices = [(ids == g).nonzero(as_tuple=True)[0] for g in torch.unique(ids)]
        else:
            Nmtx, group_indices = SmoothingSplineMixed.construct_incidence_matrix(original_vec, unique_vec, ids)
            NB = Nmtx @ B

        # [1, x] parametric part
        input_tilde = torch.stack((torch.ones_like(original_vec), original_vec), dim=1)
        input_tilde_unique = torch.stack((torch.ones_like(unique_vec), unique_vec), dim=1)
        return NB, B, input_tilde, input_tilde_unique, group_indices

    def __init__(self, y, X=None, Z=None, spline_inputs=None, obs_id=None, device=None, corr_structure = 'iid',
                 rand_D_type = 'diagonal', rand_D_estimate = 'sample', smooth_mask=None, n_knots=None,
                 standardize=False):
        """X, Z: None means no columns; DataFrame columns name the summary rows. obs_id: subject labels of
        any type. n_knots: int (or one per smooth) for a reduced-rank basis with knots placed as mgcv's
        bs="cr"; None puts a knot at every distinct x. standardize: center and scale the non-binary X
        columns internally; summary() reports their coefficients per original unit."""
        self.corr_structure = corr_structure; self.rand_D_type = rand_D_type; self.rand_D_estimate = rand_D_estimate
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        N = len(y)
        self.x_names, self.re_names, self.spline_names = (_colnames(v) for v in (X, Z, spline_inputs))
        as2d = lambda v: v.unsqueeze(1) if v.dim() == 1 else v
        self.X = as2d(_checktensor(np.zeros((N, 0)) if X is None else X).to(device=self.device))
        self.y = _checktensor(y).to(device=self.device)
        self.Z = as2d(_checktensor(np.zeros((N, 0)) if Z is None else Z).to(device=self.device))
        self.pz = self.Z.shape[1]; self.px = self.X.shape[1]
        self.spline_inputs = as2d(_checktensor(spline_inputs).to(device=self.device))
        for what, T, nm in (("y", self.y.reshape(N, -1), ["y"]), ("X", self.X, self.x_names),
                            ("Z", self.Z, self.re_names), ("spline_inputs", self.spline_inputs, self.spline_names)):
            bad = (~torch.isfinite(T)).sum(0)
            if bad.any():
                cols = [f"{nm[j] if nm else j} ({int(bad[j])} rows)" for j in bad.nonzero().flatten().tolist()]
                raise ValueError(f"{what} has missing or non-finite values in column(s) {', '.join(cols)}; drop or "
                                 f"impute those rows (from_dataframe drops them). Inactive smooth_mask rows need "
                                 f"finite placeholder values too.")
        self.x_center =torch.zeros(self.px, dtype=self.X.dtype, device=self.device)
        self.x_scale = torch.ones_like(self.x_center)
        self.standardize = bool(standardize) and self.px > 0
        if self.standardize:
            cont = torch.tensor([self.X[:, j].unique().numel() > 2 for j in range(self.px)], device=self.device)
            self.x_center[cont], self.x_scale[cont] = self.X[:, cont].mean(0), self.X[:, cont].std(0)
            self.X = (self.X - self.x_center) / self.x_scale
        ids = obs_id.cpu().numpy() if torch.is_tensor(obs_id) else np.asarray(obs_id)
        codes, self.group_labels = pd.factorize(ids.reshape(-1), sort=True)
        if (codes < 0).any():
            raise ValueError(f"obs_id has {(codes < 0).sum()} missing values; drop those rows.")
        obs_id = torch.as_tensor(codes, device=self.device)
        self.nmax = int(torch.bincount(obs_id).max())
        self.obs_id = obs_id

        S = self.spline_inputs.shape[1]
        m = self._check_mask(smooth_mask, S)
        self.smooth_mask = m
        self._flat = self._splits_subjects(m, obs_id)
        nk = list(n_knots) if isinstance(n_knots, (list, tuple)) else [n_knots] * S
        # distinct x on the rows where the spline is active (all rows when unmasked); the knots unless n_knots
        self.unique_spline_vec, self.knots = [], []
        for i in range(S):
            sp = self.spline_inputs[:, i]
            u = torch.sort(torch.unique(sp if m is None else sp[m[:, i] != 0]))[0]
            if u.numel() < 3:
                raise ValueError(f"spline {i} has {u.numel()} distinct active x-values; at least 3 are needed.")
            self.unique_spline_vec.append(u)
            self.knots.append(u if nk[i] is None or u.numel() <= nk[i] else self.place_knots(u, nk[i]))
        X_fit = self.X; NB_list = []; B_list = []; self.spline_linear_col_idx = []
        self.spline_linear_center = []
        for i in range(S):
            sp = self.spline_inputs[:, i]
            NB, B, input_tilde, _, group_indices = self.prepare_design_matrix(sp, self.knots[i], obs_id, flat=self._flat)
            if self._flat:
                NB = NB * m[:, i:i + 1]
            elif m is not None:
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

    def _splits_subjects(self, m, obs_id):
        """True when the masks split the subjects into several independent blocks of smooths (e.g. one curve
        per subject-level group) and the iid Woodbury path applies. NB_list is then per row, [N, K], and the
        REML runs per block; otherwise (no mask, masks within subjects) the padded dense path is unchanged."""
        iid = self.corr_structure is None or LinearMixedModelSmoothing._structure_kind(self.corr_structure) == "iid"
        if m is None or not iid or self.pz >= self.nmax:
            return False
        return bool(LinearMixedModelSmoothing.smooth_blocks(m != 0, obs_id)[0].max() > 0)

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
        tau_lr_mult=5.0, verbose=False, n_restarts=1, rho_init="moment", check_hessian=False):
        self.lmm = LinearMixedModelSmoothing(device=self.device, y=self.y, X=self.X_fit, Z=self.Z,
                                            spline_inputs=self.NB_list,
                                            id=self.obs_id, corr_structure=self.corr_structure, rand_D_type=self.rand_D_type, rand_D_estimate=self.rand_D_estimate)
        self.lmm.re_labels = _fill_names(self.re_names, self.pz, "b")
        self.lmm.spline_labels = _fill_names(self.spline_names, self.spline_inputs.shape[1], "spline")
        self.lmm.fit_likelihood(lr=lr, max_iter=max_iter, tol=tol, verbose=verbose, optimizer=optimizer, fit_type=fit_type, warm_start=warm_start,
                                    max_iter_inner=max_iter_inner, convg_check=convg_check, grad_control_tol=grad_control_tol, tau_lr_mult=tau_lr_mult,
                                    corr_init_method=rho_init, check_hessian=check_hessian)

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
                    check_hessian=check_hessian,
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

        # A smooth on the boundary with a positive REML score is stuck there, not at an optimum: refit once
        # with those smooths restarted at tau2 = sigma2 and keep the lower criterion.
        stuck = self.lmm.stuck_smooths()
        if stuck:
            prev, lab = copy.deepcopy(self.lmm), self.lmm._spline_labels()
            init = dict(D_params=prev.D_params.detach(), spline_params=prev.spline_params.detach().clone())
            init["spline_params"][stuck] = 0.0
            if prev._has_corr_params():
                init["rho_raw"] = prev.rho_raw.detach()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lmm.fit_likelihood(**{**prev._fit_settings, "verbose": False}, fit_type=fit_type,
                                        warm_start=warm_start, init=init, check_hessian=check_hessian)
            gain = prev.final_loss - self.lmm.final_loss
            if gain > 0:
                self.lmm._check_fit(check_hessian)               # re-issue the kept refit's own flags
            else:
                self.lmm = prev
            still = self.lmm.stuck_smooths()
            warnings.warn(f"smooth(s) {', '.join(lab[s] for s in stuck)} sat on the boundary with a positive REML "
                          f"score (stuck, not optimal). A refit from tau2 = sigma2 "
                          + (f"lowered the criterion by {gain:.3g} and was kept (converged: {bool(self.lmm.convg)})"
                             if gain > 0 else "did not lower the criterion")
                          + (f"; still stuck: {', '.join(lab[s] for s in still)}" if still else "") + ".")

        self.sigma2 = self.lmm.sigma2.detach()
        self.resid_cov = self.sigma2 * self.lmm.resid_corr().detach()
        self.lmm.resid_cov = self.resid_cov
        self.beta = self.lmm.beta.detach()
        self.D = self.lmm.D.detach()
        self.tau_sq_list = self.lmm.tuning_params.detach()
        self.boundary = self.lmm.boundary
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
        p = self.X_fit.shape[1]
        if lmm._woodbury:
            A     = self._gram()
            A_UU  = A[p + 1:, p + 1:]
            UtBir = A[p + 1:, p:p + 1] - A[p + 1:, :p] @ self.beta
        else:
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
            smooth = self._basis_at(c, Xeval) @ a_c
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

        if random_effects and lmm._woodbury:
            # b_i = D_rel R_i' T_i^-1 Q_i'(r_i - U_i â)
            b_hat = torch.zeros((G, self.pz), device=device, dtype=f64)
            if self.pz:
                D_rel = self.D / self.sigma2
                q = lmm._wb_resid(self.beta, a_hat)
                t = torch.cholesky_solve(q, lmm._woodbury_chol(D_rel))
                b_hat = (D_rel @ lmm._wb_R.transpose(-1, -2) @ t).squeeze(-1)
            self.rand_effects = b_hat.detach()
        elif random_effects:
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

    def _basis_at(self, c, x):
        """Rows of smooth c's penalized basis (knot evaluation times B) at x."""
        return self.basis_matrix(x, self.knots[c]) @ self.B_list[c].to(self.device)

    def _lam(self):
        """Prior variances of the spline coefficients, floored at sigma^2 * exp(-20), the lower
        bound the fit uses, so 1/lam stays finite when a smooth is exactly linear."""
        floor = float(self.sigma2) * float(np.exp(-20.0))
        tau = self.tau_sq_list.clamp(min=floor)
        return torch.repeat_interleave(tau, self.lmm._K_repeats)

    def _gram(self):
        """[X y U]' B^-1 [X y U] with B = sigma^2 R + Z D Z' at the fit (iid fast path)."""
        return self.lmm._woodbury_dense(self.D / self.sigma2)[1] / self.sigma2

    def _posterior_precision(self):
        """Posterior precision H of (beta, a) given the REML variance components, and the
        prior variances lam of a. Cov(beta, a | y) = H^{-1}."""
        f64 = torch.float64
        lmm = self.lmm
        G, nm, K, p = lmm.n_groups, lmm.nmax, lmm.K_tot, self.X_fit.shape[1]
        lam = self._lam()
        if lmm._woodbury:
            keep = [i for i in range(p + K + 1) if i != p]
            H = self._gram()[keep][:, keep]
            H[p:, p:] += torch.diag(1.0 / lam)
            return 0.5 * (H + H.T), lam
        L0 = torch.linalg.cholesky(lmm._group_cov(self.sigma2 * lmm.resid_corr(), self.D))

        X = lmm.X_grouped * lmm.mask.unsqueeze(-1)
        U = lmm.U_grouped
        ViX = torch.cholesky_solve(X, L0).reshape(G * nm, p)
        ViU = torch.cholesky_solve(U, L0).reshape(G * nm, K)
        X, U = X.reshape(G * nm, p), U.reshape(G * nm, K)

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
        E[:, p + off[c]: p + off[c + 1]] = self._basis_at(c, u[idx])
        return E

    def posterior_moments(self):
        """(mu, H): posterior mean [beta; a] of length p + K_tot and its precision H, so
        Cov = H^{-1}, conditional on the REML variance components."""
        if not hasattr(self, "beta"):
            raise ValueError("Model not fitted yet.")
        self.get_blup(random_effects=False)
        H, _ = self._posterior_precision()
        return torch.cat([self.beta.reshape(-1), self._a_hat]), H

    def smooth_sum_confint(self, ids, weights=None, alpha=0.05, with_mean=False, transform=None):
        """Pointwise band for sum_c w_c f_c(x) on the x values shared by the smooths in ids,
        using their joint posterior covariance (e.g. ids=[0, 1] for f_ref + delta). Smooth 0
        carries the intercept; with_mean=False centers the curve over the grid. transform: see smooth_confint."""
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
        g = (lambda v: v) if transform is None else transform
        return pd.DataFrame({"spline_var": grid.detach().cpu().numpy(), "fitted": g(fit), "se": se,
                             "lower": g(fit - z * se), "upper": g(fit + z * se)})

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
            sm = self._basis_at(c, self.spline_inputs[:, c]) @ self._a_hat[off[c]:off[c + 1]]
            fit = fit + (sm if self.smooth_mask is None else self.smooth_mask[:, c] * sm)
        if conditional:
            _, inv = torch.unique(self.obs_id.to(dev), sorted=True, return_inverse=True)
            fit = fit + (self.Z * self.rand_effects[inv]).sum(1)
        return fit.detach().cpu().numpy()

    def residuals(self, conditional=True):
        """y minus fitted_values(conditional), one per row, in input order."""
        return self.y.reshape(-1).detach().cpu().numpy() - self.fitted_values(conditional=conditional)

    def augment(self):
        """Per-row DataFrame: group, y, marginal and conditional fitted values and residuals, and the
        conditional residual standardized by its residual SD."""
        y = self.y.reshape(-1).detach().cpu().numpy()
        fm, fc = self.fitted_values(conditional=False), self.fitted_values(conditional=True)
        fidx, mask = self.obs_flat_idx, self.lmm.mask.bool()
        pos = torch.zeros(len(y), dtype=torch.long, device=self.device)
        pos[fidx[mask]] = torch.arange(self.lmm.nmax, device=self.device).expand_as(fidx)[mask]
        sd = torch.diagonal(self.resid_cov).sqrt()[pos].detach().cpu().numpy()
        return pd.DataFrame({"group": self.group_labels[self.obs_id.cpu().numpy()], "y": y,
                             "fitted_marginal": fm, "fitted_conditional": fc,
                             "resid_marginal": y - fm, "resid_conditional": y - fc, "resid_std": (y - fc) / sd})

    def random_effects(self):
        """BLUPs b_i as a DataFrame indexed by subject label, one column per random effect."""
        _, b = self.get_blup(random_effects=True)
        return pd.DataFrame(b.detach().cpu().numpy(), index=pd.Index(self.group_labels, name="group"),
                            columns=_fill_names(self.re_names, self.pz, "b"))

    def r_squared(self):
        """Nakagawa & Schielzeth (2013) marginal R2 (fixed part and smooths), conditional R2 (+ random
        effects) and ICC = var_re / (var_re + sigma2), on the model scale."""
        var_f = float(self.fitted_values(conditional=False).var())
        var_re = float(((self.Z @ self.D) * self.Z).sum(1).mean()) if self.pz else 0.0
        s2 = float(torch.diagonal(self.resid_cov).mean())
        tot = var_f + var_re + s2
        return pd.Series({"marginal_R2": var_f / tot, "conditional_R2": (var_f + var_re) / tot,
                          "ICC": var_re / (var_re + s2)})

    def convergence_diagnostics(self):
        """Gradient norm, Hessian eigenvalues and boundary components of the fit (see LinearMixedModel)."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        return self.lmm.convergence_diagnostics()

    def smooth_confint(self, alpha=0.05, col_smooth=None, z=None, with_mean=False, transform=None):
        """Pointwise confidence band for each smooth, conditional on the REML variance
        components. with_mean=False (mgcv's seWithMean=FALSE) returns the smooth centered over
        its x-grid, with the uncertainty of its shape only; with_mean=True keeps the level (the
        intercept for the first smooth) and its uncertainty, which Marra & Wood (2012) report
        gives better coverage. Refs: Wahba (1983); Nychka (1988). transform: increasing function
        applied to the curve and band ends (se stays on the model scale), e.g. np.exp for ratios of
        a centered log-scale curve, or np.expm1 with with_mean=True to undo log1p."""
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
            if transform is not None:
                for k in ("theta_fitted", "lower", "upper"):
                    sub[k] = transform(sub[k].to_numpy())
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

    def linearity_test(self, s=None, alpha=0.05, null_fit_kwargs=None, test="score"):
        """Test of H0: smooth component s is linear (tau_s^2 = 0), test="score" or "lrt", delegated to
        the fitted self.lmm. See docs/linearity_test.md. s=None tests all components."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        return self.lmm.linearity_test(s=s, alpha=alpha, null_fit_kwargs=null_fit_kwargs, test=test)

    def AIC(self):
        """AIC of the fitted model (see LinearMixedModelSmoothing.AIC)."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        return self.lmm.AIC()

    def _wald_smooth(self, c, fitted, H, Hinv, lam):
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

    def spline_effect_test(self, s=None, alpha=0.05, null_fit_kwargs=None, nonlinear_test="score"):
        """H0: f_s = 0 for spline s (None = all). One row per spline, two tests:
        Fisher: -2 (log p_linear + log p_nonlinear) ~ chi2_4, combining the t-test of the
          linear trend with the test of tau_s^2 = 0 (nonlinear_test: "score" or "lrt", on a null refit).
        Wald: Wood (2013)-style Wald test of the fitted smooth, chi2_r.
        edf_conflict flags a rejection of linearity while the main fit's EDF is ~1."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        from scipy import stats
        H, lam = self._posterior_precision()
        L_H, _ = self.lmm._robust_cholesky(H)
        Hinv = torch.cholesky_solve(torch.eye(H.shape[0], dtype=torch.float64, device=self.device), L_H)
        fdf = self.get_blup(random_effects=False)[0]

        rows = []
        for c in (range(self.spline_inputs.shape[1]) if s is None else [s]):
            lin = self.lmm.fixed_t_test(l=int(self.spline_linear_col_idx[c]))
            nonlin = self.lmm.linearity_test(s=c, alpha=alpha, null_fit_kwargs=null_fit_kwargs, test=nonlinear_test)
            stat = -2.0 * np.log(np.maximum([lin["p_value"], nonlin["p_value"]], 1e-300)).sum()
            p_fisher = float(stats.chi2.sf(stat, 4))
            edf, w_stat, w_df, p_wald = self._wald_smooth(c, fdf[fdf["spline_id"] == c], H, Hinv, lam)
            conflict = bool(nonlin["p_value"] < alpha and edf < 1.01)
            if conflict:
                warnings.warn(f"s({self.lmm._spline_labels()[c]}) (s={c}): the {nonlinear_test} test rejects linearity (p = {nonlin['p_value']:.3g}) "
                              f"but the main fit has EDF = {edf:.3f} (tau2 ~ 0), so the null refit and the main fit "
                              f"disagree (null converged: {nonlin['null_converged']}). Check both fits' convergence "
                              f"and compare nonlinear_test='lrt'.")
            rows.append(dict(spline_id=c, t_linear=lin["t"], p_linear=lin["p_value"],
                             nonlinear_test=nonlinear_test, null_converged=nonlin["null_converged"],
                             score_stat=nonlin["statistic"], p_nonlinear=nonlin["p_value"], edf_conflict=conflict,
                             fisher_stat=stat, p_fisher=p_fisher,
                             edf=edf, wald_stat=w_stat, wald_df=w_df, p_wald=p_wald,
                             reject_fisher=bool(p_fisher < alpha), reject_wald=bool(p_wald < alpha)))
        return pd.DataFrame(rows)

    def summary(self, names=None, spline_names=None, re_names=None, alpha=0.05, null_fit_kwargs=None,
                nonlinear_test="score"):
        """{"coefficients", "variance", "splines"} DataFrames; displays as formatted tables when printed.
        Coefficient rows: X, spline intercept, linear slope of each spline. Names default to the
        DataFrame columns given at construction; missing ones are filled as X1, spline1, b1, ..."""
        if not hasattr(self, "lmm"):
            raise ValueError("Model not fitted yet.")
        S = self.spline_inputs.shape[1]
        names = _fill_names(names if names is not None else self.x_names, self.px, "X")
        sp = _fill_names(spline_names if spline_names is not None else self.spline_names, S, "spline")
        re_names = _fill_names(re_names if re_names is not None else self.re_names, self.pz, "b")
        self.lmm.re_labels, self.lmm.spline_labels = re_names, sp       # warnings below use the same labels
        icpt = "(Intercept, X at mean)" if self.x_center.any() else "(Intercept)"
        coefs = self.lmm.coef_table(names=names + [icpt] + [f"s({n}) linear" for n in sp] + self.level_names,
                                        alpha=alpha)
        if self.standardize:                      # per original unit of X; t, df and p are unchanged
            cols, rows = ["estimate", "se", "ci_lower", "ci_upper"], coefs.index[:self.px]
            coefs.loc[rows, cols] = coefs.loc[rows, cols].to_numpy() / self.x_scale.cpu().numpy()[:, None]

        tau = pd.DataFrame(dict(component=[f"tau2 s({n})" for n in sp],
                                variance=self.tau_sq_list.detach().cpu().numpy()))
        var = pd.concat([self.lmm.variance_table(re_names=re_names), tau], ignore_index=True)

        tests = self.spline_effect_test(alpha=alpha, null_fit_kwargs=null_fit_kwargs, nonlinear_test=nonlinear_test)
        tests.insert(1, "term", [f"s({sp[c]})" for c in tests["spline_id"]])
        return SummaryResult(coefficients=coefs, variance=var, splines=tests)

    @classmethod
    def from_dataframe(cls, data, y, smooth, group, x=(), random_slopes=(), random_intercept=True, **kwargs):
        """Build from a DataFrame. x: columns (categoricals dummy-coded, first level as reference) or a
        formulaic right-hand side such as "age + C(cls) * sex" (pip install formulaic; keep its intercept,
        it is dropped after coding). group: subject labels. Rows with missing values are dropped."""
        smooth = [smooth] if isinstance(smooth, str) else list(smooth)
        data = data.dropna(subset=[y, group, *smooth, *random_slopes] + ([] if isinstance(x, str) else list(x)))
        if isinstance(x, str):
            from formulaic import model_matrix
            Xdf = model_matrix(x, data).drop(columns="Intercept", errors="ignore")
            data = data.loc[Xdf.index]
        else:
            Xdf = pd.get_dummies(data[list(x)], drop_first=True, dtype=float)
        Zdf = data[list(random_slopes)].astype(float)
        if random_intercept:
            Zdf.insert(0, "(Intercept)", 1.0)
        return cls(y=data[y].to_numpy(float), X=Xdf.astype(float), Z=Zdf, spline_inputs=data[smooth],
                   obs_id=data[group].to_numpy(), **kwargs)


class SmoothingSpline(SmoothingSplineMixed):
    """One penalized curve, no grouping: SmoothingSplineMixed with a single group and no random effects.
    X and Z default to zero columns, obs_id to one group."""

    def __init__(self, y, X=None, Z=None, spline_inputs=None, obs_id=None, device=None, corr_structure='iid',
                 rand_D_type='diagonal', rand_D_estimate='sample', n_knots=None):
        N = len(y)
        super().__init__(y=y, X=X, Z=Z, spline_inputs=spline_inputs, obs_id=np.zeros(N) if obs_id is None else obs_id,
                         device=device, corr_structure=corr_structure, rand_D_type=rand_D_type,
                         rand_D_estimate=rand_D_estimate, n_knots=n_knots)

    def get_fitted_theta(self, col=0):
        """Fitted curve at the distinct x values: DataFrame with spline_var and theta_fitted."""
        return self.get_blup(col_smooth=col, random_effects=False)[0]
