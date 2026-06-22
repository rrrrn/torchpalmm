import torch
from .linear_mixed_model import _checktensor, LinearMixedModel
import torch.nn.functional as F

class SmoothingSpline:
    @staticmethod
    def construct_Q(x):
        ## make sure x is a 1D torch tensor
        x = x.flatten()
        n = x.shape[0]
        h = x[1:] - x[:-1]  

        Q = torch.zeros((n, n-2), dtype=x.dtype, device=x.device)
        ## diagonals
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

        # Diagonal: (1/3)*(h_j + h_{j+1})
        cols = torch.arange(n-2)
        R[cols, cols] = (h[cols] + h[cols + 1]) / 3.0
        # Off-diagonals: 1/6 * h_{j}
        cols = torch.arange(n-3)
        R[cols + 1, cols] = h[cols+1] / 6.0
        R[cols, cols + 1] = h[cols+1] / 6.0
        return R

    @staticmethod
    def construct_Omega(x):
        Q = SmoothingSpline.construct_Q(x)
        R = SmoothingSpline.construct_R(x)

        # Omega = Q @ torch.linalg.solve(R, Q.T)

        return Q, R
    
    @staticmethod
    def construct_incidence_matrix(origin_vec, unique_vec):
        n = len(origin_vec)
        m = len(unique_vec)
        incidence_matrices = torch.zeros((n, m),
                                 dtype=unique_vec.dtype,
                                 device=unique_vec.device)

        col = 0
        for tps in unique_vec:       
            idx = (origin_vec == tps).nonzero(as_tuple=True)[0]
            incidence_matrices[idx, col] = 1.0
            col += 1  
        return incidence_matrices
    
    @staticmethod
    def prepare_design_matrix(original_vec, unique_vec):
        unique_vec = torch.sort(unique_vec, descending=False)[0]
        Q, R = SmoothingSpline.construct_Omega(unique_vec)

        ## cholesky of R inverse
        C = torch.linalg.cholesky(R)
        I = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        S = torch.linalg.solve(C, I)  # C^{-1}
        L = Q @ S ## n x n-2

        ## incidence matrix
        Nmtx = SmoothingSpline.construct_incidence_matrix(original_vec, unique_vec)

        ## random effect design matrix for smoothing splines
        LtL = L.T @ L
        chol = torch.linalg.cholesky(LtL + 1e-10*torch.eye(LtL.shape[0], device=L.device, dtype=L.dtype))
        B = torch.cholesky_solve(L.T, chol).T 
        NB = Nmtx @ B

        ## input = [1, time] as a torch tensor
        input_tilde = torch.stack((torch.ones_like(original_vec), original_vec), dim=1)
        input_tilde_unique = torch.stack((torch.ones_like(unique_vec), unique_vec), dim=1)
        return NB, B, input_tilde, input_tilde_unique

    def __init__(self, y, X, Z, spline_inputs, obs_id, device=None, corr_structure = 'iid', rand_D_type = 'diagonal',
                 rand_D_estimate = 'sample'):
        self.corr_structure = corr_structure; self.rand_D_type = rand_D_type; self.rand_D_estimate = rand_D_estimate
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.X = _checktensor(X); self.y = _checktensor(y).to(device=self.device); self.Z = _checktensor(Z).to(device=self.device)
        self.pz = Z.shape[1]; self.px = X.shape[1]
        self.spline_inputs = _checktensor(spline_inputs).to(device=self.device) ## a matrix of size n x p (p spline variables)
        if len(self.spline_inputs.shape) == 1:
            self.spline_inputs = self.spline_inputs.unsqueeze(1)
        self.unique_spline_vec = [torch.unique(self.spline_inputs[:, i]) for i in range(self.spline_inputs.shape[1])]
        self.obs_id = _checktensor(obs_id)

        X_fit = self.X; NB_list = []; B_list = []; self.spline_linear_col_idx = []
        for i in range(self.spline_inputs.shape[1]):
            sp = self.spline_inputs[:, i]
            NB, B, input_tilde, _ = self.prepare_design_matrix(sp, torch.unique(sp))
            NB_list.append(NB); B_list.append(B)
            
            if i == 0:
                X_fit = torch.hstack((X_fit, input_tilde))  
            else:
                X_fit = torch.hstack((X_fit, input_tilde[:, 1:]))  # x only
            self.spline_linear_col_idx.append(X_fit.shape[1] - 1)
        self.NB_list = NB_list; self.B_list = B_list
        ## remove repeatedly added ones in X_fit
        self.X_fit = X_fit.to(device=self.device)
        
    def fit(self, warm_start=True, lr=1e-3, max_iter=1000, optimizer=torch.optim.Adam, 
            tol=1e-6, max_iter_inner=10, convg_check="loss", fit_type="reml", grad_control_tol=1e-4,
             verbose =True):
        
        if self.pz==0:
            ## only one group
            id = torch.ones(self.y.shape[0], dtype=torch.long, device=self.device)
            self.lmm = LinearMixedModel(
                y = self.y, X = self.X_fit, Z = self.NB_list[0], id = id, corr_structure=self.corr_structure, device=self.device,
                rand_D_type="iid", rand_D_estimate=self.rand_D_estimate
            )
            self.lmm.fit_likelihood(lr=lr, max_iter=max_iter, tol = tol, verbose=verbose, optimizer=optimizer, fit_type=fit_type,warm_start=warm_start,
                                    max_iter_inner=max_iter_inner, convg_check=convg_check)
        else:
            return NotImplementedError("Smoothing spline allow only one spline variable for now. Multiple spline variables will be implemented in the future.")
            
        self.sigma2 = self.lmm.sigma2.detach()
        self.beta = self.lmm.beta.detach()
        # self.Dz = self.lmm.Dz.detach()
        self.D = self.lmm.D.detach()
        self.tau_sq_list = [torch.unique(torch.diagonal(self.lmm.D)).detach()]
        return self.lmm 
    
    def get_fitted_theta(self, col = 0):
        ## make sure the model is already fitted
        if not hasattr(self, 'beta'):
            raise ValueError("Model not fitted yet. Please call the fit() method first.")
        
        Xeval = torch.stack((torch.ones_like(self.unique_spline_vec[col]), self.unique_spline_vec[col]), dim=1).to(device=self.device)
        tau = self.tau_sq_list[col]; NB = self.NB_list[col]; B = self.B_list[col]
        V = tau * NB @ NB.T + self.sigma2 * torch.eye(NB.shape[0], device=self.device)
        deltahat = self.beta

        ahat = tau * NB.T @ torch.linalg.solve(V, self.y - self.X_fit @ deltahat)
        theta_fitted = (Xeval @ deltahat + B @ ahat).squeeze()
        
        self.fitted_df = {
            'spline_var': torch.sort(self.unique_spline_vec[col])[0].to(device=self.device).cpu().numpy(), 
            'theta_fitted': theta_fitted.to(device=self.device).cpu().numpy()
        }
        return self.fitted_df