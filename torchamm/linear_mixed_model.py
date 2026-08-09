import warnings
import torch
def _checktensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x

class LinearMixedModel:
    @staticmethod
    def construct_lower_triangular_matrix(elements, n):
        sigma_L = torch.zeros((n, n), device=elements.device, dtype=elements.dtype)
        tril_indices = torch.tril_indices(row=n, col=n, offset=0, device=elements.device)
        sigma_L[tril_indices[0], tril_indices[1]] = elements
        return sigma_L

    def initialize_correlation(self, elements=None):
        correlation_structure = self.corr_structure
        nmax = max(self.lengths_)
        if correlation_structure is None or correlation_structure.lower() in ["unstructured", "full", "unspecified"]:
            ## lower triangular matrix
            if elements is not None and elements.shape[0] == (nmax * (nmax + 1)) // 2:
                sigma_L = elements
            else:
                sigma_L = torch.ones((nmax * (nmax + 1)) // 2, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L); 
        elif correlation_structure.lower() in ["iid", "identity"]:
            if elements is not None and elements.shape[0] == 1:
                sigma_L = elements
            else:
                sigma_L = torch.ones(1, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L)
        elif correlation_structure.lower()in ["independent", "diagonal"]:
            if elements is not None and elements.shape[0] == nmax:
                sigma_L = elements
            else:   
                sigma_L = torch.ones(nmax, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L)
        elif correlation_structure.lower() in ["compoundsymmetry", "cs"]:
            if elements is not None and elements.shape[0] == 2:
                sigma_L = elements
            else:
                sigma_L = torch.ones(2, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L)
        elif correlation_structure.lower() in ["ar1", "autoregressive"]:
            if elements is not None and elements.shape[0] == 2:
                sigma_L = elements
            else:
                sigma_L = torch.ones(2, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L)
        elif correlation_structure.lower() in ['exponential', 'exp']:
            if elements is not None and elements.shape[0] == 2:
                sigma_L = elements
            else:
                sigma_L = torch.ones(2, requires_grad=True, device=self.device)
            self.sigma_L = torch.nn.Parameter(sigma_L)
        else:
            raise ValueError("Unsupported correlation structure.")
    
    @staticmethod
    def ols_matrix_elements(elements, correlation, nmax):
        ## elements: a scalar
        if correlation is None or correlation.lower() in ["unstructured", "full", "unspecified"]:
            ## element be the lower triangular matrix elements: scalar to vector of length nmax*(nmax+1)/2
            L = torch.eye(nmax, device=elements.device) * elements
            L_lower_elements = L[torch.tril_indices(row=nmax, col=nmax, offset=0)[0], torch.tril_indices(row=nmax, col=nmax, offset=0)[1]]
            return L_lower_elements
        elif correlation.lower() in ["iid", "identity"]:
            return torch.tensor([elements], device=elements.device, dtype=elements.dtype)
        elif correlation.lower() in ["independent", "diagonal"]:
            return elements * torch.ones(nmax, device=elements.device, dtype=elements.dtype)
        elif correlation.lower() in ["compoundsymmetry", "cs"]:
            # param[0]=sqrt(0.75)*σ (private), param[1]=0.5*σ (shared) → rho_init≈0.25, sigma2_init=σ².
            return torch.tensor([0.75**0.5 * elements, 0.5 * elements],
                                device=elements.device, dtype=elements.dtype)
        elif correlation.lower() in ["ar1", "autoregressive"]:
            rho = 0
            return torch.tensor([elements, rho], device=elements.device, dtype=elements.dtype)
        elif correlation.lower() in ['exponential', 'exp']:
            phi = 0
            return torch.tensor([elements, phi], device=elements.device, dtype=elements.dtype)
        else:
            raise ValueError("Unsupported correlation structure.")
        
    def param_to_sqmatrix(self, param, n, correlation_structure=None, eps=1e-6):
        if len(param)==0:
                return torch.zeros((n, n), device=param.device, dtype=self.y_.dtype)
        if correlation_structure is None or correlation_structure.lower() in ["unstructured", "full", "unspecified"]:
            L = self.construct_lower_triangular_matrix(param, n) ## lower triangular matrix
            D = (L @ L.T).to(device=param.device, dtype=self.y_.dtype)
        elif correlation_structure.lower() in ["iid", "identity"]:
            return (param**2) * self.eye_nmax[:n, :n].to(device=param.device, dtype=self.y_.dtype)
        elif correlation_structure.lower() in ["independent", "diagonal"]:
            D =  torch.diag(param**2).to(device=param.device, dtype=self.y_.dtype)
        elif correlation_structure.lower() in ["compoundsymmetry", "cs"]:
            # param[0]²·I (private) + param[1]²·J (shared): always PSD, rho=param[1]²/(param[0]²+param[1]²).
            D = param[0]**2 * torch.eye(n, device=param.device, dtype=self.y_.dtype) + param[1]**2 * torch.ones((n, n), device=param.device, dtype=self.y_.dtype)
        elif correlation_structure.lower() in ["ar1", "autoregressive"]:
            rho = torch.tanh(param[1])
            indices = torch.arange(n, device=param.device)
            D =  param[0]**2 * rho**torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).to(device=param.device, dtype=self.y_.dtype)
        elif correlation_structure.lower() in ['exponential', 'exp']:
            phi = torch.exp(param[1])
            indices = torch.arange(n, device=param.device)
            D =  param[0]**2 * torch.exp(-phi * torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))).to(device=param.device, dtype=self.y_.dtype)
        else:
            raise ValueError("Unsupported correlation structure.")
        diag_idx = torch.arange(n, device=param.device)
        D = D.clone()
        D[diag_idx, diag_idx] = D[diag_idx, diag_idx] + eps
        return D
    
    def initialize_rand_D(self, pz, type = "unstructured", estimate = "sample"):
        if type == "unstructured":
            if estimate == "sample":
                ## use sample covariance of Z as initial value
                D_hat = torch.zeros((pz, pz), device=self.device, dtype=self.y_.dtype)
                total_count = 0
                for i, l in enumerate(self.lengths_):
                    Zi = self.Z_grouped[i, :l, :]  # ni x pz
                    D_hat += Zi.T @ Zi
                    total_count += l
                D_hat /= total_count
                L_hat = torch.linalg.cholesky(D_hat + 1e-5 * torch.eye(pz, device=self.device, dtype=D_hat.dtype))
                self.L_elements = L_hat[torch.tril_indices(row=pz, col=pz, offset=0)[0], torch.tril_indices(row=pz, col=pz, offset=0)[1]]
                self.L_elements = torch.nn.Parameter(self.L_elements)
            else:
                self.L_elements = torch.ones(pz * (pz + 1) // 2, requires_grad=True, device=self.device)
                self.L_elements = torch.nn.Parameter(self.L_elements)
        elif type == "diagonal":
            self.L_elements = torch.ones(pz, requires_grad=True, device=self.device)
            self.L_elements = torch.nn.Parameter(self.L_elements)
        elif type == "iid":
            self.L_elements = torch.ones(1, requires_grad=True, device=self.device)
            self.L_elements = torch.nn.Parameter(self.L_elements)

    def __init__(self, device = 'cpu', y=None, X=None, Z=None, id=None, corr_structure=None, rand_D_type="unstructured", rand_D_estimate="sample"):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.y_ = _checktensor(y).to(self.device).view(-1, 1) ## N x 1 tensor
        self.X_ = _checktensor(X).to(self.device)## N x px tensor
        self.Z_ = _checktensor(Z).to(self.device)## N x pz tensor
        self.id_ = torch.as_tensor(id, device=self.device).view(-1)
        unique_ids = torch.unique(self.id_)
        self._group_idx = [(self.id_ == uid).nonzero(as_tuple=True)[0] for uid in unique_ids]
        self.lengths_   = [(self.id_ == uid).sum().item() for uid in unique_ids]
        
        self.n_groups = len(self.lengths_)
        self.nmax = max(self.lengths_)
        px = self.X_.shape[1]
        pz = self.Z_.shape[1]

        # Allocate padded arrays
        self.X_grouped = torch.zeros((self.n_groups, self.nmax, px), device=self.device, dtype=self.X_.dtype)
        self.Z_grouped = torch.zeros((self.n_groups, self.nmax, pz), device=self.device, dtype=self.Z_.dtype)
        self.y_grouped = torch.zeros((self.n_groups, self.nmax), device=self.device, dtype=self.y_.dtype)

        self.corr_structure = corr_structure
        self.rand_D_type = rand_D_type
        self.rand_D_estimate = rand_D_estimate

        # Fill with per-group samples
        start = 0
        for i, l in enumerate(self.lengths_):
            end = start + l
            self.X_grouped[i, :l] = self.X_[ start:end ]
            self.Z_grouped[i, :l] = self.Z_[ start:end ]
            self.y_grouped[i, :l] = self.y_[ start:end ].squeeze()
            start = end

        # Build mask
        self.mask = torch.zeros((self.n_groups, self.nmax), device=self.device)
        for i, l in enumerate(self.lengths_):
            self.mask[i, :l] = 1.0

        mask = self.mask.unsqueeze(-1)  
        self.masked_X = self.X_grouped * mask
        self.masked_Z = self.Z_grouped * mask
        self.masked_y = self.y_grouped * self.mask

        self.eye_nmax = torch.eye(self.nmax, device=self.device, dtype=self.y_grouped.dtype)
        self.pz = pz

    @staticmethod
    def stablize(x):
        eps = 1e-5
        mask = (x < eps) & (x > -eps)
        return torch.where(mask, x.abs() + eps, x)
        
    @staticmethod
    def stablize_tensor(X, eps=1e-6):
        ## check for cholesky stability
        ## X is either 2D square or 3D batch of square matrices
        if X.dim() == 3 and X.shape[1] == X.shape[2]:
            eye = torch.eye(X.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0)
            return X + eps * eye

        elif X.dim() == 2 and X.shape[0] == X.shape[1]:
            eye = torch.eye(X.size(0), device=X.device, dtype=X.dtype)
            return X + eps * eye

        else:
            raise ValueError(f"Matrix must be 2D or 3D square, got shape {X.shape}")

    @staticmethod
    def _robust_cholesky(M, eye, n_tries=10):
        # CS line-search points can drive V indefinite; progressive jitter recovers it.
        scale = torch.diagonal(M, dim1=-2, dim2=-1).mean().detach().clamp_min(1.0)
        jitter = 1e-8 * scale
        for _ in range(n_tries):
            Mj = M + jitter * eye
            L, info = torch.linalg.cholesky_ex(Mj)
            if not info.any():
                return L, Mj
            jitter = jitter * 10.0
        Mj = M + jitter * eye
        return torch.linalg.cholesky(Mj), Mj
            
    @staticmethod   
    def convg_check(prev, curr, tol=1e-6):
        rel_change = abs((curr - prev) / (prev + 1e-12))
        if rel_change < tol:
            return True
        else:
            return False
    
    def beta_hat(self, sigma2, D):
        device = self.y_grouped.device
        G = self.n_groups

        if self.mask.dim() == 2:
            mask_vec = self.mask.unsqueeze(-1)     # [G, nmax, 1]
        else:
            mask_vec = self.mask                   # [G, nmax, 1]

        # masked X and y
        X_eff = self.X_grouped * mask_vec.to(dtype=self.X_grouped.dtype)        # [G, nmax, px]
        y_eff = self.y_grouped * mask_vec.squeeze(-1).to(dtype=self.y_grouped.dtype)  # [G, nmax]
        # build V_i
        if self.corr_structure == "iid" and self.rand_D_type == "iid":
            if sigma2.ndim == 2 and sigma2.shape[0] == self.nmax:
                sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).mean(dim=-1) # scalar
            if D.ndim == 2 and D.shape[0] == self.Z_.shape[1]:
                tau2 = torch.diagonal(D, dim1=-2, dim2=-1).mean(dim=-1) # scalar

            Z = self.masked_Z * mask_vec
            ZtZ = Z.transpose(-1,-2) @ Z # pz x pz
            eps = 1e-5
            
            diag_mean = torch.mean(torch.diagonal(ZtZ, dim1=-2, dim2=-1), dim=-1)
            jitter = eps * ((1.0 / tau2) + (1.0 / sigma2) * diag_mean)
            jitter = jitter.unsqueeze(-1).unsqueeze(-1) # scalar to be added to each element of M for stability
            
            I_q = torch.eye(self.pz, device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            tau2   = tau2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.clamp_min(1e-8)
            tau2   = tau2.clamp_min(1e-8)

            M = (1.0/tau2)[..., None, None] * I_q + (1.0/sigma2)[..., None, None] * ZtZ
            L_M = torch.linalg.cholesky(M) ## pz x pz
            Zt_y = Z.transpose(-1,-2) @ self.masked_y.unsqueeze(-1) # pz x 1
            tmp = torch.cholesky_solve(Zt_y, L_M)

            Vinv_y = (1.0/sigma2)*self.masked_y.unsqueeze(-1) - (1.0/sigma2**2)*(Z @ tmp)

            Zt_X = Z.transpose(-1,-2) @ self.masked_X   # [G, pz, px]
            tmpX = torch.cholesky_solve(Zt_X, L_M)     # [G, pz, px]
            Vinv_X = (1.0/sigma2)*self.masked_X - (1.0/sigma2**2)*(Z @ tmpX)

            XtVinvX = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_X, dim=0)  # [px, px]
            XtVinvy = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_y, dim=0)  # [px, 1]

        else:
            Sigma = sigma2.unsqueeze(0).expand(G, -1, -1).to( device=device)
            ZDZ   = self.Z_grouped @ D @ self.Z_grouped.transpose(1, 2)
            V_full = Sigma + ZDZ
            V_full = 0.5 * (V_full + V_full.transpose(-1, -2))

            # apply mask: keep only observed blocks; huge variance on padded diag
            mask_mat = mask_vec @ mask_vec.transpose(1, 2)      
            Vi = V_full * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6

            Vi = 0.5 * (Vi + Vi.transpose(-1, -2))
            I = self.eye_nmax.unsqueeze(0).to(device=Vi.device, dtype=Vi.dtype)

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
        return betahat
    
    def reset_params(self, OLS=False, estimate="sample"):
        if OLS & (len(self.X_) > 0) and estimate == "sample":
            beta = torch.linalg.lstsq(self.X_, self.y_).solution.clone()
            # ols of sigma2
            residuals = self.y_ - self.X_ @ beta
            sigma2 = (residuals.T @ residuals) / (len(self.y_) - self.X_.shape[1]) ## scalar
            self.sigma_L = self.ols_matrix_elements(torch.sqrt(sigma2).squeeze(), self.corr_structure, self.nmax)
            self.sigma_L = torch.nn.Parameter(self.sigma_L)
        else:
            if self.corr_structure is None or self.corr_structure.lower() in ["unstructured", "full", "unspecified"]:
                self.sigma_L = torch.ones((self.nmax * (self.nmax + 1)) // 2, requires_grad=True, device=self.device)
            elif self.corr_structure.lower() in ["iid", "identity"]:
                self.sigma_L = torch.ones(1, requires_grad=True, device=self.device)
            elif self.corr_structure.lower() in ["independent", "diagonal"]:
                self.sigma_L = torch.ones(self.nmax, requires_grad=True, device=self.device)
            beta = torch.ones((self.X_.shape[1], 1), device=self.device)
        
        self.initialize_correlation(elements=self.sigma_L)
        self.initialize_rand_D(pz=self.Z_.shape[1], type=self.rand_D_type, estimate=self.rand_D_estimate)

        return beta, sigma2

    def loss_fn(self, sigma2, D, beta = None):
        if self.beta_ is None and beta is None:
            raise ValueError("Beta coefficients must be provided either during initialization or as an argument.")
        if beta is None:
            beta = self.beta_

        ## id: N x 1 tensor indicating individual ids
        ## D: pz x pz tensor
        ## sigma2: ni x ni tensor
        logdetVsum = torch.tensor(0.0, dtype=self.y_.dtype, device=self.y_.device)
        rVinvr = torch.tensor(0.0, dtype=self.y_.dtype, device=self.y_.device)

        ZDZ = torch.matmul(self.masked_Z @ D, self.masked_Z.transpose(-1, -2))
        Sigma = sigma2.unsqueeze(0).expand(self.n_groups, -1, -1).to( device=self.device)
        mask_mat = self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)  # [G, nmax, nmax]
        V_full = Sigma + ZDZ

        # [G, nmax, nmax]
        mask_mat = self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)  # [G, nmax, nmax]
        Vi = V_full * mask_mat + (1 - mask_mat) * self.eye_nmax * 1e6

        e = self.y_grouped - (self.X_grouped @ beta.to( device=self.device)).squeeze(-1)  # [G, nmax]

        LVi, Vi = self._robust_cholesky(Vi, self.eye_nmax)  # [G, nmax, nmax]

        e_exp = e.unsqueeze(-1)  # [G, nmax, 1]
        Vinv_e = torch.cholesky_solve(e_exp, LVi)  # [G, nmax, 1]

        rVinvr = torch.sum(e_exp.squeeze(-1) * Vinv_e.squeeze(-1), dim=1)  # [G]
        rVinvr = torch.sum(rVinvr)  # scalar

        logdetVi = 2 * torch.sum(torch.log(torch.diagonal(LVi, dim1=1, dim2=2)), dim=1)  # [G]
        logdetVsum = torch.sum(logdetVi)

        n_eff = int(self.mask.sum().item())
        neg_loglik = 0.5 * (logdetVsum + rVinvr + n_eff * torch.log(torch.tensor(2.0 * torch.pi, device=self.device)))
        return neg_loglik

    def loss_fn_reml(self, sigma2, D):
        """
        Returns negative REML log-likelihood.
        """
        G = self.n_groups
        device = self.device
        mask_vec = self.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        n_eff = int(self.mask.sum().item())

        if self.corr_structure == "iid" and self.rand_D_type == "iid":
            if sigma2.ndim == 2 and sigma2.shape[0] == self.nmax:
                sigma2 = torch.diagonal(sigma2, dim1=-2, dim2=-1).mean(dim=-1) # scalar
            if D.ndim == 2 and D.shape[0] == self.Z_.shape[1]:
                tau2 = torch.diagonal(D, dim1=-2, dim2=-1).mean(dim=-1) # scalar

            Z = self.masked_Z * mask_vec
            ZtZ = Z.transpose(-1,-2) @ Z # pz x pz
            eps = 1e-5

            ## apply transformation to improve cholesky stability
            diag_mean = torch.mean(torch.diagonal(ZtZ, dim1=-2, dim2=-1), dim=-1)
            jitter = eps * ((1.0 / tau2) + (1.0 / sigma2) * diag_mean)
            jitter = jitter.unsqueeze(-1).unsqueeze(-1)

            # ZtZ: [G, pz, pz]
            I_q = torch.eye(self.pz, device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            tau2   = tau2.to(device=ZtZ.device, dtype=ZtZ.dtype)
            sigma2 = sigma2.clamp_min(1e-8)
            tau2   = tau2.clamp_min(1e-8)

            # broadcast for batching if needed
            M = (1.0/tau2)[..., None, None] * I_q + (1.0/sigma2)[..., None, None] * ZtZ

            L_M = torch.linalg.cholesky(M) ## pz x pz
            Zt_y = Z.transpose(-1,-2) @ self.masked_y.unsqueeze(-1) # pz x 1
            tmp = torch.cholesky_solve(Zt_y, L_M)

            Vinv_y = (1.0/sigma2)*self.masked_y.unsqueeze(-1) - (1.0/sigma2**2)*(Z @ tmp)

            Zt_X = Z.transpose(-1,-2) @ self.masked_X   # [G, pz, px]
            tmpX = torch.cholesky_solve(Zt_X, L_M)     # [G, pz, px]
            Vinv_X = (1.0/sigma2)*self.masked_X - (1.0/sigma2**2)*(Z @ tmpX)

            XtVinvX = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_X, dim=0)  # [px, px]
            XtVinvy = torch.sum(self.masked_X.transpose(1, 2) @ Vinv_y, dim=0)  # [px, 1]
            beta_hat = torch.linalg.solve(XtVinvX, XtVinvy)             # [px, 1]
            r = self.masked_y - (self.masked_X @ beta_hat).squeeze(-1)  # [G, nmax]
            r_exp = r.unsqueeze(-1)
            Zt_r  = Z.transpose(-1,-2) @ r_exp
            tmp_r = torch.cholesky_solve(Zt_r, L_M)
            Vinv_r = (1.0/sigma2)*r_exp - (1.0/sigma2**2)*(Z @ tmp_r)

            logdetM = 2.0 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), dim=-1)  # [G] if batched
            logdetV = n_eff * torch.log(sigma2) + self.pz * torch.log(tau2) + logdetM  # [G] or scalar
            logdetV_sum = logdetV.sum() if logdetV.ndim > 0 else logdetV

        else:
            Sigma = sigma2.unsqueeze(0).expand(G, -1, -1).to( device=device) ## [G, nmax, nmax]
            ZDZ   = torch.matmul(self.masked_Z @ D, self.masked_Z.transpose(-1, -2)) ## [G, nmax, nmax]
            V_full = Sigma + ZDZ

            V = V_full * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6  # [G, nmax, nmax]
            L, V = self._robust_cholesky(V, self.eye_nmax)  # [G, nmax, nmax]

            logdetV = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)), dim=1)  # [G]
            logdetV_sum = torch.sum(logdetV)

            X_eff = (self.X_grouped * mask_vec).to(dtype=L.dtype)
            y_eff = (self.y_grouped * mask_vec.squeeze(-1)).to(dtype=L.dtype)

            Vinv_y = torch.cholesky_solve(y_eff.unsqueeze(-1), L)  # [G, nmax, 1]
            Vinv_X = torch.cholesky_solve(X_eff, L)                # [G, nmax, px]

            XtVinvX = torch.sum(X_eff.transpose(1, 2) @ Vinv_X, dim=0)  # [px, px]
            XtVinvy = torch.sum(X_eff.transpose(1, 2) @ Vinv_y, dim=0)  # [px, 1]

            beta_hat = torch.linalg.solve(XtVinvX, XtVinvy)             # [px, 1]
            r = y_eff - (X_eff @ beta_hat).squeeze(-1)                  # [G, nmax]
            r_exp = r.unsqueeze(-1)
            Vinv_r = torch.cholesky_solve(r_exp, L)

        rVinv_r = torch.sum(r_exp.squeeze(-1) * Vinv_r.squeeze(-1), dim=1)  # [G]
        rVinv_r = torch.sum(rVinv_r)

        logdet_XtVinvX = torch.logdet(XtVinvX + 1e-6 * torch.eye(XtVinvX.shape[0], device=XtVinvX.device, dtype=XtVinvX.dtype))
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

        ## check convergence
        with torch.no_grad():
            D = self.param_to_sqmatrix(self.L_elements, n=self.pz, correlation_structure=self.rand_D_type)
            sigma2 = self.param_to_sqmatrix(self.sigma_L, n=max(self.lengths_), correlation_structure=self.corr_structure)
            current_D = D
            current_sigma2 = sigma2

            if fit_type.lower() == "ml":
                pass
            else:
                beta_ = self.beta_hat(
                sigma2=sigma2, D=D
            )
                
            beta_change = torch.max(torch.abs((beta_ - prev_beta)/prev_beta)).item()
            eps = 1e-8
            D_change = torch.max(torch.abs((current_D.flatten() - prev_D.flatten())/prev_D.flatten()+eps)).item()
            sigma2_change = torch.max(torch.abs((current_sigma2.flatten() - prev_sigma2.flatten()) / (prev_sigma2.flatten() + eps))).item()

            current_loss = total_loss.item()
            rel_change = abs((current_loss - prev_loss) / (prev_loss + 1e-12))

        if current_loss > prev_loss:
            ## halve the learning rate if loss increases
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

        convg_criteria = 0
        if convg_check == "loss":
            if rel_change < tol:
                # print(f"Converged at step {step}, loss = {current_loss:.6f}")
                convg_criteria = 1
        elif convg_check == "all":
            if (rel_change < tol) and (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                # print(f"Converged at step {step}, loss = {current_loss:.6f}")
                convg_criteria = 1
        elif convg_check == "params":
            if (beta_change < tol) and (D_change < tol) and (sigma2_change < tol):
                # print(f"Converged at step {step}, loss = {current_loss:.6f}")
                convg_criteria = 1
        else:
            raise ValueError("Invalid convergence check method.")
        return convg_criteria, current_loss, current_sigma2, current_D, beta_, rel_change, beta_change, D_change, sigma2_change

    def fit_likelihood(self, optimizer = torch.optim.Adam, lr=1e-3, max_iter=1000, fit_type = "REML",
            tol=1e-6, warm_start=True, verbose=False, convg_check = "loss", max_iter_inner=10):
        
        pz = self.Z_.shape[1]
        self.fit_type = fit_type
        beta, sigma2 = self.reset_params(OLS=warm_start, estimate=self.rand_D_estimate)

        if verbose:
            print("OLS beta:", beta.T)

        ## check if shape of y, X, Z agrees
        if not (len(self.y_) == len(self.X_) == len(self.Z_)):
            raise ValueError("Incompatible shapes: y, X, and Z must have the same length.")
        
        if fit_type == "ML" or fit_type == "ml":
            loss_function = self.loss_fn
        else:
            loss_function = self.loss_fn_reml

        ## set up optimizer and optimization loop
        if verbose:
            print(f"Using standard optimizer: {optimizer.__name__}")
        if fit_type.lower() == "ml":
            self.beta_ = torch.nn.Parameter(beta)
            opt = torch.optim.AdamW([self.L_elements, self.sigma_L, self.beta_], lr=lr, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW([self.L_elements, self.sigma_L], lr=lr, weight_decay=0.0)

        prev_beta = beta.clone().detach()        
        D = self.param_to_sqmatrix(self.L_elements, n=pz, correlation_structure=self.rand_D_type)
        prev_D = D.detach() 

        sigma2 = self.param_to_sqmatrix(self.sigma_L, n=max(self.lengths_), correlation_structure=self.corr_structure)
        prev_sigma2 = sigma2.clone().detach()
        prev_loss = float("inf")

        if verbose:
            print("Starting optimization...")

        for step in range(max_iter):
            convg_criteria, current_loss, current_sigma2, current_D,  current_beta, rel_change, beta_change, D_change, sigma2_change = self.update_iter(
                loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss
            )

            prev_loss = current_loss
            prev_beta = current_beta.clone().detach()
            prev_D = current_D.detach()
            prev_sigma2 = self.param_to_sqmatrix(self.sigma_L, n=max(self.lengths_), correlation_structure=self.corr_structure)
            
            if step>=9:
                break
                
        if step==9 and not optimizer==torch.optim.LBFGS:
            for step in range(10, max_iter):
                convg_criteria, current_loss, current_sigma2, current_D, current_beta, rel_change, beta_change, D_change, sigma2_change = self.update_iter(
                    loss_function, opt, fit_type, tol, convg_check, prev_beta, prev_D, prev_sigma2, prev_loss
                )

                if convg_criteria == 1:
                    break

                prev_loss = current_loss
                prev_beta = current_beta.clone().detach()
                prev_D = current_D.detach()
                prev_sigma2 = current_sigma2.detach()
                if step % 10 == 0 and verbose:
                    print(f"Step {step}, loss = {current_loss:.6f}, relative change = {rel_change:.6f}")

            if step == max_iter - 1:
                convg = 0
                warnings.warn(f"LinearMixedModel: Adam did not converge in {max_iter} steps "
                              f"(rel_change={rel_change:.2e}).")
            else:
                convg = 1
                if verbose:
                    print(f"Converged at step {step}, loss = {current_loss:.6f}")
            
        if optimizer==torch.optim.LBFGS and step == 9:
            if max_iter_inner is None:
                max_iter_inner = 10
            max_iter_outer = max_iter//max_iter_inner
            if verbose:
                print(f"Using LBFGS optimizer with {max_iter_outer} epochs.")
            opt = torch.optim.LBFGS(
                [self.L_elements, self.sigma_L] if fit_type.lower() != "ml" else [self.L_elements, self.sigma_L, self.beta_],
                lr=lr,
                max_iter=max_iter_inner,         
                history_size=10,
                line_search_fn="strong_wolfe"   
            )

            def closure():
                opt.zero_grad()
                sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure)
                D = self.param_to_sqmatrix(self.L_elements, n=pz, correlation_structure=self.rand_D_type)
                loss = loss_function(sigma2=sigma2, D=D)
                loss.backward()
                return loss

            convg = 0
            for epoch in range(max_iter_outer):      # now max_iter means "outer epochs"
                total_loss = opt.step(closure)
                loss_check = total_loss.detach()
                grad_norm = 0.0
                for p in opt.param_groups[0]['params']:
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()**2
                grad_norm = grad_norm**0.5

                with torch.no_grad():
                    sigma2 = self.param_to_sqmatrix(self.sigma_L, n=self.nmax, correlation_structure=self.corr_structure)
                    D = self.param_to_sqmatrix(self.L_elements, n=pz, correlation_structure=self.rand_D_type)
                    if fit_type.lower() != "ml":
                        self.beta_ = self.beta_hat(sigma2=sigma2, D=D).clone().detach()
                    current_loss = loss_check.item()
                    current_sigma2 = sigma2.clone().detach()
                    current_D = D.clone().detach()

                    rel_loss = abs((current_loss - prev_loss) / (prev_loss + 1e-12))
                    rel_param_beta = torch.max(torch.abs((self.beta_ - prev_beta)/ (prev_beta + 1e-12))).item()
                    rel_param_D = torch.max(torch.abs((current_D.flatten() - prev_D.flatten()) / (prev_D.flatten() + 1e-12))).item()
                    rel_param_sigma2 = torch.max(torch.abs((current_sigma2.flatten() - prev_sigma2.flatten()) / (prev_sigma2.flatten() + 1e-12))).item()
                    rel_param = max(rel_param_beta, rel_param_D, rel_param_sigma2)

                prev_loss = current_loss
                prev_beta = self.beta_.detach()
                prev_D = current_D.detach()
                prev_sigma2 = current_sigma2.detach()
                if verbose and (epoch % 10 == 0 or convg==1):
                    print(f"LBFGS: Epoch {epoch}, loss = {current_loss:.6f}, grad_norm = {grad_norm:.3e}, rel_loss = {rel_loss:.6f}, rel_param = {rel_param:.6f}")
                
                if (rel_loss < tol) or grad_norm < 1e-2:
                    if convg_check in ["params", "all"]:
                        if rel_param < tol:
                            convg = 1
                    else:
                        convg = 1
                    if verbose:
                        print(f"LBFGS converged at epoch {epoch}, grad_norm = {grad_norm:.3e}, rel_loss = {rel_loss:.6f}, rel_param = {rel_param:.6f}")
                    break

            if convg == 0:
                warnings.warn(f"LinearMixedModel: LBFGS did not converge in {max_iter_outer} epochs "
                              f"(rel_loss={rel_loss:.2e}, grad_norm={grad_norm:.2e}).")

        self.final_loss = current_loss
        self.convg = convg
        self.sigma2 = current_sigma2.clone().detach()
        self.beta = self.beta_.clone().detach()
        self.D = (current_D).clone().detach()
        self.n_iter = epoch if optimizer == torch.optim.LBFGS else step
        return self
    
    
    ##########################################
    ### Hypothesis testing for fixed effects
    ##########################################
    # Satterthwaite t-test (lmerTest): t = l'b / sqrt(l'C l),
    # nu = 2 f^2 / (grad_f' A grad_f) with f(theta)=l'C(theta)l,
    # theta the variance params, A = [d2 nll/d theta2]^-1.

    def _n_resid_params(self):
        cs = self.corr_structure
        if cs is None or cs.lower() in ["unstructured", "full", "unspecified"]:
            return (self.nmax * (self.nmax + 1)) // 2
        if cs.lower() in ["iid", "identity"]:
            return 1
        if cs.lower() in ["independent", "diagonal"]:
            return self.nmax
        if cs.lower() in ["compoundsymmetry", "cs", "ar1", "autoregressive", "exponential", "exp"]:
            return 2
        raise ValueError("Unsupported correlation structure.")

    def _theta_hat(self):
        return torch.cat([self.sigma_L.detach().reshape(-1),
                          self.L_elements.detach().reshape(-1)], dim=0)

    def unpack_theta(self, theta, q):
        n_resid = self._n_resid_params()
        theta_sigma, theta_L = theta[:n_resid], theta[n_resid:]
        if self.rand_D_type == "diagonal" and theta_L.shape[0] != q:
            raise ValueError("theta_L length does not match the number of random effects.")
        if self.rand_D_type in ["unstructured", "full"] and theta_L.shape[0] != (q * (q + 1)) // 2:
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
        ZDZ = torch.matmul(self.masked_Z @ D, self.masked_Z.transpose(-1, -2))
        Sigma = sigma2.unsqueeze(0).expand(self.n_groups, -1, -1).to(device=self.device)
        V_full = Sigma + ZDZ
        mask_vec = self.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        Vi = V_full * mask_mat + (1.0 - mask_mat) * self.eye_nmax * 1e6
        Vi = 0.5 * (Vi + Vi.transpose(-1, -2))
        VinvX = torch.linalg.solve(Vi, self.masked_X)
        XtVinvX = (self.masked_X.transpose(1, 2) @ VinvX).sum(dim=0)
        return torch.linalg.inv(XtVinvX)

    def f_theta(self, theta, l_vec):
        l_vec = l_vec.to(device=self.device, dtype=self.y_.dtype)
        C = self.C_theta(theta)
        return (l_vec.transpose(0, 1) @ C @ l_vec).squeeze(), C

    def _contrast_test(self, l_vec, theta_hat, A):
        l_vec = l_vec.to(device=self.device, dtype=self.y_.dtype)
        f_val, _ = self.f_theta(theta_hat, l_vec)
        grad_f = torch.autograd.grad(f_val, theta_hat)[0]
        var_f = grad_f @ A @ grad_f
        se = torch.sqrt(f_val)
        est = (l_vec.transpose(0, 1) @ self.beta.to(self.device)).squeeze()
        t_hat = est / se
        df = (2.0 * f_val ** 2) / var_f
        from scipy import stats
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_hat.item()), df.item()))
        return {"estimate": est.item(), "se": se.item(), "t": t_hat.item(),
                "df": df.item(), "p_value": float(p_value)}

    def fixed_t_test(self, l=None, alpha=0.05):
        """Satterthwaite t-test for fixed effects (ref: lmerTest, Kuznetsova et al. 2017).

        l=None   -> per-coefficient table (a list of dicts, one unit contrast each).
        l=int    -> unit contrast e_l for a single coefficient.
        l=vector -> custom contrast of length px.
        Returns a dict with estimate, se, t, df, p_value (or a list of them).
        """
        from torch.autograd.functional import hessian
        px = self.X_.shape[1]
        theta_hat = self._theta_hat().requires_grad_(True)

        H = hessian(lambda th: self.nll(th), theta_hat)
        H = 0.5 * (H + H.transpose(0, 1))
        A = torch.linalg.inv(H + 1e-8 * torch.eye(H.shape[0], device=H.device, dtype=H.dtype))

        def unit(idx):
            v = torch.zeros((px, 1), device=self.device, dtype=self.y_.dtype)
            v[idx, 0] = 1.0
            return v

        if l is None:
            return [dict(coef=i, **self._contrast_test(unit(i), theta_hat, A)) for i in range(px)]
        if isinstance(l, int):
            if l >= px:
                raise ValueError("l must be less than the number of fixed effects.")
            l_vec = unit(l)
        else:
            l_vec = torch.as_tensor(l, device=self.device, dtype=self.y_.dtype).reshape(px, 1)
        return self._contrast_test(l_vec, theta_hat, A)

    ##########################################
    ### Hypothesis testing for random effects
    ##########################################
    # Boundary score test (Lin 1997; Zhang & Lin 2003): U = S+ - e at H0,
    # S+ = 1/2 ||Z_idx' P0 y||^2, matched to a scaled chi-square a*chi2_b.
    # See docs/linearity_test.md.

    def _chol_group_cov(self, V):
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

    def _random_effect_null_cov(self, idx, null_fit_kwargs=None):
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

        Z_keep = self.masked_Z[:, :, keep]
        ZDZ = Z_keep @ null.D @ Z_keep.transpose(-1, -2)
        V0 = null.sigma2.unsqueeze(0) + ZDZ
        return null, V0, self._chol_group_cov(V0)

    def _random_effect_score(self, idx, L_V0):
        X_eff = self.masked_X
        y_eff = self.masked_y.unsqueeze(-1)
        Z_q   = self.masked_Z[:, :, idx]

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
    def _random_effect_pvalue(S_plus, e, info):
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
        _, _, L_V0 = self._random_effect_null_cov(idx, null_fit_kwargs=null_fit_kwargs)
        S_plus, e, info = self._random_effect_score(idx, L_V0)
        out = self._random_effect_pvalue(S_plus, e, info)
        out.update(re_idx=idx if len(idx) > 1 else idx[0], reject=bool(out["p_value"] < alpha))
        return out

    def AIC(self):
        """AIC from the ML log-likelihood at the fitted estimates (use an ML fit for a proper AIC)."""
        nll_value = self.loss_fn(self.sigma2, self.D, beta=self.beta)
        k = len(self.beta) + self._theta_hat().numel()
        return 2.0 * nll_value.item() + 2.0 * k


##########################################
### LRT for model selection
##########################################
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