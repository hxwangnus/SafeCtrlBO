# GPyTorch version of SafeCtrlBO. 
# Author: H. Wang, December 2025.
import torch
import gpytorch
# from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
from device_utils import resolve_device
from model import build_gp, fit_gp


class SafeCtrlBO:
    def __init__(
        self,
        init_X,             # (n0, d)
        init_Y_perf,        # (n0, 1)
        init_Y_safe,        # (n0, 1) or None
        bounds,             # (2, d) tensor [[l1..ld],[u1..ud]]
        base_kernel,        # AdditiveKernel (frozen from DARTS search)
        safety_threshold=None,   # h_s
        switch_time=15,     # T0
        beta_fn=None,
        tau=0.1,
        device="cpu",
        init_training_iter=0,  # number of training steps at initialization (0 => use DARTS hyper as-is)
        likelihood_noise=1e-4,  # Gaussian likelihood noise variance used by both GPs
        sobol_seed=None,
        safe_retry_radius=0.05,
    ):
        self.device = resolve_device(device)
        self.bounds = bounds.to(self.device)

        # whether we have a separate safety signal g(x)
        self.use_safety = (init_Y_safe is not None) and (safety_threshold is not None)
        self.safety_threshold = safety_threshold

        self.switch_time = switch_time
        self.beta_fn = beta_fn or (lambda n: 2.0 * torch.log(torch.tensor(float(n + 1.0))))
        self.tau = tau
        self.likelihood_noise = likelihood_noise
        self.safe_retry_radius = safe_retry_radius
        self._sobol_engine = SobolEngine(
            dimension=self.bounds.shape[1],
            scramble=True,
            seed=sobol_seed,
        )

        self.X = init_X.to(self.device)
        self.Yf = init_Y_perf.to(self.device)
        self.Yg = init_Y_safe.to(self.device) if self.use_safety else None

        # current number of observations (used for beta_t etc.)
        self.n_iter = self.X.shape[0]

        # build two GPs with a frozen additive kernel learned by DARTS
        self.rebuild_models(base_kernel, training_iter=init_training_iter)

    def rebuild_models(self, base_kernel, training_iter=0):
        """
        Build GP models for f and g using the same frozen base_kernel.
        If training_iter > 0, fit_gp can be used to slightly refine noise or
        (optionally) kernel hyperparameters; with DARTS, we typically set
        training_iter=0 to keep the learned kernel unchanged.
        """
        self.model_f, self.lik_f, self.mll_f = build_gp(
            self.X, self.Yf, base_kernel, noise=self.likelihood_noise
        )

        if self.use_safety:
            self.model_g, self.lik_g, self.mll_g = build_gp(
                self.X, self.Yg, base_kernel, noise=self.likelihood_noise
            )
        else:
            self.model_g = None
            self.lik_g = None
            self.mll_g = None

        if training_iter is not None and training_iter > 0:
            fit_gp(self.model_f, self.lik_f, self.mll_f, training_iter=training_iter)
            if self.use_safety:
                fit_gp(self.model_g, self.lik_g, self.mll_g, training_iter=training_iter)

    @torch.no_grad()
    def posterior_mean_std(self, model, likelihood, Xtest):
        """
        Return the posterior over the latent GP function values.

        BO confidence bounds should be built from epistemic uncertainty in the
        latent function, not from the observation-noise distribution.
        """
        model.eval()
        if likelihood is not None:
            likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = model(Xtest)
        mean = pred.mean
        std = pred.variance.clamp_min(0.0).sqrt()
        return mean, std

    def _beta_sqrt(self, beta, dtype):
        return torch.sqrt(torch.tensor(beta, dtype=dtype, device=self.device))

    def _observed_safe_points(self, beta):
        """
        Return observed points that are still certified safe under the current GP.
        """
        if not self.use_safety:
            return self.X

        beta_sqrt = self._beta_sqrt(beta, self.X.dtype)
        mu_g_obs, std_g_obs = self.posterior_mean_std(self.model_g, self.lik_g, self.X)
        l_g_obs = mu_g_obs - beta_sqrt * std_g_obs
        safe_obs_mask = l_g_obs >= self.safety_threshold
        return self.X[safe_obs_mask]

    def _empirically_safe_observed_points(self):
        """
        Return observed points whose measured safety values satisfy the threshold.
        """
        if not self.use_safety:
            return self.X

        safe_obs_mask = self.Yg.squeeze(-1) >= self.safety_threshold
        return self.X[safe_obs_mask]

    def _local_safe_retry_candidates(self, safe_points, num_candidates):
        """
        Sample a local cloud around anchor points and include the anchors
        themselves so the retry set always contains the original points.
        """
        if safe_points.numel() == 0:
            return safe_points

        safe_points = safe_points.to(device=self.device, dtype=self.bounds.dtype)
        num_safe = safe_points.shape[0]
        num_local = max(num_candidates - num_safe, 0)
        if num_local == 0:
            return safe_points

        sample_ids = torch.randint(num_safe, (num_local,), device=self.device)
        centers = safe_points[sample_ids]

        span = (self.bounds[1] - self.bounds[0]).unsqueeze(0)
        perturb_scale = self.safe_retry_radius * span
        perturb = torch.randn_like(centers) * perturb_scale
        X_local = centers + perturb

        lb = self.bounds[0].unsqueeze(0)
        ub = self.bounds[1].unsqueeze(0)
        X_local = torch.maximum(torch.minimum(X_local, ub), lb)

        return torch.cat([safe_points, X_local], dim=0)

    def _best_safe_observed_point(self, beta):
        """
        Choose the best certified-safe observed point by the current UCB of f.
        """
        safe_points = self._observed_safe_points(beta)
        if safe_points.numel() == 0:
            raise RuntimeError(
                "SafeCtrlBO could not certify any observed point as safe. "
                "Please provide a safer initialization, relax the safety threshold, "
                "or adjust the GP uncertainty settings."
            )

        beta_sqrt = self._beta_sqrt(beta, safe_points.dtype)
        mu_f_obs, std_f_obs = self.posterior_mean_std(self.model_f, self.lik_f, safe_points)
        u_f_obs = mu_f_obs + beta_sqrt * std_f_obs
        return safe_points[torch.argmax(u_f_obs)]

    def _best_empirically_safe_observed_point(self):
        """
        Choose the best observed point among measurements that satisfied safety.
        """
        safe_obs_mask = self.Yg.squeeze(-1) >= self.safety_threshold
        if not torch.any(safe_obs_mask):
            raise RuntimeError(
                "SafeCtrlBO has no observed measurement that satisfies the safety threshold."
            )

        safe_points = self.X[safe_obs_mask]
        safe_perf = self.Yf.squeeze(-1)[safe_obs_mask]
        return safe_points[torch.argmax(safe_perf)]

    def _get_sets(self, X_cand, beta):
        """
        Calculate Sn, Bn, u_f (UCB of f), sigma_f, l_g (LCB of g)

        If self.use_safety is False, we fall back to unconstrained BO:
        S = B = all candidates, and l_g is a dummy tensor.
        """
        X_cand = X_cand.to(self.device)     # set of parameter candidates

        # posterior of f
        mu_f, std_f = self.posterior_mean_std(self.model_f, self.lik_f, X_cand)

        beta_sqrt = self._beta_sqrt(beta, X_cand.dtype)
        u_f = mu_f + beta_sqrt * std_f

        if not self.use_safety:
            # unconstrained case: everything is "safe"
            safe_mask = torch.ones(X_cand.size(0), dtype=torch.bool, device=self.device)
            boundary_mask = safe_mask.clone()
            S = X_cand
            B = X_cand
            # dummy l_g just for API compatibility
            l_g = torch.full_like(mu_f, fill_value=0.0)
            return {
                "S": S,
                "B": B,
                "safe_mask": safe_mask,
                "boundary_mask": boundary_mask,
                "u_f": u_f,
                "sigma_f": std_f,
                "l_g": l_g,
            }

        # posterior of g (safety) in constrained case
        mu_g, std_g = self.posterior_mean_std(self.model_g, self.lik_g, X_cand)
        l_g = mu_g - beta_sqrt * std_g

        # safe set Sn
        safe_mask = l_g >= self.safety_threshold
        S = X_cand[safe_mask]

        # safe boundary set Bn
        boundary_mask = safe_mask & (torch.abs(l_g - self.safety_threshold) <= self.tau)
        B = X_cand[boundary_mask]
        if B.numel() == 0 and S.numel() > 0:
            # if boundary set is empty, use the safe set
            B = S
            boundary_mask = safe_mask

        return {
            "S": S,
            "B": B,
            "safe_mask": safe_mask,
            "boundary_mask": boundary_mask,
            "u_f": u_f,
            "sigma_f": std_f,
            "l_g": l_g,
        }

    def suggest(self, num_candidates=4096):
        """
        generate next parameter x_next within [bounds]
        previously (in GPy) called as:
        x_next = opt.optimize()
        """
        # here n_iter is the current number of observations;
        # you can also use (self.n_iter + 1) if you prefer beta_{t+1}
        beta = float(self.beta_fn(self.n_iter))

        # # sample the candidates in the box (Sobol)
        # # sample n set(s) of points
        # # each set with "number of candidates" points with dimension d
        # # 4096 candidate points after squeeze(0)
        # # time complexity is O(n*q), n is the num of observed data, q is num of candidates
        # X_cand = draw_sobol_samples(
        #     bounds=self.bounds,
        #     n=1,
        #     q=num_candidates,
        # ).squeeze(0).to(self.device)

        # Use SobolEngine instead, to avoid BoTorch
        # SobolEngine draws points in [0,1]^d, then we affine-transform them to [l_i, u_i]
        # time complexity is O(n*q), n is the num of observed data, q is num of candidates
        # shape: (num_candidates, d), values in [0, 1]
        X_unit = self._sobol_engine.draw(num_candidates).to(
            device=self.device,
            dtype=self.bounds.dtype,
        )

        lb = self.bounds[0]  # (d,)
        ub = self.bounds[1]  # (d,)
        X_cand = lb + (ub - lb) * X_unit  # (num_candidates, d)

        sets = self._get_sets(X_cand, beta)
        retried_locally = False
        retry_source = None

        if self.use_safety and sets["S"].numel() == 0:
            safe_points = self._observed_safe_points(beta)
            retry_source = "certified"
            if safe_points.numel() == 0:
                safe_points = self._empirically_safe_observed_points()
                retry_source = "empirical"
                if safe_points.numel() == 0:
                    raise RuntimeError(
                        "SafeCtrlBO found no certified-safe candidate and no observed "
                        "measurement that satisfied the safety threshold."
                    )

            X_retry = self._local_safe_retry_candidates(safe_points, num_candidates)
            sets = self._get_sets(X_retry, beta)
            retried_locally = True

            if sets["S"].numel() == 0:
                if retry_source == "empirical":
                    x_next = self._best_empirically_safe_observed_point()
                    return x_next.unsqueeze(0), "empirical_safe_fallback", sets

                x_next = self._best_safe_observed_point(beta)
                return x_next.unsqueeze(0), "safe_fallback", sets

        if self.n_iter <= self.switch_time:
            # Safe exploration, maximize sigma_f in Bn
            sigma_B = sets["sigma_f"][sets["boundary_mask"]]
            idx = torch.argmax(sigma_B)
            x_next = sets["B"][idx]
            mode = "expansion"
        else:
            # Exploitation, maximize UCB_f in S_n
            u_S = sets["u_f"][sets["safe_mask"]]
            idx = torch.argmax(u_S)
            x_next = sets["S"][idx]
            mode = "optimization"

        if retried_locally:
            prefix = "empirical_" if retry_source == "empirical" else ""
            mode = f"{prefix}{mode}_local_retry"

        return x_next.unsqueeze(0), mode, sets

    def observe(
        self,
        x_new,
        y_perf_new,
        y_safe_new=None,
        train_hypers_every=None,
        training_iter=0,
    ):
        """
        Add new observation and (optionally) re-train GP.
        x_new in the shape (1,d)
        y_*_new in the shape (1,1) or a scalar

        With a DARTS-learned frozen kernel, we typically:
          - always update train data via set_train_data
          - optionally update only the likelihood noise in fit_gp
            every 'train_hypers_every' iterations (e.g., to adapt noise).
        """
        # new observation
        x_new = x_new.to(self.device)
        y_perf_new = torch.as_tensor(
            y_perf_new, dtype=self.X.dtype, device=self.device
        ).view(-1, 1)

        self.X = torch.cat([self.X, x_new], dim=0)
        self.Yf = torch.cat([self.Yf, y_perf_new], dim=0)

        if self.use_safety:
            y_safe_new = torch.as_tensor(
                y_safe_new, dtype=self.X.dtype, device=self.device
            ).view(-1, 1)
            self.Yg = torch.cat([self.Yg, y_safe_new], dim=0)

        # increase number of observations
        self.n_iter += 1

        # update train data (no change to kernel structure / hyperparameters here)
        self.model_f.set_train_data(
            inputs=self.X, targets=self.Yf.squeeze(-1), strict=False
        )
        if self.use_safety:
            self.model_g.set_train_data(
                inputs=self.X, targets=self.Yg.squeeze(-1), strict=False
            )

        # optimize hyper-parameters (e.g., noise) after K iterations
        if (
            train_hypers_every is not None
            and training_iter is not None
            and training_iter > 0
            and self.n_iter % train_hypers_every == 0
        ):
            fit_gp(self.model_f, self.lik_f, self.mll_f,
                   training_iter=training_iter,
                   train_kernel=False,
                   train_mean=False,
                   train_noise=True)
            if self.use_safety:
                fit_gp(self.model_g, self.lik_g, self.mll_g,
                       training_iter=training_iter,
                       train_kernel=False,
                       train_mean=False,
                       train_noise=True)
