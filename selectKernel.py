"""
selectKernel.py

DARTS-style bilevel kernel-structure search for Safe BO / GP regression.

Key changes vs the earlier versions
- Uses real CSV data (9D) by default.
- Kernel candidates are subsets up to max_order=2 by default (C=45 for d=9).
- Uses a *shared* lengthscale per input dimension (only d lengthscales total).
- Uses a *single global outputscale* (no per-component outputscales → better identifiability).
- Mixture weights pi = softmax(alpha / tau) with:
  (B) alpha-centering after each alpha step (keeps alpha well-conditioned).
  (C) stronger tau annealing (actually reaches a useful temperature).
- "Scientific regularization" for structure:
    • entropy penalty on pi (encourages sparsity)
    • optional Dirichlet log-prior on pi (encourages sparsity when conc < 1)
    • optional order penalty (penalize higher order; order=2 here)
    • optional L2 on alpha (weight decay style)
  and for hyperparameters:
    • Adam weight_decay on w-params (shared lengthscales, outputscale, noise)
- Robust CSV parsing (handles BOM, headers, empty cells, NaNs); skips invalid rows.

Expected CSV (header preferred):
  px1,ix1,dx1,px2,ix2,dx2,py,iy,dy,perf,safe
If no header: assumes first 9 cols are X, col9 is perf, col10 is safe.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import gpytorch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy

from device_utils import configure_torch_runtime, format_runtime, resolve_device, resolve_dtype

# -----------------------------
# Global configuration
# -----------------------------
DEVICE = resolve_device("auto")
DTYPE = torch.float64


def configure_runtime(device="auto", dtype="float64"):
    global DEVICE, DTYPE
    DEVICE = configure_torch_runtime(device)
    DTYPE = resolve_dtype(dtype)
    print(f"Running kernel search with {format_runtime(DEVICE, DTYPE)}")


# ============================================================
# 1) Robust CSV loader (handles BOM/header/empties)
# ============================================================
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_gantry_from_csv_robust(
    path: str,
    target: str = "perf",           # "perf" or "safe"
    val_ratio: float = 0.2,
    seed: int = 0,
    expect_d: int = 9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Robust CSV loader:
      - strips BOM via encoding='utf-8-sig'
      - detects header row
      - skips rows with empty/non-numeric cells
      - supports target by name if header exists, else by index

    Returns:
      X_train, y_train, X_val, y_val (torch.double on DEVICE)
      norm_info dict
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1).")
    rng = np.random.RandomState(seed)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"CSV seems empty or too short: {path}")

    # Detect header: if any entry in first row is non-float, treat as header
    first_row = rows[0]
    has_header = any((cell.strip() != "" and (not _is_float(cell.strip()))) for cell in first_row)

    colnames = None
    data_rows = rows
    if has_header:
        colnames = [c.strip() for c in first_row]
        data_rows = rows[1:]
        print("[CSV] Header detected. Using target='{}' as y column if present.".format(target))
    else:
        print("[CSV] No header detected. Using fixed column indices for X/y.")

    # Build numeric matrix, skipping invalid rows
    X_list, y_list = [], []
    skipped = 0

    # Determine column indices
    if has_header:
        name_to_idx = {name: i for i, name in enumerate(colnames)}
        # X are first 9 controller parameters; prefer by known names if present, else first 9 cols
        # We still default to "first 9 columns" for robustness.
        x_idx = list(range(expect_d))
        if target in name_to_idx:
            y_idx = name_to_idx[target]
        else:
            # fallback: perf -> 9, safe -> 10
            y_idx = 9 if target == "perf" else 10
            print(f"[CSV] Warning: target column '{target}' not found; fallback to column {y_idx}.")
    else:
        x_idx = list(range(expect_d))
        y_idx = 9 if target == "perf" else 10

    # Parse rows
    for r in data_rows:
        if len(r) <= max(max(x_idx), y_idx):
            skipped += 1
            continue
        ok = True
        x_row = []
        for j in x_idx:
            s = r[j].strip()
            if s == "" or (not _is_float(s)):
                ok = False
                break
            x_row.append(float(s))
        if not ok:
            skipped += 1
            continue
        sy = r[y_idx].strip()
        if sy == "" or (not _is_float(sy)):
            skipped += 1
            continue
        y_val = float(sy)

        # Skip NaNs/infs
        if (not np.isfinite(y_val)) or (not np.all(np.isfinite(x_row))):
            skipped += 1
            continue

        X_list.append(x_row)
        y_list.append(y_val)

    X_raw = np.asarray(X_list, dtype=np.float64)
    y_raw = np.asarray(y_list, dtype=np.float64)

    if X_raw.shape[0] < 10:
        raise ValueError(f"Too few valid rows after cleaning: {X_raw.shape[0]} (skipped {skipped}).")

    if X_raw.shape[1] != expect_d:
        raise ValueError(f"Expected X dim {expect_d}, got {X_raw.shape[1]}.")

    if skipped > 0:
        print(f"[CSV] Skipped {skipped} invalid/empty rows.")

    # Normalize X -> [-1,1]^d
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    denom = np.where(x_max > x_min, x_max - x_min, 1.0)
    X_scaled = 2.0 * (X_raw - x_min) / denom - 1.0

    # Standardize y
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std()) if float(y_raw.std()) > 0 else 1.0
    y_scaled = (y_raw - y_mean) / y_std
    print(f"[Y] standardized: mean={y_mean:.3f}, std={y_std:.3f}")

    # Train/val split
    N = X_scaled.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    N_val = max(1, int(round(N * val_ratio)))
    N_train = N - N_val
    train_idx = idx[:N_train]
    val_idx = idx[N_train:]

    X_train = torch.tensor(X_scaled[train_idx], dtype=DTYPE, device=DEVICE)
    y_train = torch.tensor(y_scaled[train_idx], dtype=DTYPE, device=DEVICE)
    X_val = torch.tensor(X_scaled[val_idx], dtype=DTYPE, device=DEVICE)
    y_val = torch.tensor(y_scaled[val_idx], dtype=DTYPE, device=DEVICE)

    norm_info = dict(
        x_min=x_min,
        x_max=x_max,
        y_mean=y_mean,
        y_std=y_std,
        skipped_rows=skipped,
        total_rows=N,
    )

    return X_train, y_train, X_val, y_val, norm_info


# ============================================================
# 2) Candidate subsets (up to max_order)
# ============================================================
def build_subsets(input_dim: int, max_order: int) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
    max_order = max(1, min(max_order, input_dim))
    subs: List[Tuple[int, ...]] = []
    orders: List[int] = []
    for order in range(1, max_order + 1):
        for idxs in combinations(range(input_dim), order):
            subs.append(tuple(idxs))
            orders.append(order)
    return subs, torch.tensor(orders, dtype=torch.int64)


# ============================================================
# 3) Shared-lengthscale RBF mixture kernel (dense, stable, identifiable)
#     K(x,x') = outputscale * Σ_c π_c * Π_{j in S_c} exp(-0.5||Δ_j||^2 / ℓ_j^2)
# ============================================================
class SharedRBFMixtureKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(self, input_dim: int, subsets: List[Tuple[int, ...]], orders: torch.Tensor, tau: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.subsets = subsets
        self.register_buffer("orders", orders.clone())
        self.num_components = len(subsets)

        # Structure logits
        self.alpha = nn.Parameter(torch.zeros(self.num_components, dtype=DTYPE))

        # Temperature
        self.register_buffer("tau", torch.tensor(float(tau), dtype=DTYPE))

        # Shared log-lengthscales per dimension (initialize moderately large)
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim, dtype=DTYPE) + math.log(2.0))

        # Global log-outputscale
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(1.0), dtype=DTYPE))

    def set_tau(self, tau: float):
        self.tau = torch.tensor(float(tau), dtype=DTYPE, device=self.alpha.device)

    def mixture_weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha / self.tau, dim=0)

    def lengthscales(self) -> torch.Tensor:
        # softplus to ensure positivity, plus small floor
        return torch.nn.functional.softplus(self.log_lengthscale) + 1e-6

    def outputscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_outputscale) + 1e-6

    def _rbf_1d(self, x1: torch.Tensor, x2: torch.Tensor, dim: int, ell: torch.Tensor) -> torch.Tensor:
        # x1: (N,d), x2: (M,d)
        a = x1[:, dim].unsqueeze(1)  # (N,1)
        b = x2[:, dim].unsqueeze(0)  # (1,M)
        diff2 = (a - b) ** 2
        return torch.exp(-0.5 * diff2 / (ell ** 2))

    def forward(self, x1, x2=None, diag: bool = False, **params):
        x2 = x1 if x2 is None else x2
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError("This kernel expects x1,x2 as 2D tensors: (N,d).")

        N = x1.size(0)
        M = x2.size(0)

        outscale = self.outputscale()
        pi = self.mixture_weights()
        ells = self.lengthscales()

        if diag:
            # Each component has diag=1, so mixture diag = outscale * sum(pi) = outscale
            return outscale * torch.ones(N, dtype=DTYPE, device=x1.device)

        # Precompute per-dim kernels (dense)
        K_dim = [self._rbf_1d(x1, x2, j, ells[j]) for j in range(self.input_dim)]  # list of (N,M)

        # Mixture sum
        K = torch.zeros((N, M), dtype=DTYPE, device=x1.device)
        for c, dims in enumerate(self.subsets):
            Kc = K_dim[dims[0]]
            if len(dims) == 2:
                Kc = Kc * K_dim[dims[1]]
            # (if max_order > 2, multiply more dims; kept generic)
            elif len(dims) > 2:
                for j in dims[1:]:
                    Kc = Kc * K_dim[j]
            K = K + pi[c] * Kc

        return outscale * K


# ============================================================
# 4) Exact GP model
# ============================================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mix_kernel: SharedRBFMixtureKernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = mix_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # GPyTorch will wrap dense covar into a LinearOperator internally
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_w_params(model: ExactGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood) -> List[torch.nn.Parameter]:
    """
    Collect parameters excluding alpha (structure).
    Here, w includes: log_lengthscale (d), log_outputscale (1), likelihood noise.
    """
    mix = model.covar_module
    w = []
    for p in list(model.parameters()) + list(likelihood.parameters()):
        if p is mix.alpha:
            continue
        w.append(p)

    # Deduplicate
    uniq, seen = [], set()
    for p in w:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq


# ============================================================
# 5) Losses
# ============================================================
def train_nll(model, likelihood, mll, X_train, y_train):
    model.train(); likelihood.train()
    out = model(X_train)
    return -mll(out, y_train)


def val_mse(model, likelihood, X_val, y_val):
    model.eval(); likelihood.eval()
    preds = likelihood(model(X_val))
    return torch.mean((preds.mean - y_val) ** 2)


def val_nll_diag(model, likelihood, X_val, y_val):
    model.eval(); likelihood.eval()
    preds = likelihood(model(X_val))
    mu = preds.mean
    var = preds.variance.clamp_min(1e-9)
    nll = 0.5 * torch.log(2 * math.pi * var) + 0.5 * (y_val - mu) ** 2 / var
    return nll.mean()


def val_combo(model, likelihood, X_val, y_val, lam=0.2):
    return val_nll_diag(model, likelihood, X_val, y_val) + lam * val_mse(model, likelihood, X_val, y_val)


@torch.no_grad()
def val_combo_nograd(model, likelihood, X_val, y_val, lam=0.2):
    return float(val_combo(model, likelihood, X_val, y_val, lam=lam).item())


# ============================================================
# 6) Alpha regularization terms (scientific & controllable)
# ============================================================
def alpha_regularization(
    mix: SharedRBFMixtureKernel,
    entropy_weight: float = 0.0,
    dirichlet_strength: float = 0.0,
    dirichlet_conc: float = 0.3,
    order_penalty: float = 0.0,
    alpha_l2: float = 0.0,
) -> torch.Tensor:
    """
    Returns a scalar penalty added to the *outer* objective.
    - entropy_weight * H(pi): minimizing encourages sparse pi
    - dirichlet_strength * NLL(Dirichlet(pi; conc)): conc<1 encourages sparse corners
    - order_penalty: penalize mass on higher order components (order-2 here)
    - alpha_l2: classic weight decay on alpha logits (does NOT force uniform like L1 does)
    """
    pi = mix.mixture_weights()
    reg = torch.zeros((), dtype=DTYPE, device=pi.device)

    if entropy_weight > 0:
        entropy = -(pi * torch.log(pi + 1e-12)).sum()
        reg = reg + entropy_weight * entropy

    if dirichlet_strength > 0:
        # Negative log-prior up to constant:  - (a-1) Σ log pi
        a = float(dirichlet_conc)
        nll = - (a - 1.0) * torch.log(pi + 1e-12).sum()
        reg = reg + dirichlet_strength * nll

    if order_penalty > 0:
        orders = mix.orders.to(pi.device).to(DTYPE)
        # penalize order-2 weight mass
        reg = reg + order_penalty * ((orders - 1.0).clamp_min(0.0) * pi).sum()

    if alpha_l2 > 0:
        reg = reg + alpha_l2 * (mix.alpha ** 2).mean()

    return reg


# ============================================================
# 7) DARTS-style second-order hypergradient (finite-diff HVP)
# ============================================================
def darts_alpha_step_second_order(
    model, likelihood, mll,
    X_train, y_train, X_val, y_val,
    xi: float = 1e-2,
    lambda_mse: float = 0.2,
    # regularization
    entropy_weight: float = 0.0,
    dirichlet_strength: float = 0.0,
    dirichlet_conc: float = 0.3,
    order_penalty: float = 0.0,
    alpha_l2: float = 0.0,
):
    mix = model.covar_module
    w_params = get_w_params(model, likelihood)
    w_vec = parameters_to_vector(w_params)

    # (A) virtual step: w' = w - xi * grad_w L_train(w, alpha)
    for p in w_params:
        p.grad = None
    if mix.alpha.grad is not None:
        mix.alpha.grad = None

    loss_tr = train_nll(model, likelihood, mll, X_train, y_train)
    loss_tr.backward()

    g_w = parameters_to_vector([
        (p.grad if p.grad is not None else torch.zeros_like(p))
        for p in w_params
    ]).detach()
    w_prime = w_vec - xi * g_w

    backup = w_vec.clone()
    with torch.no_grad():
        vector_to_parameters(w_prime, w_params)

    # (B) v = ∇_{w'} L_val(w', alpha)
    for p in w_params:
        p.grad = None
    if mix.alpha.grad is not None:
        mix.alpha.grad = None

    loss_va = val_combo(model, likelihood, X_val, y_val, lam=lambda_mse)
    loss_va.backward()
    v = parameters_to_vector([
        (p.grad if p.grad is not None else torch.zeros_like(p))
        for p in w_params
    ]).detach()

    # (B2) ∇_alpha L_val(w', alpha) + reg(alpha)
    if mix.alpha.grad is not None:
        mix.alpha.grad = None

    loss_va_alpha = val_combo(model, likelihood, X_val, y_val, lam=lambda_mse)
    loss_va_alpha = loss_va_alpha + alpha_regularization(
        mix,
        entropy_weight=entropy_weight,
        dirichlet_strength=dirichlet_strength,
        dirichlet_conc=dirichlet_conc,
        order_penalty=order_penalty,
        alpha_l2=alpha_l2,
    )
    loss_va_alpha.backward(retain_graph=True)
    grad_alpha_val = mix.alpha.grad.detach().clone()
    mix.alpha.grad = None

    # (C) Finite-difference HVP: ∂/∂alpha [∇_w L_train(w,alpha)]ᵀ v
    with torch.no_grad():
        vector_to_parameters(backup, w_params)

    eps = 0.01 / (v.norm() + 1e-12)

    def grad_alpha_train_at(w_perturb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vector_to_parameters(w_perturb, w_params)
        loss = train_nll(model, likelihood, mll, X_train, y_train)
        g = torch.autograd.grad(loss, mix.alpha, retain_graph=False, allow_unused=False)[0]
        return g.detach()

    w_plus = (w_vec + eps * v).detach()
    w_minus = (w_vec - eps * v).detach()
    g_plus = grad_alpha_train_at(w_plus)
    g_minus = grad_alpha_train_at(w_minus)
    hvp = (g_plus - g_minus) / (2.0 * eps)

    # (D) combine
    g_alpha = grad_alpha_val - xi * hvp

    # restore original w
    with torch.no_grad():
        vector_to_parameters(backup, w_params)

    return g_alpha


# ============================================================
# 8) Main bilevel loop with alpha-centering + tau anneal + early stop
# ============================================================
@dataclass
class SearchConfig:
    tau0: float = 2.0
    tau_min: float = 0.3
    tau_decay: float = 0.99
    outer_steps: int = 200
    inner_steps: int = 2

    lr_w: float = 0.03
    lr_alpha: float = 0.06
    xi: float = 1e-2

    lambda_mse: float = 0.2

    # regularization for alpha
    entropy_weight: float = 0.01
    dirichlet_strength: float = 0.0
    dirichlet_conc: float = 0.3
    order_penalty: float = 0.001
    alpha_l2: float = 0.0

    # regularization for w
    weight_decay_w: float = 1e-3

    # early stopping
    patience: int = 30
    print_every: int = 10


def run_search(
    X_train, y_train, X_val, y_val,
    max_order: int = 2,
    seed: int = 0,
    cfg: Optional[SearchConfig] = None,
):
    cfg = cfg or SearchConfig()
    torch.manual_seed(seed)
    _, d = X_train.shape

    subsets, orders = build_subsets(d, max_order=max_order)
    mix = SharedRBFMixtureKernel(d, subsets, orders, tau=cfg.tau0).to(DEVICE).to(DTYPE)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE).to(DTYPE)
    model = ExactGPModel(X_train, y_train, likelihood, mix).to(DEVICE).to(DTYPE)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    w_params = get_w_params(model, likelihood)
    opt_w = torch.optim.Adam(w_params, lr=cfg.lr_w, weight_decay=cfg.weight_decay_w)

    tau = cfg.tau0

    best_val = float("inf")
    best_step = -1
    best_state = None
    no_improve = 0

    history = {"train_nll": [], "val_combo": [], "tau": [], "pi_max": [], "pi_min": []}

    for t in range(cfg.outer_steps):
        # -------- alpha step (DARTS second-order) --------
        g_alpha = darts_alpha_step_second_order(
            model, likelihood, mll,
            X_train, y_train, X_val, y_val,
            xi=cfg.xi,
            lambda_mse=cfg.lambda_mse,
            entropy_weight=cfg.entropy_weight,
            dirichlet_strength=cfg.dirichlet_strength,
            dirichlet_conc=cfg.dirichlet_conc,
            order_penalty=cfg.order_penalty,
            alpha_l2=cfg.alpha_l2,
        )

        with torch.no_grad():
            model.covar_module.alpha.add_(-cfg.lr_alpha * g_alpha)
            # (B) alpha-centering for stability (keeps logits well-conditioned)
            model.covar_module.alpha.sub_(model.covar_module.alpha.mean())

        # (C) tau anneal
        tau = max(cfg.tau_min, tau * cfg.tau_decay)
        model.covar_module.set_tau(tau)

        # -------- inner steps on w (train NLL) --------
        train_loss_last = None
        for _ in range(cfg.inner_steps):
            opt_w.zero_grad(set_to_none=True)
            loss = train_nll(model, likelihood, mll, X_train, y_train)
            loss.backward()
            opt_w.step()
            train_loss_last = float(loss.item())

        # -------- logging / early stop --------
        vm = val_combo_nograd(model, likelihood, X_val, y_val, lam=cfg.lambda_mse)
        with torch.no_grad():
            pi = model.covar_module.mixture_weights().detach().cpu().numpy()
            pmax, pmin = float(pi.max()), float(pi.min())

        history["train_nll"].append(train_loss_last)
        history["val_combo"].append(vm)
        history["tau"].append(tau)
        history["pi_max"].append(pmax)
        history["pi_min"].append(pmin)

        if vm < best_val - 1e-6:
            best_val = vm
            best_step = t + 1
            best_state = {
                "model": deepcopy(model.state_dict()),
                "likelihood": deepcopy(likelihood.state_dict()),
            }
            no_improve = 0
        else:
            no_improve += 1

        if (t + 1) % cfg.print_every == 0 or (t == 0):
            # print top-8 pi entries
            topk = min(8, model.covar_module.num_components)
            top_idx = np.argsort(-pi)[:topk]
            tops = ", ".join([f"{int(i)}:{pi[i]:.4f}" for i in top_idx])
            print(f"[{t+1:03d}] tau={tau:.3f}  train_nll={train_loss_last:+.5f}  "
                  f"val_combo={vm:+.5f}  top_pi=({tops})")

        if no_improve >= cfg.patience:
            print(f"[EarlyStop] No improvement for {cfg.patience} steps. Stopping at step {t+1}.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    return model, likelihood, model.covar_module, history, best_val, best_step


# ============================================================
# 9) Utilities: report top components & export frozen additive kernel
# ============================================================
def top_components(mix: SharedRBFMixtureKernel, k: int = 12):
    with torch.no_grad():
        pi = mix.mixture_weights().detach().cpu()
    vals, idx = torch.topk(pi, k=min(k, mix.num_components))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        dims = mix.subsets[i]
        out.append((int(i), int(mix.orders[i].item()), float(v), tuple(int(d) for d in dims)))
    return out


def export_config(mix: SharedRBFMixtureKernel, likelihood: gpytorch.likelihoods.GaussianLikelihood, topk: int = 12):
    with torch.no_grad():
        pi = mix.mixture_weights().detach().cpu().numpy()
        ls = mix.lengthscales().detach().cpu().numpy()
        outscale = float(mix.outputscale().detach().cpu().item())
        noise = float(likelihood.noise.detach().cpu().item())

    top = np.argsort(-pi)[:min(topk, mix.num_components)]
    comps = []
    for i in top:
        dims = mix.subsets[int(i)]
        comps.append({
            "index": int(i),
            "dims": [int(d) for d in dims],
            "order": int(mix.orders[int(i)].item()),
            "pi": float(pi[int(i)]),
            "eff_outputscale": float(outscale * pi[int(i)]),
        })

    return {
        "input_dim": int(mix.input_dim),
        "max_order": int(int(mix.orders.max().item())),
        "shared_lengthscales": [float(x) for x in ls.tolist()],
        "global_outputscale": outscale,
        "likelihood_noise": noise,
        "components": comps,
    }


def render_kernel_code(cfg: dict, var_name: str = "safe_bo_kernel") -> str:
    """
    Render a copy-pasteable GPytorch kernel construction snippet:
    AdditiveKernel of top components, each ScaleKernel(Product(RBF(dim))) with outputscale=eff_outputscale,
    and shared lengthscales per dim.
    """
    lines = []
    lines.append("import gpytorch")
    lines.append("from gpytorch.kernels import RBFKernel, ProductKernel, ScaleKernel, AdditiveKernel")
    lines.append("")
    lines.append(f"def make_{var_name}():")
    lines.append(f"    # shared lengthscales per dimension")
    lines.append(f"    shared_ls = {cfg['shared_lengthscales']}")
    lines.append(f"    comps = []")
    for j, c in enumerate(cfg["components"]):
        dims = c["dims"]
        eff = c["eff_outputscale"]
        lines.append(f"    # component {j}: dims={dims}, eff_outputscale={eff:.6g}, pi={c['pi']:.6g}")
        if len(dims) == 1:
            d0 = dims[0]
            lines.append(f"    k = ScaleKernel(RBFKernel(active_dims=({d0},)))")
            lines.append(f"    k.base_kernel.initialize(lengthscale=float(shared_ls[{d0}]))")
        else:
            # build ProductKernel chain
            d0 = dims[0]
            lines.append(f"    prod = RBFKernel(active_dims=({d0},))")
            lines.append(f"    prod.initialize(lengthscale=float(shared_ls[{d0}]))")
            for dd in dims[1:]:
                lines.append(f"    ksub = RBFKernel(active_dims=({dd},))")
                lines.append(f"    ksub.initialize(lengthscale=float(shared_ls[{dd}]))")
                lines.append(f"    prod = ProductKernel(prod, ksub)")
            lines.append("    k = ScaleKernel(prod)")
        lines.append(f"    k.initialize(outputscale={eff:.6g})")
        lines.append("    comps.append(k)")
    if len(cfg["components"]) == 1:
        lines.append("    return comps[0]")
    else:
        lines.append("    return AdditiveKernel(*comps)")
    return "\n".join(lines)


# ============================================================
# 10) CLI
# ============================================================
def make_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to gantry CSV (e.g., gantry_data1.csv)")
    p.add_argument("--target", type=str, default="perf", choices=["perf", "safe"], help="Target column name")
    p.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (e.g., 0.2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<index>")
    p.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])

    # kernel search space
    p.add_argument("--max_order", type=int, default=2, help="Max interaction order (use 2 => 45 comps for d=9)")

    # optimization
    p.add_argument("--outer_steps", type=int, default=200)
    p.add_argument("--inner_steps", type=int, default=2)
    p.add_argument("--lr_w", type=float, default=0.03)
    p.add_argument("--lr_alpha", type=float, default=0.06)
    p.add_argument("--xi", type=float, default=1e-2)

    # tau schedule
    p.add_argument("--tau0", type=float, default=2.0)
    p.add_argument("--tau_decay", type=float, default=0.99)
    p.add_argument("--tau_min", type=float, default=0.3)

    # val objective
    p.add_argument("--lambda_mse", type=float, default=0.2)

    # regularization (structure)
    p.add_argument("--entropy_weight", type=float, default=0.01, help="Minimize entropy to encourage sparse pi")
    p.add_argument("--dirichlet_strength", type=float, default=0.0, help="Dirichlet prior strength on pi")
    p.add_argument("--dirichlet_conc", type=float, default=0.3, help="Dirichlet concentration (<1 => sparse)")
    p.add_argument("--order_penalty", type=float, default=0.001, help="Penalty for higher-order mass (order-2)")
    p.add_argument("--alpha_l2", type=float, default=0.0, help="L2 penalty on alpha logits")

    # regularization (hyperparams)
    p.add_argument("--weight_decay_w", type=float, default=1e-3, help="Adam weight decay on w params")

    # early stop / logging
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--print_every", type=int, default=10)

    # export
    p.add_argument("--topk", type=int, default=12, help="How many components to report/export")

    return p


def main():
    args = make_argparser().parse_args()
    configure_runtime(device=args.device, dtype=args.dtype)

    # Load data
    Xtr, ytr, Xva, yva, _norm_info = load_gantry_from_csv_robust(
        args.data, target=args.target, val_ratio=args.val_ratio, seed=args.seed, expect_d=9
    )
    print(f"Dataset loaded: N_train={Xtr.shape[0]}, N_val={Xva.shape[0]}, d={Xtr.shape[1]}")

    cfg = SearchConfig(
        tau0=args.tau0,
        tau_min=args.tau_min,
        tau_decay=args.tau_decay,
        outer_steps=args.outer_steps,
        inner_steps=args.inner_steps,
        lr_w=args.lr_w,
        lr_alpha=args.lr_alpha,
        xi=args.xi,
        lambda_mse=args.lambda_mse,
        entropy_weight=args.entropy_weight,
        dirichlet_strength=args.dirichlet_strength,
        dirichlet_conc=args.dirichlet_conc,
        order_penalty=args.order_penalty,
        alpha_l2=args.alpha_l2,
        weight_decay_w=args.weight_decay_w,
        patience=args.patience,
        print_every=args.print_every,
    )

    # Run search
    model, likelihood, mix, hist, best_val, best_step = run_search(
        Xtr, ytr, Xva, yva,
        max_order=args.max_order,
        seed=args.seed,
        cfg=cfg,
    )

    print(f"\nBest val_combo={best_val:+.6f} at step={best_step}")
    top = top_components(mix, k=args.topk)
    print("Top components (idx, order, weight, dims):")
    for item in top:
        print("  ", item)

    with torch.no_grad():
        ls = mix.lengthscales().detach().cpu().numpy()
        outscale = float(mix.outputscale().detach().cpu().item())
        noise = float(likelihood.noise.detach().cpu().item())
        pi = mix.mixture_weights().detach().cpu().numpy()
    print("\n[Shared lengthscales per dim]:", [f"{x:.4g}" for x in ls.tolist()])
    print(f"[Global outputscale]: {outscale:.6g}")
    print(f"[Likelihood noise]: {noise:.6g}")
    print(f"[pi max/min]: {pi.max():.6f} / {pi.min():.6f}")

    # Export snippet
    exp = export_config(mix, likelihood, topk=args.topk)
    snippet = render_kernel_code(exp, var_name=f"gantry_{args.target}_kernel")
    print("\n=== Exported kernel snippet (copy-paste) ===\n")
    print(snippet)
    print("\n# Reuse these too:")
    print(f"# global_outputscale={exp['global_outputscale']:.6g}")
    print(f"# likelihood_noise={exp['likelihood_noise']:.6g}")
    print(f"# shared_lengthscales={exp['shared_lengthscales']}")


if __name__ == "__main__":
    main()
