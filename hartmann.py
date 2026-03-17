# hartmann.py
from __future__ import print_function, division, absolute_import

import argparse
import os
import sys
import tempfile
import warnings
from itertools import combinations

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gpytorch
from gpytorch.utils.warnings import GPInputWarning

from device_utils import configure_torch_runtime, format_runtime, resolve_device, resolve_dtype
from safectrlbo import SafeCtrlBO


warnings.filterwarnings("ignore", category=GPInputWarning)


INPUT_DIM = 6
DOMAIN_SCALE = 4.0
GLOBAL_OPT = 3.322368011415515
PLOT_FLOOR = 1e-12
DEFAULT_LIKELIHOOD_NOISE_FLOOR = 1e-8

BOUNDS_NP = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [DOMAIN_SCALE, DOMAIN_SCALE, DOMAIN_SCALE, DOMAIN_SCALE, DOMAIN_SCALE, DOMAIN_SCALE],
])

HARTMANN_ALPHA = (1.0, 1.2, 3.0, 3.2)
HARTMANN_A = (
    (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
    (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
    (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
    (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
)
HARTMANN_P = (
    (1312, 1696, 5569, 124, 8283, 5886),
    (2329, 4135, 8307, 3736, 1004, 9991),
    (2348, 1451, 3522, 2883, 3047, 6650),
    (4047, 8828, 8732, 5743, 1091, 381),
)


def configure_reproducibility(seed, device):
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_run_seeds(num_runs, seed):
    if seed is None:
        return [None] * num_runs

    seed_seq = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in seed_seq.spawn(num_runs)]


def summarize_regret(all_simple_regret, success_threshold):
    mean_regret = np.mean(all_simple_regret, axis=1)
    median_regret = np.median(all_simple_regret, axis=1)
    std_regret = np.std(all_simple_regret, axis=1)
    q25_regret = np.quantile(all_simple_regret, 0.25, axis=1)
    q75_regret = np.quantile(all_simple_regret, 0.75, axis=1)
    success_rate = np.mean(all_simple_regret <= success_threshold, axis=1)
    return mean_regret, median_regret, std_regret, q25_regret, q75_regret, success_rate


def print_summary_report(mean_regret, median_regret, std_regret, success_rate, success_threshold):
    num_steps = mean_regret.shape[0]
    checkpoints = []
    for step in [1, 5, 10, 25, 50, 100, 150, num_steps]:
        if 1 <= step <= num_steps and step not in checkpoints:
            checkpoints.append(step)

    print("")
    print(f"{'Step':>6} {'Mean':>14} {'Median':>14} {'Std':>14} {'SuccessRate':>14}")
    for step in checkpoints:
        idx = step - 1
        print(
            f"{step:6d} "
            f"{mean_regret[idx]:14.6e} "
            f"{median_regret[idx]:14.6e} "
            f"{std_regret[idx]:14.6e} "
            f"{success_rate[idx]:13.2%}"
        )

    final_idx = num_steps - 1
    print("")
    print(
        "Final summary: "
        f"mean={mean_regret[final_idx]:.6e}, "
        f"median={median_regret[final_idx]:.6e}, "
        f"std={std_regret[final_idx]:.6e}, "
        f"success@{success_threshold:.1e}={success_rate[final_idx]:.2%}"
    )


def print_violation_report(violation_counts, iterations):
    total_violations = int(np.sum(violation_counts))
    mean_violations = float(np.mean(violation_counts))
    median_violations = float(np.median(violation_counts))
    max_violations = int(np.max(violation_counts))
    total_measurements = int(iterations * len(violation_counts))
    violation_rate = total_violations / total_measurements if total_measurements > 0 else 0.0

    print("")
    print("Safety violations:")
    print(
        f"total={total_violations}, "
        f"mean_per_run={mean_violations:.2f}, "
        f"median_per_run={median_violations:.2f}, "
        f"max_per_run={max_violations}, "
        f"rate={violation_rate:.2%}"
    )


def make_bounds(device, dtype):
    return torch.tensor(BOUNDS_NP, dtype=dtype, device=device)


def hartmann6d_torch(x: torch.Tensor, noise_std: float = 1e-4) -> torch.Tensor:
    """
    Evaluate the 6D Hartmann function.

    The benchmark domain in this script is [0, 4]^6 for compatibility with the
    older setup; points are rescaled into [0, 1]^6 before evaluation.
    """
    x = x.view(-1, INPUT_DIM)
    x_unit = x / DOMAIN_SCALE

    alpha = x.new_tensor(HARTMANN_ALPHA)
    A = x.new_tensor(HARTMANN_A)
    P = 1e-4 * x.new_tensor(HARTMANN_P)

    delta = x_unit.unsqueeze(1) - P.unsqueeze(0)
    internal_sum = (A.unsqueeze(0) * delta.square()).sum(dim=-1)
    external_sum = (alpha.unsqueeze(0) * torch.exp(-internal_sum)).sum(dim=-1)

    if noise_std is not None and noise_std > 0.0:
        external_sum = external_sum + noise_std * torch.randn_like(external_sum)

    return external_sum


def build_kernel_subsets(input_dim: int, d_effective: int):
    d_effective = int(max(1, min(d_effective, input_dim)))
    subsets = []
    for order in range(1, d_effective + 1):
        subsets.extend(combinations(range(input_dim), order))
    return subsets


def make_full_additive_kernel(
    device,
    dtype,
    input_dim: int = INPUT_DIM,
    d_effective: int = INPUT_DIM,
    lengthscale: float = 1.0,
    total_outputscale: float = 1.0,
):
    """
    Build a truncated full-additive kernel over all subsets up to d_effective.

    d_effective=1 -> additive main effects only
    d_effective=2 -> main effects + all pairwise interactions
    ...
    d_effective=6 -> all non-empty subsets of the 6 dimensions
    """
    subsets = build_kernel_subsets(input_dim, d_effective)
    if total_outputscale <= 0.0:
        raise ValueError("total_outputscale must be positive.")

    component_outputscale = float(total_outputscale) / float(len(subsets))
    components = []

    for dims in subsets:
        factors = []
        for dim in dims:
            factor = gpytorch.kernels.RBFKernel(active_dims=(int(dim),))
            factor.initialize(lengthscale=float(lengthscale))
            factors.append(factor)

        kernel = factors[0]
        for factor in factors[1:]:
            kernel = gpytorch.kernels.ProductKernel(kernel, factor)

        scaled_kernel = gpytorch.kernels.ScaleKernel(kernel)
        scaled_kernel.initialize(outputscale=component_outputscale)
        components.append(scaled_kernel)

    return gpytorch.kernels.AdditiveKernel(*components).to(device=device, dtype=dtype)


def load_initial_points(path, num_runs):
    x0_list = np.load(path)
    if x0_list.ndim == 2:
        if x0_list.shape[1] != INPUT_DIM:
            raise ValueError(f"Expected x0 file with {INPUT_DIM} columns, got shape {x0_list.shape}.")
        x0_list = x0_list[:, None, :]
    elif x0_list.ndim != 3 or x0_list.shape[1:] != (1, INPUT_DIM):
        raise ValueError(
            "Expected x0 file with shape (num_runs, 6) or (num_runs, 1, 6). "
            f"Got {x0_list.shape}."
        )

    if x0_list.shape[0] < num_runs:
        raise ValueError(
            f"x0 file contains {x0_list.shape[0]} runs, but num_runs={num_runs} was requested."
        )

    return np.asarray(x0_list[:num_runs], dtype=np.float64)


def sample_safe_initial_point(run_rng, device, dtype, safety_threshold, noise_std, max_attempts):
    for _ in range(max_attempts):
        x0_np = run_rng.uniform(low=0.0, high=DOMAIN_SCALE, size=(1, INPUT_DIM))
        x0 = torch.tensor(x0_np, dtype=dtype, device=device)
        y0 = hartmann6d_torch(x0, noise_std=noise_std).view(-1, 1)
        if y0.item() >= safety_threshold:
            return x0, y0

    raise RuntimeError(
        f"Could not sample a safe initial point after {max_attempts} attempts. "
        "Try lowering the safety threshold or supplying --x0-file."
    )


def run_experiment(
    num_runs: int = 100,
    iterations: int = 200,
    num_candidates: int = 8192,
    d_effective: int = INPUT_DIM,
    lengthscale: float = 1.0,
    total_outputscale: float = 1.0,
    safety_threshold: float = 0.3,
    tau: float = 0.2,
    switch_time: int = 15,
    safe_retry_radius: float = 0.05,
    noise_std: float = 1e-4,
    device=None,
    dtype=torch.float64,
    seed=None,
    x0_file=None,
    max_init_attempts: int = 10000,
):
    """
    Run SafeCtrlBO on the 6D Hartmann benchmark in safe mode.

    The same noisy Hartmann measurement is used as both the performance signal
    and the safety signal, matching the older single-output safe benchmark.
    """
    device = resolve_device(device or "auto")
    dtype = resolve_dtype(dtype)
    bounds = make_bounds(device, dtype)
    configure_reproducibility(seed, device)
    run_seeds = make_run_seeds(num_runs, seed)

    initial_points = load_initial_points(x0_file, num_runs) if x0_file else None
    base_kernel = make_full_additive_kernel(
        device=device,
        dtype=dtype,
        input_dim=INPUT_DIM,
        d_effective=d_effective,
        lengthscale=lengthscale,
        total_outputscale=total_outputscale,
    )

    all_simple_regret = np.zeros((iterations, num_runs), dtype=float)
    all_violation_counts = np.zeros(num_runs, dtype=int)

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        run_seed = run_seeds[run]
        run_rng = np.random.default_rng(run_seed)

        if initial_points is not None:
            x0_np = initial_points[run]
            x0 = torch.tensor(x0_np, dtype=dtype, device=device)
            y0 = hartmann6d_torch(x0, noise_std=noise_std).view(-1, 1)
            if y0.item() < safety_threshold:
                raise RuntimeError(
                    f"Initial point from --x0-file for run {run + 1} is not safe under the "
                    f"current noisy measurement: y0={y0.item():.6f} < {safety_threshold:.6f}."
                )
        else:
            x0, y0 = sample_safe_initial_point(
                run_rng=run_rng,
                device=device,
                dtype=dtype,
                safety_threshold=safety_threshold,
                noise_std=noise_std,
                max_attempts=max_init_attempts,
            )

        algo = SafeCtrlBO(
            init_X=x0,
            init_Y_perf=y0,
            init_Y_safe=y0.clone(),
            bounds=bounds,
            base_kernel=base_kernel,
            safety_threshold=safety_threshold,
            switch_time=switch_time,
            beta_fn=None,
            tau=tau,
            device=device,
            init_training_iter=0,
            likelihood_noise=max(float(noise_std) ** 2, DEFAULT_LIKELIHOOD_NOISE_FLOOR),
            sobol_seed=run_seed,
            safe_retry_radius=safe_retry_radius,
        )

        y_best = y0.item()
        safety_violations = 0

        for t in range(iterations):
            sys.stdout.write(f"\rRun {run + 1}/{num_runs} - Iteration {t + 1}/{iterations}")
            sys.stdout.flush()

            x_next, mode, sets = algo.suggest(num_candidates=num_candidates)

            y_next = hartmann6d_torch(x_next, noise_std=noise_std)
            y_next_val = y_next.item()
            y_next_tensor = y_next.view(-1, 1)

            if y_next_val < safety_threshold:
                safety_violations += 1

            if y_next_val > y_best:
                y_best = y_next_val

            simple_regret = max(GLOBAL_OPT - y_best, 0.0)
            all_simple_regret[t, run] = simple_regret

            algo.observe(
                x_new=x_next,
                y_perf_new=y_next_tensor,
                y_safe_new=y_next_tensor,
                train_hypers_every=None,
                training_iter=0,
            )

        all_violation_counts[run] = safety_violations
        final_simple_regret = all_simple_regret[iterations - 1, run]
        print(f"\nFinal Simple Regret for Run {run + 1}: {final_simple_regret:.6f}")
        print(f"Safety Violations for Run {run + 1}: {safety_violations}")

    return all_simple_regret, all_violation_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-candidates", type=int, default=8192)
    parser.add_argument("--d-effective", type=int, default=INPUT_DIM, choices=range(1, INPUT_DIM + 1))
    parser.add_argument("--lengthscale", type=float, default=1.0, help="Shared RBF lengthscale for all kernel factors")
    parser.add_argument("--total-outputscale", type=float, default=1.0, help="Total variance budget spread evenly over all components")
    parser.add_argument("--noise-std", type=float, default=1e-4, help="Observation noise added to the Hartmann value")
    parser.add_argument("--safety-threshold", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.2, help="Boundary width for the SafeCtrlBO safe set")
    parser.add_argument("--switch-time", type=int, default=15, help="Number of early iterations spent in boundary expansion mode")
    parser.add_argument("--safe-retry-radius", type=float, default=0.05)
    parser.add_argument("--x0-file", type=str, default=None, help="Optional .npy file of safe initial points with shape (runs, 6) or (runs, 1, 6)")
    parser.add_argument("--max-init-attempts", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, cuda, or cuda:<index>")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    parser.add_argument("--seed", type=int, default=0, help="Base seed for fully reproducible runs")
    parser.add_argument("--success-threshold", type=float, default=1e-2)
    args = parser.parse_args()

    device = configure_torch_runtime(args.device)
    dtype = resolve_dtype(args.dtype)
    num_components = len(build_kernel_subsets(INPUT_DIM, args.d_effective))

    print(f"Running Hartmann BO with {format_runtime(device, dtype)}")
    print(f"Reproducibility seed: {args.seed}")
    print(f"Benchmark domain: [0, {DOMAIN_SCALE}]^{INPUT_DIM}, evaluated on Hartmann6D(x / {DOMAIN_SCALE})")
    print(
        f"Kernel: truncated full-additive up to order {args.d_effective} "
        f"with {num_components} components"
    )

    all_simple_regret_matrix, all_violation_counts = run_experiment(
        num_runs=args.num_runs,
        iterations=args.iterations,
        num_candidates=args.num_candidates,
        d_effective=args.d_effective,
        lengthscale=args.lengthscale,
        total_outputscale=args.total_outputscale,
        safety_threshold=args.safety_threshold,
        tau=args.tau,
        switch_time=args.switch_time,
        safe_retry_radius=args.safe_retry_radius,
        noise_std=args.noise_std,
        device=device,
        dtype=dtype,
        seed=args.seed,
        x0_file=args.x0_file,
        max_init_attempts=args.max_init_attempts,
    )

    (
        mean_simple_regret,
        median_simple_regret,
        std_simple_regret,
        q25_simple_regret,
        q75_simple_regret,
        success_rate,
    ) = summarize_regret(
        all_simple_regret_matrix,
        success_threshold=args.success_threshold,
    )

    print_summary_report(
        mean_simple_regret,
        median_simple_regret,
        std_simple_regret,
        success_rate,
        success_threshold=args.success_threshold,
    )
    print_violation_report(all_violation_counts, args.iterations)

    print("Plotting simple regret curve...")
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(1, args.iterations + 1)
    mean_curve = np.clip(mean_simple_regret, PLOT_FLOOR, None)
    median_curve = np.clip(median_simple_regret, PLOT_FLOOR, None)
    lower_band = np.clip(q25_simple_regret, PLOT_FLOOR, None)
    upper_band = np.clip(q75_simple_regret, PLOT_FLOOR, None)

    plt.plot(x_axis, mean_curve, label="Mean Simple Regret")
    plt.plot(x_axis, median_curve, label="Median Simple Regret", linestyle="--")
    plt.fill_between(
        x_axis,
        lower_band,
        upper_band,
        alpha=0.3,
        label="Interquartile Range",
    )
    plt.yscale("log")
    plt.xlabel("Optimization Step")
    plt.ylabel("Simple Regret")
    plt.title(f"SafeCtrlBO on Hartmann6D (d_effective={args.d_effective})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = "hartmann_simple_regret.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {out_path}")
    print("Done plotting.")


if __name__ == "__main__":
    main()
