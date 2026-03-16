import argparse
import torch

from device_utils import configure_torch_runtime, format_runtime, resolve_device, resolve_dtype
from kernels import make_safe_bo_kernel
from model import build_gp, fit_gp


torch.set_default_dtype(torch.double)


def build_initial_models(device=None, dtype=torch.float64):
    """
    Build a minimal pair of performance/safety GPs for quick sanity checks.
    """
    device = resolve_device(device or "auto")
    dtype = resolve_dtype(dtype)

    train_X = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    train_Y_perf = torch.tensor([[100.0]], dtype=dtype, device=device)
    train_Y_safe = torch.tensor([[100.0]], dtype=dtype, device=device)

    kernel = make_safe_bo_kernel(device=device, dtype=dtype)

    model_f, lik_f, mll_f = build_gp(train_X, train_Y_perf, kernel)
    model_g, lik_g, mll_g = build_gp(train_X, train_Y_safe, kernel)

    return (model_f, lik_f, mll_f), (model_g, lik_g, mll_g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    args = parser.parse_args()

    device = configure_torch_runtime(args.device)
    dtype = resolve_dtype(args.dtype)
    print(f"Initializing GPs with {format_runtime(device, dtype)}")

    (model_f, lik_f, mll_f), (model_g, lik_g, mll_g) = build_initial_models(device=device, dtype=dtype)
    fit_gp(model_f, lik_f, mll_f)
    fit_gp(model_g, lik_g, mll_g)
    print("Initialized performance and safety GPs.")


if __name__ == "__main__":
    main()
