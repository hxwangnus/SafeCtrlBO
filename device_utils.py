import torch


_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _mps_backend():
    return getattr(torch.backends, "mps", None)


def mps_is_available():
    backend = _mps_backend()
    return backend is not None and backend.is_available()


def mps_is_built():
    backend = _mps_backend()
    return backend is not None and backend.is_built()


def resolve_device(device="auto"):
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                resolved = torch.device("cuda")
            elif mps_is_available():
                resolved = torch.device("mps")
            else:
                resolved = torch.device("cpu")
        else:
            resolved = torch.device(requested)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot access a GPU. "
            "Please check your PyTorch/CUDA installation and driver setup."
        )

    if resolved.type == "mps" and not mps_is_available():
        detail = (
            "this PyTorch build was not compiled with MPS support."
            if not mps_is_built()
            else "MPS is not available on this machine."
        )
        raise RuntimeError(
            "MPS was requested, but this Python environment cannot access the Apple GPU: "
            f"{detail} "
            "Please use a macOS PyTorch build with MPS support on an Apple Silicon Mac, "
            "or fall back to --device cpu."
        )

    return resolved


def resolve_dtype(dtype="float64"):
    if isinstance(dtype, torch.dtype):
        return dtype

    key = "float64" if dtype is None else str(dtype).lower()
    if key not in _DTYPE_MAP:
        valid = ", ".join(sorted(_DTYPE_MAP))
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {valid}.")
    return _DTYPE_MAP[key]


def configure_torch_runtime(device):
    device = resolve_device(device)
    if device.type == "cuda":
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    elif device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    return device


def format_runtime(device, dtype):
    label = str(device)
    if isinstance(device, torch.device) and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        label = f"{label} ({torch.cuda.get_device_name(index)})"
    elif isinstance(device, torch.device) and device.type == "mps":
        label = "mps (Apple Silicon GPU)"
    return f"device={label}, dtype={str(dtype).replace('torch.', '')}"
