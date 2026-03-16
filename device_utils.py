import torch


_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def resolve_device(device="auto"):
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).lower()
        if requested == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(requested)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot access a GPU. "
            "Please check your PyTorch/CUDA installation and driver setup."
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
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    return device


def format_runtime(device, dtype):
    label = str(device)
    if isinstance(device, torch.device) and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        label = f"{label} ({torch.cuda.get_device_name(index)})"
    return f"device={label}, dtype={str(dtype).replace('torch.', '')}"
