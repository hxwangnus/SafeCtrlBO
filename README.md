# SafeCtrlBO

Experimental GPyTorch code for safe Bayesian optimization with two current workflows:

- `selectKernel.py`: search for a sparse additive kernel from gantry controller data.
- `camelback.py`: run repeated Bayesian optimization on a 2D camelback benchmark using `SafeCtrlBO`.

The core BO loop is implemented in `safectrlbo.py`, GP model helpers live in `model.py`, and reusable runtime helpers live in `device_utils.py`.

## What This Repo Does

The main idea is:

1. Learn or select a GP kernel structure from data.
2. Freeze that kernel.
3. Use it inside a `SafeCtrlBO` optimizer that can operate in either:
   - safe mode, when a safety signal and threshold are provided
   - unconstrained mode, when they are omitted

There are currently two separate tracks in the repo:

- Gantry kernel search
  - `selectKernel.py` loads 9D controller data from `gantry_data1.csv`
  - it performs a DARTS-style bilevel search over kernel structure
  - it prints a copy-pasteable frozen kernel snippet at the end

- Camelback benchmark
  - `camelback.py` runs many BO trials on a 2D synthetic objective
  - it uses `SafeCtrlBO` in unconstrained mode
  - it writes `camelback_simple_regret.png`

## Repository Layout

- `safectrlbo.py`: main optimization loop, candidate generation, safe-set logic, and observation updates
- `model.py`: exact GP wrapper plus GP fitting utilities
- `selectKernel.py`: kernel-structure search on CSV data
- `kernels.py`: frozen additive kernel constructor currently used by `gp_initialization.py`
- `gp_initialization.py`: minimal sanity check that instantiates performance and safety GPs
- `camelback.py`: 2D benchmark script for repeated BO runs and regret plotting
- `device_utils.py`: device and dtype helpers
- `gantry_data1.csv`: sample gantry dataset with 9 inputs plus `perf` and `safe`

## Environment

The original Linux/CUDA environment for this repo used:

- Python 3.10.19
- NumPy 2.2.6
- PyTorch 2.10.0
- GPyTorch 1.15.2
- Matplotlib 3.10.8

For a fresh macOS or Linux setup, use Python `3.10` plus the
cross-platform `requirements.txt` below, and choose the appropriate PyTorch
wheel for your platform when you need GPU support.

## Installation

Create a clean virtual environment and install the core dependencies from
`requirements.txt`.

### Recommended: `uv` on macOS or Linux

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### Standard library `venv`

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- The recommended Python baseline for this repo is `3.10`.
- On Apple Silicon Macs, PyTorch uses the `mps` device for GPU acceleration. `--device auto` now prefers `cuda`, then `mps`, then `cpu`.
- If you want NVIDIA CUDA support on Linux, install the matching PyTorch wheel from the official selector first, then install `requirements.txt`. For example, CUDA 12.8:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.7,<2.8"
uv pip install -r requirements.txt
```

- On macOS, if a specific op is not implemented on MPS yet, PyTorch can fall back to CPU when `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before launching Python:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- The scripts support `--device auto`, `--device cpu`, `--device mps`, and `--device cuda[:index]`.
- `camelback.py` configures a non-interactive Matplotlib backend, so it works in headless environments.
- The commands below assume you activated a virtual environment. If you prefer to use the committed environment in this repo, replace `python` with `./env/bin/python`.

## Quick Start

### 1. Sanity Check GP Initialization

This is the fastest way to confirm the basic GP code path is working:

```bash
python gp_initialization.py --device auto --dtype float64
```

Expected output:

```text
Initializing GPs with device=..., dtype=float64
Initialized performance and safety GPs.
```

On Apple Silicon, you can also verify MPS directly:

```bash
python -c "import torch; print('mps built:', torch.backends.mps.is_built()); print('mps available:', torch.backends.mps.is_available())"
python gp_initialization.py --device mps --dtype float64
```

### 2. Run the Camelback Benchmark

```bash
python camelback.py --device auto --dtype float64
```

Useful flags:

- `--num-runs`: number of repeated BO runs, default `100`
- `--iterations`: BO steps per run, default `150`
- `--num-candidates`: Sobol candidates evaluated per step, default `16384`
- `--seed`: base seed for reproducibility
- `--success-threshold`: threshold used when reporting success rate

Output:

- console summary statistics over simple regret
- `camelback_simple_regret.png`

Example for a faster smoke test:

```bash
python camelback.py --num-runs 5 --iterations 20 --num-candidates 4096 --device auto --dtype float64
```

### 3. Run Kernel Search on the Gantry Data

```bash
python selectKernel.py --data gantry_data1.csv --target perf --device auto --dtype float64
```

Useful flags:

- `--target {perf,safe}`: choose which CSV target to model
- `--max_order`: maximum interaction order in the kernel search space
- `--outer_steps`: outer bilevel optimization steps
- `--inner_steps`: inner hyperparameter optimization steps
- `--topk`: number of components to include in the exported snippet

Example:

```bash
python selectKernel.py \
  --data gantry_data1.csv \
  --target perf \
  --max_order 2 \
  --outer_steps 100 \
  --inner_steps 2 \
  --device auto \
  --dtype float64
```

At the end of a run, the script prints:

- best validation objective
- top mixture components
- learned shared lengthscales
- GP likelihood noise
- a copy-pasteable kernel snippet for reuse

## Data Format

`gantry_data1.csv` currently uses a header of:

```text
px1,ix1,dx1,px2,ix2,dx2,py,iy,dy,perf,safe
```

The loader in `selectKernel.py` expects:

- the first 9 columns to be input features
- `perf` or `safe` as the target column when a header is present
- at least enough valid rows to survive cleaning and train/validation splitting

The current sample file has 30 data rows plus a header row.

## How SafeCtrlBO Works

`SafeCtrlBO` is initialized with:

- observed inputs `init_X`
- performance observations `init_Y_perf`
- optional safety observations `init_Y_safe`
- box constraints `bounds`
- a frozen `base_kernel`

Behavior depends on whether safety data is supplied:

- Safe mode:
  - uses a separate GP for safety
  - certifies safe candidates through a lower confidence bound
  - expands or optimizes only within the safe set
  - falls back to certified-safe observed points when needed

- Unconstrained mode:
  - if `init_Y_safe=None` and `safety_threshold=None`, all candidates are treated as safe
  - this is how `camelback.py` currently runs
