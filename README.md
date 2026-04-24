# SafeCtrlBO

SafeCtrlBO is an experimental GPyTorch implementation of safe Bayesian
optimization for higher-dimensional control problems.

The central idea is to use an additive Gaussian Process surrogate: the kernel is
a sum of low-dimensional main-effect and interaction components, so the GP can
represent structured high-dimensional objectives more efficiently than a single
dense full-dimensional kernel when the true system has additive or low-order
interaction structure.

`selectKernel.py` is an unfinished helper for exploring additive kernel
structure. `camelback.py` and `hartmann.py` are benchmark scripts. They are not
separate workflows; the core algorithm is `SafeCtrlBO`.

## Repository Layout

- `safectrlbo.py`: SafeCtrlBO loop, safe-set construction, candidate selection,
  and online observation updates.
- `model.py`: exact single-output GP wrapper and fitting utility.
- `kernels.py`: temporary additive-kernel placeholder used by sanity checks.
- `hartmann.py`: 6D safe BO benchmark using SafeCtrlBO in safe mode.
- `camelback.py`: 2D toy benchmark using SafeCtrlBO without an explicit safety
  constraint.
- `selectKernel.py`: experimental DARTS-style additive kernel search helper for
  the gantry CSV data.
- `gp_initialization.py`: minimal GP initialization sanity check.
- `gantry_data1.csv`: small sample gantry dataset with 9 inputs, `perf`, and
  `safe`.

## Installation

Use Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use the existing local environment in this repo, replace `python` with
`./env/bin/python` in the commands below.

The scripts accept `--device auto`, `--device cpu`, `--device mps`, and
`--device cuda[:index]`. For Linux CUDA, install the matching PyTorch wheel for
your driver before installing `requirements.txt`.

## Quick Start

Run the GP sanity check:

```bash
python gp_initialization.py --device auto --dtype float64
```

Run the main safe-mode benchmark:

```bash
python hartmann.py --device auto --dtype float64
```

Faster smoke test:

```bash
python hartmann.py --num-runs 1 --iterations 5 --num-candidates 512 --device cpu
```

Run the 2D unconstrained toy benchmark:

```bash
python camelback.py --num-runs 5 --iterations 20 --num-candidates 4096 --device auto
```

Optionally run the experimental kernel search helper:

```bash
python selectKernel.py --data gantry_data1.csv --target perf --max_order 2 --device auto
```

## SafeCtrlBO Usage

`SafeCtrlBO` models performance `f(x)` and, when supplied, safety `g(x)` with
separate GPs that share the same additive base kernel.

```python
algo = SafeCtrlBO(
    init_X=init_X,
    init_Y_perf=init_Y_perf,
    init_Y_safe=init_Y_safe,
    bounds=bounds,
    base_kernel=base_kernel,
    safety_threshold=safety_threshold,
)

x_next, mode, info = algo.suggest(num_candidates=4096)
algo.observe(x_next, y_perf_new, y_safe_new)
```

In safe mode, candidates are certified with the lower confidence bound of the
safety GP and optimized with the upper confidence bound of the performance GP.
If `init_Y_safe=None` and `safety_threshold=None`, SafeCtrlBO falls back to
unconstrained BO.

## Notes and Current Limitations

- `kernels.py` is only a placeholder. For real high-dimensional control use,
  replace it with an additive kernel that matches the task and input dimension.
- `selectKernel.py` normalizes inputs to `[-1, 1]` and standardizes the target.
  Any exported kernel hyperparameters assume the same scaling at BO time.
- Candidate selection currently uses Sobol samples in the full box. The current
  efficiency gain comes from the additive GP surrogate, not yet from a decomposed
  acquisition optimizer.
- Exact GP inference still scales cubically with the number of observations, so
  long BO runs may need sparse or approximate GP extensions.

# Citation

If you find this repo useful, you can consider citing our work:

```bibtex
@ARTICLE{11174949,
  author={Wang, Hongxuan and Li, Xiaocong and Zheng, Lihao and Bhaumik, Adrish and Vadakkepat, Prahlad},
  journal={IEEE Robotics and Automation Letters},
  title={Safe Bayesian Optimization for Complex Control Systems via Additive Gaussian Processes},
  year={2025},
  volume={10},
  number={11},
  pages={11538-11545},
  doi={10.1109/LRA.2025.3612756}
}
```