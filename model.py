import copy
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


class SingleOutputGP(ExactGP):
    def __init__(self, train_X, train_Y, likelihood, covar_module):
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMean()
        # we use an additive Gaussian kernel as the covariance
        self.covar_module = covar_module

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


def build_gp(train_X, train_Y, base_kernel, noise=1e-4):
    """
    train_X: (n, d) torch.double
    train_Y: (n, 1) torch.double
    base_kernel: AdditiveKernel (usually frozen from DARTS search)
    noise: Gaussian likelihood noise variance (not standard deviation)

    Returns:
        model: SingleOutputGP
        likelihood: GaussianLikelihood
        mll: ExactMarginalLogLikelihood
    """
    device = train_X.device
    dtype = train_X.dtype

    # ensure Y is 1D target vector for ExactGP
    if train_Y.dim() == 2 and train_Y.size(-1) == 1:
        train_Y_flat = train_Y.squeeze(-1)
    else:
        train_Y_flat = train_Y

    # each GP has one deepcopied kernel to avoid hyperparameters sharing
    kernel = copy.deepcopy(base_kernel).to(device=device, dtype=dtype)

    noise_value = None if noise is None else float(noise)
    noise_lower_bound = 1e-4
    if noise_value is not None:
        noise_lower_bound = max(min(noise_value * 0.5, 1e-4), 1e-12)

    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound)
    ).to(device=device, dtype=dtype)
    # if noise is provided, initialize noise; otherwise keep default gpytorch init
    if noise_value is not None:
        likelihood.initialize(noise=noise_value)

    model = SingleOutputGP(train_X, train_Y_flat, likelihood, kernel).to(device=device, dtype=dtype)

    mll = ExactMarginalLogLikelihood(likelihood, model)

    return model, likelihood, mll


def fit_gp(
    model,
    likelihood,
    mll,
    training_iter=200,
    lr=0.05,
    train_kernel=False,
    train_mean=False,
    train_noise=True,
):
    """
    Fit GP hyperparameters with Adam.

    By default (for DARTS-frozen kernels in SafeCtrlBO), we recommend:
        train_kernel=False, train_mean=False, train_noise=True
    so that:
        - kernel hyperparameters (lengthscale/outputscale) stay fixed from DARTS
        - only likelihood noise is adapted online if needed.

    Args:
        model: SingleOutputGP
        likelihood: GaussianLikelihood
        mll: ExactMarginalLogLikelihood
        training_iter: number of optimization steps (0 => skip training)
        lr: learning rate for Adam
        train_kernel: whether to update covar_module parameters
        train_mean: whether to update mean_module parameters
        train_noise: whether to update likelihood noise parameters
    """
    if training_iter is None or training_iter <= 0:
        # nothing to do
        return

    # set requires_grad flags according to training options
    for p in model.covar_module.parameters():
        p.requires_grad = bool(train_kernel)

    for p in model.mean_module.parameters():
        p.requires_grad = bool(train_mean)

    for p in likelihood.parameters():
        p.requires_grad = bool(train_noise)

    # collect parameters that are actually trainable
    params = []
    seen = set()
    for param in list(model.parameters()) + list(likelihood.parameters()):
        if not param.requires_grad or id(param) in seen:
            continue
        params.append(param)
        seen.add(id(param))

    # if no parameter is trainable, exit early
    if len(params) == 0:
        return

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(params, lr=lr)

    train_X = model.train_inputs[0]
    train_Y = model.train_targets

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        optimizer.step()
