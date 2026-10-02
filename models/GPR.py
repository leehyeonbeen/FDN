import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


"""
Baseline: channel-independent sparse variational Gaussian process regression (GPR),
point-to-point estimator.

Maps the current state x'_t = [q, qdot, qddot, u] to the current wrench W_t with one independent GP
per wrench channel and 1,024 inducing points. Implemented with GPyTorch following
https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
(Model_ below is an unused correlated-multitask variant. Performed worse than the independent version in our experiments.)

Reference: A. Dong, Z. Du, Z. Yan, A sensorless interaction forces estimator for bilateral teleoperation
system based on online sparse Gaussian process regression, Mech. Mach. Theory, 2020.
https://doi.org/10.1016/j.mechmachtheory.2019.103620
"""

class Model(ApproximateGP):
    def __init__(self, configs, inducing_points=None):
        if inducing_points is None:
            num_inducing = configs.num_inducing
            inducing_points = torch.randn(num_inducing, 24)
            enc_in = 24
        else:
            num_inducing, enc_in = inducing_points.size()
        self.num_inducing = num_inducing
        self.enc_in = enc_in
        self.c_out = configs.c_out
        inducing_points = (
            inducing_points.unsqueeze(0).contiguous().expand(self.c_out, -1, -1)
        )

        # inducing points -> N, Cout
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([self.c_out])
        )
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,  # Non-PSD K
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy,
                num_tasks=self.c_out,
            )
        )
        super(Model, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.c_out])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=self.enc_in, batch_shape=torch.Size([self.c_out])
            ),
            batch_shape=torch.Size([self.c_out]),
        )
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.c_out
        )

    @torch.no_grad()
    def inference(self, x):
        predictions = self.likelihood(self(x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        return mean, lower, upper

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Correlated multitask GPR
class Model_(ApproximateGP):
    def __init__(self, configs, inducing_points: int = None):
        if inducing_points is None:
            num_inducing = configs.num_inducing
            inducing_points = torch.randn(num_inducing, 24)
            enc_in = 24
        else:
            num_inducing, enc_in = inducing_points.size()
        self.num_inducing = num_inducing
        self.num_latents = configs.num_latents
        self.enc_in = enc_in
        self.c_out = configs.c_out
        inducing_points = (
            inducing_points.unsqueeze(0).contiguous().expand(self.num_latents, -1, -1)
        )

        # inducing points -> N, Cout
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([self.num_latents])
        )
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,  # Non-PSD K
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=self.c_out,
            num_latents=self.num_latents,
            latent_dim=-1,
        )
        super(Model, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.num_latents])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=enc_in, batch_shape=torch.Size([self.num_latents])
            ),
            batch_shape=torch.Size([self.num_latents]),
        )
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.c_out
        )

    @torch.no_grad()
    def inference(self, x):
        predictions = self.likelihood(self(x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        return mean, lower, upper

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
