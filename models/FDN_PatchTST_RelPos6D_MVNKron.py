from layers.CorrelatedGaussian import KroneckerCorrelatedGaussian
from models.FDN_PatchTST_RelPos6D import Model as IndependentGaussianModel


"""
Residual correlation study of FDN, 'Time-Channel' variant.

Separable temporal and channel-correlated residuals: the residual covariance is D R D with
R = R_T (x) R_C, where both factors are learned through Cholesky factors initialized to identity
(layers/CorrelatedGaussian.py). Otherwise identical to FDN_PatchTST_RelPos6D.
"""

class Model(IndependentGaussianModel):
    def __init__(self, configs):
        super().__init__(configs)
        # Residual distribution with correlation R = R_T (x) R_C
        self.residual_distribution = KroneckerCorrelatedGaussian(
            pred_len=self.pred_len,
            c_out=self.c_out,
        )
        self.sample_residual_is_final = True

        if self.disable_probhead:
            self.residual_distribution.requires_grad_(False)

    # Full multivariate Gaussian NLL of the residual, used by the trainer instead of the independent NLL
    def gaussian_nll(self, target, mu, logvar):
        return self.residual_distribution.nll(target, mu, logvar)

    # Correlated residual sample: mu + sigma * (L eps), with L the Cholesky factor of R
    def sample_residual(self, mu, logvar):
        return self.residual_distribution.rsample(mu, logvar)
