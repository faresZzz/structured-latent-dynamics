"""Training objectives: JEPA prediction, VICReg anti-collapse, the common
constraint factor (total-Jacobian penalty), and the spectral-family ablation
regularizer.
"""

from sld.objectives.jac_reg import TotalJacobianPenalty
from sld.objectives.jepa import EMATarget, JEPALoss
from sld.objectives.spectral_reg import SpectralSeparationReg
from sld.objectives.vicreg import covariance_loss, variance_loss

__all__ = [
    "EMATarget",
    "JEPALoss",
    "SpectralSeparationReg",
    "TotalJacobianPenalty",
    "covariance_loss",
    "variance_loss",
]
