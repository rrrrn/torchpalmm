"""torchamm -- scalable REML for additive smooth mixed models on PyTorch."""

__version__ = "0.1.0"

from .linear_mixed_model import LinearMixedModel, lrt_test
from .linear_mixed_model_smoothing import LinearMixedModelSmoothing
from .smooth_spline import SmoothingSpline
from .smooth_spline_mixed import SmoothingSplineMixed

__all__ = [
    "LinearMixedModel",
    "LinearMixedModelSmoothing",
    "SmoothingSpline",
    "SmoothingSplineMixed",
    "lrt_test",
    "__version__",
]
