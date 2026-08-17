from .attention import (
    BackendStatus,
    MAX_SEQUENCE_LENGTH,
    SUPPORTED_HEAD_DIMS,
    backend_status,
    can_use_gfx1010_kernel,
    scaled_dot_product_attention,
)
from .patch import (
    install_pytorch_patch,
    is_pytorch_patch_installed,
    uninstall_pytorch_patch,
)

from .normalization import (
    can_use_residual_layer_norm,
    residual_layer_norm,
    residual_layer_norm_status,
)
from .linear import AutocastLinear, autocast_linear

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "BackendStatus",
    "MAX_SEQUENCE_LENGTH",
    "SUPPORTED_HEAD_DIMS",
    "AutocastLinear",
    "backend_status",
    "autocast_linear",
    "can_use_gfx1010_kernel",
    "can_use_residual_layer_norm",
    "install_pytorch_patch",
    "is_pytorch_patch_installed",
    "residual_layer_norm",
    "residual_layer_norm_status",
    "scaled_dot_product_attention",
    "uninstall_pytorch_patch",
]
