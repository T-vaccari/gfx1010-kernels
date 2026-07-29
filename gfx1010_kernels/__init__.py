from gfx1010_attention import (
    BackendStatus,
    MAX_SEQUENCE_LENGTH,
    SUPPORTED_HEAD_DIMS,
    backend_status,
    can_use_gfx1010_kernel,
    install_pytorch_patch,
    is_pytorch_patch_installed,
    scaled_dot_product_attention,
    uninstall_pytorch_patch,
)

from .normalization import (
    can_use_residual_layer_norm,
    residual_layer_norm,
    residual_layer_norm_status,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "BackendStatus",
    "MAX_SEQUENCE_LENGTH",
    "SUPPORTED_HEAD_DIMS",
    "backend_status",
    "can_use_gfx1010_kernel",
    "can_use_residual_layer_norm",
    "install_pytorch_patch",
    "is_pytorch_patch_installed",
    "residual_layer_norm",
    "residual_layer_norm_status",
    "scaled_dot_product_attention",
    "uninstall_pytorch_patch",
]
