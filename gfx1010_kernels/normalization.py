from functools import lru_cache
import warnings

import torch
from torch.nn import functional as F

from .attention import BackendStatus, backend_status


MIN_HIDDEN_SIZE = 128
MAX_HIDDEN_SIZE = 1024
NATIVE_HIDDEN_SIZES = (128, 256, 384, 512, 768, 1024)


def _torch_residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p,
    eps,
    training,
):
    dropped = F.dropout(branch, dropout_p, training) if training else branch
    updated = x + dropped
    normalized = F.layer_norm(
        updated,
        (updated.shape[-1],),
        weight,
        bias,
        eps,
    )
    return updated, normalized


@lru_cache(maxsize=None)
def _triton_is_usable(device_index):
    return backend_status(torch.device("cuda", device_index))


@lru_cache(maxsize=None)
def _native_status(device_index):
    if not torch.cuda.is_available():
        return BackendStatus(False, None, None, "PyTorch CUDA/HIP is unavailable")
    if torch.version.hip is None:
        return BackendStatus(
            False,
            torch.cuda.get_device_name(device_index),
            None,
            "PyTorch was not built with ROCm",
        )
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", None)
    if architecture is None:
        architecture = getattr(properties, "gcn_arch_name", "")
    architecture = architecture.split(":", 1)[0]
    if architecture != "gfx1010":
        return BackendStatus(
            False,
            properties.name,
            architecture,
            "the native extension targets gfx1010",
        )
    from ._normalization_native import native_extension_status

    available, reason = native_extension_status()
    return BackendStatus(
        available,
        properties.name,
        architecture,
        "gfx1010 native normalization backend is ready" if available else reason,
    )


def residual_layer_norm_status(device=None):
    if not torch.cuda.is_available():
        return BackendStatus(False, None, None, "PyTorch CUDA/HIP is unavailable")
    if device is None:
        device_index = torch.cuda.current_device()
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return BackendStatus(
                False,
                str(resolved),
                None,
                "the device must be a HIP GPU",
            )
        device_index = (
            torch.cuda.current_device()
            if resolved.index is None
            else resolved.index
        )
    return _native_status(device_index)


def _input_reason(x, branch, weight, bias, dropout_p, eps):
    if not x.is_cuda:
        return "x must be a HIP tensor"
    if not branch.is_cuda or not weight.is_cuda or not bias.is_cuda:
        return "x, branch, weight and bias must be HIP tensors"
    if not (x.device == branch.device == weight.device == bias.device):
        return "x, branch, weight and bias must be on the same device"
    if x.ndim < 2 or branch.shape != x.shape:
        return "x and branch must have the same shape with at least two dimensions"
    hidden_size = x.shape[-1]
    if not MIN_HIDDEN_SIZE <= hidden_size <= MAX_HIDDEN_SIZE:
        return (
            f"hidden size must be between {MIN_HIDDEN_SIZE} and "
            f"{MAX_HIDDEN_SIZE}"
        )
    if hidden_size % 8 != 0:
        return "hidden size must be divisible by 8"
    if weight.shape != (hidden_size,) or bias.shape != (hidden_size,):
        return "weight and bias must match the final tensor dimension"
    if x.dtype != torch.float32:
        return "the optimized residual stream requires float32 x"
    if branch.dtype not in {torch.float16, torch.float32}:
        return "branch must use float16 or float32"
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        return "weight and bias must use float32"
    if not all(
        tensor.is_contiguous() for tensor in (x, branch, weight, bias)
    ):
        return "x, branch, weight and bias must be contiguous"
    if not 0.0 <= dropout_p < 1.0:
        return "dropout_p must be in [0, 1)"
    if eps <= 0.0:
        return "eps must be positive"
    if x.numel() == 0:
        return "empty tensors use the PyTorch fallback"
    return None


def can_use_residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p=0.0,
    eps=1e-5,
):
    reason = _input_reason(x, branch, weight, bias, dropout_p, eps)
    if reason is not None or x.shape[-1] not in NATIVE_HIDDEN_SIZES:
        return False
    return residual_layer_norm_status(x.device).available


def residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p=0.0,
    eps=1e-5,
    training=True,
    *,
    implementation="auto",
):
    if implementation not in {"auto", "gfx1010", "triton", "torch"}:
        raise ValueError(
            "implementation must be 'auto', 'gfx1010', 'triton' or 'torch'"
        )
    if implementation == "torch":
        return _torch_residual_layer_norm(
            x,
            branch,
            weight,
            bias,
            dropout_p,
            eps,
            training,
        )
    if implementation == "gfx1010":
        if not x.is_cuda:
            raise RuntimeError(
                "gfx1010 native residual LayerNorm is unavailable: "
                "x must be a HIP tensor"
            )
        if x.ndim == 0:
            raise RuntimeError(
                "gfx1010 native residual LayerNorm is unavailable: "
                "x must have at least two dimensions"
            )
        hidden_size = x.shape[-1]
        if hidden_size not in NATIVE_HIDDEN_SIZES:
            raise RuntimeError(
                "gfx1010 native residual LayerNorm is unavailable: "
                f"native hidden size must be one of {NATIVE_HIDDEN_SIZES}"
            )
        status = residual_layer_norm_status(x.device)
        if not status.available:
            raise RuntimeError(
                "gfx1010 native residual LayerNorm is unavailable: "
                f"{status.reason}"
            )
        from ._normalization_native import native_residual_layer_norm

        return native_residual_layer_norm(
            x,
            branch,
            weight,
            bias,
            float(dropout_p),
            float(eps),
            bool(training),
        )
    reason = _input_reason(
        x,
        branch,
        weight,
        bias,
        dropout_p,
        eps,
    )
    if reason is None and implementation == "triton":
        status = _triton_is_usable(
            x.device.index
            if x.device.index is not None
            else torch.cuda.current_device()
        )
        if not status.available:
            reason = status.reason
    if reason is not None:
        if implementation in {"gfx1010", "triton"}:
            raise RuntimeError(
                f"gfx1010 residual LayerNorm is unavailable: {reason}"
            )
        warnings.warn(
            f"gfx1010 optimized residual LayerNorm was not used: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_residual_layer_norm(
            x,
            branch,
            weight,
            bias,
            dropout_p,
            eps,
            training,
        )

    if implementation != "triton":
        if x.shape[-1] not in NATIVE_HIDDEN_SIZES:
            native_reason = (
                f"native hidden size must be one of {NATIVE_HIDDEN_SIZES}"
            )
        else:
            status = residual_layer_norm_status(x.device)
            native_available = status.available
            native_reason = status.reason
            if native_available:
                from ._normalization_native import native_residual_layer_norm

                with torch.cuda.device(x.device):
                    return native_residual_layer_norm(
                        x,
                        branch,
                        weight,
                        bias,
                        float(dropout_p),
                        float(eps),
                        bool(training),
                    )
        if implementation == "gfx1010":
            raise RuntimeError(
                f"gfx1010 native residual LayerNorm is unavailable: "
                f"{native_reason}"
            )
        warnings.warn(
            "gfx1010 native residual LayerNorm was not used: "
            f"{native_reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_residual_layer_norm(
            x,
            branch,
            weight,
            bias,
            dropout_p,
            eps,
            training,
        )

    from ._normalization_triton import triton_residual_layer_norm

    with torch.cuda.device(x.device):
        return triton_residual_layer_norm(
            x,
            branch,
            weight,
            bias,
            float(dropout_p),
            float(eps),
            bool(training),
        )
