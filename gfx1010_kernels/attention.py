from dataclasses import dataclass
from functools import lru_cache
import warnings

import torch
from torch.nn import functional as F


SUPPORTED_HEAD_DIMS = (32, 64, 128)
MAX_SEQUENCE_LENGTH = 4_096
TORCH_SCALED_DOT_PRODUCT_ATTENTION = F.scaled_dot_product_attention


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    device: str | None
    architecture: str | None
    reason: str


@lru_cache(maxsize=None)
def _backend_status(device_index):
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
        architecture = getattr(properties, "gcn_arch_name", None)
    if architecture is None:
        architecture = ""
    architecture = architecture.split(":", 1)[0]
    if architecture != "gfx1010":
        return BackendStatus(
            False,
            properties.name,
            architecture,
            "the optimized kernel targets gfx1010",
        )
    try:
        import triton

        with torch.cuda.device(device_index):
            target = triton.runtime.driver.active.get_current_target()
    except Exception as error:
        return BackendStatus(
            False,
            properties.name,
            architecture,
            f"Triton HIP is unavailable: {error}",
        )
    target_backend = getattr(target, "backend", None)
    target_architecture = str(getattr(target, "arch", "")).split(":", 1)[0]
    if target_backend != "hip" or target_architecture != "gfx1010":
        return BackendStatus(
            False,
            properties.name,
            architecture,
            "Triton is not targeting HIP gfx1010",
        )
    return BackendStatus(
        True,
        properties.name,
        architecture,
        "gfx1010 Triton backend is ready",
    )


def backend_status(device=None):
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
    return _backend_status(device_index)


def _fast_path_reason(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    enable_gqa,
):
    status = backend_status(query.device if query.is_cuda else None)
    if not status.available:
        return status.reason
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        return "query, key and value must be HIP tensors"
    if query.device != key.device or query.device != value.device:
        return "query, key and value must be on the same device"
    if (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") != torch.float16
    ):
        return "the optimized kernel requires float16 CUDA autocast"
    fp32_autocast = _uses_fp16_autocast(query, key, value)
    if query.dtype != torch.float16 and not fp32_autocast:
        return "the optimized kernel requires float16"
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return "query, key and value must have the same dtype"
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return "the optimized kernel requires [batch, heads, sequence, dim]"
    if attn_mask is not None:
        return "custom attention masks use the PyTorch fallback"
    if dropout_p != 0.0:
        return "attention dropout uses the PyTorch fallback"
    if enable_gqa:
        return "grouped-query attention uses the PyTorch fallback"
    if query.shape != key.shape or query.shape != value.shape:
        return "the optimized kernel currently requires self-attention"
    if query.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return (
            "the optimized kernel requires head_dim in "
            f"{SUPPORTED_HEAD_DIMS}"
        )
    if query.shape[0] == 0 or query.shape[1] == 0:
        return "batch and head dimensions must not be empty"
    if query.shape[-2] == 0:
        return "the sequence must not be empty"
    if query.shape[-2] > MAX_SEQUENCE_LENGTH:
        return (
            "the optimized kernel currently supports sequence lengths up to "
            f"{MAX_SEQUENCE_LENGTH}"
        )
    return None


def can_use_gfx1010_kernel(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    enable_gqa=False,
):
    return (
        _fast_path_reason(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            enable_gqa,
        )
        is None
    )


def _uses_fp16_autocast(query, key, value):
    return (
        query.is_cuda
        and query.device == key.device == value.device
        and query.dtype == key.dtype == value.dtype == torch.float32
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
    )


def _auto_training_fallback_reason(query, key, value):
    if (
        not torch.is_grad_enabled()
        or not any(tensor.requires_grad for tensor in (query, key, value))
        or query.shape[-1] == 32
    ):
        return None
    batch_heads = query.shape[0] * query.shape[1]
    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]
    use_hybrid = (
        sequence_length <= 192
        and 8 <= batch_heads <= 512
    ) or (
        head_dim == 64
        and 192 < sequence_length <= 1024
        and batch_heads >= 64
        and batch_heads * sequence_length * sequence_length
        <= 64 * 1024 * 1024
    )
    if use_hybrid:
        return None
    return (
        "the D64/D128 training shape is outside the measured faster "
        "hybrid-backward profiles"
    )


def _torch_attention(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
):
    return TORCH_SCALED_DOT_PRODUCT_ATTENTION(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _cast_fp32_inputs_for_autocast(query, key, value):
    if (
        _uses_fp16_autocast(query, key, value)
        and backend_status(query.device).available
    ):
        return tuple(
            tensor.to(dtype=torch.float16)
            for tensor in (query, key, value)
        )
    return query, key, value


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
    implementation="auto",
    _warning_stacklevel=2,
):
    if implementation not in {"auto", "gfx1010", "torch"}:
        raise ValueError("implementation must be 'auto', 'gfx1010' or 'torch'")
    if implementation == "torch":
        return _torch_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )
    query, key, value = _cast_fp32_inputs_for_autocast(query, key, value)
    reason = _fast_path_reason(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        enable_gqa,
    )
    if implementation == "auto" and reason is None:
        reason = _auto_training_fallback_reason(query, key, value)
    if implementation == "auto" and reason:
        warnings.warn(
            f"gfx1010 optimized attention was not used: {reason}",
            RuntimeWarning,
            stacklevel=_warning_stacklevel,
        )
        return _torch_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )
    if reason:
        raise RuntimeError(f"gfx1010 attention is unavailable: {reason}")

    from ._attention_triton import triton_attention

    if scale is None:
        scale = query.shape[-1] ** -0.5
    with torch.cuda.device(query.device):
        output = triton_attention(
            query,
            key,
            value,
            is_causal,
            float(scale),
        )
    return output
