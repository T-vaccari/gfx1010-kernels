import os
from contextvars import ContextVar
import warnings

import torch
from torch.nn import functional as F

from .functional import (
    TORCH_SCALED_DOT_PRODUCT_ATTENTION,
    scaled_dot_product_attention,
)


_previous_attention = None
_previous_mha_forward = None
_previous_mha_fastpath = None
_inside_mha = ContextVar("gfx1010_attention_inside_mha", default=False)
_torch_mha_forward = torch.nn.MultiheadAttention.forward
STRICT_ENVIRONMENT_VARIABLE = "GFX1010_ATTENTION_STRICT"


def _strict_enabled():
    return os.environ.get(
        STRICT_ENVIRONMENT_VARIABLE,
        "",
    ).lower() in {"1", "true", "yes", "on"}


def _patched_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    return scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        implementation="gfx1010" if _strict_enabled() else "auto",
        _warning_stacklevel=8 if _inside_mha.get() else 3,
    )


def _patched_mha_forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    if need_weights:
        reason = (
            "nn.MultiheadAttention requires need_weights=False to use "
            "scaled_dot_product_attention"
        )
        if _strict_enabled():
            raise RuntimeError(f"gfx1010 attention is unavailable: {reason}")
        warnings.warn(
            f"gfx1010 optimized attention was not used: {reason}",
            RuntimeWarning,
            stacklevel=4,
        )
    token = _inside_mha.set(True)
    try:
        return (_previous_mha_forward or _torch_mha_forward)(
            self,
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
    finally:
        _inside_mha.reset(token)


def install_pytorch_patch():
    global _previous_attention, _previous_mha_forward
    global _previous_mha_fastpath
    changed = False
    if F.scaled_dot_product_attention is not _patched_attention:
        _previous_attention = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _patched_attention
        changed = True
    if torch.nn.MultiheadAttention.forward is not _patched_mha_forward:
        _previous_mha_forward = torch.nn.MultiheadAttention.forward
        torch.nn.MultiheadAttention.forward = _patched_mha_forward
        changed = True
    if hasattr(torch.backends, "mha"):
        if _previous_mha_fastpath is None:
            _previous_mha_fastpath = (
                torch.backends.mha.get_fastpath_enabled()
            )
        if torch.backends.mha.get_fastpath_enabled():
            torch.backends.mha.set_fastpath_enabled(False)
            changed = True
    return changed


def uninstall_pytorch_patch():
    global _previous_attention, _previous_mha_forward
    global _previous_mha_fastpath
    attention_installed = (
        F.scaled_dot_product_attention is _patched_attention
    )
    mha_forward_installed = (
        torch.nn.MultiheadAttention.forward is _patched_mha_forward
    )
    mha_state_tracked = _previous_mha_fastpath is not None
    if attention_installed:
        F.scaled_dot_product_attention = (
            _previous_attention or TORCH_SCALED_DOT_PRODUCT_ATTENTION
        )
    if mha_forward_installed:
        torch.nn.MultiheadAttention.forward = (
            _previous_mha_forward or _torch_mha_forward
        )
    _previous_attention = None
    _previous_mha_forward = None
    if mha_state_tracked:
        torch.backends.mha.set_fastpath_enabled(_previous_mha_fastpath)
        _previous_mha_fastpath = None
    return attention_installed or mha_forward_installed or mha_state_tracked


def is_pytorch_patch_installed():
    if F.scaled_dot_product_attention is not _patched_attention:
        return False
    if torch.nn.MultiheadAttention.forward is not _patched_mha_forward:
        return False
    return (
        not hasattr(torch.backends, "mha")
        or not torch.backends.mha.get_fastpath_enabled()
    )
