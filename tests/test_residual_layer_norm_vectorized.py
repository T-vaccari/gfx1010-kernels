import pytest
import torch
from torch.nn import functional as F

from gfx1010_kernels import (
    residual_layer_norm,
    residual_layer_norm_status,
)


REQUIRES_NATIVE = pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 native residual LayerNorm backend",
)


@REQUIRES_NATIVE
@pytest.mark.parametrize("hidden_size", [256, 384, 512, 768, 1024])
@pytest.mark.parametrize("dropout_p", [0.0, 0.15])
@pytest.mark.parametrize("branch_dtype", [torch.float16, torch.float32])
def test_vectorized_backward_matches_pytorch(
    hidden_size,
    dropout_p,
    branch_dtype,
):
    torch.manual_seed(401 + hidden_size)
    rows = 19
    inputs = (
        torch.randn(rows, hidden_size, device="cuda"),
        torch.ones(
            rows,
            hidden_size,
            device="cuda",
            dtype=branch_dtype,
        ),
        torch.randn(hidden_size, device="cuda"),
        torch.randn(hidden_size, device="cuda"),
    )
    actual_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in inputs
    )
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in inputs
    )
    actual = residual_layer_norm(
        *actual_inputs,
        dropout_p=dropout_p,
        training=True,
        implementation="gfx1010",
    )
    if dropout_p:
        mask_scale = (
            actual[0].detach() - actual_inputs[0].detach()
        )
    else:
        mask_scale = torch.ones_like(actual_inputs[0])
    expected_updated = (
        expected_inputs[0]
        + expected_inputs[1].float() * mask_scale
    )
    expected_normalized = F.layer_norm(
        expected_updated,
        (hidden_size,),
        expected_inputs[2],
        expected_inputs[3],
    )
    gradients = (
        torch.randn_like(actual[0]),
        torch.randn_like(actual[1]),
    )
    actual_gradients = torch.autograd.grad(
        actual,
        actual_inputs,
        gradients,
    )
    expected_gradients = torch.autograd.grad(
        (expected_updated, expected_normalized),
        expected_inputs,
        gradients,
    )

    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5e-3,
            atol=5e-3,
        )


@REQUIRES_NATIVE
def test_misaligned_contiguous_gradient_uses_scalar_fallback():
    from gfx1010_kernels import _C

    torch.manual_seed(409)
    rows = 23
    hidden_size = 384
    x = torch.randn(rows, hidden_size, device="cuda")
    branch = torch.randn(
        rows,
        hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.randn(hidden_size, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")
    outputs = _C.residual_layer_norm_forward(
        x,
        branch,
        weight,
        bias,
        0.15,
        1e-5,
        True,
        128,
    )
    gradient_storage = torch.randn(
        rows * hidden_size + 1,
        device="cuda",
    )
    misaligned_gradient = gradient_storage[1:].view(rows, hidden_size)
    aligned_gradient = misaligned_gradient.clone()
    grad_normalized = torch.randn_like(x)
    assert misaligned_gradient.is_contiguous()
    assert misaligned_gradient.data_ptr() % 16 != 0

    def backward(grad_updated):
        return _C.residual_layer_norm_backward(
            grad_updated,
            grad_normalized,
            outputs[0],
            weight,
            outputs[2],
            outputs[3],
            outputs[4],
            0.15,
            True,
            True,
            128,
            16,
            64,
            1,
            False,
        )

    vectorized = backward(aligned_gradient)
    scalar = backward(misaligned_gradient)
    torch.testing.assert_close(
        vectorized[0],
        scalar[0],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        vectorized[1],
        scalar[1],
        rtol=2e-3,
        atol=2e-3,
    )


@REQUIRES_NATIVE
def test_vectorized_backward_deterministic_replay_is_bitwise_exact():
    from gfx1010_kernels import _C

    torch.manual_seed(419)
    rows = 31
    hidden_size = 768
    x = torch.randn(rows, hidden_size, device="cuda")
    branch = torch.randn(
        rows,
        hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.randn(hidden_size, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")
    outputs = _C.residual_layer_norm_forward(
        x,
        branch,
        weight,
        bias,
        0.15,
        1e-5,
        True,
        128,
    )
    grad_updated = torch.randn_like(x)
    grad_normalized = torch.randn_like(x)

    def backward():
        return _C.residual_layer_norm_backward(
            grad_updated,
            grad_normalized,
            outputs[0],
            weight,
            outputs[2],
            outputs[3],
            outputs[4],
            0.15,
            True,
            True,
            128,
            16,
            64,
            1,
            False,
        )

    deterministic_was_enabled = (
        torch.are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        torch.use_deterministic_algorithms(True)
        first = backward()
        second = backward()
    finally:
        torch.use_deterministic_algorithms(
            deterministic_was_enabled,
            warn_only=deterministic_warn_only,
        )

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
