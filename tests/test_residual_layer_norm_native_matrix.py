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


def _inputs(rows, hidden_size, branch_dtype):
    return (
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        ),
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=branch_dtype,
        ),
        torch.randn(hidden_size, device="cuda", dtype=torch.float32),
        torch.randn(hidden_size, device="cuda", dtype=torch.float32),
    )


def _leaves(inputs):
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def _reference(x, branch, weight, bias, eps=1e-5):
    updated = x + branch
    normalized = F.layer_norm(
        updated,
        (updated.shape[-1],),
        weight,
        bias,
        eps,
    )
    return updated, normalized


def _assert_forward_backward_close(data, upstream_scale=1.0):
    actual_inputs = _leaves(data)
    expected_inputs = _leaves(data)
    actual = residual_layer_norm(
        *actual_inputs,
        implementation="gfx1010",
    )
    expected = _reference(*expected_inputs)
    grad_updated = torch.randn_like(actual[0]) * upstream_scale
    grad_normalized = torch.randn_like(actual[1]) * upstream_scale
    actual_gradients = torch.autograd.grad(
        actual,
        actual_inputs,
        (grad_updated, grad_normalized),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        expected_inputs,
        (grad_updated, grad_normalized),
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=6e-4, atol=6e-4)
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
def test_native_float32_branch_forward_and_backward_matches_pytorch():
    torch.manual_seed(211)
    _assert_forward_backward_close(
        _inputs(rows=37, hidden_size=384, branch_dtype=torch.float32),
    )


@REQUIRES_NATIVE
def test_native_dropout_hidden_1024_forward_and_backward_matches_masked_reference():
    torch.manual_seed(223)
    hidden_size = 1024
    dropout_p = 0.2
    x = torch.randn(
        9,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )
    branch = torch.ones(
        9,
        hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    actual_inputs = _leaves((x, branch, weight, bias))
    expected_inputs = _leaves((x, branch, weight, bias))

    actual = residual_layer_norm(
        *actual_inputs,
        dropout_p=dropout_p,
        training=True,
        implementation="gfx1010",
    )
    mask_scale = (
        (actual[0].detach() - actual_inputs[0].detach())
        / actual_inputs[1].detach().float()
    )
    valid_scales = torch.tensor(
        [0.0, 1.0 / (1.0 - dropout_p)],
        device="cuda",
    )
    distances = (mask_scale[..., None] - valid_scales).abs().amin(dim=-1)
    assert distances.max().item() < 2e-3

    expected_updated = (
        expected_inputs[0] + expected_inputs[1].float() * mask_scale
    )
    expected_normalized = F.layer_norm(
        expected_updated,
        (hidden_size,),
        expected_inputs[2],
        expected_inputs[3],
    )
    expected = (expected_updated, expected_normalized)
    grad_updated = torch.randn_like(actual[0])
    grad_normalized = torch.randn_like(actual[1])
    actual_gradients = torch.autograd.grad(
        actual,
        actual_inputs,
        (grad_updated, grad_normalized),
    )
    expected_gradients = torch.autograd.grad(
        expected,
        expected_inputs,
        (grad_updated, grad_normalized),
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=6e-4, atol=6e-4)
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
@pytest.mark.parametrize(
    "rows",
    [1, 15, 16, 17, 511, 512, 513, 2047, 2048, 2049],
)
def test_native_row_count_boundaries_match_pytorch(rows):
    torch.manual_seed(227 + rows)
    _assert_forward_backward_close(
        _inputs(rows=rows, hidden_size=128, branch_dtype=torch.float16),
        upstream_scale=rows ** -0.5,
    )


@REQUIRES_NATIVE
def test_updated_only_backward_leaves_layer_norm_parameters_unused():
    torch.manual_seed(229)
    actual_inputs = _leaves(
        _inputs(rows=13, hidden_size=384, branch_dtype=torch.float16),
    )
    expected_inputs = _leaves(tuple(value.detach() for value in actual_inputs))
    actual_updated = residual_layer_norm(
        *actual_inputs,
        implementation="gfx1010",
    )[0]
    expected_updated = _reference(*expected_inputs)[0]
    upstream = torch.randn_like(actual_updated)

    actual_gradients = torch.autograd.grad(
        actual_updated,
        actual_inputs,
        upstream,
        allow_unused=True,
        materialize_grads=False,
    )
    expected_gradients = torch.autograd.grad(
        expected_updated,
        expected_inputs,
        upstream,
        allow_unused=True,
        materialize_grads=False,
    )

    torch.testing.assert_close(
        actual_gradients[0],
        expected_gradients[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_gradients[1],
        expected_gradients[1],
        rtol=0,
        atol=0,
    )
    assert actual_gradients[2:] == expected_gradients[2:] == (None, None)


@REQUIRES_NATIVE
def test_raw_backward_rejects_cpu_inputs():
    from gfx1010_kernels import _C

    updated = torch.zeros(2, 128)
    stats = torch.zeros(2)
    with pytest.raises(RuntimeError, match="updated must be a HIP tensor"):
        _C.residual_layer_norm_backward(
            updated,
            updated,
            updated,
            torch.ones(128),
            stats,
            stats,
            torch.empty(0, dtype=torch.int32),
            0.0,
            True,
            True,
            128,
            16,
            64,
            1,
            True,
        )


@REQUIRES_NATIVE
def test_raw_backward_rejects_empty_rows():
    from gfx1010_kernels import _C

    updated = torch.empty(0, 128, device="cuda")
    stats = torch.empty(0, device="cuda")
    with pytest.raises(RuntimeError, match="empty tensors are unsupported"):
        _C.residual_layer_norm_backward(
            updated,
            updated,
            updated,
            torch.ones(128, device="cuda"),
            stats,
            stats,
            torch.empty(0, device="cuda", dtype=torch.int32),
            0.0,
            True,
            True,
            128,
            16,
            64,
            1,
            True,
        )


@REQUIRES_NATIVE
def test_raw_parameter_gradient_reduction_modes_match_reference():
    from gfx1010_kernels import _C

    torch.manual_seed(233)
    rows = 257
    hidden_size = 384
    x, branch, weight, bias = _inputs(
        rows,
        hidden_size,
        torch.float16,
    )
    outputs = _C.residual_layer_norm_forward(
        x,
        branch,
        weight,
        bias,
        0.0,
        1e-5,
        True,
        128,
    )
    grad_updated = torch.randn_like(outputs[0])
    grad_normalized = torch.randn_like(outputs[1])

    def backward(parameter_groups):
        return _C.residual_layer_norm_backward(
            grad_updated,
            grad_normalized,
            outputs[0],
            weight,
            outputs[2],
            outputs[3],
            outputs[4],
            0.0,
            True,
            True,
            128,
            32,
            64,
            parameter_groups,
            True,
        )

    with pytest.raises(
        RuntimeError,
        match=r"parameter_groups must be in \[1, row_count\]",
    ):
        backward(0)

    deterministic_was_enabled = (
        torch.are_deterministic_algorithms_enabled()
    )
    warn_only_was_enabled = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        torch.use_deterministic_algorithms(False)
        atomic_one = backward(1)
        atomic_twelve = backward(12)
        torch.use_deterministic_algorithms(True)
        deterministic_one = backward(1)
        deterministic_twelve = backward(12)
    finally:
        torch.use_deterministic_algorithms(
            deterministic_was_enabled,
            warn_only=warn_only_was_enabled,
        )

    xhat = (
        outputs[0] - outputs[2].unsqueeze(-1)
    ) * outputs[3].unsqueeze(-1)
    expected_weight = (grad_normalized * xhat).sum(dim=0)
    expected_bias = grad_normalized.sum(dim=0)
    for gradients in (
        atomic_one,
        atomic_twelve,
        deterministic_one,
        deterministic_twelve,
    ):
        torch.testing.assert_close(
            gradients[2],
            expected_weight,
            rtol=5e-4,
            atol=5e-4,
        )
        torch.testing.assert_close(
            gradients[3],
            expected_bias,
            rtol=5e-4,
            atol=5e-4,
        )
    torch.testing.assert_close(
        deterministic_one[2],
        deterministic_twelve[2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        deterministic_one[3],
        deterministic_twelve[3],
        rtol=0,
        atol=0,
    )
