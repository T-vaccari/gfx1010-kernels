import pytest
import torch
from torch.nn import functional as F

from gfx1010_kernels import (
    residual_layer_norm,
    residual_layer_norm_status,
)


BACKENDS = [
    pytest.param(torch.device("cpu"), "torch", id="cpu-torch"),
    pytest.param(
        torch.device("cuda"),
        "gfx1010",
        id="gpu-gfx1010",
        marks=pytest.mark.skipif(
            not residual_layer_norm_status().available,
            reason="requires the gfx1010 ROCm server",
        ),
    ),
]


def _reference(x, branch, weight, bias, dropout_p=0.0, eps=1e-5, training=True):
    branch = F.dropout(branch, dropout_p, training)
    updated = x + branch
    normalized = F.layer_norm(
        updated,
        (updated.shape[-1],),
        weight,
        bias,
        eps,
    )
    return updated, normalized


def _random_inputs(device, implementation, hidden_size, rows=7):
    branch_dtype = torch.float16 if implementation == "gfx1010" else torch.float32
    return (
        torch.randn(rows, hidden_size, device=device, dtype=torch.float32),
        torch.randn(rows, hidden_size, device=device, dtype=branch_dtype),
        torch.randn(hidden_size, device=device, dtype=torch.float32),
        torch.randn(hidden_size, device=device, dtype=torch.float32),
    )


def _leaves(inputs):
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def _rng_state(device):
    cpu_state = torch.random.get_rng_state()
    device_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    return cpu_state, device_state


def _restore_rng_state(device, state):
    cpu_state, device_state = state
    torch.random.set_rng_state(cpu_state)
    if device_state is not None:
        torch.cuda.set_rng_state(device_state, device)


def _assert_optional_gradient_close(actual, expected, like):
    if expected is None:
        if actual is not None:
            torch.testing.assert_close(actual, torch.zeros_like(like))
        return
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("device,implementation", BACKENDS)
def test_eval_bypasses_dropout_without_consuming_rng(device, implementation):
    torch.manual_seed(101)
    inputs = _random_inputs(device, implementation, hidden_size=384)
    state = _rng_state(device)

    actual = residual_layer_norm(
        *inputs,
        dropout_p=0.8,
        training=False,
        implementation=implementation,
    )
    expected = _reference(*inputs, dropout_p=0.8, training=False)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-4, atol=5e-4)
    current_state = _rng_state(device)
    assert torch.equal(current_state[0], state[0])
    if state[1] is not None:
        assert torch.equal(current_state[1], state[1])


@pytest.mark.parametrize("device,implementation", BACKENDS)
@pytest.mark.parametrize("used_output", [0, 1], ids=["updated-only", "normalized-only"])
def test_each_output_can_drive_backward_alone(device, implementation, used_output):
    torch.manual_seed(103)
    data = _random_inputs(device, implementation, hidden_size=384)
    actual_inputs = _leaves(data)
    expected_inputs = _leaves(data)

    actual_outputs = residual_layer_norm(
        *actual_inputs,
        implementation=implementation,
    )
    expected_outputs = _reference(*expected_inputs)
    upstream = torch.randn_like(actual_outputs[used_output])

    actual_gradients = torch.autograd.grad(
        actual_outputs[used_output],
        actual_inputs,
        upstream,
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_outputs[used_output],
        expected_inputs,
        upstream,
        allow_unused=True,
    )

    for actual, expected, like in zip(
        actual_gradients,
        expected_gradients,
        actual_inputs,
    ):
        _assert_optional_gradient_close(actual, expected, like)


@pytest.mark.parametrize("device,implementation", BACKENDS)
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_nearly_constant_rows_respect_eps(device, implementation, eps):
    hidden_size = 384
    columns = torch.linspace(
        -1.0,
        1.0,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    row_centers = torch.tensor(
        [-3.0, 0.25, 4.0],
        device=device,
        dtype=torch.float32,
    )[:, None]
    x = row_centers + columns[None, :] * 2e-4
    branch = (-columns[None, :] * 5e-5).expand_as(x).contiguous()
    if implementation == "gfx1010":
        branch = branch.to(torch.float16)
    weight = torch.linspace(
        0.75,
        1.25,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    bias = torch.linspace(
        -0.1,
        0.1,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    actual_inputs = _leaves((x, branch, weight, bias))
    expected_inputs = _leaves((x, branch, weight, bias))

    actual = residual_layer_norm(
        *actual_inputs,
        eps=eps,
        implementation=implementation,
    )
    expected = _reference(*expected_inputs, eps=eps)
    upstream = torch.sin(columns)[None, :].expand_as(actual[1]).contiguous()
    actual_gradients = torch.autograd.grad(actual[1], actual_inputs, upstream)
    expected_gradients = torch.autograd.grad(expected[1], expected_inputs, upstream)

    assert all(torch.isfinite(value).all() for value in (*actual, *actual_gradients))
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=8e-3, atol=8e-3)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize("device,implementation", BACKENDS)
@pytest.mark.parametrize("hidden_size", [384, 768])
def test_non_power_of_two_hidden_sizes_forward_and_backward(
    device,
    implementation,
    hidden_size,
):
    torch.manual_seed(107)
    data = _random_inputs(device, implementation, hidden_size, rows=5)
    actual_inputs = _leaves(data)
    expected_inputs = _leaves(data)

    actual = residual_layer_norm(*actual_inputs, implementation=implementation)
    expected = _reference(*expected_inputs)
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
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-4, atol=5e-4)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3e-3,
            atol=3e-3,
        )


@pytest.mark.parametrize("device,implementation", BACKENDS)
def test_dropout_replays_after_rng_state_restore(device, implementation):
    torch.manual_seed(109)
    data = _random_inputs(device, implementation, hidden_size=384, rows=11)
    first_inputs = _leaves(data)
    second_inputs = _leaves(data)
    grad_updated = torch.linspace(
        -0.5,
        0.5,
        first_inputs[0].numel(),
        device=device,
        dtype=torch.float32,
    ).reshape_as(first_inputs[0])
    grad_normalized = torch.cos(grad_updated)
    state = _rng_state(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    if implementation == "gfx1010":
        torch.use_deterministic_algorithms(True)
    try:
        first_outputs = residual_layer_norm(
            *first_inputs,
            dropout_p=0.25,
            training=True,
            implementation=implementation,
        )
        _restore_rng_state(device, state)
        second_outputs = residual_layer_norm(
            *second_inputs,
            dropout_p=0.25,
            training=True,
            implementation=implementation,
        )
        first_gradients = torch.autograd.grad(
            first_outputs,
            first_inputs,
            (grad_updated, grad_normalized),
        )
        second_gradients = torch.autograd.grad(
            second_outputs,
            second_inputs,
            (grad_updated, grad_normalized),
        )

        for first, second in zip(
            (*first_outputs, *first_gradients),
            (*second_outputs, *second_gradients),
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        if implementation == "gfx1010":
            torch.use_deterministic_algorithms(
                deterministic,
                warn_only=deterministic_warn_only,
            )


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_native_backend_does_not_depend_on_triton_status(monkeypatch):
    import gfx1010_kernels.normalization as normalization

    def fail_if_called(*args, **kwargs):
        raise AssertionError("native dispatch queried Triton")

    monkeypatch.setattr(normalization, "_triton_is_usable", fail_if_called)
    assert residual_layer_norm_status().available
    inputs = _random_inputs(
        torch.device("cuda"),
        "gfx1010",
        hidden_size=384,
    )
    residual_layer_norm(*inputs, implementation="gfx1010")


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_forced_native_hot_path_skips_python_validation_and_device_guard(
    monkeypatch,
):
    import gfx1010_kernels.normalization as normalization

    inputs = _random_inputs(
        torch.device("cuda"),
        "gfx1010",
        hidden_size=384,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("forced native hot path used Python dispatch work")

    monkeypatch.setattr(normalization, "_input_reason", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device", fail_if_called)
    residual_layer_norm(*inputs, implementation="gfx1010")


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_consecutive_dropout_calls_do_not_overlap_philox_subsequences():
    device = torch.device("cuda")
    hidden_size = 768
    rows = 3
    x = torch.zeros(rows, hidden_size, device=device)
    branch = torch.ones(
        rows,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )
    weight = torch.ones(hidden_size, device=device)
    bias = torch.zeros(hidden_size, device=device)
    torch.manual_seed(127)

    first = residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        dropout_p=0.5,
        implementation="gfx1010",
    )[0]
    second = residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        dropout_p=0.5,
        implementation="gfx1010",
    )[0]

    assert not torch.equal(first[:, 512:768], second[:, :256])


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_raw_extension_rejects_invalid_inputs():
    from gfx1010_kernels import _C

    x = torch.zeros(2, 128)
    with pytest.raises(RuntimeError, match="x must be a HIP tensor"):
        _C.residual_layer_norm_forward(
            x,
            x,
            torch.ones(128),
            torch.zeros(128),
            0.0,
            1e-5,
            True,
            128,
        )
