import pytest
import torch
from torch.nn import functional as F

from gfx1010_kernels import (
    can_use_residual_layer_norm,
    residual_layer_norm,
    residual_layer_norm_status,
)


def torch_reference(
    x,
    branch,
    weight,
    bias,
    dropout_p=0.0,
    eps=1e-5,
    training=True,
):
    if training:
        branch = F.dropout(branch, dropout_p, True)
    updated = x + branch
    normalized = F.layer_norm(
        updated,
        (updated.shape[-1],),
        weight,
        bias,
        eps,
    )
    return updated, normalized


def test_cpu_falls_back_to_torch():
    x = torch.randn(2, 4, 16)
    branch = torch.randn_like(x)
    weight = torch.randn(16)
    bias = torch.randn(16)
    with pytest.warns(RuntimeWarning, match="was not used"):
        actual = residual_layer_norm(x, branch, weight, bias)
    expected = torch_reference(x, branch, weight, bias)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert not can_use_residual_layer_norm(x, branch, weight, bias)


def test_forced_kernel_reports_cpu_input_before_backend_lookup(monkeypatch):
    import gfx1010_kernels.normalization as normalization

    def fail_if_called(*args, **kwargs):
        raise AssertionError("forced CPU input reached backend lookup")

    monkeypatch.setattr(
        normalization,
        "residual_layer_norm_status",
        fail_if_called,
    )
    x = torch.randn(2, 4, 128)
    with pytest.raises(
        RuntimeError,
        match="x must be a HIP tensor",
    ):
        residual_layer_norm(
            x,
            x,
            torch.ones(128),
            torch.zeros(128),
            implementation="gfx1010",
        )


def test_forced_kernel_reports_hidden_size_before_backend_lookup(monkeypatch):
    import gfx1010_kernels.normalization as normalization

    class FakeCudaTensor:
        is_cuda = True
        ndim = 2
        shape = (4, 160)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid hidden size reached backend lookup")

    monkeypatch.setattr(
        normalization,
        "residual_layer_norm_status",
        fail_if_called,
    )
    tensor = FakeCudaTensor()
    with pytest.raises(
        RuntimeError,
        match="native hidden size must be one of",
    ):
        residual_layer_norm(
            tensor,
            tensor,
            tensor,
            tensor,
            implementation="gfx1010",
        )


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("hidden_size", [128, 256, 384, 512, 768, 1024])
def test_gfx1010_forward_matches_pytorch(hidden_size):
    torch.manual_seed(17)
    device = torch.device("cuda")
    x = torch.randn(1, 37, hidden_size, device=device)
    branch = torch.randn(
        1,
        37,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )
    weight = torch.randn(hidden_size, device=device)
    bias = torch.randn(hidden_size, device=device)
    actual = residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        implementation="gfx1010",
    )
    expected = torch_reference(x, branch, weight, bias)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("hidden_size", [128, 256, 384, 512, 768, 1024])
def test_gfx1010_backward_matches_pytorch(hidden_size):
    torch.manual_seed(23)
    device = torch.device("cuda")
    x = torch.randn(1, 41, hidden_size, device=device, requires_grad=True)
    branch = torch.randn(
        1,
        41,
        hidden_size,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(hidden_size, device=device, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, requires_grad=True)
    actual_updated, actual_normalized = residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        implementation="gfx1010",
    )
    grad_updated = torch.randn_like(actual_updated)
    grad_normalized = torch.randn_like(actual_normalized)
    torch.autograd.backward(
        (actual_updated, actual_normalized),
        (grad_updated, grad_normalized),
    )
    actual_gradients = tuple(
        tensor.grad.detach().clone()
        for tensor in (x, branch, weight, bias)
    )

    reference_inputs = (
        x.detach().clone().requires_grad_(True),
        branch.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
    )
    expected_updated, expected_normalized = torch_reference(*reference_inputs)
    torch.autograd.backward(
        (expected_updated, expected_normalized),
        (grad_updated, grad_normalized),
    )
    expected_gradients = tuple(
        tensor.grad for tensor in reference_inputs
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not residual_layer_norm_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("hidden_size", [128, 256, 384, 512, 768, 1024])
def test_gfx1010_dropout_forward_and_backward_are_consistent(hidden_size):
    torch.manual_seed(29)
    device = torch.device("cuda")
    dropout_p = 0.15
    x = torch.randn(1, 43, hidden_size, device=device, requires_grad=True)
    branch = torch.ones(
        1,
        43,
        hidden_size,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(hidden_size, device=device, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, requires_grad=True)
    actual_updated, actual_normalized = residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        dropout_p=dropout_p,
        implementation="gfx1010",
    )
    mask_scale = ((actual_updated - x) / branch.float()).detach()
    expected_scales = torch.tensor(
        [0.0, 1.0 / (1.0 - dropout_p)],
        device=device,
    )
    distances = (
        mask_scale[..., None] - expected_scales
    ).abs().amin(dim=-1)
    assert distances.max().item() < 2e-3

    expected_normalized = F.layer_norm(
        actual_updated,
        (hidden_size,),
        weight,
        bias,
    )
    torch.testing.assert_close(
        actual_normalized,
        expected_normalized,
        rtol=3e-4,
        atol=3e-4,
    )

    grad_updated = torch.randn_like(actual_updated)
    grad_normalized = torch.randn_like(actual_normalized)
    torch.autograd.backward(
        (actual_updated, actual_normalized),
        (grad_updated, grad_normalized),
    )
    actual_gradients = tuple(
        tensor.grad.detach().clone()
        for tensor in (x, branch, weight, bias)
    )

    reference_inputs = (
        x.detach().clone().requires_grad_(True),
        branch.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
    )
    reference_x, reference_branch, reference_weight, reference_bias = (
        reference_inputs
    )
    reference_updated = reference_x + reference_branch * mask_scale
    reference_normalized = F.layer_norm(
        reference_updated,
        (hidden_size,),
        reference_weight,
        reference_bias,
    )
    torch.autograd.backward(
        (reference_updated, reference_normalized),
        (grad_updated, grad_normalized),
    )
    expected_gradients = tuple(tensor.grad for tensor in reference_inputs)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_native_parameter_dispatch_matches_tuned_profiles():
    from gfx1010_kernels._normalization_native import (
        _parameter_groups,
        _parameter_threads,
    )

    assert _parameter_threads(384, 1024) == 64
    assert _parameter_threads(512, 256) == 256
    assert _parameter_threads(256, 512) == 256
    assert _parameter_threads(1024, 4096) == 256
    assert _parameter_groups(384, 128) == 8
    assert _parameter_groups(384, 1024) == 64
    assert _parameter_groups(128, 4096) == 128
    assert _parameter_groups(1024, 4096) == 24
