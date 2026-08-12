import gfx1010_kernels
import pytest
import torch
from torch.nn import functional as F


def test_attention_api_is_public():
    assert callable(gfx1010_kernels.scaled_dot_product_attention)


def test_autocast_linear_api_is_public():
    assert callable(gfx1010_kernels.autocast_linear)
    assert issubclass(gfx1010_kernels.AutocastLinear, torch.nn.Linear)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
def test_autocast_linear_accepts_half_activation():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(16, 8, device="cuda", requires_grad=True)
    bias = torch.randn(16, device="cuda", requires_grad=True)

    actual = gfx1010_kernels.autocast_linear(x, weight, bias)
    expected = F.linear(x, weight.half(), bias.half())
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()

    assert weight.grad is not None
    assert weight.grad.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
def test_autocast_linear_module_compiles():
    module = torch.compile(gfx1010_kernels.AutocastLinear(8, 16).cuda())
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    output = module(x)
    output.float().square().mean().backward()

    assert output.dtype == torch.float16
    assert module.weight.grad is not None
    assert module.weight.grad.dtype == torch.float32


def test_package_version():
    assert gfx1010_kernels.__version__ == "0.4.0"


def test_normalization_status_is_public():
    assert callable(gfx1010_kernels.residual_layer_norm_status)
