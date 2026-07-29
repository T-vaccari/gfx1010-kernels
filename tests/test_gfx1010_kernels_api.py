import gfx1010_attention
import gfx1010_kernels


def test_attention_compatibility_api_uses_same_function():
    assert (
        gfx1010_kernels.scaled_dot_product_attention
        is gfx1010_attention.scaled_dot_product_attention
    )


def test_package_version():
    assert gfx1010_kernels.__version__ == "0.3.0"


def test_normalization_status_is_public():
    assert callable(gfx1010_kernels.residual_layer_norm_status)
