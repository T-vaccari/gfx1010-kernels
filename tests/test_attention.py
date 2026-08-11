import pytest
import torch
from torch.nn import functional as F

from gfx1010_kernels import (
    __version__,
    backend_status,
    can_use_gfx1010_kernel,
    install_pytorch_patch,
    is_pytorch_patch_installed,
    scaled_dot_product_attention,
    uninstall_pytorch_patch,
)


def test_package_version():
    assert __version__ == "0.4.0"


def fp32_reference(query, key, value, is_causal, scale=None):
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = query.float() @ key.float().transpose(-2, -1)
    scores *= scale
    if is_causal:
        sequence_length = query.shape[-2]
        mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ value.float()


def test_cpu_falls_back_to_torch():
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with pytest.warns(RuntimeWarning, match="was not used"):
        actual = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
    expected = scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="torch",
    )
    torch.testing.assert_close(actual, expected)
    assert not can_use_gfx1010_kernel(
        query,
        key,
        value,
        is_causal=True,
    )


def test_cpu_backend_status_is_unavailable():
    assert not backend_status("cpu").available


def test_pytorch_patch_installs_and_uninstalls(monkeypatch):
    query = torch.randn(1, 2, 5, 8)
    expected = scaled_dot_product_attention(
        query,
        query,
        query,
        is_causal=True,
        implementation="torch",
    )
    initially_installed = is_pytorch_patch_installed()
    initial_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
    if initially_installed:
        uninstall_pytorch_patch()
    try:
        assert install_pytorch_patch()
        assert is_pytorch_patch_installed()
        assert not torch.backends.mha.get_fastpath_enabled()
        with pytest.warns(RuntimeWarning, match="was not used"):
            actual = F.scaled_dot_product_attention(
                query,
                query,
                query,
                is_causal=True,
            )
        torch.testing.assert_close(actual, expected)
        monkeypatch.setenv("GFX1010_KERNELS_ATTENTION_STRICT", "1")
        with pytest.raises(
            RuntimeError,
            match="gfx1010 attention is unavailable",
        ):
            F.scaled_dot_product_attention(
                query,
                query,
                query,
                is_causal=True,
            )
    finally:
        uninstall_pytorch_patch()
        if initially_installed:
            install_pytorch_patch()
        else:
            assert (
                torch.backends.mha.get_fastpath_enabled()
                == initial_mha_fastpath
            )


def test_patch_composes_with_later_sdpa_replacement():
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    original_attention = F.scaled_dot_product_attention
    initial_mha_fastpath = torch.backends.mha.get_fastpath_enabled()

    def replacement(*args, **kwargs):
        return original_attention(*args, **kwargs)

    try:
        install_pytorch_patch()
        torch.backends.mha.set_fastpath_enabled(True)
        assert not is_pytorch_patch_installed()
        assert install_pytorch_patch()
        assert not torch.backends.mha.get_fastpath_enabled()
        F.scaled_dot_product_attention = replacement
        assert uninstall_pytorch_patch()
        assert F.scaled_dot_product_attention is replacement
        assert (
            torch.backends.mha.get_fastpath_enabled()
            == initial_mha_fastpath
        )
    finally:
        uninstall_pytorch_patch()
        F.scaled_dot_product_attention = original_attention
        torch.backends.mha.set_fastpath_enabled(initial_mha_fastpath)
        if initially_installed:
            install_pytorch_patch()


def test_patch_composes_with_later_mha_replacement():
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    original_forward = torch.nn.MultiheadAttention.forward
    initial_mha_fastpath = torch.backends.mha.get_fastpath_enabled()

    def replacement(self, *args, **kwargs):
        return original_forward(self, *args, **kwargs)

    try:
        install_pytorch_patch()
        torch.nn.MultiheadAttention.forward = replacement
        assert not is_pytorch_patch_installed()
        assert install_pytorch_patch()
        assert is_pytorch_patch_installed()
        assert uninstall_pytorch_patch()
        assert torch.nn.MultiheadAttention.forward is replacement
        assert (
            torch.backends.mha.get_fastpath_enabled()
            == initial_mha_fastpath
        )
    finally:
        uninstall_pytorch_patch()
        torch.nn.MultiheadAttention.forward = original_forward
        torch.backends.mha.set_fastpath_enabled(initial_mha_fastpath)
        if initially_installed:
            install_pytorch_patch()


def test_multihead_attention_default_is_diagnosed(monkeypatch):
    module = torch.nn.MultiheadAttention(
        8,
        1,
        dropout=0.0,
        batch_first=True,
    )
    inputs = torch.randn(2, 5, 8)
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    try:
        install_pytorch_patch()
        with pytest.warns(
            RuntimeWarning,
            match="requires need_weights=False",
        ) as warning:
            output, weights = module(inputs, inputs, inputs)
        assert warning[0].filename.endswith(
            "test_attention.py"
        )
        assert output.shape == inputs.shape
        assert weights is not None
        monkeypatch.setenv("GFX1010_KERNELS_ATTENTION_STRICT", "1")
        with pytest.raises(
            RuntimeError,
            match="requires need_weights=False",
        ):
            module(inputs, inputs, inputs)
    finally:
        uninstall_pytorch_patch()
        if initially_installed:
            install_pytorch_patch()


def test_multihead_attention_fallback_warning_points_to_caller():
    module = torch.nn.MultiheadAttention(
        8,
        1,
        dropout=0.0,
        batch_first=True,
    )
    inputs = torch.randn(2, 5, 8)
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    try:
        install_pytorch_patch()
        with pytest.warns(RuntimeWarning, match="was not used") as warning:
            output, weights = module(
                inputs,
                inputs,
                inputs,
                need_weights=False,
            )
        assert warning[0].filename.endswith(
            "test_attention.py"
        )
        assert output.shape == inputs.shape
        assert weights is None
    finally:
        uninstall_pytorch_patch()
        if initially_installed:
            install_pytorch_patch()


def test_forced_kernel_reports_unsupported_input():
    query = torch.randn(1, 1, 8, 32)
    with pytest.raises(RuntimeError, match="gfx1010 attention is unavailable"):
        scaled_dot_product_attention(
            query,
            query,
            query,
            is_causal=True,
            implementation="gfx1010",
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize(
    "sequence_length",
    [1, 17, 31, 32, 55, 64, 74, 80, 96, 105, 128, 192],
)
def test_forward_and_backward_match_reference(sequence_length):
    torch.manual_seed(1234 + sequence_length)
    shape = (2, 3, sequence_length, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [tensor.detach().clone().requires_grad_() for tensor in source]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=3e-2,
            rtol=3e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_noncontiguous_inputs():
    torch.manual_seed(7)
    source = torch.randn(
        2,
        55,
        3,
        4,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    packed = source.detach().clone().requires_grad_()
    reference_packed = source.detach().clone().requires_grad_()
    query, key, value = (
        tensor.transpose(1, 2) for tensor in packed.unbind(dim=2)
    )
    reference_query, reference_key, reference_value = (
        tensor.transpose(1, 2)
        for tensor in reference_packed.unbind(dim=2)
    )
    assert not query.is_contiguous()
    output = scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        implementation="gfx1010",
    )
    reference = scaled_dot_product_attention(
        reference_query,
        reference_key,
        reference_value,
        is_causal=True,
        implementation="torch",
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        packed.grad,
        reference_packed.grad,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.0, 0.17, -0.17])
def test_matches_fp32_reference(is_causal, scale):
    torch.manual_seed(99)
    shape = (2, 3, 37, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().float().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        scale=scale,
        implementation="gfx1010",
    )
    expected = fp32_reference(
        *reference,
        is_causal=is_causal,
        scale=scale,
    )
    actual.backward(grad)
    expected.backward(grad.float())

    assert (actual.float() - expected).abs().max().item() < 4e-3
    for actual_tensor, expected_tensor in zip(fast, reference):
        error = (
            actual_tensor.grad.float() - expected_tensor.grad
        ).abs().max().item()
        assert error < 8e-3


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_fp32_inputs_use_fast_path_under_fp16_autocast():
    torch.manual_seed(808)
    shape = (2, 3, 37, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float32)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    with torch.autocast("cuda", dtype=torch.float16):
        assert can_use_gfx1010_kernel(
            *fast,
            is_causal=True,
        )
        actual = scaled_dot_product_attention(
            *fast,
            is_causal=True,
            implementation="gfx1010",
        )
        expected = scaled_dot_product_attention(
            *reference,
            is_causal=True,
            implementation="torch",
        )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert actual.dtype == expected.dtype == torch.float16
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=3e-2,
            rtol=3e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_bfloat16_autocast_takes_fallback():
    query = torch.randn(
        1,
        2,
        37,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not can_use_gfx1010_kernel(
            query,
            query,
            query,
            is_causal=True,
        )
        with pytest.warns(
            RuntimeWarning,
            match="requires float16 CUDA autocast",
        ):
            actual = scaled_dot_product_attention(
                query,
                query,
                query,
                is_causal=True,
            )
        expected = scaled_dot_product_attention(
            query,
            query,
            query,
            is_causal=True,
            implementation="torch",
        )
        with pytest.raises(
            RuntimeError,
            match="requires float16 CUDA autocast",
        ):
            scaled_dot_product_attention(
                query,
                query,
                query,
                is_causal=True,
                implementation="gfx1010",
            )
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("sequence_length", [37, 64, 191])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_supported_head_dimensions(
    head_dim,
    sequence_length,
    is_causal,
):
    torch.manual_seed(head_dim + sequence_length)
    shape = (1, 2, sequence_length, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale", [0.0, -0.17])
def test_matmul_backward_nonpositive_scale(
    head_dim,
    is_causal,
    scale,
):
    torch.manual_seed(1700 + head_dim + is_causal)
    shape = (1, 2, 37, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        scale=scale,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        scale=scale,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        assert torch.isfinite(actual_tensor.grad).all()
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_aliased_inputs_and_strided_grad_output():
    torch.manual_seed(123)
    shape = (2, 3, 55, 32)
    source = torch.randn(shape, device="cuda", dtype=torch.float16)
    fast = source.detach().clone().requires_grad_()
    reference = source.detach().clone().requires_grad_()
    actual = scaled_dot_product_attention(
        fast,
        fast,
        fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        reference,
        reference,
        reference,
        is_causal=True,
        implementation="torch",
    )
    grad_storage = torch.randn(
        *shape[:-1],
        shape[-1] * 2,
        device="cuda",
        dtype=torch.float16,
    )
    grad = grad_storage[..., ::2]
    assert not grad.is_contiguous()
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        fast.grad,
        reference.grad,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_zero_stride_grad_output():
    torch.manual_seed(124)
    shape = (2, 3, 55, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    grad = torch.randn(
        *shape[:-1],
        1,
        device="cuda",
        dtype=torch.float16,
    ).expand(shape)
    assert grad.stride(-1) == 0
    actual.backward(grad)
    expected.backward(grad)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_subset_of_inputs_requires_gradient():
    torch.manual_seed(321)
    shape = (2, 3, 55, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    query = source[0].detach().clone().requires_grad_()
    reference_query = source[0].detach().clone().requires_grad_()
    output = scaled_dot_product_attention(
        query,
        source[1],
        source[2],
        is_causal=True,
        implementation="gfx1010",
    )
    reference = scaled_dot_product_attention(
        reference_query,
        source[1],
        source[2],
        is_causal=True,
        implementation="torch",
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(
        query.grad,
        reference_query.grad,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "grad_mask",
    [
        (False, False, True),
        (False, True, False),
        (True, False, True),
    ],
)
def test_matmul_backward_subset_of_inputs(head_dim, grad_mask):
    torch.manual_seed(654 + head_dim)
    shape = (2, 3, 55, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [
        tensor.detach().clone().requires_grad_(requires_grad)
        for tensor, requires_grad in zip(source, grad_mask)
    ]
    reference = [
        tensor.detach().clone().requires_grad_(requires_grad)
        for tensor, requires_grad in zip(source, grad_mask)
    ]
    output = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    for actual_tensor, expected_tensor, requires_grad in zip(
        fast,
        reference,
        grad_mask,
    ):
        if requires_grad:
            torch.testing.assert_close(
                actual_tensor.grad,
                expected_tensor.grad,
                atol=4e-2,
                rtol=4e-2,
            )
        else:
            assert actual_tensor.grad is None


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_auto_falls_back_outside_measured_training_profiles():
    torch.manual_seed(755)
    shape = (1, 2, 193, 128)
    source = [
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    with pytest.warns(RuntimeWarning, match="outside the measured"):
        actual = scaled_dot_product_attention(
            *source,
            is_causal=True,
        )
    expected = scaled_dot_product_attention(
        *source,
        is_causal=True,
        implementation="torch",
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [32, 64])
def test_strided_head_dimension(head_dim):
    torch.manual_seed(987)
    shape = (2, 3, 55, head_dim)
    fast = [
        torch.as_strided(
            tensor,
            shape,
            (*tensor.stride()[:-1], 2),
            storage_offset=1,
        )
        for tensor in [
            torch.randn(
                *shape[:-1],
                shape[-1] * 2 + 1,
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
    ]
    reference = [
        tensor.detach().clone().requires_grad_()
        for tensor in fast
    ]
    assert all(tensor.stride(-1) == 2 for tensor in fast)
    output = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    grad = torch.randn_like(output)
    gradients = torch.autograd.grad(output, fast, grad)
    expected_gradients = torch.autograd.grad(expected, reference, grad)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    for actual, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(
            actual,
            expected_gradient,
            atol=3e-2,
            rtol=3e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("sequence_length", [256, 511, 1024])
def test_long_context_forward_and_backward(sequence_length):
    torch.manual_seed(sequence_length)
    shape = (1, 2, sequence_length, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_high_parallelism_long_context_dispatch(is_causal):
    torch.manual_seed(2048 + is_causal)
    shape = (8, 8, 1024, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    with torch.no_grad():
        actual = scaled_dot_product_attention(
            *source,
            is_causal=is_causal,
            implementation="gfx1010",
        )
        expected = scaled_dot_product_attention(
            *source,
            is_causal=is_causal,
            implementation="torch",
        )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_high_parallelism_long_context_backward(is_causal):
    torch.manual_seed(3072 + is_causal)
    shape = (8, 8, 1024, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_gpt2_d64_batch_six_backward():
    torch.manual_seed(1010)
    shape = (6, 12, 1024, 64)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [tensor.detach().clone().requires_grad_() for tensor in source]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=True,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=True,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_matmul_backward_long_context(head_dim, is_causal):
    torch.manual_seed(4096 + head_dim + is_causal)
    shape = (1, 2, 1024, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=4e-2,
            rtol=4e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_long_context_head_dimension_inference_dispatch(head_dim):
    torch.manual_seed(6144 + head_dim)
    shape = (1, 8, 1024, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    with torch.no_grad():
        actual = scaled_dot_product_attention(
            *source,
            is_causal=True,
            implementation="gfx1010",
        )
        expected = scaled_dot_product_attention(
            *source,
            is_causal=True,
            implementation="torch",
        )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_matmul_backward_maximum_context(head_dim, is_causal):
    torch.manual_seed(8192 + head_dim + is_causal)
    shape = (1, 1, 4096, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_d32_backward_maximum_context(is_causal):
    torch.manual_seed(12288 + is_causal)
    shape = (1, 1, 4096, 32)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    fast = [tensor.detach().clone().requires_grad_() for tensor in source]
    reference = [
        tensor.detach().clone().requires_grad_() for tensor in source
    ]
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)
    actual = scaled_dot_product_attention(
        *fast,
        is_causal=is_causal,
        implementation="gfx1010",
    )
    expected = scaled_dot_product_attention(
        *reference,
        is_causal=is_causal,
        implementation="torch",
    )
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    for actual_tensor, expected_tensor in zip(fast, reference):
        torch.testing.assert_close(
            actual_tensor.grad,
            expected_tensor.grad,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_maximum_context_and_boundary(head_dim):
    torch.manual_seed(4096 + head_dim)
    shape = (1, 1, 4096, head_dim)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    with torch.no_grad():
        actual = scaled_dot_product_attention(
            *source,
            is_causal=True,
            implementation="gfx1010",
        )
        expected = scaled_dot_product_attention(
            *source,
            is_causal=True,
            implementation="torch",
        )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    unsupported = [
        torch.empty(
            1,
            1,
            4097,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    assert not can_use_gfx1010_kernel(
        *unsupported,
        is_causal=True,
    )
    with pytest.raises(RuntimeError, match="up to 4096"):
        scaled_dot_product_attention(
            *unsupported,
            is_causal=True,
            implementation="gfx1010",
        )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_nn_multihead_attention_uses_global_patch(monkeypatch):
    torch.manual_seed(11)
    reference_module = torch.nn.MultiheadAttention(
        256,
        8,
        dropout=0.0,
        batch_first=True,
        device="cuda",
        dtype=torch.float16,
    ).train()
    fast_module = torch.nn.MultiheadAttention(
        256,
        8,
        dropout=0.0,
        batch_first=True,
        device="cuda",
        dtype=torch.float16,
    ).train()
    fast_module.load_state_dict(reference_module.state_dict())
    source = torch.randn(
        2,
        37,
        256,
        device="cuda",
        dtype=torch.float16,
    )
    reference_input = source.detach().clone().requires_grad_()
    fast_input = source.detach().clone().requires_grad_()
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    expected, _ = reference_module(
        reference_input,
        reference_input,
        reference_input,
        need_weights=False,
    )
    install_pytorch_patch()
    monkeypatch.setenv("GFX1010_KERNELS_ATTENTION_STRICT", "1")
    try:
        actual, _ = fast_module(
            fast_input,
            fast_input,
            fast_input,
            need_weights=False,
        )
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
    finally:
        uninstall_pytorch_patch()
        if initially_installed:
            install_pytorch_patch()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fast_input.grad,
        reference_input.grad,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_nn_multihead_attention_eval_avoids_native_fastpath(monkeypatch):
    module = torch.nn.MultiheadAttention(
        256,
        8,
        dropout=0.0,
        batch_first=True,
        device="cuda",
        dtype=torch.float16,
    ).eval()
    inputs = torch.randn(
        2,
        37,
        256,
        device="cuda",
        dtype=torch.float16,
    )
    initially_installed = is_pytorch_patch_installed()
    if initially_installed:
        uninstall_pytorch_patch()
    install_pytorch_patch()
    monkeypatch.setenv("GFX1010_KERNELS_ATTENTION_STRICT", "1")

    def reject_native_fastpath(*args, **kwargs):
        raise AssertionError("native MHA fast path bypassed gfx1010 attention")

    monkeypatch.setattr(
        torch,
        "_native_multi_head_attention",
        reject_native_fastpath,
    )
    try:
        with torch.inference_mode():
            output, weights = module(inputs, inputs, inputs, need_weights=False)
    finally:
        uninstall_pytorch_patch()
        if initially_installed:
            install_pytorch_patch()
    assert output.shape == inputs.shape
    assert weights is None


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_dropout_and_mask_take_fallback():
    query = torch.randn(
        1,
        2,
        32,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    assert not can_use_gfx1010_kernel(
        query,
        query,
        query,
        dropout_p=0.1,
        is_causal=True,
    )
    assert not can_use_gfx1010_kernel(
        query,
        query,
        query,
        attn_mask=torch.ones(32, 32, device="cuda", dtype=torch.bool),
    )
    mask = torch.ones(32, 32, device="cuda", dtype=torch.bool).tril()
    with pytest.warns(RuntimeWarning, match="custom attention masks"):
        actual = scaled_dot_product_attention(
            query,
            query,
            query,
            attn_mask=mask,
        )
    expected = scaled_dot_product_attention(
        query,
        query,
        query,
        attn_mask=mask,
        implementation="torch",
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_other_unsupported_operations_take_fallback():
    query = torch.randn(
        1,
        4,
        32,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        1,
        2,
        24,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    key_gqa = torch.randn(
        1,
        2,
        32,
        32,
        device="cuda",
        dtype=torch.float16,
    )
    value_gqa = torch.randn_like(key_gqa)
    self_attention = query[:, :2]
    cases = [
        (
            (self_attention, self_attention, self_attention),
            {"dropout_p": 0.1},
            "attention dropout",
        ),
        (
            (query[:, :2], key, value),
            {},
            "requires self-attention",
        ),
        (
            (query, key_gqa, value_gqa),
            {"enable_gqa": True},
            "grouped-query attention",
        ),
    ]
    for tensors, kwargs, reason in cases:
        torch.cuda.manual_seed(909)
        with pytest.warns(RuntimeWarning, match=reason):
            actual = scaled_dot_product_attention(
                *tensors,
                **kwargs,
            )
        torch.cuda.manual_seed(909)
        expected = scaled_dot_product_attention(
            *tensors,
            implementation="torch",
            **kwargs,
        )
        torch.testing.assert_close(actual, expected)
        with pytest.raises(RuntimeError, match=reason):
            scaled_dot_product_attention(
                *tensors,
                implementation="gfx1010",
                **kwargs,
            )


@pytest.mark.skipif(
    not backend_status().available,
    reason="requires the gfx1010 ROCm server",
)
def test_bfloat16_inputs_take_fallback():
    query = torch.randn(
        1,
        2,
        32,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.warns(RuntimeWarning, match="requires float16"):
        actual = scaled_dot_product_attention(query, query, query)
    expected = scaled_dot_product_attention(
        query,
        query,
        query,
        implementation="torch",
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(RuntimeError, match="requires float16"):
        scaled_dot_product_attention(
            query,
            query,
            query,
            implementation="gfx1010",
        )
