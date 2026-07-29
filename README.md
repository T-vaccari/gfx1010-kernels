# gfx1010-kernels

Reusable PyTorch training kernels for AMD `gfx1010`, developed and measured on
a Radeon RX 5600 XT with a machine-specific PyTorch/ROCm/Triton stack. The
library exposes standalone operators and does not depend on a model
implementation.

The canonical import is `gfx1010_kernels`. The historical
`gfx1010_attention` import remains supported and exposes the same attention
API, patching utilities and precompile command.

The current library contains:

- a native HIP fused residual-add, dropout and LayerNorm operation with
  first-order autograd;
- fused scaled dot-product attention implemented with Triton and ROCm
  operations;
- explicit compatibility checks, strict fast-path selection and transparent
  PyTorch fallbacks.

## Residual + LayerNorm

The implementation, backward equations, tuned runtime dispatch and validation
procedure are documented in
[`docs/residual_layer_norm.md`](docs/residual_layer_norm.md).

The public operation is equivalent to:

```python
dropped = torch.nn.functional.dropout(branch, dropout_p, training)
updated = x + dropped
normalized = torch.nn.functional.layer_norm(
    updated,
    (updated.shape[-1],),
    weight,
    bias,
    eps,
)
```

It returns both tensors:

```python
from gfx1010_kernels import residual_layer_norm

updated, normalized = residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p=0.15,
    eps=1e-5,
    training=True,
    implementation="gfx1010",
)
```

`updated` is the FP32 residual stream that continues through the skip
connections. `normalized` is the LayerNorm result consumed by the following
attention or MLP sublayer. Returning both avoids recomputing or materializing
the residual addition outside the fused operation.

### Native fast-path contract

| Property | Supported native HIP path |
|---|---|
| GPU architecture | AMD `gfx1010` |
| Execution | PyTorch eager mode on ROCm |
| `x` | Contiguous FP32 tensor |
| `branch` | Same shape as `x`, contiguous FP16 or FP32 |
| `weight`, `bias` | Contiguous FP32 vectors |
| Hidden size | 128, 256, 384, 512, 768, 1024 |
| Leading dimensions | Any non-empty shape accepted by the memory budget |
| Dropout | `0 <= dropout_p < 1`, fused during training |
| Autograd | First-order forward and backward |

Only the hidden dimension is compile-time specialized. Batch size, sequence
length and all other leading dimensions are flattened into a runtime row
count, so changing context length does not require compiling another kernel.
The implementation includes tail-row handling; context length is not restricted
to a particular alignment.

The native forward fuses dropout, residual addition, mean/variance reduction
and affine LayerNorm. It saves the updated residual, row mean, reciprocal
standard deviation and a wave32 bit-packed dropout mask required by backward.
Backward computes input gradients and the reductions for `weight` and `bias`.
Its default fast path uses grouped FP32 atomics for the two parameter
gradients. Those reductions match PyTorch within the documented numerical
tolerances, but their final few bits are not guaranteed to be reproducible.
`torch.use_deterministic_algorithms(True)` selects a separate fixed-order
partial-reduction path for bitwise replay.

### Selection and diagnostics

```python
from gfx1010_kernels import (
    can_use_residual_layer_norm,
    residual_layer_norm_status,
)

print(residual_layer_norm_status())
print(can_use_residual_layer_norm(x, branch, weight, bias, dropout_p=0.15))
```

`residual_layer_norm_status()` checks the native HIP extension directly; it
does not require the Triton attention backend to be available.

`implementation` controls dispatch:

- `gfx1010` requires the native fast path and raises `RuntimeError` with the
  rejected condition when it cannot be used;
- `auto` selects the native path when compatible, otherwise emits
  `RuntimeWarning` and calls the PyTorch reference;
- `torch` always calls the PyTorch decomposition;
- `triton` explicitly selects the experimental Triton normalization path and
  raises if that backend is unavailable.

Use `implementation="gfx1010"` in validation and performance runs when a
fallback must be treated as an error.

## Attention

Attention sequence length is a runtime value, not a fixed kernel constant. The
dispatcher selects a tuned tile configuration while the same compiled kernel
handles aligned and tail sequence lengths. `head_dim` remains compile-time
specialized because that lets Triton generate substantially better code.

The D=32 path streams attention and its first-order backward without creating
an N×N attention matrix. D=64/128 use the same fused streaming forward plus a
hybrid backward: two Triton row-reduction kernels handle masked softmax and its
derivative, while ROCm matmuls compute the matrix products. The hybrid route is
faster on the measured high-parallelism profiles but uses O(batch·heads·N²)
temporary memory; `auto` falls back to PyTorch outside the measured faster
training regions.

### Attention fast-path support

| Property | Supported fast path |
|---|---|
| GPU architecture | AMD `gfx1010` |
| Backend | PyTorch ROCm with a working Triton HIP backend |
| Dtype | `torch.float16`, FP32 softmax and gradient accumulation |
| Tensor shape | `[batch, heads, sequence, head_dim]` |
| Tensor layout | Contiguous or strided Q/K/V |
| Operation | Self-attention with identical Q/K/V shapes |
| `head_dim` | 32, 64, 128 |
| Sequence length | 1–4096 |
| Causal mode | Causal and non-causal |
| Scale | Default or explicit `scale=` |
| Autograd | First-order forward and backward |

Attention dropout, custom masks, GQA, cross-attention and BF16 use the PyTorch
fallback. FP32 also falls back unless CUDA autocast is active with FP16 as its
target dtype, in which case Q/K/V are cast into the fast path. Higher-order
gradients are not supported.

## Tuned dispatch

`BH` below means `batch * heads`; `w`, `s` and `v` are warps, pipeline stages
and AMD waves per execution unit.

| Path | Condition | Configuration |
|---|---|---|
| D32 inference | batch=1, N≤96 | BM16/BN16, w2/s2/v1 |
| D32 inference | batch≤8, remaining short/medium profiles | BM16/BN8, w1/s2/v1 |
| D32 inference | BH≤8, 1024≤N<4096 | BM32/BN4, w1/s1/v2 |
| D32 inference | BH≥64 and N≥1024, or N≥4096 | BM32/BN8, w1/s1/v1 |
| D32 forward | remaining profiles, including training forward | BM16/BN16, w1/s1; v1 at N≤64, v3 at N≤128, v4 otherwise |
| D32 training | default | BMK8/BNK16 and BMQ16/BNQ8, w1/s1 |
| D32 training | BH≥64 and N≥1024, or N≥4096 | BMK8/BNK32 and BMQ32/BNQ8, w2/s1/v1 |
| D64 inference | default | BM16/BN8, w1/s1/v1 |
| D64 causal inference | BH≥8 and N≥1024 | BM32/BN4, w2; s2 below 4096, otherwise s1 |
| D128 inference | default | BM8/BN4, w1/s1/v1 |
| D128 causal inference | BH≥8 and N≥1024 | BM16/BN4, w2/s1/v1 |

For the D64/D128 hybrid backward, reduction blocks are the next power of two
of N. Measured launch choices use one warp for blocks up to 64 and for blocks
256/1024, four warps for blocks 128/512, and eight for 2048/4096.

## Installation

Do not install another PyTorch or Triton wheel: the environment already
needs to contain compatible ROCm builds of both packages. Build isolation must
remain disabled because `setup.py` imports the existing PyTorch installation to
compile the HIP extension.

```bash
cd gfx1010-kernels
CXX=/usr/bin/c++ PYTORCH_ROCM_ARCH=gfx1010 \
  python -m pip install -e . --no-deps --no-build-isolation
```

Verify both independent backends in a fresh process:

```bash
python -c "import gfx1010_kernels as g; print(g.__version__); print(g.residual_layer_norm_status()); print(g.backend_status())"
```

The distribution was called `gfx1010-attention` before version 0.3. Remove its
stale editable metadata once when migrating, then repeat the installation
above:

```bash
python -m pip uninstall gfx1010-attention
```

### Optional global attention patch

Residual LayerNorm uses the explicit `gfx1010_kernels` API. Attention can
additionally be installed as a process-wide PyTorch patch:

```bash
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp deploy/gfx1010_attention_autoload.pth "$SITE_PACKAGES/"
```

The `.pth` file runs when Python starts, so
`torch.nn.functional.scaled_dot_product_attention` and
`torch.nn.MultiheadAttention` can use the optimized dispatcher without
project-specific imports. While the patch is active, PyTorch's separate native
MHA/Transformer fast path is disabled so eval and inference calls cannot bypass
the dispatcher; its previous setting is restored on uninstall.

`nn.MultiheadAttention` reaches SDPA only when called with
`need_weights=False`. Its default `need_weights=True` computes attention
through a different PyTorch path and therefore bypasses the kernel. The global
patch warns for that call in `auto` mode and raises in strict mode.

Verify the optional patch in another fresh process:

```bash
python -c "import gfx1010_attention as g; print(g.backend_status()); print(g.is_pytorch_patch_installed())"
```

Disable the startup import for one process with
`GFX1010_ATTENTION_AUTOLOAD=0`; `false`, `no` and `off` are equivalent. A full
uninstall must remove both the editable package and the manually installed
startup file:

```bash
rm "$SITE_PACKAGES/gfx1010_attention_autoload.pth"
python -m pip uninstall gfx1010-kernels
```

## Attention direct API

```python
from gfx1010_kernels import scaled_dot_product_attention

output = scaled_dot_product_attention(
    query,
    key,
    value,
    is_causal=True,
    implementation="gfx1010",
)
```

Existing code can keep the compatibility import unchanged:

```python
from gfx1010_attention import scaled_dot_product_attention
```

`implementation` controls fallback behavior:

- `gfx1010`: require the optimized path and raise `RuntimeError` otherwise.
- `auto`: use the optimized path when supported and emit `RuntimeWarning` on
  every distinct fallback call site.
- `torch`: force the original PyTorch SDPA implementation silently.

For compatible D=32 calls, `auto` always selects the custom path. During
training, D=64/128 select the hybrid backward at N≤192 when 8≤BH≤512; D=64
also selects it for 192<N≤1024 when BH≥64 and
BH·N²≤64·1024². Other D=64/128 training profiles fall back because PyTorch was
faster in measurement. `gfx1010` bypasses this performance guard and forces the
custom implementation, while retaining all compatibility checks.

The global patch defaults to `auto`. Make every unsupported call a hard error
when validating a workload:

```bash
GFX1010_ATTENTION_STRICT=1 python your_program.py
```

The patch can also be controlled explicitly:

```python
from gfx1010_attention import install_pytorch_patch, uninstall_pytorch_patch

install_pytorch_patch()
uninstall_pytorch_patch()
```

The patch affects the current process. A function alias imported before the
patch was installed cannot be replaced retroactively. Direct ATen/C++ calls
also bypass this Python-level integration.

## PyTorch execution compatibility

The supported production mode in the current `ml` environment is eager
execution. Its PyTorch/Triton combination fails `torch.compile`, export and
AOTAutograd with `ImportError: cannot import name 'triton_key'`; the custom
autograd wrapper is also not registered as a `torch.library.triton_op`.
`torch.func`, `vmap`, `jvp` and higher-order gradients are unsupported.

## Precompile

Warm the Triton cache for D=32/64/128, causal and non-causal execution, aligned
and tail lengths, inference dispatches at batch 1/8/64, and training. The
default long profiles are N=1024 and N=4096:

```bash
gfx1010-kernels-precompile
```

The profiles can be selected explicitly:

```bash
gfx1010-kernels-precompile \
  --head-dim 32 64 128 \
  --batch 1 8 64 \
  --heads 8 \
  --sequence 64 96 128 192 \
  --long-sequence 1024 4096
```

Short-profile training is compiled at batch 1. Long profiles compile batch 1
for every head dimension, plus batch 8 for D=32 at N=1024.

The cache is specific to the user, GPU and PyTorch/Triton/ROCm versions. Long
contexts do not require one binary per exact length; they reuse the applicable
runtime-length kernel.

## Validate and benchmark

The published measurements use this exact environment:

| Component | Version |
|---|---|
| GPU | AMD Radeon RX 5600 XT, `gfx1010`, 6 GiB |
| Python | 3.10.20 |
| PyTorch | 2.8.0a0+gitba56102 |
| ROCm reported by PyTorch | 7.2.53211-671d39a71e |
| Triton | 3.5.1 |
| Execution mode | Eager |

```bash
python -m pytest -q

python benchmarks/benchmark_residual_layer_norm_matrix.py \
  --batch 1 \
  --sequence 1024 \
  --dropout 0.15 \
  --dtype float16 \
  --warmup 100 \
  --reps 1000

python benchmarks/benchmark_forward_vectorization.py
python benchmarks/benchmark_backward_vectorization.py --dropout 0.15
python benchmarks/benchmark_parameter_reduction.py --best-only

python benchmarks/benchmark_gfx1010_attention.py \
  --batch 64 \
  --heads 8 \
  --sequence 55 64 96 128 192 \
  --head-dim 32
```

The residual LayerNorm matrix covers hidden sizes 128, 256, 384, 512, 768 and
1024. It measures the public API for both forward and complete
forward-plus-backward execution against the equivalent PyTorch operations. It
emits deterministically ordered JSON Lines with median, p20 and p80 CUDA-event
times. Run the same matrix with `--dtype float32` to cover the FP32 branch
specializations.

The focused residual scripts measure the aligned vectorized forward path, the
vectorized input backward and the tuned parameter-gradient reduction
respectively. Their dispatch rules and intended use are described in the
[residual LayerNorm implementation guide](docs/residual_layer_norm.md).

The following residual LayerNorm results were measured on the RX 5600 XT with
batch 1, hidden width 384, an FP32 residual stream, an FP16 branch and dropout
0.15. Each median uses 100 warmup iterations and 1000 measured iterations.
`Training` is the complete public forward-plus-backward call; speedup is
PyTorch time divided by `gfx1010` time.

| T | Forward speedup | Training speedup |
|---:|---:|---:|
| 128 | 1.273× | 1.269× |
| 256 | 1.262× | 1.265× |
| 512 | 1.303× | 1.292× |
| 1024 | 1.242× | 1.299× |
| 2048 | 1.792× | 1.846× |
| 4096 | 1.839× | 2.021× |

These measurements compare against the decomposition in this exact
machine-specific PyTorch/ROCm build. They are representative of the stated
shape and dtype, not a guarantee for a complete model or another system.

Representative causal FP16 results measured on the RX 5600 XT on 2026-07-23
are below. Times cover attention only, not a complete Transformer.

| Mode | B | H | N | D | gfx1010 ms | PyTorch ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| inference | 64 | 8 | 64 | 32 | 0.164 | 0.877 | 5.33× |
| training total | 64 | 8 | 64 | 32 | 0.617 | 1.858 | 3.01× |
| inference | 64 | 8 | 192 | 32 | 0.987 | 4.610 | 4.67× |
| training total | 64 | 8 | 192 | 32 | 4.245 | 10.288 | 2.42× |
| inference | 1 | 8 | 1024 | 32 | 0.495 | 2.708 | 5.48× |
| inference | 1 | 8 | 2048 | 32 | 1.291 | 8.181 | 6.34× |
| inference | 1 | 8 | 4096 | 32 | 4.897 | 30.742 | 6.28× |
| training total | 8 | 8 | 1024 | 32 | 12.818 | 32.581 | 2.54× |
| training total | 8 | 8 | 2048 | 32 | 48.517 | 122.755 | 2.53× |
| training total | 64 | 8 | 64 | 64 | 1.193 | 2.304 | 1.93× |
| training total | 64 | 8 | 192 | 64 | 8.735 | 11.868 | 1.36× |
| inference | 1 | 8 | 1024 | 64 | 1.079 | 2.896 | 2.69× |
| inference | 1 | 8 | 2048 | 64 | 2.908 | 8.704 | 2.99× |
| inference | 1 | 8 | 4096 | 64 | 11.612 | 32.650 | 2.81× |
| training total | 64 | 8 | 64 | 128 | 2.118 | 3.330 | 1.57× |
| training total | 64 | 8 | 192 | 128 | 12.166 | 15.281 | 1.26× |
| inference | 1 | 8 | 1024 | 128 | 2.399 | 3.296 | 1.37× |
| inference | 1 | 8 | 2048 | 128 | 8.723 | 9.866 | 1.13× |
| inference | 1 | 8 | 4096 | 128 | 32.685 | 36.766 | 1.12× |

The PyTorch comparison is the implementation available in this exact
machine-specific build; it reports that memory-efficient SDPA was not compiled
in. Results therefore must not be extrapolated to another ROCm build. The
performance guard is material: forced D64 and D128 training at B=1, H=8,
N=1024 measured 0.46× and 0.27× PyTorch respectively, so `auto` routes both to
PyTorch.

The attention benchmark also emits JSON Lines and measures forward and
backward separately. Compare only runs from the same GPU state and software
build. The test suite checks every native residual hidden specialization,
forward and backward gradients, dropout behavior, edge cases, public dispatch
and attention against PyTorch references.

`benchmark_backward_formula.py` is an experimental diagnostic for comparing
backward decompositions; it is not the production path.

## Fallback diagnostics

Fallback is decided before launching a GPU kernel, so an unsupported operation
cannot silently enter a generic branch inside the optimized kernel. In `auto`
mode the warning includes the exact reason. In `gfx1010` mode the same
condition is an error.

The two status calls intentionally answer different questions:

- `residual_layer_norm_status()` reports whether the compiled native HIP
  extension can execute on the current GPU;
- `backend_status()` reports whether the Triton HIP attention backend targets
  `gfx1010`.

`GFX1010_ATTENTION_STRICT=1` applies only to the global attention patch.
Residual LayerNorm is made strict per call with
`implementation="gfx1010"`.

## License

MIT. See [`LICENSE`](LICENSE).
