# gfx1010-attention

Fused scaled dot-product attention for AMD `gfx1010`, packaged independently
from any model repository. It is tuned and tested on a Radeon RX 5600 XT with
the ROCm build of PyTorch and Triton from the `ml` Conda environment.

The sequence length is a runtime value, not a fixed kernel constant. The
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

## Fast-path support

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

## Install in Conda `ml`

Do not install another PyTorch or Triton wheel: the environment already
contains the machine-specific ROCm builds.

```bash
source /home/tom/miniforge3/etc/profile.d/conda.sh
conda activate ml
cd /home/tom/machine-learning/gfx1010-attention
python -m pip install -e . --no-deps --no-build-isolation

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp deploy/gfx1010_attention_autoload.pth "$SITE_PACKAGES/"
```

The `.pth` file installs a process-wide patch when Python starts, so
`torch.nn.functional.scaled_dot_product_attention` and
`torch.nn.MultiheadAttention` can use the optimized dispatcher without
project-specific imports. While the patch is active, PyTorch's separate native
MHA/Transformer fast path is disabled so eval and inference calls cannot bypass
the dispatcher; its previous setting is restored on uninstall.

`nn.MultiheadAttention` reaches SDPA only when called with
`need_weights=False`. Its default `need_weights=True` computes attention
through a different PyTorch path and therefore bypasses the kernel. The global
patch warns for that call in `auto` mode and raises in strict mode.

Verify the installation in a fresh Python process:

```bash
python -c "import gfx1010_attention as g; print(g.backend_status()); print(g.is_pytorch_patch_installed())"
```

Disable the startup import for one process with
`GFX1010_ATTENTION_AUTOLOAD=0`; `false`, `no` and `off` are equivalent. A full
uninstall must remove both the editable package and the manually installed
startup file:

```bash
rm "$SITE_PACKAGES/gfx1010_attention_autoload.pth"
python -m pip uninstall gfx1010-attention
```

## Direct API

```python
from gfx1010_attention import scaled_dot_product_attention

output = scaled_dot_product_attention(
    query,
    key,
    value,
    is_causal=True,
    implementation="gfx1010",
)
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
gfx1010-attention-precompile
```

The profiles can be selected explicitly:

```bash
gfx1010-attention-precompile \
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

```bash
python -m pytest -q
python benchmarks/benchmark_gfx1010_attention.py \
  --batch 64 \
  --heads 8 \
  --sequence 55 64 96 128 192 \
  --head-dim 32
```

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

The benchmark emits JSON Lines and measures forward and backward separately.
Compare only runs from the same GPU state and software build. The test suite
checks the fused result against PyTorch and, for stricter numerical checks,
against an FP32 reference.

`benchmark_backward_formula.py` is an experimental diagnostic for comparing
backward decompositions; it is not the production path.

## Fallback diagnostics

Fallback is decided before launching a GPU kernel, so an unsupported operation
cannot silently enter a generic branch inside the optimized kernel. In `auto`
mode the warning includes the exact reason. In `gfx1010` or global strict mode
the same condition is an error.
