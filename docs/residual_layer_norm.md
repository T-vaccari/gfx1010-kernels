# Residual + LayerNorm

`gfx1010_kernels.residual_layer_norm` is a native HIP operation specialized
for the AMD `gfx1010` architecture. It fuses the residual update, optional
dropout and the LayerNorm that feeds the next Transformer sublayer.

## Public API

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

The arguments are:

- `x`: contiguous FP32 residual stream;
- `branch`: contiguous FP16 or FP32 attention/MLP output with the same shape
  as `x`;
- `weight`, `bias`: contiguous FP32 LayerNorm vectors whose length is the
  final tensor dimension;
- `dropout_p`: branch dropout probability in `[0, 1)`;
- `eps`: positive LayerNorm epsilon;
- `training`: dropout is active only when this is `True`;
- `implementation`: `gfx1010`, `auto`, `torch` or the separate experimental
  `triton` normalization path.

`implementation="gfx1010"` is the recommended mode for correctness tests and
performance measurements because any rejected fast path becomes an error.
`auto` uses the native operation when compatible and otherwise emits a
`RuntimeWarning` before using the PyTorch decomposition.

The operation returns two FP32 tensors with the same shape as `x`:

- `updated` is the new residual stream;
- `normalized` is the input to the following attention or MLP sublayer.

Both are required by a pre-norm Transformer block. Returning them together
lets the kernel write the residual update once and reuse it immediately for
normalization.

## Pre-norm Transformer integration

For an attention branch followed by `ln2`, replace:

```python
x = x + attention
normalized = ln2(x)
```

with:

```python
x, normalized = residual_layer_norm(
    x,
    attention,
    ln2.weight,
    ln2.bias,
    dropout_p=0.0,
    eps=ln2.eps,
    training=self.training,
    implementation="gfx1010",
)
```

To fuse an MLP whose last module is dropout, call the MLP only through its
pre-dropout projection and pass that result to `residual_layer_norm` with the
original dropout probability. The normalization parameters must belong to the
next consumer: the next block's `ln1`, or the model's final LayerNorm after the
last block. Do not leave the original dropout active as well.

An integration can keep the existing `LayerNorm` modules and pass their
parameters to the fused operation. This preserves checkpoint `state_dict`
keys while replacing only the execution path.

## Forward

For every row of hidden width \(H\), the operation computes:

\[
d_i =
\begin{cases}
b_i & \text{during evaluation or when }p=0 \\
b_i m_i/(1-p) & \text{during training}
\end{cases}
\]

\[
u_i = x_i + d_i
\]

\[
\mu = \frac{1}{H}\sum_i u_i,\qquad
r = \frac{1}{\sqrt{\frac{1}{H}\sum_i(u_i-\mu)^2+\epsilon}}
\]

\[
\hat{x}_i=(u_i-\mu)r,\qquad
y_i=\hat{x}_i\gamma_i+\beta_i
\]

Here `u` is `updated`, `y` is `normalized`, `b` is `branch`, and `m` is the
dropout keep mask.

The native forward performs the following work in one HIP launch:

1. load `x` and `branch`;
2. generate Philox dropout values when dropout is active;
3. apply inverted-dropout scaling and add the branch to the residual;
4. reduce the FP32 row sum and squared centered sum with wave32 reductions;
5. apply the affine LayerNorm;
6. write `updated`, `normalized`, the row mean and reciprocal standard
   deviation.

The dropout mask is stored as one `int32` word per wave32 group of 32 hidden
elements instead of one element per activation. Mean, reciprocal standard
deviation and the packed mask are internal tensors saved for backward.
Dropout uses the active PyTorch GPU generator, so restoring its RNG state
replays the mask and consecutive calls consume distinct Philox subsequences.

When dropout is inactive, aligned and sufficiently large rows use four-element
vector loads and stores. Dropout uses the scalar-per-lane kernel because RNG
generation and bit packing dominate that path.

## Backward

Let `g_u` and `g_y` be the incoming gradients for `updated` and `normalized`.
The input-gradient kernel computes:

\[
q_i=g_{y,i}\gamma_i
\]

\[
g_{\mathrm{LN},i}
=r\left(q_i-\operatorname{mean}(q)
-\hat{x}_i\operatorname{mean}(q\hat{x})\right)
\]

\[
g_i=g_{u,i}+g_{\mathrm{LN},i}
\]

\[
\frac{\partial L}{\partial x_i}=g_i,\qquad
\frac{\partial L}{\partial b_i}=
\begin{cases}
g_i & \text{without dropout}\\
g_i m_i/(1-p) & \text{with dropout}
\end{cases}
\]

The two row reductions for `mean(q)` and `mean(q*xhat)` are combined into one
wave32/block reduction. Eligible shapes use four-element vector memory
operations; the scalar path remains available for small or misaligned inputs.
The branch gradient is written in the original branch dtype.

LayerNorm parameter gradients are:

\[
\frac{\partial L}{\partial\gamma_i}
=\sum_{\text{rows}}g_{y,i}\hat{x}_i,\qquad
\frac{\partial L}{\partial\beta_i}
=\sum_{\text{rows}}g_{y,i}
\]

The default reduction partitions rows into tuned groups. For the common
64-thread configuration, four wave32 groups accumulate independent partials
in LDS and wave zero combines them before issuing one FP32 atomic update per
parameter. This reduces global atomic pressure compared with one update per
wave. Selected profiles use a 256-thread scalar-group variant when it is
faster.

If only `updated` contributes to the loss, autograd does not materialize
gradients for `weight` or `bias` and the parameter-reduction kernel is skipped.
Only first-order autograd is supported.

## Compile-time specializations and runtime dimensions

The native extension contains hidden-width specializations for:

```text
128, 256, 384, 512, 768, 1024
```

Hidden width is compile-time because it fixes loop trip counts, reduction
topology and launch bounds. Batch, sequence length and any other leading
dimensions are flattened into:

```text
row_count = x.numel() / hidden_size
```

`row_count` is a runtime value. Increasing context length therefore does not
require another compiled kernel, and tail row counts are supported.

The base forward/input-backward thread counts are:

| Hidden width | Threads |
|---:|---:|
| 128 | 128 |
| 256 | 128 |
| 384 | 128 |
| 512 | 128 |
| 768 | 128 |
| 1024 | 192 |

## Runtime dispatch

### Vectorized forward

The four-element forward path requires compatible pointer alignment, inactive
dropout and at least the following number of rows:

| Hidden width | Minimum rows |
|---:|---:|
| 128 | 2048 |
| 256, 384 | 1024 |
| 512, 768 | 512 |
| 1024 | 256 |

Below the threshold, or when alignment is insufficient, dispatch selects the
scalar-per-lane specialization. The thresholds are measured profitability
guards, not correctness restrictions.

### Vectorized input backward

Hidden width 128 always uses the scalar path. Other widths select vectorized
input backward when alignment is compatible and:

| Hidden width | Condition |
|---:|---|
| 256, 384 | `row_count >= 512` |
| 512, 768 | dropout inactive, or `row_count >= 256` |
| 1024 | all non-empty row counts |

### Parameter-gradient reduction

The default non-deterministic path uses 64 threads except for these measured
profiles, which use 256:

- hidden 512 with exactly 256 rows;
- hidden 256 with exactly 512 rows;
- hidden 768 or 1024 with at least 4096 rows.

The number of row groups is selected from this table and capped at
`row_count`:

| Hidden | rows ≤128 | ≤256 | ≤512 | ≤1024 | ≤2048 | >2048 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 80 | 12 | 48 | 80 | 96 | 128 |
| 256 | 48 | 32 | 12 | 64 | 80 | 80 |
| 384 | 8 | 16 | 64 | 64 | 80 | 80 |
| 512 | 48 | 48 | 48 | 64 | 48 | 48 |
| 768 | 32 | 48 | 48 | 48 | 48 | 32 |
| 1024 | 32 | 32 | 32 | 24 | 24 | 24 |

These values are internal launch policy, not part of the public API. They were
selected by sweeping the real complete native backward rather than timing the
parameter kernel in isolation.

## Determinism

The fast parameter-gradient reduction uses FP32 atomics. Its values match the
PyTorch reference within the test tolerances, but the order of additions can
change and the final bits are not guaranteed to replay.

```python
torch.use_deterministic_algorithms(True)
```

selects a two-stage fixed-order parameter reduction:

1. each block writes an FP32 partial for a fixed contiguous row group;
2. a second kernel reduces those partials in a fixed order.

This path uses an intermediate tensor and is slower, but repeated backward
calls are bitwise stable. Input and branch gradients are row-local and are
also bitwise stable for identical inputs. Dropout reproducibility additionally
depends on the PyTorch RNG state.

## Native-path limits

- GPU architecture must report exactly `gfx1010`.
- Execution must use the installed ROCm PyTorch eager runtime.
- `x`, `weight` and `bias` must be FP32.
- `branch` may be FP16 or FP32.
- All four public inputs must be contiguous and on the same HIP device.
- Hidden width must be one of the six compiled specializations.
- Empty tensors use the PyTorch fallback.
- Dropout during HIP graph capture is unsupported.
- Higher-order gradients, `torch.func`, `vmap`, `jvp`, export and
  `torch.compile` are unsupported in the current machine-specific stack.

Use the status and compatibility helpers before constructing an optional
integration:

```python
from gfx1010_kernels import (
    can_use_residual_layer_norm,
    residual_layer_norm_status,
)

print(residual_layer_norm_status())
print(can_use_residual_layer_norm(x, branch, weight, bias, dropout_p=0.15))
```

## Tests

Run the complete package suite:

```bash
python -m pytest -q
```

Run only residual LayerNorm coverage:

```bash
python -m pytest -q \
  tests/test_residual_layer_norm.py \
  tests/test_residual_layer_norm_edge_cases.py \
  tests/test_residual_layer_norm_native_matrix.py \
  tests/test_residual_layer_norm_vectorized.py
```

The tests compare forward outputs and all first-order gradients with PyTorch.
They cover every hidden specialization, FP16 and FP32 branches, dropout mask
and RNG behavior, row-count boundaries, strict and fallback dispatch,
misaligned vector fallbacks, tuned parameter dispatch and deterministic
bitwise replay.

## Benchmarks

The public end-to-end matrix measures both the fused forward and the complete
forward-plus-backward call:

```bash
python benchmarks/benchmark_residual_layer_norm_matrix.py \
  --batch 1 \
  --sequence 1024 \
  --dropout 0.15 \
  --dtype float16 \
  --warmup 100 \
  --reps 1000
```

Each JSON Lines result includes median, p20 and p80 CUDA-event times for
`implementation="gfx1010"` and `implementation="torch"`. The reported speedup
is `torch median / gfx1010 median`.

The focused scripts isolate dispatch decisions:

```bash
python benchmarks/benchmark_forward_vectorization.py
python benchmarks/benchmark_backward_vectorization.py --dropout 0.15
python benchmarks/benchmark_parameter_reduction.py --best-only
```

- `benchmark_forward_vectorization.py` compares aligned vectorized and
  deliberately misaligned scalar forward paths without dropout.
- `benchmark_backward_vectorization.py` makes the same comparison for the
  input-gradient kernel.
- `benchmark_parameter_reduction.py` sweeps thread and row-group choices while
  timing the complete native backward.

Use `benchmark_residual_layer_norm.py` for one hidden width and
`benchmark_native_variants.py` or `benchmark_backward_input_matrix.py` for
lower-level investigations. Compare only measurements from the same GPU power
state and exact PyTorch/ROCm build.

On a character-level GPT with `B=1, T=1024, C=384, L=8`, 10 warmups and 40
CUDA-event samples measured the complete AMP forward and backward microstep at
150.119 ms with the original normalization flow and 148.946 ms with the full
fused flow, a 1.008× model-level speedup. This is deliberately reported
separately from the larger isolated-operator speedups because attention, MLP
matmuls and the vocabulary projection dominate complete-model time.
