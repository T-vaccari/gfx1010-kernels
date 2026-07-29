from threading import Lock

import torch
from torch.autograd.function import once_differentiable

import triton
import triton.language as tl


_dropout_lock = Lock()
_dropout_offsets = {}


def _dropout_seed_offset(device, element_count):
    device_index = device.index or torch.cuda.current_device()
    seed = torch.initial_seed() & 0xFFFFFFFF
    key = (device_index, seed)
    with _dropout_lock:
        offset = _dropout_offsets.get(key, 0)
        _dropout_offsets[key] = offset + element_count
    return seed, offset


def _row_launch(hidden_size):
    block_size = triton.next_power_of_2(hidden_size)
    if block_size <= 256:
        return block_size, 2
    if block_size <= 512:
        return block_size, 4
    return block_size, 8


@triton.jit(
    do_not_specialize=["seed", "random_offset"],
    do_not_specialize_on_alignment=["seed", "random_offset"],
)
def _residual_layer_norm_forward(
    x,
    branch,
    weight,
    bias,
    updated,
    normalized,
    mean,
    rstd,
    hidden_size: tl.constexpr,
    eps,
    dropout_p: tl.constexpr,
    seed,
    random_offset,
    training: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    valid = columns < hidden_size
    offsets = row * hidden_size + columns
    x_values = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32)
    branch_values = tl.load(
        branch + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    if training and dropout_p > 0.0:
        random = tl.rand(seed, random_offset + offsets)
        keep = random >= dropout_p
        branch_values = tl.where(
            keep,
            branch_values / (1.0 - dropout_p),
            0.0,
        )
    values = x_values + branch_values
    row_mean = tl.sum(values, axis=0) / hidden_size
    centered = tl.where(valid, values - row_mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    row_rstd = tl.rsqrt(variance + eps)
    weights = tl.load(weight + columns, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(bias + columns, mask=valid, other=0.0).to(tl.float32)
    output = centered * row_rstd * weights + biases
    tl.store(updated + offsets, values, mask=valid)
    tl.store(normalized + offsets, output, mask=valid)
    tl.store(mean + row, row_mean)
    tl.store(rstd + row, row_rstd)


@triton.jit(
    do_not_specialize=["seed", "random_offset"],
    do_not_specialize_on_alignment=["seed", "random_offset"],
)
def _residual_layer_norm_backward_inputs(
    grad_updated,
    grad_normalized,
    updated,
    weight,
    mean,
    rstd,
    grad_x,
    grad_branch,
    hidden_size: tl.constexpr,
    dropout_p: tl.constexpr,
    seed,
    random_offset,
    training: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    valid = columns < hidden_size
    offsets = row * hidden_size + columns
    values = tl.load(updated + offsets, mask=valid, other=0.0).to(tl.float32)
    grad_residual = tl.load(
        grad_updated + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    grad_output = tl.load(
        grad_normalized + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(weight + columns, mask=valid, other=0.0).to(tl.float32)
    row_mean = tl.load(mean + row)
    row_rstd = tl.load(rstd + row)
    xhat = tl.where(valid, (values - row_mean) * row_rstd, 0.0)
    weighted_grad = grad_output * weights
    grad_mean = tl.sum(weighted_grad, axis=0) / hidden_size
    grad_projection = (
        tl.sum(weighted_grad * xhat, axis=0) / hidden_size
    )
    grad_norm_input = (
        weighted_grad - grad_mean - xhat * grad_projection
    ) * row_rstd
    total_grad = grad_residual + grad_norm_input
    tl.store(grad_x + offsets, total_grad, mask=valid)
    if training and dropout_p > 0.0:
        random = tl.rand(seed, random_offset + offsets)
        keep = random >= dropout_p
        total_grad = tl.where(
            keep,
            total_grad / (1.0 - dropout_p),
            0.0,
        )
    tl.store(grad_branch + offsets, total_grad, mask=valid)


@triton.jit(
    do_not_specialize=["row_count"],
    do_not_specialize_on_alignment=["row_count"],
)
def _residual_layer_norm_backward_partials(
    grad_normalized,
    updated,
    mean,
    rstd,
    partial_weight,
    partial_bias,
    row_count,
    hidden_size: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    row_group = tl.program_id(0)
    column_group = tl.program_id(1)
    rows = row_group * block_rows + tl.arange(0, block_rows)
    columns = (
        column_group * block_columns + tl.arange(0, block_columns)
    )
    offsets = rows[:, None] * hidden_size + columns[None, :]
    valid = (rows[:, None] < row_count) & (
        columns[None, :] < hidden_size
    )
    values = tl.load(updated + offsets, mask=valid, other=0.0).to(tl.float32)
    grad_output = tl.load(
        grad_normalized + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    row_mean = tl.load(
        mean + rows,
        mask=rows < row_count,
        other=0.0,
    )
    row_rstd = tl.load(
        rstd + rows,
        mask=rows < row_count,
        other=0.0,
    )
    xhat = (values - row_mean[:, None]) * row_rstd[:, None]
    weight_partial = tl.sum(grad_output * xhat, axis=0)
    bias_partial = tl.sum(grad_output, axis=0)
    partial_offsets = row_group * hidden_size + columns
    tl.store(
        partial_weight + partial_offsets,
        weight_partial,
        mask=columns < hidden_size,
    )
    tl.store(
        partial_bias + partial_offsets,
        bias_partial,
        mask=columns < hidden_size,
    )


@triton.jit(
    do_not_specialize=["group_count"],
    do_not_specialize_on_alignment=["group_count"],
)
def _residual_layer_norm_backward_reduce(
    partial_weight,
    partial_bias,
    grad_weight,
    grad_bias,
    group_count,
    hidden_size: tl.constexpr,
    block_groups: tl.constexpr,
    block_columns: tl.constexpr,
):
    column_group = tl.program_id(0)
    groups = tl.arange(0, block_groups)
    columns = (
        column_group * block_columns + tl.arange(0, block_columns)
    )
    offsets = groups[:, None] * hidden_size + columns[None, :]
    valid = (groups[:, None] < group_count) & (
        columns[None, :] < hidden_size
    )
    weight_values = tl.load(
        partial_weight + offsets,
        mask=valid,
        other=0.0,
    )
    bias_values = tl.load(
        partial_bias + offsets,
        mask=valid,
        other=0.0,
    )
    tl.store(
        grad_weight + columns,
        tl.sum(weight_values, axis=0),
        mask=columns < hidden_size,
    )
    tl.store(
        grad_bias + columns,
        tl.sum(bias_values, axis=0),
        mask=columns < hidden_size,
    )


class TritonResidualLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        branch,
        weight,
        bias,
        dropout_p,
        eps,
        training,
    ):
        hidden_size = x.shape[-1]
        row_count = x.numel() // hidden_size
        block_size, warps = _row_launch(hidden_size)
        updated = torch.empty_like(x)
        normalized = torch.empty_like(x)
        mean = torch.empty(row_count, dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
        if training and dropout_p > 0.0:
            seed, random_offset = _dropout_seed_offset(x.device, x.numel())
        else:
            seed, random_offset = 0, 0
        _residual_layer_norm_forward[(row_count,)](
            x,
            branch,
            weight,
            bias,
            updated,
            normalized,
            mean,
            rstd,
            hidden_size=hidden_size,
            eps=eps,
            dropout_p=dropout_p,
            seed=seed,
            random_offset=random_offset,
            training=training,
            block_size=block_size,
            num_warps=warps,
            num_stages=1,
            waves_per_eu=1,
            allow_flush_denorm=True,
        )
        if any(ctx.needs_input_grad[:4]):
            ctx.save_for_backward(updated, weight, mean, rstd)
            ctx.branch_dtype = branch.dtype
            ctx.dropout_p = dropout_p
            ctx.training = training
            ctx.seed = seed
            ctx.random_offset = random_offset
        return updated, normalized

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_updated, grad_normalized):
        updated, weight, mean, rstd = ctx.saved_tensors
        hidden_size = updated.shape[-1]
        row_count = updated.numel() // hidden_size
        block_size, warps = _row_launch(hidden_size)
        if grad_updated is None:
            grad_updated = torch.zeros_like(updated)
        else:
            grad_updated = grad_updated.contiguous()
        if grad_normalized is None:
            grad_normalized = torch.zeros_like(updated)
        else:
            grad_normalized = grad_normalized.contiguous()
        grad_x = torch.empty_like(updated)
        grad_branch = torch.empty(
            updated.shape,
            dtype=ctx.branch_dtype,
            device=updated.device,
        )
        _residual_layer_norm_backward_inputs[(row_count,)](
            grad_updated,
            grad_normalized,
            updated,
            weight,
            mean,
            rstd,
            grad_x,
            grad_branch,
            hidden_size=hidden_size,
            dropout_p=ctx.dropout_p,
            seed=ctx.seed,
            random_offset=ctx.random_offset,
            training=ctx.training,
            block_size=block_size,
            num_warps=warps,
            num_stages=1,
            waves_per_eu=1,
            allow_flush_denorm=True,
        )

        block_rows = 16
        block_columns = 32
        group_count = triton.cdiv(row_count, block_rows)
        partial_weight = torch.empty(
            (group_count, hidden_size),
            dtype=torch.float32,
            device=updated.device,
        )
        partial_bias = torch.empty_like(partial_weight)
        partial_grid = (
            group_count,
            triton.cdiv(hidden_size, block_columns),
        )
        _residual_layer_norm_backward_partials[partial_grid](
            grad_normalized,
            updated,
            mean,
            rstd,
            partial_weight,
            partial_bias,
            row_count,
            hidden_size=hidden_size,
            block_rows=block_rows,
            block_columns=block_columns,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
            allow_flush_denorm=True,
        )
        reduce_columns = 8
        reduce_groups = triton.next_power_of_2(group_count)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(weight)
        _residual_layer_norm_backward_reduce[
            (triton.cdiv(hidden_size, reduce_columns),)
        ](
            partial_weight,
            partial_bias,
            grad_weight,
            grad_bias,
            group_count,
            hidden_size=hidden_size,
            block_groups=reduce_groups,
            block_columns=reduce_columns,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
            allow_flush_denorm=True,
        )
        return grad_x, grad_branch, grad_weight, grad_bias, None, None, None


def triton_residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p,
    eps,
    training,
):
    return TritonResidualLayerNorm.apply(
        x,
        branch,
        weight,
        bias,
        dropout_p,
        eps,
        training,
    )
