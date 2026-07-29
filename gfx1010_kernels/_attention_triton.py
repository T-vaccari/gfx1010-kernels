import torch
from torch.autograd.function import once_differentiable

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["sequence_length"],
    do_not_specialize_on_alignment=["sequence_length"],
)
def _attention_probabilities(
    scores,
    scale,
    sequence_length,
    block_n: tl.constexpr,
    causal: tl.constexpr,
):
    row = tl.program_id(0)
    column_offsets = tl.arange(0, block_n)
    query_position = row % sequence_length
    logits = tl.load(
        scores + row * sequence_length + column_offsets,
        mask=column_offsets < sequence_length,
        other=0.0,
    ).to(tl.float32)
    logits *= scale
    valid = column_offsets < sequence_length
    if causal:
        valid &= column_offsets <= query_position
    logits = tl.where(valid, logits, -float("inf"))
    maximum = tl.max(logits, axis=0)
    probabilities = tl.math.exp2(
        (logits - maximum) * 1.4426950408889634
    )
    probabilities /= tl.sum(probabilities, axis=0)
    tl.store(
        scores + row * sequence_length + column_offsets,
        probabilities,
        mask=column_offsets < sequence_length,
    )


@triton.jit(
    do_not_specialize=["sequence_length"],
    do_not_specialize_on_alignment=["sequence_length"],
)
def _attention_backward_score(
    probabilities,
    grad_score,
    sequence_length,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    column_offsets = tl.arange(0, block_n)
    valid = column_offsets < sequence_length
    offsets = row * sequence_length + column_offsets
    probability = tl.load(
        probabilities + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    grad_probability = tl.load(
        grad_score + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(probability * grad_probability, axis=0)
    tl.store(
        grad_score + offsets,
        probability * (grad_probability - delta),
        mask=valid,
    )


@triton.jit
def _attention_forward_inner(
    accumulator,
    denominator,
    maximum,
    query,
    key,
    value,
    key_base,
    value_base,
    stride_key_sequence,
    stride_key_dim,
    stride_value_sequence,
    stride_value_dim,
    query_offsets,
    key_offsets,
    dim_offsets,
    query_block,
    scale_log2,
    sequence_length,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    stage: tl.constexpr,
):
    if stage == 1:
        low = 0
        high = query_block * block_m
    elif stage == 2:
        low = query_block * block_m
        high = (query_block + 1) * block_m
        low = tl.multiple_of(low, block_m)
    else:
        low = 0
        high = sequence_length

    for key_block in tl.range(low, high, block_n):
        key_positions = key_block + key_offsets
        key_tile = tl.load(
            key_base
            + key_positions[:, None] * stride_key_sequence
            + dim_offsets[None, :] * stride_key_dim,
            mask=(key_positions[:, None] < sequence_length)
            & (dim_offsets[None, :] < head_dim),
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(key_tile)) * scale_log2
        valid = key_positions[None, :] < sequence_length
        if stage == 2:
            valid = valid & (query_offsets[:, None] >= key_positions[None, :])
        logits = tl.where(valid, logits, -float("inf"))

        next_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        probabilities = tl.math.exp2(logits - next_maximum[:, None])
        correction = tl.math.exp2(maximum - next_maximum)
        next_denominator = tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]

        value_tile = tl.load(
            value_base
            + key_positions[:, None] * stride_value_sequence
            + dim_offsets[None, :] * stride_value_dim,
            mask=(key_positions[:, None] < sequence_length)
            & (dim_offsets[None, :] < head_dim),
            other=0.0,
        )
        accumulator = tl.dot(probabilities.to(tl.float16), value_tile, accumulator)
        denominator = denominator * correction + next_denominator
        maximum = next_maximum

    return accumulator, denominator, maximum


@triton.jit(
    do_not_specialize=["sequence_length"],
    do_not_specialize_on_alignment=["sequence_length"],
)
def _attention_forward(
    query,
    key,
    value,
    output,
    logsumexp,
    scale,
    stride_query_batch,
    stride_query_head,
    stride_query_sequence,
    stride_query_dim,
    stride_key_batch,
    stride_key_head,
    stride_key_sequence,
    stride_key_dim,
    stride_value_batch,
    stride_value_head,
    stride_value_sequence,
    stride_value_dim,
    stride_output_batch,
    stride_output_head,
    stride_output_sequence,
    stride_output_dim,
    head_count: tl.constexpr,
    sequence_length,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    causal: tl.constexpr,
    save_stats: tl.constexpr,
):
    tl.static_assert(block_m % block_n == 0)
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // head_count
    head = batch_head % head_count

    query_base = query + batch * stride_query_batch + head * stride_query_head
    key_base = key + batch * stride_key_batch + head * stride_key_head
    value_base = value + batch * stride_value_batch + head * stride_value_head
    output_base = output + batch * stride_output_batch + head * stride_output_head

    query_offsets = query_block * block_m + tl.arange(0, block_m)
    key_offsets = tl.arange(0, block_n)
    dim_offsets = tl.arange(0, head_dim)
    query_tile = tl.load(
        query_base
        + query_offsets[:, None] * stride_query_sequence
        + dim_offsets[None, :] * stride_query_dim,
        mask=query_offsets[:, None] < sequence_length,
        other=0.0,
    )

    maximum = tl.full([block_m], -float("inf"), tl.float32)
    denominator = tl.full([block_m], 1.0, tl.float32)
    accumulator = tl.zeros([block_m, head_dim], tl.float32)
    scale_log2 = scale * 1.4426950408889634

    if causal:
        accumulator, denominator, maximum = _attention_forward_inner(
            accumulator,
            denominator,
            maximum,
            query_tile,
            key,
            value,
            key_base,
            value_base,
            stride_key_sequence,
            stride_key_dim,
            stride_value_sequence,
            stride_value_dim,
            query_offsets,
            key_offsets,
            dim_offsets,
            query_block,
            scale_log2,
            sequence_length,
            head_dim,
            block_m,
            block_n,
            1,
        )
        accumulator, denominator, maximum = _attention_forward_inner(
            accumulator,
            denominator,
            maximum,
            query_tile,
            key,
            value,
            key_base,
            value_base,
            stride_key_sequence,
            stride_key_dim,
            stride_value_sequence,
            stride_value_dim,
            query_offsets,
            key_offsets,
            dim_offsets,
            query_block,
            scale_log2,
            sequence_length,
            head_dim,
            block_m,
            block_n,
            2,
        )
    else:
        accumulator, denominator, maximum = _attention_forward_inner(
            accumulator,
            denominator,
            maximum,
            query_tile,
            key,
            value,
            key_base,
            value_base,
            stride_key_sequence,
            stride_key_dim,
            stride_value_sequence,
            stride_value_dim,
            query_offsets,
            key_offsets,
            dim_offsets,
            query_block,
            scale_log2,
            sequence_length,
            head_dim,
            block_m,
            block_n,
            3,
        )

    accumulator /= denominator[:, None]
    maximum += tl.math.log2(denominator)
    tl.store(
        output_base
        + query_offsets[:, None] * stride_output_sequence
        + dim_offsets[None, :] * stride_output_dim,
        accumulator.to(tl.float16),
        mask=query_offsets[:, None] < sequence_length,
    )
    if save_stats:
        tl.store(
            logsumexp + batch_head * sequence_length + query_offsets,
            maximum,
            mask=query_offsets < sequence_length,
        )


@triton.jit(
    do_not_specialize=["sequence_length"],
    do_not_specialize_on_alignment=["sequence_length"],
)
def _attention_backward_preprocess(
    output,
    grad_output,
    delta,
    stride_output_batch,
    stride_output_head,
    stride_output_sequence,
    stride_output_dim,
    stride_grad_output_batch,
    stride_grad_output_head,
    stride_grad_output_sequence,
    stride_grad_output_dim,
    head_count: tl.constexpr,
    sequence_length,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    aligned: tl.constexpr,
):
    sequence_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    batch_head = tl.program_id(1)
    batch = batch_head // head_count
    head = batch_head % head_count
    dim_offsets = tl.arange(0, head_dim)
    output_base = output + batch * stride_output_batch + head * stride_output_head
    grad_output_base = (
        grad_output
        + batch * stride_grad_output_batch
        + head * stride_grad_output_head
    )
    output_ptrs = (
        output_base
        + sequence_offsets[:, None] * stride_output_sequence
        + dim_offsets[None, :] * stride_output_dim
    )
    grad_ptrs = (
        grad_output_base
        + sequence_offsets[:, None] * stride_grad_output_sequence
        + dim_offsets[None, :] * stride_grad_output_dim
    )
    if aligned:
        output_tile = tl.load(output_ptrs)
        grad_tile = tl.load(grad_ptrs).to(tl.float32)
        tl.store(
            delta + batch_head * sequence_length + sequence_offsets,
            tl.sum(output_tile * grad_tile, axis=1),
        )
    else:
        valid = sequence_offsets < sequence_length
        output_tile = tl.load(
            output_ptrs,
            mask=valid[:, None],
            other=0.0,
        )
        grad_tile = tl.load(
            grad_ptrs,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            delta + batch_head * sequence_length + sequence_offsets,
            tl.sum(output_tile * grad_tile, axis=1),
            mask=valid,
        )


@triton.jit
def _attention_backward_key_value(
    grad_key,
    grad_value,
    query,
    key_tile,
    value_tile,
    scale_log2,
    grad_output,
    logsumexp,
    delta,
    query_sequence_stride,
    query_dim_stride,
    grad_output_sequence_stride,
    grad_output_dim_stride,
    sequence_length,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    start_key,
    start_query,
    steps,
    masked: tl.constexpr,
    aligned: tl.constexpr,
):
    query_offsets = start_query + tl.arange(0, block_m)
    key_offsets = start_key + tl.arange(0, block_n)
    dim_offsets = tl.arange(0, head_dim)
    query_ptrs = (
        query
        + query_offsets[None, :] * query_sequence_stride
        + dim_offsets[:, None] * query_dim_stride
    )
    grad_output_ptrs = (
        grad_output
        + query_offsets[:, None] * grad_output_sequence_stride
        + dim_offsets[None, :] * grad_output_dim_stride
    )
    tl.static_assert(block_n % block_m == 0)

    current_query = start_query
    for _ in tl.range(0, steps):
        query_offsets = current_query + tl.arange(0, block_m)
        if aligned:
            query_transposed = tl.load(query_ptrs)
            maximum = tl.load(logsumexp + query_offsets)
        else:
            query_valid = query_offsets < sequence_length
            query_transposed = tl.load(
                query_ptrs,
                mask=query_valid[None, :],
                other=0.0,
            )
            maximum = tl.load(
                logsumexp + query_offsets,
                mask=query_valid,
                other=0.0,
            )
        logits_transposed = (
            tl.dot(key_tile, query_transposed) * scale_log2
        )
        probabilities_transposed = tl.math.exp2(
            logits_transposed - maximum[None, :]
        )
        if masked:
            valid = query_offsets[None, :] >= key_offsets[:, None]
            if not aligned:
                valid &= (
                    query_valid[None, :]
                    & (key_offsets[:, None] < sequence_length)
                )
            probabilities_transposed = tl.where(
                valid,
                probabilities_transposed,
                0.0,
            )
        elif not aligned:
            valid = (
                query_valid[None, :]
                & (key_offsets[:, None] < sequence_length)
            )
            probabilities_transposed = tl.where(
                valid,
                probabilities_transposed,
                0.0,
            )

        if aligned:
            grad_output_tile = tl.load(grad_output_ptrs)
        else:
            grad_output_tile = tl.load(
                grad_output_ptrs,
                mask=query_valid[:, None],
                other=0.0,
            )
        grad_value += tl.dot(
            probabilities_transposed.to(tl.float16),
            grad_output_tile,
        )
        if aligned:
            delta_tile = tl.load(delta + query_offsets)
        else:
            delta_tile = tl.load(
                delta + query_offsets,
                mask=query_valid,
                other=0.0,
            )
        grad_probability = tl.dot(
            value_tile,
            tl.trans(grad_output_tile),
        ).to(tl.float32)
        grad_score = probabilities_transposed * (
            grad_probability - delta_tile[None, :]
        )
        grad_key += tl.dot(
            grad_score.to(tl.float16),
            tl.trans(query_transposed),
        )

        current_query += block_m
        query_ptrs += block_m * query_sequence_stride
        grad_output_ptrs += block_m * grad_output_sequence_stride

    return grad_key, grad_value


@triton.jit
def _attention_backward_query(
    grad_query,
    query_tile,
    key,
    value,
    scale_log2,
    grad_output_tile,
    maximum,
    delta,
    key_sequence_stride,
    key_dim_stride,
    value_sequence_stride,
    value_dim_stride,
    sequence_length,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
    start_query,
    start_key,
    steps,
    masked: tl.constexpr,
    aligned: tl.constexpr,
):
    query_offsets = start_query + tl.arange(0, block_m)
    key_offsets = start_key + tl.arange(0, block_n)
    dim_offsets = tl.arange(0, head_dim)
    key_ptrs = (
        key
        + key_offsets[None, :] * key_sequence_stride
        + dim_offsets[:, None] * key_dim_stride
    )
    value_ptrs = (
        value
        + key_offsets[None, :] * value_sequence_stride
        + dim_offsets[:, None] * value_dim_stride
    )
    if aligned:
        delta_tile = tl.load(delta + query_offsets)
    else:
        query_valid = query_offsets < sequence_length
        delta_tile = tl.load(
            delta + query_offsets,
            mask=query_valid,
            other=0.0,
        )
    tl.static_assert(block_m % block_n == 0)

    current_key = start_key
    for _ in tl.range(0, steps):
        key_offsets = current_key + tl.arange(0, block_n)
        if aligned:
            key_transposed = tl.load(key_ptrs)
            value_transposed = tl.load(value_ptrs)
        else:
            key_valid = key_offsets < sequence_length
            key_transposed = tl.load(
                key_ptrs,
                mask=key_valid[None, :],
                other=0.0,
            )
            value_transposed = tl.load(
                value_ptrs,
                mask=key_valid[None, :],
                other=0.0,
            )
        logits = tl.dot(query_tile, key_transposed) * scale_log2
        probabilities = tl.math.exp2(logits - maximum)
        if masked:
            valid = query_offsets[:, None] >= key_offsets[None, :]
            if not aligned:
                valid &= query_valid[:, None] & key_valid[None, :]
            probabilities = tl.where(valid, probabilities, 0.0)
        elif not aligned:
            valid = query_valid[:, None] & key_valid[None, :]
            probabilities = tl.where(valid, probabilities, 0.0)

        grad_probability = tl.dot(
            grad_output_tile,
            value_transposed,
        ).to(tl.float32)
        grad_score = probabilities * (
            grad_probability - delta_tile[:, None]
        )
        grad_query += tl.dot(
            grad_score.to(tl.float16),
            tl.trans(key_transposed),
        )

        current_key += block_n
        key_ptrs += block_n * key_sequence_stride
        value_ptrs += block_n * value_sequence_stride

    return grad_query


@triton.jit(
    do_not_specialize=["sequence_length"],
    do_not_specialize_on_alignment=["sequence_length"],
)
def _attention_backward(
    query,
    key,
    value,
    scale,
    grad_output,
    grad_query,
    grad_key,
    grad_value,
    logsumexp,
    delta,
    stride_query_batch,
    stride_query_head,
    stride_query_sequence,
    stride_query_dim,
    stride_key_batch,
    stride_key_head,
    stride_key_sequence,
    stride_key_dim,
    stride_value_batch,
    stride_value_head,
    stride_value_sequence,
    stride_value_dim,
    stride_grad_output_batch,
    stride_grad_output_head,
    stride_grad_output_sequence,
    stride_grad_output_dim,
    stride_grad_query_batch,
    stride_grad_query_head,
    stride_grad_query_sequence,
    stride_grad_query_dim,
    stride_grad_key_batch,
    stride_grad_key_head,
    stride_grad_key_sequence,
    stride_grad_key_dim,
    stride_grad_value_batch,
    stride_grad_value_head,
    stride_grad_value_sequence,
    stride_grad_value_dim,
    head_count: tl.constexpr,
    sequence_length,
    block_m_key: tl.constexpr,
    block_n_key: tl.constexpr,
    block_m_query: tl.constexpr,
    block_n_query: tl.constexpr,
    slice_factor: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
    aligned: tl.constexpr,
):
    batch_head = tl.program_id(2)
    batch = batch_head // head_count
    head = batch_head % head_count
    stats_offset = batch_head * sequence_length
    block = tl.program_id(0)
    tl.static_assert(block_n_key == block_m_query)

    query += batch * stride_query_batch + head * stride_query_head
    key += batch * stride_key_batch + head * stride_key_head
    value += batch * stride_value_batch + head * stride_value_head
    grad_output += (
        batch * stride_grad_output_batch
        + head * stride_grad_output_head
    )
    grad_query += batch * stride_grad_query_batch + head * stride_grad_query_head
    grad_key += batch * stride_grad_key_batch + head * stride_grad_key_head
    grad_value += (
        batch * stride_grad_value_batch
        + head * stride_grad_value_head
    )
    logsumexp += stats_offset
    delta += stats_offset

    dim_offsets = tl.arange(0, head_dim)
    scale_log2 = scale * 1.4426950408889634

    start_key = block * block_n_key
    start_query = 0
    masked_block_m: tl.constexpr = block_m_key // slice_factor
    key_offsets = start_key + tl.arange(0, block_n_key)
    key_ptrs = (
        key
        + key_offsets[:, None] * stride_key_sequence
        + dim_offsets[None, :] * stride_key_dim
    )
    value_ptrs = (
        value
        + key_offsets[:, None] * stride_value_sequence
        + dim_offsets[None, :] * stride_value_dim
    )
    if aligned:
        key_tile = tl.load(key_ptrs)
        value_tile = tl.load(value_ptrs)
    else:
        key_valid = key_offsets < sequence_length
        key_tile = tl.load(
            key_ptrs,
            mask=key_valid[:, None],
            other=0.0,
        )
        value_tile = tl.load(
            value_ptrs,
            mask=key_valid[:, None],
            other=0.0,
        )
    grad_key_tile = tl.zeros([block_n_key, head_dim], tl.float32)
    grad_value_tile = tl.zeros([block_n_key, head_dim], tl.float32)

    if causal:
        start_query = start_key
        steps = block_n_key // masked_block_m
        grad_key_tile, grad_value_tile = _attention_backward_key_value(
            grad_key_tile,
            grad_value_tile,
            query,
            key_tile,
            value_tile,
            scale_log2,
            grad_output,
            logsumexp,
            delta,
            stride_query_sequence,
            stride_query_dim,
            stride_grad_output_sequence,
            stride_grad_output_dim,
            sequence_length,
            masked_block_m,
            block_n_key,
            head_dim,
            start_key,
            start_query,
            steps,
            True,
            aligned,
        )
        start_query += steps * masked_block_m

    steps = tl.cdiv(
        tl.maximum(sequence_length - start_query, 0),
        block_m_key,
    )
    grad_key_tile, grad_value_tile = _attention_backward_key_value(
        grad_key_tile,
        grad_value_tile,
        query,
        key_tile,
        value_tile,
        scale_log2,
        grad_output,
        logsumexp,
        delta,
        stride_query_sequence,
        stride_query_dim,
        stride_grad_output_sequence,
        stride_grad_output_dim,
        sequence_length,
        block_m_key,
        block_n_key,
        head_dim,
        start_key,
        start_query,
        steps,
        False,
        aligned,
    )
    grad_key_tile *= scale
    grad_key_ptrs = (
        grad_key
        + key_offsets[:, None] * stride_grad_key_sequence
        + dim_offsets[None, :] * stride_grad_key_dim
    )
    grad_value_ptrs = (
        grad_value
        + key_offsets[:, None] * stride_grad_value_sequence
        + dim_offsets[None, :] * stride_grad_value_dim
    )
    if aligned:
        tl.store(grad_key_ptrs, grad_key_tile)
        tl.store(grad_value_ptrs, grad_value_tile)
    else:
        key_valid = key_offsets < sequence_length
        tl.store(
            grad_key_ptrs,
            grad_key_tile,
            mask=key_valid[:, None],
        )
        tl.store(
            grad_value_ptrs,
            grad_value_tile,
            mask=key_valid[:, None],
        )

    start_query = block * block_m_query
    start_key = 0
    masked_block_n: tl.constexpr = block_n_query // slice_factor
    query_offsets = start_query + tl.arange(0, block_m_query)
    query_ptrs = (
        query
        + query_offsets[:, None] * stride_query_sequence
        + dim_offsets[None, :] * stride_query_dim
    )
    grad_output_ptrs = (
        grad_output
        + query_offsets[:, None] * stride_grad_output_sequence
        + dim_offsets[None, :] * stride_grad_output_dim
    )
    if aligned:
        query_tile = tl.load(query_ptrs)
        grad_output_tile = tl.load(grad_output_ptrs)
        maximum = tl.load(logsumexp + query_offsets)[:, None]
    else:
        query_valid = query_offsets < sequence_length
        query_tile = tl.load(
            query_ptrs,
            mask=query_valid[:, None],
            other=0.0,
        )
        grad_output_tile = tl.load(
            grad_output_ptrs,
            mask=query_valid[:, None],
            other=0.0,
        )
        maximum = tl.load(
            logsumexp + query_offsets,
            mask=query_valid,
            other=0.0,
        )[:, None]
    grad_query_tile = tl.zeros([block_m_query, head_dim], tl.float32)

    if causal:
        end_key = start_query + block_m_query
        steps = block_m_query // masked_block_n
        grad_query_tile = _attention_backward_query(
            grad_query_tile,
            query_tile,
            key,
            value,
            scale_log2,
            grad_output_tile,
            maximum,
            delta,
            stride_key_sequence,
            stride_key_dim,
            stride_value_sequence,
            stride_value_dim,
            sequence_length,
            block_m_query,
            masked_block_n,
            head_dim,
            start_query,
            end_key - steps * masked_block_n,
            steps,
            True,
            aligned,
        )
        end_key -= steps * masked_block_n
        steps = end_key // block_n_query
        start_key = end_key - steps * block_n_query
    else:
        steps = tl.cdiv(sequence_length, block_n_query)

    grad_query_tile = _attention_backward_query(
        grad_query_tile,
        query_tile,
        key,
        value,
        scale_log2,
        grad_output_tile,
        maximum,
        delta,
        stride_key_sequence,
        stride_key_dim,
        stride_value_sequence,
        stride_value_dim,
        sequence_length,
        block_m_query,
        block_n_query,
        head_dim,
        start_query,
        start_key,
        steps,
        False,
        aligned,
    )
    grad_query_tile *= scale
    grad_query_ptrs = (
        grad_query
        + query_offsets[:, None] * stride_grad_query_sequence
        + dim_offsets[None, :] * stride_grad_query_dim
    )
    if aligned:
        tl.store(grad_query_ptrs, grad_query_tile)
    else:
        tl.store(
            grad_query_ptrs,
            grad_query_tile,
            mask=query_offsets[:, None] < sequence_length,
        )


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal, scale):
        batch, heads, sequence_length, head_dim = query.shape
        needs_backward = any(ctx.needs_input_grad[:3])
        use_fused_backward = needs_backward and head_dim == 32
        output = torch.empty(
            query.shape,
            dtype=query.dtype,
            device=query.device,
        )
        if use_fused_backward:
            logsumexp = torch.empty(
                (batch, heads, sequence_length),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            logsumexp = torch.empty(1, device=query.device)
        if (
            head_dim == 128
            and not needs_backward
            and causal
            and batch * heads >= 8
            and sequence_length >= 1024
        ):
            block_m = 16
            block_n = 4
            warps = 2
            stages = 1
            waves_per_eu = 1
        elif head_dim == 128:
            block_m = 8
            block_n = 4
            warps = 1
            stages = 1
            waves_per_eu = 1
        elif (
            head_dim == 64
            and not needs_backward
            and causal
            and batch * heads >= 8
            and sequence_length >= 1024
        ):
            block_m = 32
            block_n = 4
            warps = 2
            stages = 2 if sequence_length < 4096 else 1
            waves_per_eu = 1
        elif head_dim == 64:
            block_m = 16
            block_n = 8
            warps = 1
            stages = 1
            waves_per_eu = 1
        elif not needs_backward and sequence_length >= 4096:
            block_m = 32
            block_n = 8
            warps = 1
            stages = 1
            waves_per_eu = 1
        elif (
            not needs_backward
            and batch * heads >= 64
            and sequence_length >= 1024
        ):
            block_m = 32
            block_n = 8
            warps = 1
            stages = 1
            waves_per_eu = 1
        elif (
            not needs_backward
            and batch * heads <= 8
            and sequence_length >= 1024
        ):
            block_m = 32
            block_n = 4
            warps = 1
            stages = 1
            waves_per_eu = 2
        elif not needs_backward and batch == 1 and sequence_length <= 96:
            block_m = 16
            block_n = 16
            warps = 2
            stages = 2
            waves_per_eu = 1
        elif not needs_backward and batch <= 8:
            block_m = 16
            block_n = 8
            warps = 1
            stages = 2
            waves_per_eu = 1
        else:
            block_m = 16
            block_n = 16
            warps = 1
            stages = 1
            waves_per_eu = 1 if sequence_length <= 64 else (
                3 if sequence_length <= 128 else 4
            )
        grid = (triton.cdiv(sequence_length, block_m), batch * heads)
        _attention_forward[grid](
            query,
            key,
            value,
            output,
            logsumexp,
            scale,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            head_count=heads,
            sequence_length=sequence_length,
            head_dim=head_dim,
            block_m=block_m,
            block_n=block_n,
            causal=causal,
            save_stats=use_fused_backward,
            num_warps=warps,
            num_stages=stages,
            waves_per_eu=waves_per_eu,
            allow_flush_denorm=True,
        )
        if needs_backward:
            if use_fused_backward:
                ctx.save_for_backward(
                    query,
                    key,
                    value,
                    output,
                    logsumexp,
                )
            else:
                ctx.save_for_backward(query, key, value)
            ctx.use_fused_backward = use_fused_backward
            ctx.input_grad_mask = ctx.needs_input_grad[:3]
            ctx.causal = causal
            ctx.scale = scale
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query = ctx.saved_tensors[0]
        with torch.cuda.device(query.device):
            return TritonAttention._backward_impl(ctx, grad_output)

    @staticmethod
    def _backward_impl(ctx, grad_output):
        if not ctx.use_fused_backward:
            return TritonAttention._backward_matmul(ctx, grad_output)
        query, key, value, output, logsumexp = ctx.saved_tensors
        grad_query = torch.empty(
            query.shape,
            dtype=query.dtype,
            device=query.device,
        )
        grad_key = torch.empty(
            key.shape,
            dtype=key.dtype,
            device=key.device,
        )
        grad_value = torch.empty(
            value.shape,
            dtype=value.dtype,
            device=value.device,
        )
        batch, heads, sequence_length, head_dim = query.shape
        preprocess_block = 16
        use_long_context_config = (
            sequence_length >= 4096
            or (
                batch * heads >= 64
                and sequence_length >= 1024
            )
        )
        block_m_key = 8
        block_n_key = 32 if use_long_context_config else 16
        block_m_query = 32 if use_long_context_config else 16
        block_n_query = 8
        warps = 2 if use_long_context_config else 1
        waves_per_eu = (
            1
            if use_long_context_config
            else (
                1 if sequence_length <= 64 else (
                    3 if sequence_length <= 128 else 4
                )
            )
        )
        delta = torch.empty_like(logsumexp)

        preprocess_grid = (
            triton.cdiv(sequence_length, preprocess_block),
            batch * heads,
        )
        _attention_backward_preprocess[preprocess_grid](
            output,
            grad_output,
            delta,
            *output.stride(),
            *grad_output.stride(),
            head_count=heads,
            sequence_length=sequence_length,
            head_dim=head_dim,
            block_m=preprocess_block,
            aligned=sequence_length % preprocess_block == 0,
            num_warps=1,
            num_stages=1,
            waves_per_eu=waves_per_eu,
            allow_flush_denorm=True,
        )
        grid = (
            triton.cdiv(sequence_length, block_n_key),
            1,
            batch * heads,
        )
        _attention_backward[grid](
            query,
            key,
            value,
            ctx.scale,
            grad_output,
            grad_query,
            grad_key,
            grad_value,
            logsumexp,
            delta,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *grad_query.stride(),
            *grad_key.stride(),
            *grad_value.stride(),
            head_count=heads,
            sequence_length=sequence_length,
            block_m_key=block_m_key,
            block_n_key=block_n_key,
            block_m_query=block_m_query,
            block_n_query=block_n_query,
            slice_factor=2,
            head_dim=head_dim,
            causal=ctx.causal,
            aligned=sequence_length % block_n_key == 0,
            num_warps=warps,
            num_stages=1,
            waves_per_eu=waves_per_eu,
            allow_flush_denorm=True,
        )
        return grad_query, grad_key, grad_value, None, None

    @staticmethod
    def _backward_matmul(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        sequence_length = query.shape[-2]
        probabilities = torch.matmul(
            query,
            key.transpose(-2, -1),
        )
        probability_block = triton.next_power_of_2(sequence_length)
        probability_rows = probabilities.numel() // sequence_length
        probability_warps = (
            1
            if (
                probability_block <= 64
                or probability_block in (256, 1024)
            )
            else (4 if probability_block <= 1024 else 8)
        )
        _attention_probabilities[(probability_rows,)](
            probabilities,
            ctx.scale,
            sequence_length,
            block_n=probability_block,
            causal=ctx.causal,
            num_warps=probability_warps,
            num_stages=1,
            waves_per_eu=1,
            allow_flush_denorm=True,
        )
        need_query, need_key, need_value = ctx.input_grad_mask
        grad_value = None
        if need_value:
            grad_value = torch.matmul(
                probabilities.transpose(-2, -1),
                grad_output,
            )
        grad_query = None
        grad_key = None
        if need_query or need_key:
            grad_score = torch.matmul(
                grad_output,
                value.transpose(-2, -1),
            )
            _attention_backward_score[(probability_rows,)](
                probabilities,
                grad_score,
                sequence_length,
                block_n=probability_block,
                num_warps=probability_warps,
                num_stages=1,
                waves_per_eu=1,
                allow_flush_denorm=True,
            )
            if need_query:
                grad_query = torch.matmul(
                    grad_score,
                    key,
                ).mul_(ctx.scale)
            if need_key:
                grad_key = torch.matmul(
                    grad_score.transpose(-2, -1),
                    query,
                ).mul_(ctx.scale)
        return grad_query, grad_key, grad_value, None, None


triton_attention = TritonAttention.apply
