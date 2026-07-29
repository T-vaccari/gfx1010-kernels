import argparse
import json

import torch

from gfx1010_kernels.attention import (
    TORCH_SCALED_DOT_PRODUCT_ATTENTION,
)


def benchmark(callable, warmup, iterations):
    for _ in range(warmup):
        callable()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


@torch.no_grad()
def backward_formula(query, key, value, grad_output, scale):
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    sequence_length = query.shape[-2]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=query.device,
    ).tril()
    scores.masked_fill_(~causal, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    grad_value = torch.matmul(
        probabilities.transpose(-2, -1),
        grad_output,
    )
    grad_probability = torch.matmul(
        grad_output,
        value.transpose(-2, -1),
    )
    delta = (
        grad_probability * probabilities
    ).sum(
        dim=-1,
        keepdim=True,
    )
    grad_score = grad_probability.sub_(delta).mul_(probabilities)
    grad_query = torch.matmul(grad_score, key) * scale
    grad_key = torch.matmul(
        grad_score.transpose(-2, -1),
        query,
    ) * scale
    return grad_query, grad_key, grad_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, nargs="+", default=[64, 192])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    for sequence_length in args.sequence:
        shape = (
            args.batch,
            args.heads,
            sequence_length,
            args.head_dim,
        )
        tensors = [
            torch.randn(
                shape,
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        grad_output = torch.randn_like(tensors[0])
        scale = args.head_dim**-0.5
        reference_output = TORCH_SCALED_DOT_PRODUCT_ATTENTION(
            *tensors,
            is_causal=True,
        )

        def torch_backward():
            return torch.autograd.grad(
                reference_output,
                tensors,
                grad_output,
                retain_graph=True,
            )

        formula_result = backward_formula(
            *tensors,
            grad_output,
            scale,
        )
        torch_result = torch_backward()
        maximum_error = max(
            (actual - expected).abs().max().item()
            for actual, expected in zip(formula_result, torch_result)
        )
        row = {
            "batch": args.batch,
            "heads": args.heads,
            "sequence": sequence_length,
            "head_dim": args.head_dim,
            "formula_backward_ms": benchmark(
                lambda: backward_formula(
                    *tensors,
                    grad_output,
                    scale,
                ),
                args.warmup,
                args.iterations,
            ),
            "torch_backward_ms": benchmark(
                torch_backward,
                args.warmup,
                args.iterations,
            ),
            "maximum_error": maximum_error,
        }
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
