import argparse
import json

import torch

from gfx1010_kernels import (
    backend_status,
    scaled_dot_product_attention,
)


def benchmark(callable, warmup=25, iterations=100):
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


def measure_training(
    batch,
    heads,
    sequence,
    head_dim,
    warmup,
    iterations,
):
    tensors = [
        torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    grad = torch.randn_like(tensors[0])
    rows = []
    for implementation in ("gfx1010", "torch"):
        def forward():
            return scaled_dot_product_attention(
                *tensors,
                is_causal=True,
                implementation=implementation,
            )

        output = forward()

        def backward():
            torch.autograd.grad(
                output,
                tensors,
                grad,
                retain_graph=True,
            )

        forward_ms = benchmark(forward, warmup, iterations)
        backward_ms = benchmark(backward, warmup, iterations)
        rows.append(
            {
                "mode": "training",
                "implementation": implementation,
                "batch": batch,
                "heads": heads,
                "sequence": sequence,
                "head_dim": head_dim,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "total_ms": forward_ms + backward_ms,
            }
        )
    return rows


def measure_inference(
    batch,
    heads,
    sequence,
    head_dim,
    warmup,
    iterations,
):
    tensors = [
        torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    rows = []
    for implementation in ("gfx1010", "torch"):
        @torch.no_grad()
        def forward():
            return scaled_dot_product_attention(
                *tensors,
                is_causal=True,
                implementation=implementation,
            )

        forward_ms = benchmark(forward, warmup, iterations)
        rows.append(
            {
                "mode": "inference",
                "implementation": implementation,
                "batch": batch,
                "heads": heads,
                "sequence": sequence,
                "head_dim": head_dim,
                "forward_ms": forward_ms,
                "backward_ms": None,
                "total_ms": forward_ms,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gfx1010 attention against PyTorch SDPA",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[32, 55, 64, 74, 96, 105, 128, 192],
    )
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument(
        "--mode",
        choices=("inference", "training", "both"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    status = backend_status()
    if not status.available:
        raise SystemExit(status.reason)
    for sequence in args.sequence:
        if args.mode in {"inference", "both"}:
            for row in measure_inference(
                args.batch,
                args.heads,
                sequence,
                args.head_dim,
                args.warmup,
                args.iterations,
            ):
                print(json.dumps(row), flush=True)
        if args.mode in {"training", "both"}:
            for row in measure_training(
                args.batch,
                args.heads,
                sequence,
                args.head_dim,
                args.warmup,
                args.iterations,
            ):
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
