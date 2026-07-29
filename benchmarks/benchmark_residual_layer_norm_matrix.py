import argparse
import json
import math

import torch

from gfx1010_kernels import (
    residual_layer_norm,
    residual_layer_norm_status,
)


HIDDEN_SIZES = (128, 256, 384, 512, 768, 1024)
SEED = 29_072_026
DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def emit(row):
    print(
        json.dumps(
            row,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples):
    return {
        "median_ms": quantile(samples, 0.5),
        "p20_ms": quantile(samples, 0.2),
        "p80_ms": quantile(samples, 0.8),
    }


def benchmark(call, warmup, reps, seed):
    torch.manual_seed(seed)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    torch.manual_seed(seed)
    event_pairs = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        event_pairs.append((start, end))

    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in event_pairs]


def make_inputs(batch, sequence, hidden, branch_dtype):
    x = torch.randn(
        batch,
        sequence,
        hidden,
        device="cuda",
        dtype=torch.float32,
    )
    branch = torch.randn(
        batch,
        sequence,
        hidden,
        device="cuda",
        dtype=branch_dtype,
    )
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden, device="cuda", dtype=torch.float32)
    return x, branch, weight, bias


def measure(
    implementation,
    mode,
    source_inputs,
    grad_outputs,
    dropout_p,
    warmup,
    reps,
    seed,
):
    inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in source_inputs
    )

    if mode == "forward":

        def call():
            return residual_layer_norm(
                *inputs,
                dropout_p=dropout_p,
                training=True,
                implementation=implementation,
            )

    else:

        def call():
            outputs = residual_layer_norm(
                *inputs,
                dropout_p=dropout_p,
                training=True,
                implementation=implementation,
            )
            return torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs,
            )

    return summarize(benchmark(call, warmup, reps, seed))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end residual LayerNorm matrix for gfx1010 and PyTorch"
        ),
    )
    parser.add_argument("-B", "--batch", type=int, default=1)
    parser.add_argument("-T", "--sequence", type=int, default=1024)
    parser.add_argument("-p", "--dropout", type=float, default=0.15)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="float16",
        help="dtype of the residual branch",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=500)
    args = parser.parse_args()

    if args.batch <= 0 or args.sequence <= 0:
        parser.error("batch and sequence must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if args.warmup < 0 or args.reps <= 0:
        parser.error("warmup must be non-negative and reps must be positive")

    status = residual_layer_norm_status()
    if not status.available:
        raise SystemExit(status.reason)

    emit(
        {
            "B": args.batch,
            "T": args.sequence,
            "architecture": status.architecture,
            "branch_dtype": args.dtype,
            "device": status.device,
            "dropout_p": args.dropout,
            "hidden_sizes": HIDDEN_SIZES,
            "kind": "config",
            "reps": args.reps,
            "seed": SEED,
            "timing": "CUDA events",
            "warmup": args.warmup,
            "x_weight_bias_dtype": "float32",
        }
    )

    for hidden in HIDDEN_SIZES:
        input_seed = SEED + hidden
        torch.manual_seed(input_seed)
        source_inputs = make_inputs(
            args.batch,
            args.sequence,
            hidden,
            DTYPES[args.dtype],
        )
        grad_outputs = (
            torch.randn_like(source_inputs[0]),
            torch.randn_like(source_inputs[0]),
        )

        for mode in ("forward", "train"):
            results = {}
            measurement_seed = input_seed + (0 if mode == "forward" else 1)
            for implementation in ("torch", "gfx1010"):
                results[implementation] = measure(
                    implementation,
                    mode,
                    source_inputs,
                    grad_outputs,
                    args.dropout,
                    args.warmup,
                    args.reps,
                    measurement_seed,
                )
            emit(
                {
                    "B": args.batch,
                    "C": hidden,
                    "T": args.sequence,
                    "branch_dtype": args.dtype,
                    "dropout_p": args.dropout,
                    "gfx1010": results["gfx1010"],
                    "kind": "benchmark",
                    "mode": mode,
                    "speedup": (
                        results["torch"]["median_ms"]
                        / results["gfx1010"]["median_ms"]
                    ),
                    "torch": results["torch"],
                }
            )


if __name__ == "__main__":
    main()
