import argparse
import json
import math

import torch
from torch.nn import functional as F

from gfx1010_kernels import residual_layer_norm, residual_layer_norm_status


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


def benchmark(call, warmup, reps):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    events = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def make_inputs(batch, sequence, hidden):
    x = torch.randn(
        batch,
        sequence,
        hidden,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    branch = torch.randn(
        batch,
        sequence,
        hidden,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    weight = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.randn(
        hidden,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    return x, branch, weight, bias


def maximum_absolute_difference(actual, expected):
    return max(
        (actual_value.float() - expected_value.float()).abs().max().item()
        for actual_value, expected_value in zip(actual, expected)
    )


def numerical_check(batch, sequence, hidden, dropout_p, check_backward):
    torch.manual_seed(17)
    source = make_inputs(batch, sequence, hidden)
    torch_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source
    )
    gfx_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in source
    )
    torch_outputs = residual_layer_norm(
        *torch_inputs,
        dropout_p=0.0,
        training=True,
        implementation="torch",
    )
    gfx_outputs = residual_layer_norm(
        *gfx_inputs,
        dropout_p=0.0,
        training=True,
        implementation="gfx1010",
    )
    torch.testing.assert_close(
        gfx_outputs[0],
        torch_outputs[0],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        gfx_outputs[1],
        torch_outputs[1],
        rtol=3e-4,
        atol=3e-4,
    )
    result = {
        "kind": "check",
        "dropout_p": 0.0,
        "forward_max_abs": maximum_absolute_difference(
            gfx_outputs,
            torch_outputs,
        ),
    }
    if check_backward:
        grad_outputs = tuple(torch.randn_like(output) for output in gfx_outputs)
        torch_gradients = torch.autograd.grad(
            torch_outputs,
            torch_inputs,
            grad_outputs,
        )
        gfx_gradients = torch.autograd.grad(
            gfx_outputs,
            gfx_inputs,
            grad_outputs,
        )
        for actual, expected in zip(gfx_gradients, torch_gradients):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2e-3,
                atol=2e-3,
            )
        result["backward_max_abs"] = maximum_absolute_difference(
            gfx_gradients,
            torch_gradients,
        )

    if dropout_p > 0.0:
        dropout_result = {}
        for implementation in ("torch", "gfx1010"):
            x = torch.randn(
                batch,
                sequence,
                hidden,
                device="cuda",
                dtype=torch.float32,
            )
            branch = torch.ones(
                batch,
                sequence,
                hidden,
                device="cuda",
                dtype=torch.float16,
            )
            weight = torch.randn(hidden, device="cuda")
            bias = torch.randn(hidden, device="cuda")
            updated, normalized = residual_layer_norm(
                x,
                branch,
                weight,
                bias,
                dropout_p=dropout_p,
                training=True,
                implementation=implementation,
            )
            scales = updated - x
            expected_scales = torch.tensor(
                [0.0, 1.0 / (1.0 - dropout_p)],
                device="cuda",
            )
            distances = (
                scales[..., None] - expected_scales
            ).abs().amin(dim=-1)
            if distances.max().item() > 2e-3:
                raise AssertionError(
                    f"{implementation} dropout scaling check failed"
                )
            expected_normalized = F.layer_norm(
                updated,
                (hidden,),
                weight,
                bias,
            )
            torch.testing.assert_close(
                normalized,
                expected_normalized,
                rtol=3e-4,
                atol=3e-4,
            )
            dropout_result[implementation] = {
                "keep_ratio": (scales != 0.0).float().mean().item(),
                "normalization_max_abs": (
                    normalized - expected_normalized
                ).abs().max().item(),
            }
        result["dropout"] = dropout_result
        result["requested_dropout_p"] = dropout_p

    torch.cuda.synchronize()
    return result


def measure(
    mode,
    inputs,
    grad_outputs,
    dropout_p,
    warmup,
    reps,
):
    results = {}
    for implementation in ("torch", "gfx1010"):
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

        results[implementation] = summarize(
            benchmark(call, warmup, reps)
        )
    return {
        "kind": "benchmark",
        "mode": mode,
        "torch": results["torch"],
        "gfx1010": results["gfx1010"],
        "speedup": (
            results["torch"]["median_ms"]
            / results["gfx1010"]["median_ms"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused residual LayerNorm against PyTorch",
    )
    parser.add_argument("-B", "--batch", type=int, default=1)
    parser.add_argument("-T", "--sequence", type=int, default=1024)
    parser.add_argument("-C", "--hidden", type=int, default=384)
    parser.add_argument("-p", "--dropout", type=float, default=0.15)
    parser.add_argument(
        "--mode",
        choices=("forward", "train", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()
    if args.batch <= 0 or args.sequence <= 0 or args.hidden <= 0:
        parser.error("B, T and C must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must be in [0, 1)")
    if args.warmup < 0 or args.reps <= 0:
        parser.error("warmup must be non-negative and reps must be positive")

    status = residual_layer_norm_status()
    if not status.available:
        raise SystemExit(status.reason)

    print(
        json.dumps(
            {
                "kind": "config",
                "B": args.batch,
                "T": args.sequence,
                "C": args.hidden,
                "dropout_p": args.dropout,
                "warmup": args.warmup,
                "reps": args.reps,
                "timing": "CUDA events",
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            numerical_check(
                args.batch,
                args.sequence,
                args.hidden,
                args.dropout,
                args.mode in {"train", "all"},
            )
        ),
        flush=True,
    )

    torch.manual_seed(31)
    inputs = make_inputs(args.batch, args.sequence, args.hidden)
    grad_outputs = (
        torch.randn_like(inputs[0]),
        torch.randn_like(inputs[0]),
    )
    modes = ("forward", "train") if args.mode == "all" else (args.mode,)
    for mode in modes:
        row = measure(
            mode,
            inputs,
            grad_outputs,
            args.dropout,
            args.warmup,
            args.reps,
        )
        row.update(
            {
                "B": args.batch,
                "T": args.sequence,
                "C": args.hidden,
                "dropout_p": args.dropout,
            }
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
