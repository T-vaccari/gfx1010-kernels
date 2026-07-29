import argparse
import json
import statistics

import torch

from gfx1010_kernels import _C


def benchmark(call, warmup, reps):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=1)
    parser.add_argument("-T", "--sequence", type=int, default=1024)
    parser.add_argument("-C", "--hidden", type=int, default=384)
    parser.add_argument("-p", "--dropout", type=float, default=0.15)
    parser.add_argument("--input-threads", type=int, default=128)
    parser.add_argument(
        "--parameter-threads",
        type=int,
        nargs="+",
        default=(64, 128, 256),
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        nargs="+",
        default=(16, 32, 64, 128),
    )
    parser.add_argument(
        "--parameter-groups",
        type=int,
        nargs="+",
        default=(12,),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--disable-deterministic-fill",
        action="store_true",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(args.deterministic)
    if args.disable_deterministic_fill:
        torch.utils.deterministic.fill_uninitialized_memory = False

    x = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.float32,
    )
    branch = torch.randn_like(x, dtype=torch.float16)
    weight = torch.randn(args.hidden, device="cuda")
    bias = torch.randn(args.hidden, device="cuda")

    threads = args.input_threads
    for parameter_threads in args.parameter_threads:
        forward = lambda: _C.residual_layer_norm_forward(
            x,
            branch,
            weight,
            bias,
            args.dropout,
            1e-5,
            True,
            threads,
        )
        outputs = forward()
        grad_updated = torch.randn_like(x)
        grad_normalized = torch.randn_like(x)
        backward_inputs = lambda: _C.residual_layer_norm_backward(
            grad_updated,
            grad_normalized,
            outputs[0],
            weight,
            outputs[2],
            outputs[3],
            outputs[4],
            args.dropout,
            True,
            True,
            threads,
            args.block_rows[0],
            parameter_threads,
            args.parameter_groups[0],
            False,
        )
        input_backward_ms = benchmark(
            backward_inputs,
            args.warmup,
            args.reps,
        )
        zeros_ms = benchmark(
            lambda: torch.zeros(
                2,
                args.hidden,
                device="cuda",
                dtype=torch.float32,
            ),
            args.warmup,
            args.reps,
        )
        for parameter_groups in args.parameter_groups:
            for block_rows in args.block_rows:
                backward = lambda: _C.residual_layer_norm_backward(
                    grad_updated,
                    grad_normalized,
                    outputs[0],
                    weight,
                    outputs[2],
                    outputs[3],
                    outputs[4],
                    args.dropout,
                    True,
                    True,
                    threads,
                    block_rows,
                    parameter_threads,
                    parameter_groups,
                    True,
                )
                print(
                    json.dumps(
                        {
                            "C": args.hidden,
                            "dropout_p": args.dropout,
                            "threads": threads,
                            "parameter_threads": parameter_threads,
                            "block_rows": block_rows,
                            "parameter_groups": parameter_groups,
                            "deterministic": args.deterministic,
                            "forward_ms": benchmark(
                                forward,
                                args.warmup,
                                args.reps,
                            ),
                            "backward_ms": benchmark(
                                backward,
                                args.warmup,
                                args.reps,
                            ),
                            "input_backward_ms": input_backward_ms,
                            "zeros_ms": zeros_ms,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
