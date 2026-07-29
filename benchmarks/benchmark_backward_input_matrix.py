import argparse
import json
import statistics

import torch

from gfx1010_kernels import _C


THREADS = {
    128: 128,
    256: 128,
    384: 128,
    512: 128,
    768: 128,
    1024: 192,
}


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
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=(256, 1024, 4096),
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=(128, 256, 384, 512, 768, 1024),
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    for sequence in args.sequences:
        for hidden in args.hidden:
            x = torch.randn(
                1,
                sequence,
                hidden,
                device="cuda",
                dtype=torch.float32,
            )
            branch = torch.randn_like(x, dtype=torch.float16)
            weight = torch.randn(hidden, device="cuda")
            bias = torch.randn(hidden, device="cuda")
            outputs = _C.residual_layer_norm_forward(
                x,
                branch,
                weight,
                bias,
                args.dropout,
                1e-5,
                True,
                THREADS[hidden],
            )
            grad_updated = torch.randn_like(x)
            grad_normalized = torch.randn_like(x)

            def backward_input():
                return _C.residual_layer_norm_backward(
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
                    THREADS[hidden],
                    16,
                    64,
                    1,
                    False,
                )

            samples = [
                benchmark(
                    backward_input,
                    args.warmup,
                    args.reps,
                )
                for _ in range(args.rounds)
            ]
            print(
                json.dumps(
                    {
                        "T": sequence,
                        "C": hidden,
                        "dropout_p": args.dropout,
                        "threads": THREADS[hidden],
                        "input_backward_ms": statistics.median(samples),
                        "rounds_ms": samples,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
