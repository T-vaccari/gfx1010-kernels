import argparse
import itertools
import json
import statistics

import torch

from gfx1010_kernels import _C


INPUT_THREADS = {
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
    events = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in events
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-T",
        "--sequence",
        type=int,
        nargs="+",
        default=(256, 512, 1024, 2048, 4096),
    )
    parser.add_argument(
        "-C",
        "--hidden",
        type=int,
        nargs="+",
        default=(128, 256, 384, 512, 768, 1024),
    )
    parser.add_argument(
        "--parameter-threads",
        type=int,
        nargs="+",
        default=(64, 128, 256),
    )
    parser.add_argument(
        "--parameter-groups",
        type=int,
        nargs="+",
        default=(16, 24, 32, 48, 64, 80, 96),
    )
    parser.add_argument("-p", "--dropout", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    for sequence in args.sequence:
        for hidden in args.hidden:
            torch.manual_seed(sequence * 1024 + hidden)
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
                INPUT_THREADS[hidden],
            )
            grad_updated = torch.randn_like(x)
            grad_normalized = torch.randn_like(x)
            configs = list(
                itertools.product(
                    args.parameter_threads,
                    (
                        group
                        for group in args.parameter_groups
                        if group <= sequence
                    ),
                )
            )
            samples = {config: [] for config in configs}

            for round_index in range(args.rounds):
                order = configs if round_index % 2 == 0 else configs[::-1]
                for parameter_threads, parameter_groups in order:
                    call = lambda: _C.residual_layer_norm_backward(
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
                        INPUT_THREADS[hidden],
                        32,
                        parameter_threads,
                        parameter_groups,
                        True,
                    )
                    samples[
                        parameter_threads,
                        parameter_groups,
                    ].append(benchmark(call, args.warmup, args.reps))

            rows = []
            for parameter_threads, parameter_groups in configs:
                round_samples = samples[parameter_threads, parameter_groups]
                rows.append(
                    {
                        "T": sequence,
                        "C": hidden,
                        "parameter_threads": parameter_threads,
                        "parameter_groups": parameter_groups,
                        "median_ms": statistics.median(round_samples),
                        "rounds_ms": round_samples,
                    }
                )

            if args.best_only:
                best_by_threads = {}
                for parameter_threads in args.parameter_threads:
                    candidates = [
                        row
                        for row in rows
                        if row["parameter_threads"] == parameter_threads
                    ]
                    if candidates:
                        best_by_threads[str(parameter_threads)] = min(
                            candidates,
                            key=lambda row: row["median_ms"],
                        )
                print(
                    json.dumps(
                        {
                            "T": sequence,
                            "C": hidden,
                            "best": min(
                                rows,
                                key=lambda row: row["median_ms"],
                            ),
                            "by_parameter_threads": best_by_threads,
                        }
                    ),
                    flush=True,
                )
            else:
                for row in rows:
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
