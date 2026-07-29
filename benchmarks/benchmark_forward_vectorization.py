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
        default=(128, 256, 512, 1024, 2048, 4096),
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=(128, 256, 384, 512, 768, 1024),
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    for sequence in args.sequences:
        for hidden in args.hidden:
            x_storage = torch.randn(
                sequence * hidden + 1,
                device="cuda",
                dtype=torch.float32,
            )
            scalar_x = x_storage[1:].view(sequence, hidden)
            vector_x = scalar_x.clone()
            branch = torch.randn_like(vector_x, dtype=torch.float16)
            weight = torch.randn(hidden, device="cuda")
            bias = torch.randn(hidden, device="cuda")
            assert scalar_x.data_ptr() % 16 != 0
            assert vector_x.data_ptr() % 16 == 0

            def forward(x):
                return _C.residual_layer_norm_forward(
                    x,
                    branch,
                    weight,
                    bias,
                    0.0,
                    1e-5,
                    True,
                    THREADS[hidden],
                )

            vector_samples = []
            scalar_samples = []
            for round_index in range(args.rounds):
                calls = (
                    (
                        ("vector", lambda: forward(vector_x)),
                        ("scalar", lambda: forward(scalar_x)),
                    )
                    if round_index % 2 == 0
                    else (
                        ("scalar", lambda: forward(scalar_x)),
                        ("vector", lambda: forward(vector_x)),
                    )
                )
                for name, call in calls:
                    sample = benchmark(call, args.warmup, args.reps)
                    if name == "vector":
                        vector_samples.append(sample)
                    else:
                        scalar_samples.append(sample)

            vector_ms = statistics.median(vector_samples)
            scalar_ms = statistics.median(scalar_samples)
            print(
                json.dumps(
                    {
                        "T": sequence,
                        "C": hidden,
                        "vector_ms": vector_ms,
                        "scalar_ms": scalar_ms,
                        "speedup": scalar_ms / vector_ms,
                        "vector_rounds_ms": vector_samples,
                        "scalar_rounds_ms": scalar_samples,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
