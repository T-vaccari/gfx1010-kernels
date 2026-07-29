import argparse
import json

import torch

from .functional import (
    SUPPORTED_HEAD_DIMS,
    backend_status,
    scaled_dot_product_attention,
)


def compile_inference(batch, heads, head_dim, sequence_length, causal):
    tensors = [
        torch.randn(
            batch,
            heads,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    with torch.no_grad():
        scaled_dot_product_attention(
            *tensors,
            is_causal=causal,
            implementation="gfx1010",
        )


def compile_training(
    batch,
    heads,
    head_dim,
    sequence_length,
    causal,
):
    tensors = [
        torch.randn(
            batch,
            heads,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    output = scaled_dot_product_attention(
        *tensors,
        is_causal=causal,
        implementation="gfx1010",
    )
    output.sum().backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head-dim",
        type=int,
        nargs="+",
        default=list(SUPPORTED_HEAD_DIMS),
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[63, 64, 95, 96, 127, 128, 191, 192],
    )
    parser.add_argument(
        "--long-sequence",
        type=int,
        nargs="*",
        default=[1024, 4096],
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()
    status = backend_status()
    if not status.available:
        raise SystemExit(status.reason)

    for head_dim in args.head_dim:
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise SystemExit(f"unsupported head_dim: {head_dim}")
        for sequence_length in args.sequence:
            for causal in (False, True):
                for batch in args.batch:
                    compile_inference(
                        batch,
                        args.heads,
                        head_dim,
                        sequence_length,
                        causal,
                    )
                compile_training(
                    1,
                    args.heads,
                    head_dim,
                    sequence_length,
                    causal,
                )
                torch.cuda.synchronize()
                print(
                    json.dumps(
                        {
                            "inference_batches": args.batch,
                            "heads": args.heads,
                            "head_dim": head_dim,
                            "sequence": sequence_length,
                            "causal": causal,
                            "compiled": True,
                        }
                    ),
                    flush=True,
                )
        for sequence_length in args.long_sequence:
            for causal in (False, True):
                inference_batches = [1]
                training_batches = [1]
                if head_dim == 32 and sequence_length < 4096:
                    inference_batches.append(8)
                    training_batches.append(8)
                for batch in inference_batches:
                    compile_inference(
                        batch,
                        args.heads,
                        head_dim,
                        sequence_length,
                        causal,
                    )
                for batch in training_batches:
                    compile_training(
                        batch,
                        args.heads,
                        head_dim,
                        sequence_length,
                        causal,
                    )
                torch.cuda.synchronize()
                print(
                    json.dumps(
                        {
                            "inference_batches": inference_batches,
                            "training_batches": training_batches,
                            "heads": args.heads,
                            "head_dim": head_dim,
                            "sequence": sequence_length,
                            "causal": causal,
                            "compiled": True,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
