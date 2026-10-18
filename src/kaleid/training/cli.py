"""Train the one-shot algorithm with explicit offline observation/pose pairs."""

import argparse
import json
from pathlib import Path
import torch
from .runner import train


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kaleid-train", description=__doc__)
    parser.add_argument("system", choices=("nikon", "trepan", "olympus"))
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data", type=Path, help="Dataset manifest JSON")
    data.add_argument(
        "--smoke",
        action="store_true",
        help="Small synthetic-interface training check, not an accuracy benchmark",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--micro-batch",
        type=int,
        help="Gradient-accumulation chunk size; lower this for limited memory",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--init",
        help="pretrained, or a Kaleid deployment checkpoint; omit for random parameters",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause after this many additional optimizer updates",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--validation-every", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    manifest = args.data
    if args.smoke:
        from .smoke import make_smoke_data

        manifest = make_smoke_data(args.system, args.output / "smoke-data")
        if args.batch_size is None:
            args.batch_size = 2
        if args.max_steps is None:
            args.max_steps = 1
        print(
            "SMOKE: sensor-shaped packets and arbitrary pose labels; tests execution only.",
            flush=True,
        )
    try:
        result = train(
            args.system,
            manifest,
            args.output,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            micro_batch=args.micro_batch,
            seed=args.seed,
            calibration=args.calibration,
            init=args.init,
            resume=args.resume,
            max_steps=args.max_steps,
            save_every=args.save_every,
            validation_every=args.validation_every,
        )
        print(json.dumps(result, allow_nan=False), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"kaleid-train: {exc}\n")
