"""Small file-based interface to the one-shot predictor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .predictor import Kaleid, available_systems, system_info


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kaleid", description="One observation packet. One joint correction.")
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("list", help="List sensor and actuator contracts")
    for name in ("predict", "demo"):
        command = commands.add_parser(name)
        command.add_argument("system", choices=available_systems())
        if name == "predict":
            command.add_argument("input", type=Path, help="NPY array or NPZ with an observations array")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
        command.add_argument("--threads", type=int, default=4, help="CPU worker threads")
    args = parser.parse_args(argv)
    if args.operation == "list":
        print(json.dumps({name: system_info(name) for name in available_systems()}, indent=2))
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        model = Kaleid(args.system, device=args.device)
        if args.operation == "demo":
            observations = model.demo_observation()
        elif args.input.suffix.lower() == ".npz":
            with np.load(args.input, allow_pickle=False) as packet:
                observations = packet["observations"].copy()
        else:
            observations = np.load(args.input, allow_pickle=False)
        if args.operation == "predict" and args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        result = model.predict(observations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Opening the stream preserves the exact output filename.
        with args.output.open("wb") as stream:
            np.savez_compressed(stream, command_normalized=result,
                                command_physical=model.to_physical(result), system=args.system,
                                coordinates=model.info["coordinates"], units=model.info["physical_units"])
        print(json.dumps({"system": args.system, "shape": list(result.shape),
                          "model_epoch": model.epoch, "output": str(args.output),
                          "input_kind": "synthetic interface demo" if args.operation == "demo" else "observations"}))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"kaleid: {exc}\n")
