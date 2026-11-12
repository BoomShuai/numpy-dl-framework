import argparse
import json
from pathlib import Path
import sys
from .checks import gradient_checks
from .demo import train_demo, benchmark


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="NumPy explicit-backprop teaching framework")
    parser.add_argument("command", choices=("check", "train", "benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--checkpoint", type=Path, help="Create a new .npz file after training")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report = gradient_checks(args.seed)
        elif args.command == "benchmark":
            report = benchmark(args.seed)
        else:
            model, report = train_demo(args.epochs, args.seed)
            if args.checkpoint:
                model.save(args.checkpoint)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, AssertionError):
        print("Operation failed: check arguments, numeric checks and output destination.", file=sys.stderr)
        return 2
