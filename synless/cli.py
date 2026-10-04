import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select synthetic contrastive negatives using LESS-style influence")
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .experiment import run, summarize
    if args.command == "summarize":
        summarize(args.output)
    else:
        if args.config is None:
            parser.error("run requires --config")
        run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
