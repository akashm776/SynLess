import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Separate LLM generator pilot; preflight -> run (D only) -> report (test)")
    parser.add_argument("command", choices=["preflight", "run", "report"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-units", type=int, help="Pause after this many updates/evaluations; useful for resume testing")
    args = parser.parse_args()
    if args.max_units is not None and args.max_units < 1:
        parser.error("max-units must be positive")
    from .experiment import execute, Paused
    try:
        execute(args.command, json.loads(args.config.read_text()), args.output, args.max_units)
    except Paused as exc:
        print(str(exc), flush=True)


if __name__ == "__main__":
    main()
