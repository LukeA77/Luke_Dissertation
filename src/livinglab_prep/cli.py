"""Command-line entry point (README.md §13 Makefile-style targets).

Usage:
    python -m src.livinglab_prep.cli <target> [--config PATH]

Targets:
    config     validate + print the config (§6)
    preflight  Stage 0 task-duration report (blocks if window unsafe)
    run        Stages 1-9 per recording + reports (§8/§11)
    validate   Stage 12 mechanical checks on the outputs (§10)
    all        preflight -> run -> validate, halting on first failure
"""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_pipeline
from .preflight import run_preflight
from .validate import run_validation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="livinglab_prep")
    parser.add_argument("target",
                        choices=["config", "preflight", "run", "validate", "all"])
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"[cli] config_hash={cfg.config_hash} git_sha={cfg.git_sha}")

    if args.target == "config":
        print("[cli] config valid.")
        return
    if args.target == "preflight":
        run_preflight(cfg)
        return
    if args.target == "run":
        run_pipeline(cfg)
        return
    if args.target == "validate":
        run_validation(cfg)
        return
    if args.target == "all":
        run_preflight(cfg)
        run_pipeline(cfg)
        run_validation(cfg)
        print("[cli] make all: COMPLETE.")


if __name__ == "__main__":
    main()
