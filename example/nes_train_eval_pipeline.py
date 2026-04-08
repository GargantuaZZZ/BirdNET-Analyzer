#!/usr/bin/env python3
"""Run BirdNET training/evaluation for NES using existing clips.

What this script does:
1) Locate NES train/test clip folders (existing clips only, no clip building).
2) Run birdnet_analyzer.train with the same training/evaluation method as HSN pipeline.
3) Save model outputs under NES workspace (sibling directory of HSN output by default).

Typical directory layout:
- <parent>/HSN
- <parent>/NES

Expected clip folders under NES workspace:
- <NES>/train_clips
- <NES>/test_clips
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ensure_non_empty_class_dirs(root: Path):
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Required directory not found: {root}")

    classes = [p for p in root.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found under: {root}")

    non_empty = [p for p in classes if any(p.iterdir())]
    if not non_empty:
        raise RuntimeError(f"No clips found under class folders in: {root}")


def run_train_command(
    repo_root: Path,
    train_dir: Path,
    test_dir: Path,
    model_out_prefix: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threads: int,
    python_exec: str,
):
    cmd = [
        python_exec,
        "-m",
        "birdnet_analyzer.train",
        str(train_dir),
        "-o",
        str(model_out_prefix),
        "--test_data",
        str(test_dir),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--threads",
        str(threads),
        "--crop_mode",
        "center",
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NES training/evaluation pipeline using existing train/test clips"
    )

    parser.add_argument("--repo", required=True, help="BirdNET-Analyzer repository path")

    parser.add_argument(
        "--hsn_out",
        help="HSN output workspace path. If set and --nes_out is omitted, NES path is derived as sibling: <parent>/NES",
    )
    parser.add_argument(
        "--nes_out",
        help="NES workspace path. Defaults to sibling of --hsn_out: <parent>/NES",
    )

    parser.add_argument(
        "--train_dir",
        help="Existing train clips folder. Default: <NES>/train_clips",
    )
    parser.add_argument(
        "--test_dir",
        help="Existing test clips folder. Default: <NES>/test_clips",
    )

    parser.add_argument("--model_name", default="nes_custom", help="Output model prefix name")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Training learning rate")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")

    parser.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable used to run birdnet_analyzer.train (default: current interpreter)",
    )

    return parser.parse_args()


def resolve_nes_out(args) -> Path:
    if args.nes_out:
        return Path(args.nes_out).resolve()

    if args.hsn_out:
        hsn_out = Path(args.hsn_out).resolve()
        return hsn_out.parent / "NES"

    raise ValueError("Provide either --nes_out or --hsn_out (to derive sibling NES path).")


def main():
    args = parse_args()

    repo_root = Path(args.repo).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repo path not found: {repo_root}")

    nes_out = resolve_nes_out(args)
    train_dir = Path(args.train_dir).resolve() if args.train_dir else nes_out / "train_clips"
    test_dir = Path(args.test_dir).resolve() if args.test_dir else nes_out / "test_clips"
    model_out_prefix = nes_out / "models" / args.model_name

    ensure_non_empty_class_dirs(train_dir)
    ensure_non_empty_class_dirs(test_dir)

    (nes_out / "models").mkdir(parents=True, exist_ok=True)

    print(f"Using NES workspace: {nes_out}", flush=True)
    print(f"Train clips: {train_dir}", flush=True)
    print(f"Test clips: {test_dir}", flush=True)
    print(f"Model output prefix: {model_out_prefix}", flush=True)

    run_train_command(
        repo_root=repo_root,
        train_dir=train_dir,
        test_dir=test_dir,
        model_out_prefix=model_out_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threads=args.threads,
        python_exec=args.python_exec,
    )

    print("Done. Check model outputs under:", nes_out / "models", flush=True)


if __name__ == "__main__":
    main()
