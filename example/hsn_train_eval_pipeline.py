#!/usr/bin/env python3
"""End-to-end HSN training/evaluation pipeline for BirdNET-Analyzer.

What this script does:
1) Read HSN train/test parquet metadata.
2) Create BirdNET training folder structure with class subfolders and wav clips.
3) Create BirdNET test folder structure from HSN test metadata.
4) Run birdnet_analyzer.train using generated train/test clip folders.

Notes:
- Requires ffmpeg in PATH.
- Assumes audio files are under DATA root (recursively), mostly .ogg.
- Uses 3-second clips centered on event intervals for train and test.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BuildStats:
    created: int = 0
    skipped: int = 0
    missing_audio: int = 0
    bad_rows: int = 0


def parse_list(value):
    if value is None:
        return []

    # pandas NA / NaN
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    # numpy/pyarrow list-like containers
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg first.")


def build_audio_index(data_root: Path, audio_ext: str) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    pattern = f"*{audio_ext}"
    for p in data_root.rglob(pattern):
        idx[p.name] = p
    return idx


def resolve_audio(path_or_name, data_root: Path, audio_index: dict[str, Path]) -> Path | None:
    if not path_or_name:
        return None

    p = Path(str(path_or_name))
    candidates = []

    if p.is_absolute():
        candidates.append(p)

    candidates.append(data_root / p)
    candidates.append(data_root / p.name)

    for c in candidates:
        if c.exists():
            return c

    return audio_index.get(p.name)


def cut_clip(src: Path, dst: Path, start_sec: float, duration_sec: float, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "48000",
        str(dst),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def event_to_3s_window(start: float, end: float) -> tuple[float, float]:
    if end <= start:
        raise ValueError("end must be greater than start")
    mid = (start + end) / 2.0
    window_start = max(0.0, mid - 1.5)
    return window_start, 3.0


def build_train_clips(
    train_parquet: Path,
    data_root: Path,
    output_dir: Path,
    audio_index: dict[str, Path],
    overwrite_clips: bool,
    progress_interval: int,
) -> BuildStats:
    df = pd.read_parquet(train_parquet)
    stats = BuildStats()
    total_rows = len(df)

    required_cols = ["filepath", "detected_events"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Train parquet missing required columns: {missing}")

    for row_idx, (ridx, row) in enumerate(df.iterrows(), start=1):
        if progress_interval > 0 and (row_idx % progress_interval == 0 or row_idx == total_rows):
            print(
                f"[train] {row_idx}/{total_rows} rows | created={stats.created} skipped={stats.skipped} "
                f"missing_audio={stats.missing_audio} bad_rows={stats.bad_rows}",
                flush=True,
            )

        src = resolve_audio(row.get("filepath"), data_root, audio_index)
        if src is None:
            stats.missing_audio += 1
            continue

        labels = []
        labels.extend([str(x) for x in parse_list(row.get("ebird_code_multilabel")) if x])

        ebird_code = row.get("ebird_code")
        if isinstance(ebird_code, str) and ebird_code.strip():
            labels.append(ebird_code.strip())

        labels = sorted(set(labels))
        if not labels:
            stats.bad_rows += 1
            continue

        events = parse_list(row.get("detected_events"))
        if not events:
            stats.bad_rows += 1
            continue

        for eidx, ev in enumerate(events):
            if hasattr(ev, "tolist") and not isinstance(ev, (str, bytes)):
                try:
                    ev = ev.tolist()
                except Exception:
                    pass

            if not isinstance(ev, (list, tuple)) or len(ev) < 2:
                stats.bad_rows += 1
                continue

            try:
                start = float(ev[0])
                end = float(ev[1])
                clip_start, clip_dur = event_to_3s_window(start, end)
            except Exception:
                stats.bad_rows += 1
                continue

            for label in labels:
                clip_name = f"{src.stem}_{ridx}_{eidx}_{label}.wav"
                dst = output_dir / label / clip_name
                try:
                    created = cut_clip(src, dst, clip_start, clip_dur, overwrite=overwrite_clips)
                    if created:
                        stats.created += 1
                    else:
                        stats.skipped += 1
                except Exception:
                    stats.bad_rows += 1

    return stats


def build_test_clips(
    test_parquet: Path,
    data_root: Path,
    output_dir: Path,
    audio_index: dict[str, Path],
    overwrite_clips: bool,
    progress_interval: int,
) -> BuildStats:
    df = pd.read_parquet(test_parquet)
    stats = BuildStats()
    total_rows = len(df)

    required_cols = ["start_time", "end_time", "ebird_code"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Test parquet missing required columns: {missing}")

    for row_idx, (idx, row) in enumerate(df.iterrows(), start=1):
        if progress_interval > 0 and (row_idx % progress_interval == 0 or row_idx == total_rows):
            print(
                f"[test] {row_idx}/{total_rows} rows | created={stats.created} skipped={stats.skipped} "
                f"missing_audio={stats.missing_audio} bad_rows={stats.bad_rows}",
                flush=True,
            )

        recording = idx if isinstance(idx, str) else row.get("filepath")
        src = resolve_audio(recording, data_root, audio_index)
        if src is None:
            stats.missing_audio += 1
            continue

        label = row.get("ebird_code")
        if not isinstance(label, str) or not label.strip():
            stats.bad_rows += 1
            continue
        label = label.strip()

        try:
            start = float(row.get("start_time"))
            end = float(row.get("end_time"))
            clip_start, clip_dur = event_to_3s_window(start, end)
        except Exception:
            stats.bad_rows += 1
            continue

        clip_name = f"{src.stem}_{int(start * 1000)}_{int(end * 1000)}_{label}.wav"
        dst = output_dir / label / clip_name

        try:
            created = cut_clip(src, dst, clip_start, clip_dur, overwrite=overwrite_clips)
            if created:
                stats.created += 1
            else:
                stats.skipped += 1
        except Exception:
            stats.bad_rows += 1

    return stats


def run_train_command(
    repo_root: Path,
    train_dir: Path,
    test_dir: Path,
    model_out_prefix: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threads: int,
):
    cmd = [
        "python",
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


def ensure_non_empty_class_dirs(root: Path):
    classes = [p for p in root.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders created under: {root}")

    non_empty = [p for p in classes if any(p.iterdir())]
    if not non_empty:
        raise RuntimeError(f"No clips created under any class folder in: {root}")


def parse_args():
    parser = argparse.ArgumentParser(description="HSN -> BirdNET train/test pipeline")
    parser.add_argument("--repo", required=True, help="BirdNET-Analyzer repository path")
    parser.add_argument("--data", required=True, help="HSN dataset root path")
    parser.add_argument("--out", required=True, help="Output workspace path")

    parser.add_argument("--train_parquet", default="HSN_metadata_train.parquet", help="Train parquet filename")
    parser.add_argument("--test_parquet", default="HSN_metadata_test.parquet", help="Test parquet filename")

    parser.add_argument("--audio_ext", default=".ogg", help="Audio extension to index (default: .ogg)")
    parser.add_argument("--overwrite_clips", action="store_true", help="Overwrite already generated clips")
    parser.add_argument("--rebuild_clips", action="store_true", help="Delete existing generated clip dirs before building")
    parser.add_argument("--skip_build_train", action="store_true", help="Skip building train clips")
    parser.add_argument("--skip_build_test", action="store_true", help="Skip building test clips")
    parser.add_argument("--skip_train", action="store_true", help="Only build clips, do not run training")
    parser.add_argument("--progress_interval", type=int, default=200, help="Print progress every N parquet rows")

    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Training learning rate")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")

    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(args.repo).resolve()
    data_root = Path(args.data).resolve()
    out_root = Path(args.out).resolve()

    train_parquet = data_root / args.train_parquet
    test_parquet = data_root / args.test_parquet

    train_dir = out_root / "train_clips"
    test_dir = out_root / "test_clips"
    model_out_prefix = out_root / "models" / "hsn_custom"

    if not train_parquet.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_parquet}")
    if not test_parquet.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_parquet}")

    check_ffmpeg()

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "models").mkdir(parents=True, exist_ok=True)

    if args.rebuild_clips:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if test_dir.exists():
            shutil.rmtree(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing audio files...", flush=True)
    audio_index = build_audio_index(data_root, args.audio_ext)
    print(f"Indexed {len(audio_index)} audio files", flush=True)

    train_stats = BuildStats()
    if args.skip_build_train:
        print("Skipping train clip build (--skip_build_train).", flush=True)
    else:
        print("Building train clips...", flush=True)
        train_stats = build_train_clips(
            train_parquet,
            data_root,
            train_dir,
            audio_index,
            args.overwrite_clips,
            args.progress_interval,
        )
        print(f"Train stats: {train_stats}", flush=True)

    test_stats = BuildStats()
    if args.skip_build_test:
        print("Skipping test clip build (--skip_build_test).", flush=True)
    else:
        print("Building test clips...", flush=True)
        test_stats = build_test_clips(
            test_parquet,
            data_root,
            test_dir,
            audio_index,
            args.overwrite_clips,
            args.progress_interval,
        )
        print(f"Test stats: {test_stats}", flush=True)

    ensure_non_empty_class_dirs(train_dir)
    ensure_non_empty_class_dirs(test_dir)

    if args.skip_train:
        print("skip_train enabled. Clip generation completed.", flush=True)
        return

    run_train_command(
        repo_root=repo_root,
        train_dir=train_dir,
        test_dir=test_dir,
        model_out_prefix=model_out_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threads=args.threads,
    )

    print("Done. Check model outputs under:", out_root / "models", flush=True)


if __name__ == "__main__":
    main()
