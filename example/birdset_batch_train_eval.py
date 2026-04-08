#!/usr/bin/env python3
"""Batch train/evaluate BirdNET custom heads for BirdSet subsets.

This script mirrors the HSN workflow and applies it to multiple subsets:
- build BirdNET-style train/test clip folders
- run `python -m birdnet_analyzer.train ... --test_data ...`
- collect each subset evaluation CSV path

Supported parquet variants:
1) <SUBSET>_metadata_train.parquet + <SUBSET>_metadata_test.parquet
2) fallback to <SUBSET>_metadata_test_5s.parquet when test parquet is missing
3) single <SUBSET>_metadata.parquet (XCL/XCM): deterministic recording-level split

Output layout (per subset):
<out_root>/<SUBSET>/
  train_clips/
  test_clips/
  models/
    <subset_lower>_custom.tflite
    <subset_lower>_custom_evaluation.csv
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class BuildStats:
    created: int = 0
    skipped: int = 0
    missing_audio: int = 0
    bad_rows: int = 0


@dataclass
class TrainAttempt:
    batch_size: int
    threads: int
    force_cpu: bool
    cuda_visible_devices: str | None


@dataclass
class TrainRunResult:
    return_code: int
    lines: list[str]


class _RunLock:
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            msg = (
                f"Lock file exists: {self.lock_file}. Another batch run may still be active. "
                "Remove this file only if you confirmed no active process is running."
            )
            raise RuntimeError(msg)

        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "lock_file": str(self.lock_file),
        }
        self.lock_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception:
            pass


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg first.")


def _parse_list(v):
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass

    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            vv = v.tolist()
            if isinstance(vv, list):
                return vv
        except Exception:
            pass

    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            x = ast.literal_eval(s)
            return x if isinstance(x, list) else []
        except Exception:
            return []
    return []


def _normalize_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []

    if "ebird_code_multilabel" in row.index:
        labels.extend([str(x).strip() for x in _parse_list(row.get("ebird_code_multilabel")) if str(x).strip()])

    if "ebird_code" in row.index:
        c = row.get("ebird_code")
        if isinstance(c, str) and c.strip():
            labels.append(c.strip())

    # Remove placeholders
    cleaned = []
    for lb in labels:
        if lb.lower() in {"none", "nan", ""}:
            continue
        cleaned.append(lb)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for lb in cleaned:
        if lb not in seen:
            seen.add(lb)
            out.append(lb)
    return out


def _events_from_row(row: pd.Series) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []

    if "detected_events" in row.index:
        for ev in _parse_list(row.get("detected_events")):
            if hasattr(ev, "tolist") and not isinstance(ev, (str, bytes)):
                try:
                    ev = ev.tolist()
                except Exception:
                    pass
            if isinstance(ev, (list, tuple)) and len(ev) >= 2:
                try:
                    s = float(ev[0])
                    e = float(ev[1])
                    if e > s:
                        events.append((s, e))
                except Exception:
                    pass

    if events:
        return events

    if "start_time" in row.index and "end_time" in row.index:
        try:
            s = float(row.get("start_time"))
            e = float(row.get("end_time"))
            if e > s:
                events.append((s, e))
        except Exception:
            pass

    return events


def _event_to_window(s: float, e: float, clip_seconds: float) -> tuple[float, float]:
    mid = (s + e) / 2.0
    start = max(0.0, mid - clip_seconds / 2.0)
    return start, clip_seconds


def _recording_hint_from_row(row_idx, row: pd.Series) -> str | None:
    # Prefer filepath-like columns
    for c in ("filepath", "path", "file", "filename", "audio"):
        if c in row.index:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return Path(v).name

    # Some test parquet keep recording name in index
    if isinstance(row_idx, str) and row_idx.strip():
        return Path(row_idx).name

    return None


def _build_audio_index(subset_dir: Path, audio_ext: str) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in subset_dir.rglob(f"*{audio_ext}"):
        idx[p.name] = p
    return idx


def _resolve_audio(recording_hint: str | None, subset_dir: Path, audio_index: dict[str, Path]) -> Path | None:
    if not recording_hint:
        return None

    p = Path(recording_hint)
    cands = [subset_dir / p, subset_dir / p.name]
    for c in cands:
        if c.exists():
            return c

    return audio_index.get(p.name)


def _cut_clip(src: Path, dst: Path, start_s: float, dur_s: float, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur_s:.3f}",
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


def _hash_split_key(s: str, test_ratio: float) -> bool:
    # True => test, False => train
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16) / 0xFFFFFFFF
    return v < test_ratio


def _split_single_metadata(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []

    for ridx, row in df.iterrows():
        hint = _recording_hint_from_row(ridx, row)
        key = hint if hint else str(ridx)
        if _hash_split_key(key, test_ratio):
            test_rows.append((ridx, row))
        else:
            train_rows.append((ridx, row))

    train_df = pd.DataFrame([r for _, r in train_rows])
    test_df = pd.DataFrame([r for _, r in test_rows])

    # Keep original index semantics where possible
    if train_rows:
        train_df.index = [i for i, _ in train_rows]
    if test_rows:
        test_df.index = [i for i, _ in test_rows]

    return train_df, test_df


def _build_clips_from_df(
    df: pd.DataFrame,
    subset_dir: Path,
    target_dir: Path,
    audio_index: dict[str, Path],
    clip_seconds: float,
    overwrite: bool,
    progress_interval: int,
    phase: str,
) -> BuildStats:
    stats = BuildStats()
    total = len(df)

    for n, (ridx, row) in enumerate(df.iterrows(), start=1):
        if progress_interval > 0 and (n % progress_interval == 0 or n == total):
            print(
                f"[{phase}] {n}/{total} | created={stats.created} skipped={stats.skipped} "
                f"missing_audio={stats.missing_audio} bad_rows={stats.bad_rows}",
                flush=True,
            )

        recording = _recording_hint_from_row(ridx, row)
        src = _resolve_audio(recording, subset_dir, audio_index)
        if src is None:
            stats.missing_audio += 1
            continue

        labels = _normalize_labels(row)
        if not labels:
            stats.bad_rows += 1
            continue

        events = _events_from_row(row)
        if not events:
            stats.bad_rows += 1
            continue

        for eidx, (s, e) in enumerate(events):
            try:
                win_s, win_d = _event_to_window(s, e, clip_seconds)
            except Exception:
                stats.bad_rows += 1
                continue

            for lb in labels:
                clip_name = f"{src.stem}_{ridx}_{eidx}_{lb}.wav"
                dst = target_dir / lb / clip_name
                try:
                    created = _cut_clip(src, dst, win_s, win_d, overwrite)
                    if created:
                        stats.created += 1
                    else:
                        stats.skipped += 1
                except Exception:
                    stats.bad_rows += 1

    return stats


def _pick_parquet(subset_dir: Path, subset_name: str, kind: str) -> Path | None:
    # kind in {"train", "test", "test5s", "single"}
    name = subset_name.upper()

    exact = {
        "train": subset_dir / f"{name}_metadata_train.parquet",
        "test": subset_dir / f"{name}_metadata_test.parquet",
        "test5s": subset_dir / f"{name}_metadata_test_5s.parquet",
        "single": subset_dir / f"{name}_metadata.parquet",
    }
    if exact[kind].exists():
        return exact[kind]

    # Fallback by glob
    candidates = sorted(subset_dir.glob("*.parquet"))
    if not candidates:
        return None

    if kind == "train":
        for p in candidates:
            s = p.name.lower()
            if "train" in s:
                return p
    elif kind == "test":
        for p in candidates:
            s = p.name.lower()
            if "test.parquet" in s and "5s" not in s and "old" not in s:
                return p
    elif kind == "test5s":
        for p in candidates:
            s = p.name.lower()
            if "test_5s" in s and "old" not in s:
                return p
    elif kind == "single":
        for p in candidates:
            s = p.name.lower()
            if "metadata" in s and "train" not in s and "test" not in s:
                return p

    return None


def _run_train_once(
    repo_root: Path,
    train_dir: Path,
    test_dir: Path,
    model_prefix: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threads: int,
    cache_mode: str | None,
    cache_file: Path | None,
    force_cpu: bool,
    cuda_visible_devices: str | None,
    log_file: Path | None,
) -> TrainRunResult:
    cmd = [
        "python",
        "-m",
        "birdnet_analyzer.train",
        str(train_dir),
        "-o",
        str(model_prefix),
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

    if cache_mode in {"save", "load"}:
        cmd.extend(["--cache_mode", cache_mode])
        if cache_file:
            cmd.extend(["--cache_file", str(cache_file)])

    print("Running:", " ".join(cmd), flush=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] Running: {' '.join(cmd)}\n")

    # Reduce noisy TensorFlow/absl logs in nohup output while preserving meaningful errors.
    suppress_tokens = (
        "WARNING:absl:Importing a function",
        "with ops with unsaved custom gradients. Will likely fail if a gradient is requested.",
    )

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    with subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if all(token in line for token in suppress_tokens):
                continue
            print(line, end="", flush=True)
            if log_file:
                with log_file.open("a", encoding="utf-8") as lf:
                    lf.write(line)

        return_code = proc.wait()

    return TrainRunResult(return_code=return_code, lines=[])


def _read_tail_lines(path: Path, max_lines: int = 500) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def _looks_like_cuda_runtime_issue(lines: list[str]) -> bool:
    joined = "\n".join(lines).lower()
    patterns = [
        r"cuda_error_invalid_handle",
        r"resource_exhausted",
        r"failed to allocate",
        r"could not create cudnn handle",
        r"cudnn_status",
        r"tensorflow was not built with cuda kernel binaries compatible with compute capability",
    ]
    return any(re.search(p, joined) for p in patterns)


def _looks_like_cache_load_issue(lines: list[str]) -> bool:
    joined = "\n".join(lines).lower()
    patterns = [
        r"cache",
        r"npz",
        r"failed to interpret file",
        r"cannot load file",
        r"no data left in file",
        r"badzipfile",
        r"eoferror",
    ]

    # Require at least one explicit load/caching token and one error token.
    has_cache_token = any(re.search(p, joined) for p in patterns[:2])
    has_error_token = any(re.search(p, joined) for p in patterns[2:])
    return has_cache_token and has_error_token


def _make_attempt_plan(
    batch_size: int,
    threads: int,
    force_cpu: bool,
    cuda_visible_devices: str | None,
    auto_cpu_fallback: bool,
) -> list[TrainAttempt]:
    plan: list[TrainAttempt] = []

    # Attempt 1: requested settings.
    plan.append(
        TrainAttempt(
            batch_size=max(1, batch_size),
            threads=max(1, threads),
            force_cpu=force_cpu,
            cuda_visible_devices=cuda_visible_devices,
        )
    )

    # Attempt 2: smaller batch and fewer threads to reduce pressure.
    plan.append(
        TrainAttempt(
            batch_size=max(4, batch_size // 2),
            threads=max(1, min(threads, 4)),
            force_cpu=force_cpu,
            cuda_visible_devices=cuda_visible_devices,
        )
    )

    # Attempt 3: force CPU for guaranteed completion if GPU runtime is unstable.
    if auto_cpu_fallback and not force_cpu:
        plan.append(
            TrainAttempt(
                batch_size=max(4, batch_size // 2),
                threads=max(1, min(threads, 4)),
                force_cpu=True,
                cuda_visible_devices=None,
            )
        )

    # Deduplicate identical attempts while preserving order.
    out: list[TrainAttempt] = []
    seen: set[tuple[int, int, bool, str | None]] = set()
    for a in plan:
        key = (a.batch_size, a.threads, a.force_cpu, a.cuda_visible_devices)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _run_train_with_retries(
    repo_root: Path,
    train_dir: Path,
    test_dir: Path,
    model_prefix: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threads: int,
    cache_mode: str | None,
    cache_file: Path | None,
    force_cpu: bool,
    cuda_visible_devices: str | None,
    auto_cpu_fallback: bool,
    log_file: Path | None,
) -> None:
    attempts = _make_attempt_plan(
        batch_size=batch_size,
        threads=threads,
        force_cpu=force_cpu,
        cuda_visible_devices=cuda_visible_devices,
        auto_cpu_fallback=auto_cpu_fallback,
    )

    last_error: Exception | None = None
    current_cache_mode = cache_mode
    tried_cache_rebuild = False

    for idx, attempt in enumerate(attempts, start=1):
        _log(
            (
                f"Train attempt {idx}/{len(attempts)}: batch_size={attempt.batch_size}, "
                f"threads={attempt.threads}, force_cpu={attempt.force_cpu}, "
                f"cuda_visible_devices={attempt.cuda_visible_devices}, "
                f"cache_mode={current_cache_mode}"
            ),
            log_file,
        )

        result = _run_train_once(
            repo_root=repo_root,
            train_dir=train_dir,
            test_dir=test_dir,
            model_prefix=model_prefix,
            epochs=epochs,
            batch_size=attempt.batch_size,
            learning_rate=learning_rate,
            threads=attempt.threads,
            cache_mode=current_cache_mode,
            cache_file=cache_file,
            force_cpu=attempt.force_cpu,
            cuda_visible_devices=attempt.cuda_visible_devices,
            log_file=log_file,
        )

        if result.return_code == 0:
            _log("Training finished successfully.", log_file)
            return

        tail = _read_tail_lines(log_file, max_lines=500) if log_file else []
        is_cuda_issue = _looks_like_cuda_runtime_issue(tail)
        is_cache_issue = _looks_like_cache_load_issue(tail)
        last_error = subprocess.CalledProcessError(result.return_code, "birdnet_analyzer.train")

        if is_cache_issue and current_cache_mode == "load" and not tried_cache_rebuild:
            _log("Detected cache-load failure; retrying with cache_mode=save to rebuild cache.", log_file)
            current_cache_mode = "save"
            tried_cache_rebuild = True
            continue

        if not is_cuda_issue:
            _log("Training failed with non-CUDA error. Stop retrying.", log_file)
            break

        if idx < len(attempts):
            _log("Detected CUDA runtime instability; retrying with safer settings...", log_file)

    if last_error:
        raise last_error
    raise RuntimeError("Training failed with unknown error")


def _resolve_cache_mode(requested_mode: str, cache_file: Path) -> str | None:
    if requested_mode == "none":
        return None
    if requested_mode in {"save", "load"}:
        return requested_mode
    if requested_mode == "auto":
        return "load" if cache_file.exists() and cache_file.stat().st_size > 0 else "save"
    raise ValueError(f"Unsupported cache_mode: {requested_mode}")


def _parse_gpu_pool(cuda_visible_devices: str | None) -> list[str]:
    if cuda_visible_devices is None:
        return []
    parts = [p.strip() for p in str(cuda_visible_devices).split(",")]
    return [p for p in parts if p]


def _ensure_non_empty(root: Path, name: str) -> None:
    if not root.exists():
        raise RuntimeError(f"{name} dir does not exist: {root}")
    classes = [p for p in root.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class directories under {name}: {root}")
    if not any(any(c.iterdir()) for c in classes):
        raise RuntimeError(f"All class directories are empty in {name}: {root}")


def _log(msg: str, log_file: Path | None = None) -> None:
    print(msg, flush=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(msg + "\n")


def _default_subsets(all_dirs: Iterable[Path], include_hsn: bool) -> list[str]:
    names = sorted([d.name for d in all_dirs if d.is_dir()])
    if include_hsn:
        return names
    return [n for n in names if n.upper() != "HSN"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch BirdSet subset training/evaluation")
    p.add_argument("--repo", required=True, help="BirdNET-Analyzer repository path")
    p.add_argument("--birdset_root", required=True, help="Root path containing subset directories (HSN/NBP/...)" )
    p.add_argument("--out_root", required=True, help="Output root like .../BirdNET-Analyzer/birdset_train")

    p.add_argument("--subsets", nargs="+", help="Subset names. Default: all under birdset_root except HSN")
    p.add_argument("--include_hsn", action="store_true", help="Include HSN in default subset list")

    p.add_argument("--audio_ext", default=".ogg")
    p.add_argument("--clip_seconds", type=float, default=3.0)
    p.add_argument("--test_ratio_single", type=float, default=0.2, help="Only used when subset has single metadata parquet")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--threads", type=int, default=8)

    p.add_argument(
        "--cache_mode",
        choices=["none", "save", "load", "auto"],
        default="none",
        help="Train cache mode for each subset. auto=load if cache exists else save.",
    )
    p.add_argument("--cache_dirname", default="cache", help="Per-subset cache directory name under subset output")
    p.add_argument("--force_cpu", action="store_true", help="Run train subprocess on CPU only (sets CUDA_VISIBLE_DEVICES='')")
    p.add_argument("--cuda_visible_devices", default=None, help="Pass through CUDA_VISIBLE_DEVICES value to train subprocess")
    p.add_argument(
        "--gpu_round_robin",
        action="store_true",
        help="If CUDA_VISIBLE_DEVICES has multiple GPU ids (e.g. 3,4,5,6,7), assign one GPU per subset in round-robin.",
    )
    p.add_argument(
        "--auto_cpu_fallback",
        action="store_true",
        help="If CUDA runtime is unstable (OOM/invalid handle), retry once with CPU to keep batch running.",
    )
    p.add_argument(
        "--run_lock_file",
        default=None,
        help="Optional lock file path to prevent accidentally running multiple batch jobs concurrently.",
    )

    p.add_argument("--skip_build", action="store_true", help="Skip clip building and only run training")
    p.add_argument("--skip_train", action="store_true", help="Only build clips")
    p.add_argument("--overwrite_clips", action="store_true")
    p.add_argument("--rebuild_clips", action="store_true")
    p.add_argument("--progress_interval", type=int, default=1000, help="Print clip-building progress every N parquet rows")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo).resolve()
    birdset_root = Path(args.birdset_root).resolve()
    out_root = Path(args.out_root).resolve()

    _check_ffmpeg()
    out_root.mkdir(parents=True, exist_ok=True)

    all_subset_dirs = [d for d in birdset_root.iterdir() if d.is_dir()]
    subsets = args.subsets if args.subsets else _default_subsets(all_subset_dirs, args.include_hsn)

    lock_file = Path(args.run_lock_file).resolve() if args.run_lock_file else (out_root / ".batch_train_eval.lock")

    summary_rows = []
    gpu_pool = _parse_gpu_pool(args.cuda_visible_devices)

    with _RunLock(lock_file):
        for subset_idx, subset in enumerate(subsets):
            subset_name = subset.upper()
            subset_src = birdset_root / subset_name
            subset_out = out_root / subset_name
            train_dir = subset_out / "train_clips"
            test_dir = subset_out / "test_clips"
            models_dir = subset_out / "models"
            model_prefix = models_dir / f"{subset_name.lower()}_custom"
            eval_csv = Path(str(model_prefix) + "_evaluation.csv")
            cache_dir = subset_out / args.cache_dirname
            cache_file = cache_dir / f"{subset_name.lower()}_train_cache.npz"
            subset_logs_dir = subset_out / "logs"
            subset_log_file = subset_logs_dir / f"{subset_name.lower()}_run.log"

            print("=" * 90)
            print(f"Subset: {subset_name}")
            print(f"Source: {subset_src}")
            print(f"Output: {subset_out}")

            row = {
                "subset": subset_name,
                "status": "ok",
                "train_parquet": "",
                "test_parquet": "",
                "evaluation_csv": str(eval_csv),
                "subset_log": str(subset_log_file),
                "message": "",
            }

            try:
                if not subset_src.exists():
                    raise FileNotFoundError(f"Subset directory not found: {subset_src}")

                train_pq = _pick_parquet(subset_src, subset_name, "train")
                test_pq = _pick_parquet(subset_src, subset_name, "test")
                if test_pq is None:
                    test_pq = _pick_parquet(subset_src, subset_name, "test5s")

                single_pq = _pick_parquet(subset_src, subset_name, "single")

                row["train_parquet"] = str(train_pq) if train_pq else ""
                row["test_parquet"] = str(test_pq) if test_pq else ""

                if train_pq is None and single_pq is None:
                    raise RuntimeError("No usable train parquet found")
                if test_pq is None and single_pq is None:
                    raise RuntimeError("No usable test parquet found")

                subset_out.mkdir(parents=True, exist_ok=True)
                models_dir.mkdir(parents=True, exist_ok=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
                subset_logs_dir.mkdir(parents=True, exist_ok=True)

                _log("=" * 90, subset_log_file)
                _log(f"Subset: {subset_name}", subset_log_file)
                _log(f"Source: {subset_src}", subset_log_file)
                _log(f"Output: {subset_out}", subset_log_file)

                if args.rebuild_clips and not args.skip_build:
                    if train_dir.exists():
                        shutil.rmtree(train_dir)
                    if test_dir.exists():
                        shutil.rmtree(test_dir)

                train_dir.mkdir(parents=True, exist_ok=True)
                test_dir.mkdir(parents=True, exist_ok=True)

                if not args.skip_build:
                    _log("Indexing audio files...", subset_log_file)
                    audio_index = _build_audio_index(subset_src, args.audio_ext)
                    _log(f"Indexed {len(audio_index)} audio files", subset_log_file)

                    if train_pq and test_pq:
                        df_train = pd.read_parquet(train_pq)
                        df_test = pd.read_parquet(test_pq)
                    else:
                        # XCL/XCM style: split a single metadata parquet deterministically
                        if not single_pq:
                            raise RuntimeError("Single metadata parquet split required but not found")
                        _log(f"Using deterministic split from single parquet: {single_pq}", subset_log_file)
                        df_single = pd.read_parquet(single_pq)
                        df_train, df_test = _split_single_metadata(df_single, test_ratio=args.test_ratio_single)

                    _log("Building train clips...", subset_log_file)
                    train_stats = _build_clips_from_df(
                        df_train,
                        subset_src,
                        train_dir,
                        audio_index,
                        clip_seconds=args.clip_seconds,
                        overwrite=args.overwrite_clips,
                        progress_interval=args.progress_interval,
                        phase=f"{subset_name}:train",
                    )
                    _log(f"Train stats: {train_stats}", subset_log_file)

                    _log("Building test clips...", subset_log_file)
                    test_stats = _build_clips_from_df(
                        df_test,
                        subset_src,
                        test_dir,
                        audio_index,
                        clip_seconds=args.clip_seconds,
                        overwrite=args.overwrite_clips,
                        progress_interval=args.progress_interval,
                        phase=f"{subset_name}:test",
                    )
                    _log(f"Test stats: {test_stats}", subset_log_file)

                _ensure_non_empty(train_dir, "train_clips")
                _ensure_non_empty(test_dir, "test_clips")

                if args.skip_train:
                    row["status"] = "built_only"
                    row["message"] = "skip_train enabled"
                    _log("skip_train enabled", subset_log_file)
                else:
                    selected_cuda_devices = args.cuda_visible_devices
                    if args.gpu_round_robin and gpu_pool and not args.force_cpu:
                        selected_cuda_devices = gpu_pool[subset_idx % len(gpu_pool)]
                        _log(
                            f"Round-robin GPU assignment for {subset_name}: CUDA_VISIBLE_DEVICES={selected_cuda_devices}",
                            subset_log_file,
                        )

                    resolved_cache_mode = _resolve_cache_mode(args.cache_mode, cache_file)
                    _log(
                        f"Cache mode for {subset_name}: requested={args.cache_mode}, resolved={resolved_cache_mode}",
                        subset_log_file,
                    )

                    _run_train_with_retries(
                        repo_root=repo_root,
                        train_dir=train_dir,
                        test_dir=test_dir,
                        model_prefix=model_prefix,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        threads=args.threads,
                        cache_mode=resolved_cache_mode,
                        cache_file=cache_file,
                        force_cpu=args.force_cpu,
                        cuda_visible_devices=selected_cuda_devices,
                        auto_cpu_fallback=args.auto_cpu_fallback,
                        log_file=subset_log_file,
                    )

                    if not eval_csv.exists():
                        row["status"] = "warn"
                        row["message"] = "training finished but evaluation csv not found"
                        _log("WARNING: training finished but evaluation csv not found", subset_log_file)

            except Exception as e:
                row["status"] = "failed"
                row["message"] = str(e)
                _log(f"ERROR [{subset_name}]: {e}", subset_log_file)

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "batch_train_eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("=" * 90)
    print("Batch finished")
    print(f"Summary: {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()
