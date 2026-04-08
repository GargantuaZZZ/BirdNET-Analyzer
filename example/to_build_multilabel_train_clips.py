#!/usr/bin/env python3
"""Build a multi-label BirdNET training set by mixing clips from multiple classes.

Input layout (single-label):
  train_clips/
    label_a/*.wav
    label_b/*.wav

Output layout (multi-label):
  train_clips_multilabel/
    label_a,label_b/*.wav
    label_a,label_c,label_d/*.wav

BirdNET multi-label training is triggered by folder names that contain commas.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class BuildStats:
    created_mixes: int = 0
    skipped_same_label: int = 0
    skipped_missing_audio: int = 0
    skipped_too_short: int = 0


def list_label_folders(input_root: Path) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}

    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        label = d.name.strip()
        if not label:
            continue
        files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        if files:
            mapping[label] = files

    return mapping


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return y.astype(np.float32, copy=False)


def choose_distinct_labels(labels: list[str], k: int, rng: random.Random) -> list[str]:
    return rng.sample(labels, k)


def mix_signals(signals: list[np.ndarray], rng: random.Random, peak: float) -> np.ndarray:
    min_len = min(len(x) for x in signals)
    if min_len < 16:
        return np.array([], dtype=np.float32)

    trimmed = [x[:min_len] for x in signals]

    # Random gain per source in [-6 dB, 0 dB].
    gains = [10 ** (rng.uniform(-6.0, 0.0) / 20.0) for _ in trimmed]

    mix = np.zeros(min_len, dtype=np.float32)
    for g, sig in zip(gains, trimmed, strict=True):
        mix += g * sig

    mix /= max(1, len(trimmed))

    max_abs = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
    if max_abs > peak and max_abs > 0:
        mix = mix * (peak / max_abs)

    return mix


def maybe_copy_originals(input_map: dict[str, list[Path]], output_root: Path, link_mode: str) -> int:
    copied = 0
    for label, files in input_map.items():
        out_dir = output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / src.name
            if dst.exists():
                continue
            if link_mode == "hardlink":
                dst.hardlink_to(src)
            elif link_mode == "symlink":
                dst.symlink_to(src)
            else:
                data, sr = sf.read(src)
                sf.write(dst, data, sr)
            copied += 1
    return copied


def build_dataset(
    input_root: Path,
    output_root: Path,
    num_mixes: int,
    min_sources: int,
    max_sources: int,
    sample_rate: int,
    seed: int,
    peak: float,
    include_originals: bool,
    link_mode: str,
) -> BuildStats:
    stats = BuildStats()
    rng = random.Random(seed)

    input_map = list_label_folders(input_root)
    labels = sorted(input_map.keys())

    if len(labels) < 2:
        raise RuntimeError("Need at least 2 label folders in input_root to create multi-label mixes.")

    output_root.mkdir(parents=True, exist_ok=True)

    if include_originals:
        copied = maybe_copy_originals(input_map, output_root, link_mode)
        print(f"Copied/linked original clips: {copied}", flush=True)

    for i in range(1, num_mixes + 1):
        k = rng.randint(min_sources, max_sources)
        if k > len(labels):
            k = len(labels)

        picked_labels = choose_distinct_labels(labels, k, rng)
        unique_labels = sorted(set(picked_labels))

        if len(unique_labels) < 2:
            stats.skipped_same_label += 1
            continue

        picked_files: list[Path] = []
        for lb in unique_labels:
            files = input_map.get(lb, [])
            if not files:
                picked_files = []
                break
            picked_files.append(rng.choice(files))

        if len(picked_files) < 2:
            stats.skipped_missing_audio += 1
            continue

        signals = [load_audio(p, sample_rate) for p in picked_files]
        if any(len(x) == 0 for x in signals):
            stats.skipped_missing_audio += 1
            continue

        mixed = mix_signals(signals, rng, peak=peak)
        if mixed.size == 0:
            stats.skipped_too_short += 1
            continue

        combo = ",".join(unique_labels)
        out_dir = output_root / combo
        out_dir.mkdir(parents=True, exist_ok=True)

        src_stems = "__".join(p.stem for p in picked_files)
        out_name = f"mix_{i:07d}_{combo}__{src_stems}.wav"
        out_path = out_dir / out_name

        sf.write(out_path, mixed, sample_rate)
        stats.created_mixes += 1

        if i % 500 == 0 or i == num_mixes:
            print(
                (
                    f"[{i}/{num_mixes}] created={stats.created_mixes} "
                    f"skipped_same_label={stats.skipped_same_label} "
                    f"skipped_missing_audio={stats.skipped_missing_audio} "
                    f"skipped_too_short={stats.skipped_too_short}"
                ),
                flush=True,
            )

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create BirdNET multi-label train clips by overlaying clips from different species")
    p.add_argument("--input_root", required=True, help="Path to existing single-label train_clips directory")
    p.add_argument("--output_root", required=True, help="Path to output train_clips_multilabel directory")
    p.add_argument("--num_mixes", type=int, default=50000, help="Number of mixed clips to generate")
    p.add_argument("--min_sources", type=int, default=2, help="Minimum number of source clips per mix")
    p.add_argument("--max_sources", type=int, default=3, help="Maximum number of source clips per mix")
    p.add_argument("--sample_rate", type=int, default=48000, help="Output sample rate")
    p.add_argument("--peak", type=float, default=0.95, help="Peak normalization cap to avoid clipping")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    p.add_argument(
        "--include_originals",
        action="store_true",
        help="Also include original single-label clips in output_root",
    )
    p.add_argument(
        "--link_mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to include originals when --include_originals is set",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_sources < 2:
        raise ValueError("min_sources must be >= 2")
    if args.max_sources < args.min_sources:
        raise ValueError("max_sources must be >= min_sources")
    if args.num_mixes <= 0:
        raise ValueError("num_mixes must be > 0")

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"input_root not found or not a directory: {input_root}")

    stats = build_dataset(
        input_root=input_root,
        output_root=output_root,
        num_mixes=args.num_mixes,
        min_sources=args.min_sources,
        max_sources=args.max_sources,
        sample_rate=args.sample_rate,
        seed=args.seed,
        peak=args.peak,
        include_originals=args.include_originals,
        link_mode=args.link_mode,
    )

    print("Done building multi-label train clips", flush=True)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
