#!/usr/bin/env python3
"""Project BirdNET embeddings to 2D and plot class distribution.

Expected input CSV format (from birdnet_analyzer.embeddings --file_output):
file_path,start,end,embedding
/path/to/audio.wav,0.0,3.0,"0.01,0.23,..."

By default, class labels are inferred from parent directory of file_path.
Example: /data/train_clips/amecro/clip.wav -> label amecro
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_embedding_cell(cell: str) -> np.ndarray:
    text = str(cell).strip().strip('"')
    if not text:
        raise ValueError("Empty embedding cell.")
    return np.fromstring(text, sep=",", dtype=np.float32)


def infer_label_from_path(file_path: str) -> str:
    p = Path(str(file_path))
    if p.parent.name:
        return p.parent.name
    if p.stem:
        return p.stem
    return "unknown"


def pca_project(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    comps = u[:, :n_components] * s[:n_components]
    return comps.astype(np.float32, copy=False)


def tsne_project(x: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("t-SNE requires scikit-learn. Install via: pip install scikit-learn") from exc

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    )
    return tsne.fit_transform(x)


def umap_project(x: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    try:
        umap = importlib.import_module("umap")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("UMAP requires umap-learn. Install via: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(x)


def sample_per_class(df: pd.DataFrame, max_per_class: int, random_state: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df

    sampled = []
    for _, g in df.groupby("label", sort=False):
        if len(g) <= max_per_class:
            sampled.append(g)
        else:
            sampled.append(g.sample(n=max_per_class, random_state=random_state))
    return pd.concat(sampled, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot 2D projection of BirdNET embeddings")
    p.add_argument("--input_csv", required=True, help="CSV exported from BirdNET embeddings")
    p.add_argument("--output_png", required=True, help="Output plot path (.png)")
    p.add_argument("--output_csv", default=None, help="Optional CSV to save 2D points with labels")

    p.add_argument("--label_column", default=None, help="Use an explicit label column if present in CSV")
    p.add_argument("--top_k_classes", type=int, default=20, help="Keep top-K most frequent classes; 0 means keep all")
    p.add_argument("--max_points_per_class", type=int, default=800, help="Max sampled points per class; 0 means keep all")

    p.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fig_width", type=float, default=12)
    p.add_argument("--fig_height", type=float, default=8)
    p.add_argument("--point_size", type=float, default=8)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--dpi", type=int, default=180)
    return p


def main() -> None:
    args = build_parser().parse_args()

    in_path = Path(args.input_csv)
    out_png = Path(args.output_png)
    out_csv = Path(args.output_csv) if args.output_csv else None

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if "embedding" not in df.columns:
        raise ValueError("Input CSV must contain an 'embedding' column.")

    if args.label_column:
        if args.label_column not in df.columns:
            raise ValueError(f"label_column not found: {args.label_column}")
        df["label"] = df[args.label_column].astype(str)
    else:
        if "file_path" not in df.columns:
            raise ValueError("Input CSV must contain 'file_path' when label_column is not provided.")
        df["label"] = df["file_path"].map(infer_label_from_path)

    # Filter invalid rows
    df = df[df["label"].notna() & (df["label"].astype(str).str.len() > 0)].copy()
    if df.empty:
        raise ValueError("No valid rows after label filtering.")

    if args.top_k_classes > 0:
        top_labels = df["label"].value_counts().head(args.top_k_classes).index
        df = df[df["label"].isin(top_labels)].copy()

    if df.empty:
        raise ValueError("No rows left after top_k_classes filter.")

    df = sample_per_class(df, args.max_points_per_class, args.seed)

    embeddings = [parse_embedding_cell(v) for v in df["embedding"].tolist()]
    dims = {e.shape[0] for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {sorted(dims)}")

    x = np.vstack(embeddings)

    if args.method == "pca":
        z = pca_project(x)
    elif args.method == "tsne":
        z = tsne_project(x, perplexity=args.perplexity, random_state=args.seed)
    else:
        z = umap_project(x, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, random_state=args.seed)

    df_plot = pd.DataFrame({
        "x": z[:, 0],
        "y": z[:, 1],
        "label": df["label"].to_numpy(),
    })

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(args.fig_width, args.fig_height))
    labels = sorted(df_plot["label"].unique())
    cmap = plt.get_cmap("tab20", max(20, len(labels)))

    for i, lb in enumerate(labels):
        sub = df_plot[df_plot["label"] == lb]
        plt.scatter(sub["x"], sub["y"], s=args.point_size, alpha=args.alpha, label=lb, color=cmap(i))

    plt.title(f"BirdNET Embeddings 2D Projection ({args.method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    # Put legend outside to keep plot area readable.
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    plt.close()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_plot.to_csv(out_csv, index=False)

    print(f"Saved figure: {out_png}")
    if out_csv:
        print(f"Saved points: {out_csv}")
    print(f"Rows plotted: {len(df_plot)} | Classes: {df_plot['label'].nunique()}")


if __name__ == "__main__":
    main()
