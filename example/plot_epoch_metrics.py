#!/usr/bin/env python3
"""Plot per-epoch training metrics for BirdNET training runs.

Supported inputs:
1) History CSV with columns like:
   epoch,loss,AUPRC,AUROC,val_loss,val_AUPRC,val_AUROC,learning_rate
2) Raw training log text containing Keras epoch outputs.

Outputs:
- One PNG figure with 4 line-chart subplots:
  loss, AUPRC, AUROC, learning rate.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_ALIASES = {
    "loss": ["loss", "train_loss"],
    "val_loss": ["val_loss", "valid_loss", "validation_loss"],
    "AUPRC": ["AUPRC", "auprc", "pr_auc", "auc_pr"],
    "val_AUPRC": ["val_AUPRC", "val_auprc", "val_pr_auc", "val_auc_pr"],
    "AUROC": ["AUROC", "auroc", "roc_auc", "auc"],
    "val_AUROC": ["val_AUROC", "val_auroc", "val_roc_auc", "val_auc"],
    "learning_rate": ["learning_rate", "lr", "LearningRate"],
}


# Local editable template: change these arrays and run with `--format array`.
MANUAL_TEMPLATE = {
    "train_loss": [277.9491, 10.485, 6.0046, 4.7672, 4.0606, 3.6154, 3.3404, 3.1492, 3.0068, 2.8942, 2.8025, 2.7268, 2.6623, 2.6057, 2.5565, 2.5135, 2.4753],
    "val_loss": [20.8946, 7.9603, 5.9398, 4.7914, 4.1641, 3.8751, 3.7386, 3.6693, 3.6264, 3.6065, 3.6078, 3.599, 3.6074, 3.6089, 3.6115, 3.6261, 3.6309],
    "train_auprc": [0.0001912, 0.0761, 0.2761, 0.4084, 0.4831, 0.5322, 0.5657, 0.5909, 0.6117, 0.6292, 0.6443, 0.657, 0.668, 0.678, 0.687, 0.6944, 0.7014],
    "val_auprc": [0.00028544, 0.0496, 0.2015, 0.344, 0.4302, 0.4755, 0.4997, 0.5133, 0.5234, 0.5293, 0.5317, 0.5352, 0.5357, 0.5367, 0.5387, 0.5374, 0.5375],
    "train_auroc": [0.5321, 0.6404, 0.747, 0.806, 0.838, 0.8599, 0.8748, 0.8858, 0.8951, 0.9021, 0.9081, 0.9129, 0.9172, 0.9206, 0.9237, 0.9261, 0.9282],
    "val_auroc": [0.5307, 0.5965, 0.7126, 0.7827, 0.8213, 0.8397, 0.8488, 0.8532, 0.8567, 0.8585, 0.8591, 0.86, 0.8598, 0.8608, 0.8603, 0.8603, 0.8596],
    "learning_rate": [2e-05, 4e-05, 6e-05, 8e-05, 0.0001, 0.0001, 9.989e-05, 9.9562e-05, 9.9017e-05, 9.8257e-05, 9.7286e-05, 9.611e-05, 9.4733e-05, 9.3162e-05, 9.1406e-05, 8.9472e-05, 8.737e-05],
}


PRESET_ARRAY_TEMPLATES = {
    "hsn_multilabel": {
        "train_loss": [2.7320, 2.3371, 2.2900, 2.2567, 2.2407, 2.2306],
        "val_loss": [1.9773, 2.2478, 2.4352, 2.3741, 2.4451, 2.4188],
        "train_auprc": [0.7436, 0.7969, 0.8034, 0.8082, 0.8106, 0.8118],
        "val_auprc": [0.4773, 0.4657, 0.4602, 0.4623, 0.4533, 0.4490],
        "train_auroc": [0.9144, 0.9303, 0.9320, 0.9339, 0.9347, 0.9354],
        "val_auroc": [0.9060, 0.9003, 0.8934, 0.8910, 0.8898, 0.8896],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "hsn": {
        "train_loss": [1.7791, 0.8296, 0.6845, 0.6172, 0.5793, 0.5539, 0.535],
        "val_loss": [2.706, 2.6931, 2.7916, 3.2568, 3.3502, 3.3864, 3.3226],
        "train_auprc": [0.7547, 0.9257, 0.9425, 0.9511, 0.9561, 0.9592, 0.9615],
        "val_auprc": [0.5818, 0.608, 0.6238, 0.571, 0.584, 0.591, 0.5802],
        "train_auroc": [0.9519, 0.9878, 0.9905, 0.9916, 0.9924, 0.9928, 0.9931],
        "val_auroc": [0.8518, 0.8725, 0.866, 0.8351, 0.8341, 0.8349, 0.8356],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394, 0.00048643],
    },
    "nbp": {
        "train_loss": [1.4868, 1.1721, 1.1336, 1.1045, 1.09, 1.0826],
        "val_loss": [3.8332, 4.3665, 4.6697, 4.7759, 4.7174, 4.7924],
        "train_auprc": [0.8125, 0.8663, 0.873, 0.8784, 0.8815, 0.883],
        "val_auprc": [0.3804, 0.3264, 0.306, 0.3084, 0.3132, 0.3054],
        "train_auroc": [0.9668, 0.9731, 0.9737, 0.9748, 0.9752, 0.9754],
        "val_auroc": [0.7927, 0.7592, 0.7548, 0.7463, 0.7507, 0.7527],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "nes": {
        "train_loss": [1.4636, 0.8551, 0.7719, 0.7173, 0.6874, 0.6681],
        "val_loss": [4.1603, 5.0338, 5.6299, 5.9277, 6.0172, 6.0914],
        "train_auprc": [0.8257, 0.9178, 0.9296, 0.9379, 0.9426, 0.9457],
        "val_auprc": [0.3326, 0.2848, 0.2582, 0.2429, 0.2429, 0.2439],
        "train_auroc": [0.9711, 0.9829, 0.985, 0.9867, 0.9877, 0.9885],
        "val_auroc": [0.8099, 0.7691, 0.738, 0.7312, 0.7241, 0.7243],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "per": {
        "train_loss": [1.853, 1.066, 0.9446, 0.8659, 0.8235, 0.7961],
        "val_loss": [5.5527, 6.3889, 7.152, 7.5268, 7.7416, 7.986],
        "train_auprc": [0.7613, 0.8886, 0.9072, 0.9199, 0.9271, 0.9318],
        "val_auprc": [0.1002, 0.0899, 0.0741, 0.0682, 0.0674, 0.0653],
        "train_auroc": [0.9547, 0.9749, 0.9789, 0.9816, 0.9834, 0.9843],
        "val_auroc": [0.7014, 0.6725, 0.6527, 0.642, 0.6428, 0.6407],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "pow": {
        "train_loss": [1.5016, 1.0884, 1.0178, 0.9705, 0.9445, 0.9292],
        "val_loss": [4.7076, 5.5409, 6.0864, 6.2866, 6.5316, 6.7439],
        "train_auprc": [0.812, 0.8801, 0.8914, 0.8995, 0.904, 0.9067],
        "val_auprc": [0.1724, 0.1469, 0.1386, 0.1245, 0.1313, 0.1208],
        "train_auroc": [0.966, 0.9759, 0.9779, 0.9795, 0.9803, 0.9809],
        "val_auroc": [0.78, 0.7236, 0.7023, 0.6888, 0.6914, 0.6797],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "sne": {
        "train_loss": [1.4174, 1.028, 0.9713, 0.9303, 0.9086, 0.8959],
        "val_loss": [4.0819, 4.5698, 5.0295, 5.1851, 5.3725, 5.4924],
        "train_auprc": [0.8283, 0.89, 0.8989, 0.9058, 0.9098, 0.9122],
        "val_auprc": [0.2691, 0.2484, 0.227, 0.2064, 0.2012, 0.197],
        "train_auroc": [0.9681, 0.9773, 0.9787, 0.9801, 0.9809, 0.9815],
        "val_auroc": [0.8022, 0.7742, 0.7391, 0.7424, 0.7301, 0.7247],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "ssw": {
        "train_loss": [1.4913, 1.1235, 1.0784, 1.0436, 1.0265, 1.0174],
        "val_loss": [3.8025, 4.2504, 4.4441, 4.7375, 4.684, 4.8196],
        "train_auprc": [0.8166, 0.8762, 0.884, 0.8906, 0.8941, 0.8961],
        "val_auprc": [0.3119, 0.2682, 0.2579, 0.234, 0.2518, 0.2419],
        "train_auroc": [0.9656, 0.9745, 0.9754, 0.977, 0.9776, 0.9782],
        "val_auroc": [0.8339, 0.7906, 0.7778, 0.755, 0.7608, 0.7522],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
    "uhh": {
        "train_loss": [1.071, 0.4029, 0.2959, 0.2426, 0.2124, 0.193],
        "val_loss": [4.5072, 5.0343, 5.1966, 5.4812, 5.7134, 5.8817],
        "train_auprc": [0.8954, 0.979, 0.9867, 0.9903, 0.9925, 0.9937],
        "val_auprc": [0.2051, 0.205, 0.2119, 0.2136, 0.2205, 0.212],
        "train_auroc": [0.9841, 0.9971, 0.9977, 0.9984, 0.9987, 0.999],
        "val_auroc": [0.7231, 0.7151, 0.7046, 0.6952, 0.696, 0.6875],
        "learning_rate": [0.00016667, 0.00033333, 0.0005, 0.0005, 0.00049848, 0.00049394],
    },
}


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        c = lowered.get(alias.lower())
        if c is not None:
            return c
    return None


def parse_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    out = pd.DataFrame()
    epoch_col = _find_col(df, ["epoch", "Epoch"])
    if epoch_col is not None:
        out["epoch"] = pd.to_numeric(df[epoch_col], errors="coerce") + 1
    else:
        out["epoch"] = range(1, len(df) + 1)

    for canonical, aliases in METRIC_ALIASES.items():
        c = _find_col(df, aliases)
        if c is not None:
            out[canonical] = pd.to_numeric(df[c], errors="coerce")

    out = out.dropna(subset=["epoch"]).reset_index(drop=True)
    return out


def _parse_metric_pairs(line: str) -> dict[str, float]:
    # Matches fragments like "loss: 0.1234" or "learning_rate: 1.0e-4"
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
    result: dict[str, float] = {}
    for k, v in pairs:
        try:
            result[k] = float(v)
        except ValueError:
            continue
    return result


def parse_training_log(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    rows: list[dict[str, float]] = []
    current_epoch: int | None = None

    for line in lines:
        ep_match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
        if ep_match:
            current_epoch = int(ep_match.group(1))
            continue

        if current_epoch is None:
            continue

        if ":" not in line:
            continue

        metrics = _parse_metric_pairs(line)
        if not metrics:
            continue

        # Heuristic: consider this line an epoch-summary if it has any main metric.
        if not any(k in metrics for k in ("loss", "val_loss", "AUPRC", "AUROC", "val_AUPRC", "val_AUROC", "learning_rate", "lr")):
            continue

        row: dict[str, float] = {"epoch": float(current_epoch)}
        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                if alias in metrics:
                    row[canonical] = metrics[alias]
                    break
        rows.append(row)

    if not rows:
        raise ValueError("No epoch metrics found in log file.")

    df = pd.DataFrame(rows)
    df = df.groupby("epoch", as_index=False).last()
    return df


def synthesize_learning_rate(df: pd.DataFrame, initial_lr: float, total_epochs: int) -> pd.Series:
    # Mirrors birdnet_analyzer.model.train_linear_classifier lr schedule.
    warmup_epochs = min(5, int(total_epochs * 0.1))
    values = []

    for e in df["epoch"].astype(int).tolist():
        epoch_index = e - 1
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            lr = initial_lr * (epoch_index + 1) / warmup_epochs
        else:
            denom = max(1, total_epochs - warmup_epochs)
            progress = (epoch_index - warmup_epochs) / denom
            lr = initial_lr * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        values.append(lr)

    return pd.Series(values, index=df.index, dtype="float64")


def parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    tokens = [t.strip() for t in str(text).split(",")]
    out: list[float] = []
    for t in tokens:
        if not t:
            continue
        out.append(float(t))
    return out


def build_df_from_manual_args(args: argparse.Namespace) -> pd.DataFrame:
    manual_data = {
        "loss": parse_float_list(args.train_loss),
        "val_loss": parse_float_list(args.val_loss),
        "AUPRC": parse_float_list(args.train_auprc),
        "val_AUPRC": parse_float_list(args.val_auprc),
        "AUROC": parse_float_list(args.train_auroc),
        "val_AUROC": parse_float_list(args.val_auroc),
        "learning_rate": parse_float_list(args.learning_rate_list),
    }

    lengths = [len(v) for v in manual_data.values() if len(v) > 0]
    if not lengths:
        raise ValueError("Manual mode requires at least one metric list.")

    n_epochs = args.manual_epochs if args.manual_epochs is not None else lengths[0]
    if n_epochs <= 0:
        raise ValueError("manual_epochs must be > 0")

    df = pd.DataFrame({"epoch": range(1, n_epochs + 1)})

    for col, values in manual_data.items():
        if not values:
            continue
        if len(values) != n_epochs:
            raise ValueError(f"Manual metric '{col}' length={len(values)} does not match epochs={n_epochs}")
        df[col] = values

    return df


def build_df_from_template_data(template_data: dict[str, list[float]]) -> pd.DataFrame:
    manual_data = {
        "loss": template_data.get("train_loss", []),
        "val_loss": template_data.get("val_loss", []),
        "AUPRC": template_data.get("train_auprc", []),
        "val_AUPRC": template_data.get("val_auprc", []),
        "AUROC": template_data.get("train_auroc", []),
        "val_AUROC": template_data.get("val_auroc", []),
        "learning_rate": template_data.get("learning_rate", []),
    }

    lengths = [len(v) for v in manual_data.values() if len(v) > 0]
    if not lengths:
        raise ValueError("MANUAL_TEMPLATE is empty. Fill arrays in the script first.")

    n_epochs = lengths[0]
    for col, values in manual_data.items():
        if values and len(values) != n_epochs:
            raise ValueError(f"Template metric '{col}' length={len(values)} does not match length={n_epochs}")

    df = pd.DataFrame({"epoch": range(1, n_epochs + 1)})
    for col, values in manual_data.items():
        if values:
            df[col] = values
    return df


def build_df_from_template(array_name: str) -> pd.DataFrame:
    if array_name == "custom":
        return build_df_from_template_data(MANUAL_TEMPLATE)

    key = array_name.lower()
    if key not in PRESET_ARRAY_TEMPLATES:
        available = ", ".join(["custom", *sorted(PRESET_ARRAY_TEMPLATES.keys())])
        raise ValueError(f"Unknown array_name '{array_name}'. Available: {available}")

    return build_df_from_template_data(PRESET_ARRAY_TEMPLATES[key])


def plot_metrics(df: pd.DataFrame, output_png: Path, title: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax_loss, ax_auprc, ax_auroc, ax_lr = axes.flatten()

    epochs = df["epoch"]

    if "loss" in df.columns:
        ax_loss.plot(epochs, df["loss"], label="train_loss", linewidth=2)
    if "val_loss" in df.columns:
        ax_loss.plot(epochs, df["val_loss"], label="val_loss", linewidth=2)
    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best")

    if "AUPRC" in df.columns:
        ax_auprc.plot(epochs, df["AUPRC"], label="train_AUPRC", linewidth=2)
    if "val_AUPRC" in df.columns:
        ax_auprc.plot(epochs, df["val_AUPRC"], label="val_AUPRC", linewidth=2)
    ax_auprc.set_title("AUPRC")
    ax_auprc.set_ylabel("AUPRC")
    ax_auprc.grid(alpha=0.3)
    ax_auprc.legend(loc="best")

    if "AUROC" in df.columns:
        ax_auroc.plot(epochs, df["AUROC"], label="train_AUROC", linewidth=2)
    if "val_AUROC" in df.columns:
        ax_auroc.plot(epochs, df["val_AUROC"], label="val_AUROC", linewidth=2)
    ax_auroc.set_title("AUROC")
    ax_auroc.set_xlabel("epoch")
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.grid(alpha=0.3)
    ax_auroc.legend(loc="best")

    if "learning_rate" in df.columns:
        ax_lr.plot(epochs, df["learning_rate"], label="learning_rate", linewidth=2)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("epoch")
    ax_lr.set_ylabel("lr")
    ax_lr.grid(alpha=0.3)
    ax_lr.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot BirdNET per-epoch metrics")
    p.add_argument("--input", default=None, help="History CSV or training log path")
    p.add_argument("--output", required=True, help="Output PNG file path")
    p.add_argument("--format", choices=["auto", "csv", "log", "manual", "array"], default="auto", help="Input file format")
    p.add_argument(
        "--array_name",
        default="custom",
        help="Template name for --format array. Use 'custom' or one of: hsn_multilabel, hsn, nbp, nes, per, pow, sne, ssw, uhh",
    )
    p.add_argument("--title", default="BirdNET Training Curves", help="Figure title")
    p.add_argument(
        "--initial_lr",
        type=float,
        default=None,
        help="If learning rate is missing from input, provide initial lr to synthesize schedule.",
    )
    p.add_argument(
        "--total_epochs",
        type=int,
        default=None,
        help="If synthesizing learning rate, provide planned total epochs.",
    )

    # Manual mode inputs (comma-separated values)
    p.add_argument("--manual_epochs", type=int, default=None, help="Epoch count for manual mode. If omitted, inferred from first metric list.")
    p.add_argument("--train_loss", default=None, help="Comma-separated train loss values")
    p.add_argument("--val_loss", default=None, help="Comma-separated validation loss values")
    p.add_argument("--train_auprc", default=None, help="Comma-separated train AUPRC values")
    p.add_argument("--val_auprc", default=None, help="Comma-separated validation AUPRC values")
    p.add_argument("--train_auroc", default=None, help="Comma-separated train AUROC values")
    p.add_argument("--val_auroc", default=None, help="Comma-separated validation AUROC values")
    p.add_argument("--learning_rate_list", default=None, help="Comma-separated learning rate values")

    p.add_argument("--save_csv", default=None, help="Optional path to save parsed metrics table")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.format == "array":
        df = build_df_from_template(args.array_name)
    elif args.format == "manual":
        df = build_df_from_manual_args(args)
    else:
        if not args.input:
            raise ValueError("--input is required unless --format manual is used")

        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if args.format == "csv":
            df = parse_history_csv(input_path)
        elif args.format == "log":
            df = parse_training_log(input_path)
        elif input_path.suffix.lower() in {".csv", ".tsv"}:
            df = parse_history_csv(input_path)
        else:
            df = parse_training_log(input_path)

    # Synthesize learning-rate curve if input doesn't include it.
    if "learning_rate" not in df.columns and args.initial_lr is not None and args.total_epochs is not None:
        df["learning_rate"] = synthesize_learning_rate(df, args.initial_lr, args.total_epochs)

    required = ["loss", "val_loss", "AUPRC", "val_AUPRC", "AUROC", "val_AUROC"]
    missing = [m for m in required if m not in df.columns]
    if missing:
        print(f"Warning: missing columns for some requested curves: {missing}")

    plot_metrics(df, Path(args.output), args.title)

    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    print(f"Saved plot: {args.output}")
    if args.save_csv:
        print(f"Saved parsed table: {args.save_csv}")
    print(f"Parsed epochs: {len(df)}")


if __name__ == "__main__":
    main()
