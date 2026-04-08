#!/usr/bin/env python3
"""Compute BirdSet-style metrics from BirdNET evaluation inputs.

Metrics:
- cmAP: class-mean Average Precision (macro AP)
- AUROC: macro AUROC
- T1-Acc: top-1 accuracy

Inputs follow birdnet_analyzer.evaluation conventions:
- annotation file/folder
- prediction file/folder
- optional column mappings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from birdnet_analyzer.evaluation import process_data


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.sum(y_true == 1) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _top1_acc(predictions: np.ndarray, labels: np.ndarray, positive_only: bool = True) -> float:
    if predictions.size == 0:
        return float("nan")

    top1 = np.argmax(predictions, axis=1)

    if positive_only:
        has_pos = labels.sum(axis=1) > 0
        if np.sum(has_pos) == 0:
            return float("nan")
        idx = np.where(has_pos)[0]
        hits = [int(labels[i, top1[i]] == 1) for i in idx]
        return float(np.mean(hits))

    hits = [int(labels[i, top1[i]] == 1) for i in range(labels.shape[0])]
    return float(np.mean(hits))


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cmAP, AUROC and T1-Acc from BirdNET evaluation inputs")
    parser.add_argument("--annotation_path", required=True, help="Path to annotation file or folder")
    parser.add_argument("--prediction_path", required=True, help="Path to prediction file or folder")
    parser.add_argument("--mapping_path", help="Optional class mapping JSON path")
    parser.add_argument("--sample_duration", type=float, default=3.0)
    parser.add_argument("--min_overlap", type=float, default=0.5)
    parser.add_argument("--recording_duration", type=float)
    parser.add_argument("--columns_annotations", type=json.loads, help="JSON string for annotation column mapping")
    parser.add_argument("--columns_predictions", type=json.loads, help="JSON string for prediction column mapping")
    parser.add_argument("--selected_classes", nargs="+", help="Optional selected classes")
    parser.add_argument("--selected_recordings", nargs="+", help="Optional selected recordings")
    parser.add_argument("--output_json", help="Optional output json file path")

    args = parser.parse_args()

    _, _, predictions, labels = process_data(
        annotation_path=args.annotation_path,
        prediction_path=args.prediction_path,
        mapping_path=args.mapping_path,
        sample_duration=args.sample_duration,
        min_overlap=args.min_overlap,
        recording_duration=args.recording_duration,
        columns_annotations=args.columns_annotations,
        columns_predictions=args.columns_predictions,
        selected_classes=args.selected_classes,
        selected_recordings=args.selected_recordings,
        metrics_list=("ap", "auroc"),
        threshold=0.5,
        class_wise=True,
    )

    num_classes = predictions.shape[1]

    ap_per_class = [_safe_ap(labels[:, i], predictions[:, i]) for i in range(num_classes)]
    auroc_per_class = [_safe_auroc(labels[:, i], predictions[:, i]) for i in range(num_classes)]

    result = {
        "num_samples": int(predictions.shape[0]),
        "num_classes": int(num_classes),
        "cmAP": _nanmean(ap_per_class),
        "AUROC": _nanmean(auroc_per_class),
        "T1_Acc": _top1_acc(predictions, labels, positive_only=True),
        "T1_Acc_all_samples": _top1_acc(predictions, labels, positive_only=False),
        "valid_ap_classes": int(np.sum(~np.isnan(ap_per_class))),
        "valid_auroc_classes": int(np.sum(~np.isnan(auroc_per_class))),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
