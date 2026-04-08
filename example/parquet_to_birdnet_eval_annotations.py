import argparse
import ast
import os

import pandas as pd


def _normalize_events(value):
    """Normalize detected_events to a list of [start, end] pairs."""
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
        except (SyntaxError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []

    return []


def _recording_basename(path_or_name):
    if not isinstance(path_or_name, str):
        return ""
    return os.path.basename(path_or_name)


def _is_missing(v):
    if v is None:
        return True

    # Array-like containers are not "missing" as a whole; parse elements later.
    if isinstance(v, (list, tuple, set, dict)):
        return False

    if isinstance(v, str):
        return v.strip() == ""

    try:
        is_na = pd.isna(v)
    except Exception:
        return False

    # pd.isna can return scalar bool or array-like booleans.
    if isinstance(is_na, bool):
        return is_na
    if hasattr(is_na, "all"):
        try:
            return bool(is_na.all())
        except Exception:
            return False

    return False


def _normalize_labels(value):
    """Normalize a single label or a multilabel field into a list of class codes."""
    if _is_missing(value):
        return []

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if not _is_missing(x)]
            except (SyntaxError, ValueError):
                pass
        return [s]

    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if not _is_missing(x)]

    # Handle numpy array and other list-like containers coming from parquet object columns.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            as_list = value.tolist()
        except Exception:
            as_list = None
        if isinstance(as_list, list):
            return [str(x) for x in as_list if not _is_missing(x)]

    return [str(value)]


def _pick_recording(r, recording_col):
    # HSN test/test_5s often store filename in index, train uses filepath column.
    if recording_col in r and isinstance(r.get(recording_col), str):
        return _recording_basename(r.get(recording_col))

    idx = r.name
    if isinstance(idx, str) and idx:
        return _recording_basename(idx)

    return ""


def convert_parquet_to_annotation_tsv(
    parquet_path,
    output_tsv,
    recording_col="filepath",
    class_col="ebird_code",
    multilabel_col="ebird_code_multilabel",
    events_col="detected_events",
    duration_col="length",
):
    df = pd.read_parquet(parquet_path)

    has_recording = recording_col in df.columns or isinstance(df.index, pd.Index)
    if not has_recording:
        raise ValueError("Could not find recording names in either filepath column or dataframe index")

    has_events = events_col in df.columns
    has_segments = "start_time" in df.columns and "end_time" in df.columns

    if not has_events and not has_segments:
        raise ValueError("Parquet must contain detected_events or (start_time, end_time) columns")

    rows = []
    for _, r in df.iterrows():
        recording = _pick_recording(r, recording_col)

        labels = []
        if class_col in df.columns:
            labels.extend(_normalize_labels(r.get(class_col)))
        if multilabel_col in df.columns:
            labels.extend(_normalize_labels(r.get(multilabel_col)))

        # Deduplicate while preserving order
        labels = list(dict.fromkeys([lb for lb in labels if lb and lb.lower() != "none"]))

        duration = r.get(duration_col) if duration_col in df.columns else None

        if has_events:
            events = _normalize_events(r.get(events_col))
            for ev in events:
                if not isinstance(ev, (list, tuple)) or len(ev) < 2:
                    continue
                start_time = ev[0]
                end_time = ev[1]
                if start_time is None or end_time is None:
                    continue
                for klass in labels:
                    rows.append(
                        {
                            "Start Time": float(start_time),
                            "End Time": float(end_time),
                            "Class": str(klass),
                            "Recording": recording,
                            "Duration": float(duration) if duration is not None else None,
                        }
                    )

        elif has_segments:
            start_time = r.get("start_time")
            end_time = r.get("end_time")
            if _is_missing(start_time) or _is_missing(end_time):
                continue
            for klass in labels:
                rows.append(
                    {
                        "Start Time": float(start_time),
                        "End Time": float(end_time),
                        "Class": str(klass),
                        "Recording": recording,
                        "Duration": float(duration) if duration is not None else None,
                    }
                )

    out_df = pd.DataFrame(rows, columns=["Start Time", "End Time", "Class", "Recording", "Duration"])
    out_df = out_df.sort_values(["Recording", "Start Time", "End Time"], na_position="last") if not out_df.empty else out_df

    out_dir = os.path.dirname(output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Wrote {len(out_df)} annotation rows to {output_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet metadata to BirdNET evaluation annotation TSV.")
    parser.add_argument("--parquet", required=True, help="Input parquet file path")
    parser.add_argument("--output", required=True, help="Output annotation .txt/.tsv path")
    parser.add_argument("--recording_col", default="filepath", help="Column containing audio filename or path")
    parser.add_argument("--class_col", default="ebird_code", help="Column containing class label")
    parser.add_argument("--multilabel_col", default="ebird_code_multilabel", help="Column containing multilabel values")
    parser.add_argument("--events_col", default="detected_events", help="Column containing event list")
    parser.add_argument("--duration_col", default="length", help="Column containing file duration (optional)")

    args = parser.parse_args()

    convert_parquet_to_annotation_tsv(
        parquet_path=args.parquet,
        output_tsv=args.output,
        recording_col=args.recording_col,
        class_col=args.class_col,
        multilabel_col=args.multilabel_col,
        events_col=args.events_col,
        duration_col=args.duration_col,
    )
