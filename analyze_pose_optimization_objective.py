"""Summarize one pose-optimization run's trajectory objective contributions.

The script takes a Markdown run record written by ``optimize_precomputed_poses``,
extracts its embedded JSON config, finds matching per-object debug NPZ files
under the recorded data root, and reports how score, translation penalty, and
rotation penalty contribute to the selected trajectory objective:

    score - trans_lambda * trans_cost - rot_lambda * rot_cost
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SUMMARY_FILENAME = "summary.md"
OBJECT_SUMMARY_FILENAME = "object_summary.csv"
FRAME_DETAILS_FILENAME = "frame_details.csv"
EPSILON = 1e-12

REQUIRED_DEBUG_KEYS = (
    "frame_index",
    "selected_candidate_index",
    "selected_adjusted_score",
    "score_rank",
    "translation_cost",
    "rotation_cost",
    "weighted_translation_penalty",
    "weighted_rotation_penalty",
    "net_contribution",
    "cumulative_score",
)

OBJECT_SUMMARY_FIELDNAMES = [
    "object_dir",
    "num_frames",
    "score_sum",
    "score_mean",
    "score_median",
    "score_p90",
    "score_p95",
    "score_max",
    "translation_penalty_sum",
    "translation_penalty_mean",
    "translation_penalty_p95",
    "translation_penalty_max",
    "rotation_penalty_sum",
    "rotation_penalty_mean",
    "rotation_penalty_p95",
    "rotation_penalty_max",
    "total_penalty_sum",
    "total_penalty_to_score_ratio",
    "translation_penalty_to_score_ratio",
    "rotation_penalty_to_score_ratio",
    "net_sum",
    "net_mean",
    "negative_net_frame_count",
    "negative_net_frame_ratio",
    "non_rank1_frame_count",
    "non_rank1_frame_ratio",
    "mean_score_rank",
    "max_score_rank",
    "translation_cost_mean",
    "translation_cost_p95",
    "translation_cost_max",
    "rotation_cost_mean",
    "rotation_cost_p95",
    "rotation_cost_max",
]

FRAME_DETAILS_FIELDNAMES = [
    "object_dir",
    "frame_index",
    "selected_candidate_index",
    "selected_adjusted_score",
    "score_rank",
    "translation_cost",
    "rotation_cost",
    "weighted_translation_penalty",
    "weighted_rotation_penalty",
    "total_penalty",
    "total_penalty_to_score_ratio",
    "translation_penalty_to_score_ratio",
    "rotation_penalty_to_score_ratio",
    "net_contribution",
    "cumulative_score",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze one pose optimization run's objective contributions."
    )
    parser.add_argument(
        "run_record",
        help="Markdown run record path. Relative paths are resolved from repo root.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for summary.md and CSV outputs. Defaults to "
            "<run_record_dir>/<run_id>_objective_summary/."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of highest-contribution objects/frames to show in summary.md.",
    )
    parser.add_argument(
        "--write-frame-details",
        action="store_true",
        help="Write per-frame frame_details.csv in addition to summary outputs.",
    )
    return parser.parse_args(argv)


def resolve_run_record_path(path_arg):
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_run_record(path):
    text = Path(path).read_text()
    for match in re.finditer(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"{path} does not contain a parseable fenced json block")


def find_debug_files(data_root, debug_filename):
    return sorted(Path(data_root).rglob(debug_filename))


def load_debug_npz(path):
    with np.load(path) as data:
        missing = [key for key in REQUIRED_DEBUG_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"missing required keys: {', '.join(missing)}")
        arrays = {
            key: np.asarray(data[key]).reshape(-1)
            for key in REQUIRED_DEBUG_KEYS
        }

    lengths = {key: len(value) for key, value in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        details = ", ".join(f"{key}={value}" for key, value in sorted(lengths.items()))
        raise ValueError(f"debug arrays have inconsistent lengths: {details}")
    if next(iter(unique_lengths), 0) == 0:
        raise ValueError("debug file has no selected frames")
    return arrays


def safe_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "count": 0,
            "sum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(len(finite)),
        "sum": float(np.sum(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def safe_ratio(numerator, denominator):
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return np.nan
    if abs(denominator) <= EPSILON:
        return np.nan
    return numerator / denominator


def _object_dir_for_debug(debug_path, data_root):
    object_path = Path(debug_path).parent
    try:
        return object_path.relative_to(data_root).as_posix()
    except ValueError:
        return str(object_path)


def build_object_summary(debug_path, data_root, arrays):
    object_dir = _object_dir_for_debug(debug_path, data_root)
    scores = arrays["selected_adjusted_score"].astype(np.float64)
    trans_penalty = arrays["weighted_translation_penalty"].astype(np.float64)
    rot_penalty = arrays["weighted_rotation_penalty"].astype(np.float64)
    total_penalty = trans_penalty + rot_penalty
    net = arrays["net_contribution"].astype(np.float64)
    score_rank = arrays["score_rank"].astype(np.float64)
    trans_cost = arrays["translation_cost"].astype(np.float64)
    rot_cost = arrays["rotation_cost"].astype(np.float64)

    score_stats = safe_stats(scores)
    trans_penalty_stats = safe_stats(trans_penalty)
    rot_penalty_stats = safe_stats(rot_penalty)
    net_stats = safe_stats(net)
    rank_stats = safe_stats(score_rank)
    trans_cost_stats = safe_stats(trans_cost)
    rot_cost_stats = safe_stats(rot_cost)
    num_frames = len(scores)
    score_sum = score_stats["sum"]
    trans_penalty_sum = trans_penalty_stats["sum"]
    rot_penalty_sum = rot_penalty_stats["sum"]
    total_penalty_sum = float(np.nansum(total_penalty))
    negative_net_count = int(np.sum(np.isfinite(net) & (net < 0)))
    non_rank1_count = int(np.sum(np.isfinite(score_rank) & (score_rank != 1)))

    return {
        "object_dir": object_dir,
        "num_frames": num_frames,
        "score_sum": score_sum,
        "score_mean": score_stats["mean"],
        "score_median": score_stats["median"],
        "score_p90": score_stats["p90"],
        "score_p95": score_stats["p95"],
        "score_max": score_stats["max"],
        "translation_penalty_sum": trans_penalty_sum,
        "translation_penalty_mean": trans_penalty_stats["mean"],
        "translation_penalty_p95": trans_penalty_stats["p95"],
        "translation_penalty_max": trans_penalty_stats["max"],
        "rotation_penalty_sum": rot_penalty_sum,
        "rotation_penalty_mean": rot_penalty_stats["mean"],
        "rotation_penalty_p95": rot_penalty_stats["p95"],
        "rotation_penalty_max": rot_penalty_stats["max"],
        "total_penalty_sum": total_penalty_sum,
        "total_penalty_to_score_ratio": safe_ratio(total_penalty_sum, score_sum),
        "translation_penalty_to_score_ratio": safe_ratio(trans_penalty_sum, score_sum),
        "rotation_penalty_to_score_ratio": safe_ratio(rot_penalty_sum, score_sum),
        "net_sum": net_stats["sum"],
        "net_mean": net_stats["mean"],
        "negative_net_frame_count": negative_net_count,
        "negative_net_frame_ratio": safe_ratio(negative_net_count, num_frames),
        "non_rank1_frame_count": non_rank1_count,
        "non_rank1_frame_ratio": safe_ratio(non_rank1_count, num_frames),
        "mean_score_rank": rank_stats["mean"],
        "max_score_rank": rank_stats["max"],
        "translation_cost_mean": trans_cost_stats["mean"],
        "translation_cost_p95": trans_cost_stats["p95"],
        "translation_cost_max": trans_cost_stats["max"],
        "rotation_cost_mean": rot_cost_stats["mean"],
        "rotation_cost_p95": rot_cost_stats["p95"],
        "rotation_cost_max": rot_cost_stats["max"],
    }


def build_frame_rows(debug_path, data_root, arrays):
    object_dir = _object_dir_for_debug(debug_path, data_root)
    scores = arrays["selected_adjusted_score"].astype(np.float64)
    trans_penalty = arrays["weighted_translation_penalty"].astype(np.float64)
    rot_penalty = arrays["weighted_rotation_penalty"].astype(np.float64)
    total_penalty = trans_penalty + rot_penalty
    rows = []
    for idx in range(len(scores)):
        score = scores[idx]
        rows.append(
            {
                "object_dir": object_dir,
                "frame_index": int(arrays["frame_index"][idx]),
                "selected_candidate_index": int(arrays["selected_candidate_index"][idx]),
                "selected_adjusted_score": score,
                "score_rank": int(arrays["score_rank"][idx]),
                "translation_cost": float(arrays["translation_cost"][idx]),
                "rotation_cost": float(arrays["rotation_cost"][idx]),
                "weighted_translation_penalty": float(trans_penalty[idx]),
                "weighted_rotation_penalty": float(rot_penalty[idx]),
                "total_penalty": float(total_penalty[idx]),
                "total_penalty_to_score_ratio": safe_ratio(total_penalty[idx], score),
                "translation_penalty_to_score_ratio": safe_ratio(trans_penalty[idx], score),
                "rotation_penalty_to_score_ratio": safe_ratio(rot_penalty[idx], score),
                "net_contribution": float(arrays["net_contribution"][idx]),
                "cumulative_score": float(arrays["cumulative_score"][idx]),
            }
        )
    return rows


def aggregate_dataset(object_rows, frame_rows):
    score_sum = float(np.nansum([row["score_sum"] for row in object_rows]))
    trans_penalty_sum = float(
        np.nansum([row["translation_penalty_sum"] for row in object_rows])
    )
    rot_penalty_sum = float(
        np.nansum([row["rotation_penalty_sum"] for row in object_rows])
    )
    total_penalty_sum = trans_penalty_sum + rot_penalty_sum
    net_sum = float(np.nansum([row["net_sum"] for row in object_rows]))
    frame_count = int(np.nansum([row["num_frames"] for row in object_rows]))
    negative_net_count = int(
        np.nansum([row["negative_net_frame_count"] for row in object_rows])
    )
    non_rank1_count = int(
        np.nansum([row["non_rank1_frame_count"] for row in object_rows])
    )
    return {
        "object_count": len(object_rows),
        "frame_count": frame_count,
        "score_sum": score_sum,
        "translation_penalty_sum": trans_penalty_sum,
        "rotation_penalty_sum": rot_penalty_sum,
        "total_penalty_sum": total_penalty_sum,
        "net_sum": net_sum,
        "total_penalty_to_score_ratio": safe_ratio(total_penalty_sum, score_sum),
        "translation_penalty_to_score_ratio": safe_ratio(trans_penalty_sum, score_sum),
        "rotation_penalty_to_score_ratio": safe_ratio(rot_penalty_sum, score_sum),
        "negative_net_frame_ratio": safe_ratio(negative_net_count, frame_count),
        "non_rank1_frame_ratio": safe_ratio(non_rank1_count, frame_count),
    }


def _format_csv_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"
    return value


def write_object_summary_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OBJECT_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _format_csv_value(row.get(key, "")) for key in OBJECT_SUMMARY_FIELDNAMES}
            )


def write_frame_details_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_DETAILS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _format_csv_value(row.get(key, "")) for key in FRAME_DETAILS_FIELDNAMES}
            )


def _format_md_number(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def _append_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def _top_rows(rows, metric, top_k):
    finite_rows = [
        row for row in rows if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
    ]
    return sorted(finite_rows, key=lambda row: row[metric], reverse=True)[:top_k]


def _append_top_object_section(lines, title, metric, rows, top_k):
    lines.append(f"## {title}")
    top = _top_rows(rows, metric, top_k)
    if not top:
        lines.extend(["No finite rows.", ""])
        return
    _append_table(
        lines,
        ["object_dir", metric, "num_frames", "score_sum", "net_sum"],
        [
            [
                row["object_dir"],
                _format_md_number(row[metric]),
                row["num_frames"],
                _format_md_number(row["score_sum"]),
                _format_md_number(row["net_sum"]),
            ]
            for row in top
        ],
    )


def _append_top_frame_section(lines, title, metric, rows, top_k):
    lines.append(f"## {title}")
    top = _top_rows(rows, metric, top_k)
    if not top:
        lines.extend(["No finite rows.", ""])
        return
    _append_table(
        lines,
        ["object_dir", "frame_index", metric, "score", "net_contribution"],
        [
            [
                row["object_dir"],
                row["frame_index"],
                _format_md_number(row[metric]),
                _format_md_number(row["selected_adjusted_score"]),
                _format_md_number(row["net_contribution"]),
            ]
            for row in top
        ],
    )


def write_summary_md(
    path,
    run_record,
    data_root,
    debug_filename,
    debug_file_count,
    object_rows,
    frame_rows,
    dataset,
    warnings,
    top_k,
):
    params = run_record.get("optimization_parameters", {})
    lines = [
        f"# Pose Optimization Objective Summary: {run_record.get('run_id')}",
        "",
        f"- run_id: `{run_record.get('run_id')}`",
        f"- data_root: `{data_root}`",
        f"- debug file name: `{debug_filename}`",
        f"- debug files found: `{debug_file_count}`",
        "",
        "## Optimization Parameters",
        "",
    ]
    _append_table(
        lines,
        ["parameter", "value"],
        [
            ["trans_lambda", _format_md_number(params.get("trans_lambda", np.nan))],
            ["rot_lambda", _format_md_number(params.get("rot_lambda", np.nan))],
            ["max_invalid_gap", _format_md_number(params.get("max_invalid_gap", np.nan))],
            ["smooth_window", _format_md_number(params.get("smooth_window", np.nan))],
            ["smooth_polyorder", _format_md_number(params.get("smooth_polyorder", np.nan))],
        ],
    )

    lines.append("## Dataset-Level Contributions")
    _append_table(
        lines,
        ["component", "sum"],
        [
            ["score", _format_md_number(dataset["score_sum"])],
            ["translation penalty", _format_md_number(dataset["translation_penalty_sum"])],
            ["rotation penalty", _format_md_number(dataset["rotation_penalty_sum"])],
            ["total penalty", _format_md_number(dataset["total_penalty_sum"])],
            ["net contribution", _format_md_number(dataset["net_sum"])],
        ],
    )

    lines.append("## Dataset-Level Ratios")
    _append_table(
        lines,
        ["metric", "value"],
        [
            [
                "total penalty / score",
                _format_md_number(dataset["total_penalty_to_score_ratio"]),
            ],
            [
                "translation penalty / score",
                _format_md_number(dataset["translation_penalty_to_score_ratio"]),
            ],
            [
                "rotation penalty / score",
                _format_md_number(dataset["rotation_penalty_to_score_ratio"]),
            ],
            [
                "negative net frame ratio",
                _format_md_number(dataset["negative_net_frame_ratio"]),
            ],
            [
                "non-rank1 frame ratio",
                _format_md_number(dataset["non_rank1_frame_ratio"]),
            ],
        ],
    )

    _append_top_object_section(
        lines,
        f"Top-{top_k} Objects by Total Penalty / Score",
        "total_penalty_to_score_ratio",
        object_rows,
        top_k,
    )
    _append_top_object_section(
        lines,
        f"Top-{top_k} Objects by Translation Penalty / Score",
        "translation_penalty_to_score_ratio",
        object_rows,
        top_k,
    )
    _append_top_object_section(
        lines,
        f"Top-{top_k} Objects by Rotation Penalty / Score",
        "rotation_penalty_to_score_ratio",
        object_rows,
        top_k,
    )
    _append_top_object_section(
        lines,
        f"Top-{top_k} Objects by Negative Net Frame Ratio",
        "negative_net_frame_ratio",
        object_rows,
        top_k,
    )
    _append_top_object_section(
        lines,
        f"Top-{top_k} Objects by Non-Rank1 Frame Ratio",
        "non_rank1_frame_ratio",
        object_rows,
        top_k,
    )
    _append_top_frame_section(
        lines,
        f"Top-{top_k} Frames by Total Penalty / Score",
        "total_penalty_to_score_ratio",
        frame_rows,
        top_k,
    )
    _append_top_frame_section(
        lines,
        f"Top-{top_k} Frames by Weighted Translation Penalty",
        "weighted_translation_penalty",
        frame_rows,
        top_k,
    )
    _append_top_frame_section(
        lines,
        f"Top-{top_k} Frames by Weighted Rotation Penalty",
        "weighted_rotation_penalty",
        frame_rows,
        top_k,
    )

    lines.append("## Warnings")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _validate_run_record(run_record):
    run_id = run_record.get("run_id")
    data_root = run_record.get("data_root")
    if not run_id:
        raise ValueError("run record JSON is missing required key: run_id")
    if not data_root:
        raise ValueError("run record JSON is missing required key: data_root")
    debug_filename = run_record.get("output_naming", {}).get("debug")
    if not debug_filename:
        debug_filename = f"poses_optimized_{run_id}_debug.npz"
    return run_id, Path(data_root).expanduser(), debug_filename


def _process_debug_files(debug_files, data_root):
    warnings = []
    object_rows = []
    frame_rows = []
    for debug_path in debug_files:
        try:
            arrays = load_debug_npz(debug_path)
            object_rows.append(build_object_summary(debug_path, data_root, arrays))
            frame_rows.extend(build_frame_rows(debug_path, data_root, arrays))
        except Exception as exc:
            warnings.append(f"{debug_path}: {exc}")
    return object_rows, frame_rows, warnings


def main(argv=None):
    args = parse_args(argv)
    try:
        run_record_path = resolve_run_record_path(args.run_record)
        if not run_record_path.exists():
            print(f"ERROR: run record does not exist: {run_record_path}", file=sys.stderr)
            return 1

        run_record = load_run_record(run_record_path)
        run_id, data_root, debug_filename = _validate_run_record(run_record)
        if not data_root.exists():
            print(f"ERROR: data_root does not exist: {data_root}", file=sys.stderr)
            return 1

        debug_files = find_debug_files(data_root, debug_filename)
        if not debug_files:
            print(
                f"ERROR: no debug npz files named {debug_filename} under {data_root}",
                file=sys.stderr,
            )
            return 1

        object_rows, frame_rows, warnings = _process_debug_files(debug_files, data_root)
        if not object_rows:
            print("ERROR: no valid debug npz files were loaded", file=sys.stderr)
            for warning in warnings:
                print(f"WARNING: {warning}", file=sys.stderr)
            return 1

        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else run_record_path.parent / f"{run_id}_objective_summary"
        )
        dataset = aggregate_dataset(object_rows, frame_rows)
        write_object_summary_csv(object_rows, output_dir / OBJECT_SUMMARY_FILENAME)
        if args.write_frame_details:
            write_frame_details_csv(frame_rows, output_dir / FRAME_DETAILS_FILENAME)
        write_summary_md(
            output_dir / SUMMARY_FILENAME,
            run_record,
            data_root,
            debug_filename,
            len(debug_files),
            object_rows,
            frame_rows,
            dataset,
            warnings,
            max(args.top_k, 0),
        )
        print(f"Wrote summary to {output_dir / SUMMARY_FILENAME}")
        print(f"Wrote object summary to {output_dir / OBJECT_SUMMARY_FILENAME}")
        if args.write_frame_details:
            print(f"Wrote frame details to {output_dir / FRAME_DETAILS_FILENAME}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
