"""Analyze per-frame pose candidate score distributions as confidence signals."""

import argparse
import csv
import io
import math
import pickle
import traceback
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"
CANDIDATES_FILENAME = "all_poses&scores.pkl"
DEBUG_DIRNAME = "debug"
FRAME_CSV_FILENAME = "score_confidence_by_frame.csv"
OBJECT_CSV_FILENAME = "score_confidence_by_object.csv"
CURVES_FILENAME = "score_confidence_curves.png"
DISTRIBUTIONS_FILENAME = "score_confidence_distributions.png"
HISTOGRAM_DIRNAME = "score_histograms"
HISTOGRAM_FILENAME = "frame_{frame_idx:06d}.png"

IDENTITY_FIELDNAMES = ["seq_dir", "object_dir", "object_name", "frame_idx"]
METRIC_FIELDNAMES = [
    "candidate_count",
    "finite_score_count",
    "nonfinite_score_count",
    "top1_score",
    "top2_score",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "score_range",
    "top1_top2_gap",
    "top1_mean_gap",
    "top1_median_gap",
    "top1_zscore",
    "top1_prob",
    "top2_prob",
    "prob_gap",
    "top5_prob_mass",
    "entropy_nats",
    "entropy_norm",
    "confidence_entropy",
    "effective_candidate_count",
]
FRAME_CSV_FIELDNAMES = IDENTITY_FIELDNAMES + METRIC_FIELDNAMES
OBJECT_AGGREGATE_METRICS = [
    "top1_top2_gap",
    "entropy_norm",
    "confidence_entropy",
    "top1_prob",
    "effective_candidate_count",
    "finite_score_count",
]
OBJECT_AGGREGATE_STATS = ["mean", "std", "p10", "p50", "p90"]
OBJECT_AGGREGATE_FIELDNAMES = ["seq_dir", "object_dir", "object_name"] + [
    f"{metric}_{stat}"
    for metric in OBJECT_AGGREGATE_METRICS
    for stat in OBJECT_AGGREGATE_STATS
]


def positive_float(value):
    """Parse a strictly positive float for argparse."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args(argv=None):
    """Parse command-line arguments for score confidence analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pose candidate score distributions as per-frame confidence "
            "signals."
        )
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root of the processed DexYCB dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate debug visualizations even when expected outputs exist.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Number of bins to use for optional per-frame score histograms.",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=1.0,
        help="Positive softmax temperature for probability and entropy metrics.",
    )
    parser.add_argument(
        "--save-score-histograms",
        action="store_true",
        help="Save raw score histograms for every frame.",
    )
    return parser.parse_args(argv)


def load_scores(candidates_path):
    """Load per-frame candidate scores from an ``all_poses&scores.pkl`` file."""
    with open(candidates_path, "rb") as f:
        candidates = pickle.load(f)
    if "scores" not in candidates:
        raise KeyError(f"{candidates_path} must contain 'scores'")
    return candidates["scores"]


def flattened_scores(scores):
    """Return one frame's scores as a flat float64 array."""
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def finite_scores(scores):
    """Return one frame's finite scores as a flat float64 array."""
    scores = flattened_scores(scores)
    return scores[np.isfinite(scores)]


def stable_softmax(scores, temperature=1.0):
    """Compute stable softmax probabilities from finite raw scores."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0:
        return np.empty((0,), dtype=np.float64)

    scaled_scores = scores / temperature
    shifted = scaled_scores - scaled_scores.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def _nan_metric_dict():
    return {field: np.nan for field in METRIC_FIELDNAMES}


def compute_frame_score_metrics(scores, temperature=1.0):
    """Compute score-distribution confidence metrics for one frame."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    raw_scores = flattened_scores(scores)
    finite = raw_scores[np.isfinite(raw_scores)]
    finite_desc = np.sort(finite)[::-1]
    finite_count = len(finite_desc)

    metrics = _nan_metric_dict()
    metrics["candidate_count"] = int(len(raw_scores))
    metrics["finite_score_count"] = int(finite_count)
    metrics["nonfinite_score_count"] = int(len(raw_scores) - finite_count)

    if finite_count == 0:
        return metrics

    probabilities = stable_softmax(finite_desc, temperature=temperature)
    positive_probabilities = probabilities[probabilities > 0]
    entropy_nats = float(
        -np.sum(positive_probabilities * np.log(positive_probabilities))
    )
    score_mean = float(np.mean(finite_desc))
    score_std = float(np.std(finite_desc))
    top1_score = float(finite_desc[0])

    metrics.update(
        {
            "top1_score": top1_score,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_min": float(np.min(finite_desc)),
            "score_max": float(np.max(finite_desc)),
            "score_range": float(np.max(finite_desc) - np.min(finite_desc)),
            "top1_mean_gap": top1_score - score_mean,
            "top1_median_gap": top1_score - float(np.median(finite_desc)),
            "top1_zscore": (
                (top1_score - score_mean) / score_std if score_std > 0 else np.nan
            ),
            "top1_prob": float(probabilities[0]),
            "top5_prob_mass": float(np.sum(probabilities[:5])),
            "entropy_nats": entropy_nats,
            "entropy_norm": (
                entropy_nats / math.log(finite_count) if finite_count > 1 else 0.0
            ),
            "effective_candidate_count": float(math.exp(entropy_nats)),
        }
    )
    metrics["confidence_entropy"] = 1.0 - metrics["entropy_norm"]

    if finite_count > 1:
        metrics.update(
            {
                "top2_score": float(finite_desc[1]),
                "top1_top2_gap": float(finite_desc[0] - finite_desc[1]),
                "top2_prob": float(probabilities[1]),
                "prob_gap": float(probabilities[0] - probabilities[1]),
            }
        )

    return metrics


def frame_score_records(object_dir, all_scores, temperature=1.0):
    """Return CSV-ready per-frame score confidence records for one object."""
    object_dir = Path(object_dir)
    seq_dir = object_dir.parents[2]
    records = []
    for frame_idx, scores in enumerate(all_scores):
        record = OrderedDict(
            [
                ("seq_dir", str(seq_dir)),
                ("object_dir", str(object_dir)),
                ("object_name", object_dir.name),
                ("frame_idx", frame_idx),
            ]
        )
        record.update(compute_frame_score_metrics(scores, temperature=temperature))
        records.append(record)
    return records


def _write_csv(records, csv_path, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_frame_csv(records, csv_path):
    """Write all per-frame score confidence records to a CSV file."""
    _write_csv(records, csv_path, FRAME_CSV_FIELDNAMES)


def write_object_csv(records, csv_path):
    """Write per-object aggregate score confidence records to a CSV file."""
    _write_csv(records, csv_path, OBJECT_AGGREGATE_FIELDNAMES)


def _finite_record_values(records, metric):
    values = np.array([record[metric] for record in records], dtype=np.float64)
    return values[np.isfinite(values)]


def _aggregate_values(values):
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def aggregate_object_records(frame_records):
    """Aggregate per-frame records into one row per object."""
    grouped = OrderedDict()
    for record in frame_records:
        key = (record["seq_dir"], record["object_dir"], record["object_name"])
        grouped.setdefault(key, []).append(record)

    aggregate_records = []
    for (seq_dir, object_dir, object_name), records in grouped.items():
        row = OrderedDict(
            [
                ("seq_dir", seq_dir),
                ("object_dir", object_dir),
                ("object_name", object_name),
            ]
        )
        for metric in OBJECT_AGGREGATE_METRICS:
            values = _finite_record_values(records, metric)
            stats = _aggregate_values(values)
            for stat in OBJECT_AGGREGATE_STATS:
                row[f"{metric}_{stat}"] = stats[stat]
        aggregate_records.append(row)

    return aggregate_records


def _records_to_metric_array(records, metric):
    return np.array([record[metric] for record in records], dtype=np.float64)


def plot_confidence_curves(records, save_path):
    """Write time curves for key score confidence metrics."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _records_to_metric_array(records, "frame_idx")
    plot_specs = [
        ("top1_top2_gap", "Top1-top2 gap"),
        ("entropy_norm", "Normalized entropy"),
        ("top1_prob", "Top1 probability"),
        ("effective_candidate_count", "Effective candidates"),
        ("finite_score_count", "Finite score count"),
    ]

    fig, axes = plt.subplots(
        len(plot_specs), 1, figsize=(10, 10), sharex=True, constrained_layout=True
    )
    for ax, (metric, ylabel) in zip(axes, plot_specs):
        ax.plot(frame_indices, _records_to_metric_array(records, metric), linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Pose score confidence metrics", fontsize=12)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confidence_distributions(records, save_path, bins):
    """Write cross-frame histograms for key score confidence metrics."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("top1_top2_gap", "Top1-top2 gap"),
        ("entropy_norm", "Normalized entropy"),
        ("top1_prob", "Top1 probability"),
        ("effective_candidate_count", "Effective candidates"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.reshape(-1), plot_specs):
        values = _records_to_metric_array(records, metric)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_ylabel("Frame count")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Pose score confidence distributions", fontsize=12)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_histogram(scores, save_path, bins):
    """Write one raw-score histogram PNG, annotating frames without finite scores."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scores = finite_scores(scores)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if len(scores) == 0:
        ax.text(
            0.5,
            0.5,
            "No finite scores",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(scores, bins=bins)
    ax.set_xlabel("Raw score")
    ax.set_ylabel("Count")
    ax.set_title("Frame score histogram")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def output_paths(object_dir, frame_count, save_score_histograms=False):
    """Return expected debug output paths for an object."""
    debug_dir = Path(object_dir) / DEBUG_DIRNAME
    histogram_dir = debug_dir / HISTOGRAM_DIRNAME
    paths = {
        "curves_path": debug_dir / CURVES_FILENAME,
        "distributions_path": debug_dir / DISTRIBUTIONS_FILENAME,
        "histogram_paths": [],
    }
    if save_score_histograms:
        paths["histogram_paths"] = [
            histogram_dir / HISTOGRAM_FILENAME.format(frame_idx=frame_idx)
            for frame_idx in range(frame_count)
        ]
    return paths


def expected_outputs_exist(object_dir, frame_count, save_score_histograms=False):
    """Return whether expected debug outputs already exist."""
    paths = output_paths(
        object_dir, frame_count, save_score_histograms=save_score_histograms
    )
    expected_paths = [paths["curves_path"], paths["distributions_path"]]
    expected_paths.extend(paths["histogram_paths"])
    return all(path.exists() for path in expected_paths)


def process_object(object_dir, args):
    """Create score confidence debug visualizations for one object directory."""
    object_dir = Path(object_dir)
    candidates_path = object_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates", []

    all_scores = load_scores(candidates_path)
    frame_count = len(all_scores)
    records = frame_score_records(
        object_dir, all_scores, temperature=args.temperature
    )

    if not args.overwrite and expected_outputs_exist(
        object_dir,
        frame_count,
        save_score_histograms=args.save_score_histograms,
    ):
        return "skipped_existing", records

    paths = output_paths(
        object_dir,
        frame_count,
        save_score_histograms=args.save_score_histograms,
    )
    plot_confidence_curves(records, paths["curves_path"])
    plot_confidence_distributions(records, paths["distributions_path"], args.hist_bins)
    for scores, histogram_path in zip(all_scores, paths["histogram_paths"]):
        plot_score_histogram(scores, histogram_path, args.hist_bins)

    return "processed", records


def main():
    """Run score confidence analysis for every processed object directory."""
    args = parse_args()

    data_root = Path(args.data_root)
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]
    records = []

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                _, object_records = process_object(object_dir, args)
                records.extend(object_records)

        except Exception:
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())

    debug_dir = Path(__file__).resolve().parent / DEBUG_DIRNAME
    write_frame_csv(records, debug_dir / FRAME_CSV_FILENAME)
    write_object_csv(aggregate_object_records(records), debug_dir / OBJECT_CSV_FILENAME)


if __name__ == "__main__":
    main()
