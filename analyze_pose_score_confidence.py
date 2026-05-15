"""Analyze NMS-filtered pose candidate score distributions as confidence signals.

Purpose:
    FoundationPose can save multiple pose candidates and their scores for each
    frame in ``all_poses&scores.pkl``. This script measures per-frame confidence
    signals after greedy rotation NMS, without rerunning pose estimation or
    requiring ground-truth poses.

What it does:
    For every processed object directory under ``--data-root``, the script loads
    per-frame candidate poses and scores, filters candidates whose score or pose
    matrix contains non-finite values, runs rotation-only NMS, and computes a
    compact score-distribution record for the remaining candidates. It writes a
    per-frame CSV plus one dataset summary CSV under the repository ``debug/``
    directory and saves per-object plots under each ``<object_dir>/debug/``
    directory. Per-frame histograms are available with
    ``--save-score-histograms`` but are disabled by default.

How to use:
    Run from the repository root, for example:

        python analyze_pose_score_confidence.py \
            --data-root /data/datasets/DexYCB/processed \
            --temperature 1.0 \
            --nms-threshold 5.0

    Main outputs are ``debug/score_confidence_by_frame.csv`` and
    ``debug/score_confidence_summary.csv``. Per-object visualizations are
    ``score_confidence_curves.png`` and ``score_confidence_distributions.png``.
    ``--nms-threshold`` is measured in degrees; 0 still runs NMS and suppresses
    duplicate candidates with identical rotations.
"""

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
SUMMARY_CSV_FILENAME = "score_confidence_summary.csv"
CURVES_FILENAME = "score_confidence_curves.png"
DISTRIBUTIONS_FILENAME = "score_confidence_distributions.png"
HISTOGRAM_DIRNAME = "score_histograms"
HISTOGRAM_FILENAME = "frame_{frame_idx:06d}.png"

FRAME_CSV_FIELDNAMES = [
    "object_dir",
    "frame_idx",
    "valid_candidate_count",
    "post_nms_candidate_count",
    "top1_score",
    "top1_top2_gap",
    "top1_prob",
    "entropy",
]
SUMMARY_COUNT_METRICS = ["valid_candidate_count", "post_nms_candidate_count"]
SUMMARY_FINITE_METRICS = ["top1_score", "top1_top2_gap", "top1_prob", "entropy"]
SUMMARY_METRICS = SUMMARY_COUNT_METRICS + SUMMARY_FINITE_METRICS
SUMMARY_STATS = ["mean", "std", "p10", "p50", "p90"]
SUMMARY_CSV_FIELDNAMES = ["object_count", "frame_count", "valid_frame_count"] + [
    f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in SUMMARY_STATS
]


def positive_float(value):
    """Parse a strictly positive finite float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def nonnegative_float(value):
    """Parse a non-negative finite float for argparse."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
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
        help="Save NMS-filtered score histograms for every frame.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=nonnegative_float,
        default=5.0,
        help=(
            "Rotation NMS threshold in degrees. Use 0 to suppress only "
            "identical rotations."
        ),
    )
    return parser.parse_args(argv)


def load_pose_candidates(candidates_path):
    """Load per-frame candidate poses and scores from a pickle file."""
    with open(candidates_path, "rb") as f:
        candidates = pickle.load(f)
    if "poses" not in candidates or "scores" not in candidates:
        raise KeyError(f"{candidates_path} must contain 'poses' and 'scores'")
    return candidates["poses"], candidates["scores"]


def flattened_scores(scores):
    """Return one frame's scores as a flat float64 array."""
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def pose_array(poses):
    """Return one frame's poses as an ``N x 4 x 4`` float64 array."""
    return np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)


def prepare_pose_score_frames(all_poses, all_scores):
    """Validate frame and candidate alignment, returning normalized arrays."""
    if len(all_poses) != len(all_scores):
        raise ValueError("poses and scores must have the same number of frames")

    frames = []
    for frame_idx, (poses, scores) in enumerate(zip(all_poses, all_scores)):
        frame_poses = pose_array(poses)
        frame_scores = flattened_scores(scores)
        if len(frame_poses) != len(frame_scores):
            raise ValueError(f"frame {frame_idx} pose and score counts do not match")
        frames.append((frame_poses, frame_scores))
    return frames


def valid_candidate_mask(poses, scores):
    """Return candidates with finite score and finite pose matrix entries."""
    poses = pose_array(poses)
    scores = flattened_scores(scores)
    if len(poses) != len(scores):
        raise ValueError("pose and score counts do not match")
    return np.isfinite(scores) & np.isfinite(poses).all(axis=(1, 2))


def valid_pose_score_candidates(poses, scores):
    """Return pose and score arrays filtered to valid candidates."""
    poses = pose_array(poses)
    scores = flattened_scores(scores)
    mask = valid_candidate_mask(poses, scores)
    return poses[mask], scores[mask]


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


def _nan_score_metrics():
    return {
        "top1_score": np.nan,
        "top1_top2_gap": np.nan,
        "top1_prob": np.nan,
        "entropy": np.nan,
    }


def _score_distribution_metrics(scores, temperature=1.0):
    """Compute emitted score-distribution metrics for valid NMS scores."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    sorted_scores = np.sort(scores)[::-1]
    candidate_count = len(sorted_scores)
    metrics = _nan_score_metrics()
    if candidate_count == 0:
        return metrics

    probabilities = stable_softmax(sorted_scores, temperature=temperature)
    positive_probabilities = probabilities[probabilities > 0]
    entropy_nats = float(
        -np.sum(positive_probabilities * np.log(positive_probabilities))
    )
    entropy = entropy_nats / math.log(candidate_count) if candidate_count > 1 else 0.0
    metrics.update(
        {
            "top1_score": float(sorted_scores[0]),
            "top1_prob": float(probabilities[0]),
            "entropy": float(entropy),
        }
    )
    if candidate_count > 1:
        metrics["top1_top2_gap"] = float(sorted_scores[0] - sorted_scores[1])
    return metrics


def rotation_angle(rotation_a, rotation_b):
    """Return the geodesic angle in radians between two rotation matrices."""
    relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(
        rotation_b, dtype=np.float64
    )
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_distance_degrees(pose_a, pose_b):
    """Return the geodesic rotation distance between two poses in degrees."""
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    return float(math.degrees(rotation_angle(pose_a[:3, :3], pose_b[:3, :3])))


def pose_nms_scores(poses, scores, threshold_degrees):
    """Greedily suppress lower-scored candidates by rotation distance."""
    poses = pose_array(poses)
    scores = flattened_scores(scores)
    if len(poses) != len(scores):
        raise ValueError("pose and score counts do not match")
    if threshold_degrees < 0:
        raise ValueError("threshold must be greater than or equal to 0")

    if len(scores) == 0:
        return poses, scores, np.empty((0,), dtype=np.int64)

    order = np.argsort(-scores, kind="mergesort")
    kept_indices = []
    for candidate_idx in order:
        candidate_pose = poses[candidate_idx]
        if all(
            rotation_distance_degrees(candidate_pose, poses[kept_idx])
            > threshold_degrees
            for kept_idx in kept_indices
        ):
            kept_indices.append(int(candidate_idx))

    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    return poses[kept_indices], scores[kept_indices], kept_indices


def compute_frame_score_metrics(
    poses,
    scores,
    temperature=1.0,
    nms_threshold=5.0,
):
    """Compute NMS-filtered confidence metrics for one frame."""
    poses = pose_array(poses)
    scores = flattened_scores(scores)
    if len(poses) != len(scores):
        raise ValueError("pose and score counts do not match")

    valid_poses, valid_scores = valid_pose_score_candidates(poses, scores)
    kept_poses, kept_scores, _ = pose_nms_scores(
        valid_poses, valid_scores, threshold_degrees=nms_threshold
    )
    metrics = OrderedDict(
        [
            ("valid_candidate_count", int(len(valid_scores))),
            ("post_nms_candidate_count", int(len(kept_poses))),
        ]
    )
    metrics.update(_score_distribution_metrics(kept_scores, temperature=temperature))
    return metrics


def frame_score_records(
    object_dir,
    all_poses,
    all_scores,
    temperature=1.0,
    nms_threshold=5.0,
):
    """Return CSV-ready per-frame score confidence records for one object."""
    object_dir = Path(object_dir)
    records = []
    for frame_idx, (poses, scores) in enumerate(
        prepare_pose_score_frames(all_poses, all_scores)
    ):
        record = OrderedDict(
            [
                ("object_dir", str(object_dir)),
                ("frame_idx", frame_idx),
            ]
        )
        record.update(
            compute_frame_score_metrics(
                poses,
                scores,
                temperature=temperature,
                nms_threshold=nms_threshold,
            )
        )
        records.append(record)
    return records


def _write_csv(records, csv_path, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_frame_csv(records, csv_path):
    """Write all per-frame score confidence records to a CSV file."""
    _write_csv(records, csv_path, FRAME_CSV_FIELDNAMES)


def _record_values(records, metric):
    return np.array([record.get(metric, np.nan) for record in records], dtype=np.float64)


def _finite_record_values(records, metric):
    values = _record_values(records, metric)
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


def aggregate_dataset_summary(frame_records):
    """Aggregate per-frame records into a single dataset summary row."""
    records = list(frame_records)
    row = OrderedDict(
        [
            ("object_count", len({record["object_dir"] for record in records})),
            ("frame_count", len(records)),
            (
                "valid_frame_count",
                int(
                    np.sum(
                        _record_values(records, "post_nms_candidate_count") > 0
                    )
                ),
            ),
        ]
    )

    for metric in SUMMARY_COUNT_METRICS:
        stats = _aggregate_values(_record_values(records, metric))
        for stat in SUMMARY_STATS:
            row[f"{metric}_{stat}"] = stats[stat]

    for metric in SUMMARY_FINITE_METRICS:
        stats = _aggregate_values(_finite_record_values(records, metric))
        for stat in SUMMARY_STATS:
            row[f"{metric}_{stat}"] = stats[stat]

    return [row]


def write_summary_csv(records, csv_path):
    """Write the dataset summary CSV."""
    _write_csv(records, csv_path, SUMMARY_CSV_FIELDNAMES)


def plot_confidence_curves(records, save_path):
    """Write time curves for key score confidence metrics."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _record_values(records, "frame_idx")
    plot_specs = [
        ("top1_top2_gap", "Top1-top2 gap"),
        ("top1_prob", "Top1 probability"),
        ("entropy", "Normalized entropy"),
        ("post_nms_candidate_count", "Post-NMS candidate count"),
    ]

    fig, axes = plt.subplots(
        len(plot_specs),
        1,
        figsize=(10, 2.4 * len(plot_specs)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (metric, ylabel) in zip(axes, plot_specs):
        ax.plot(frame_indices, _record_values(records, metric), linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Pose score confidence metrics after rotation NMS", fontsize=12)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confidence_distributions(records, save_path, bins):
    """Write cross-frame histograms for key score confidence metrics."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("top1_top2_gap", "Top1-top2 gap"),
        ("top1_prob", "Top1 probability"),
        ("entropy", "Normalized entropy"),
        ("post_nms_candidate_count", "Post-NMS candidate count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    for ax, (metric, title) in zip(np.ravel(axes), plot_specs):
        values = _finite_record_values(records, metric)
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

    fig.suptitle("Pose score confidence distributions after rotation NMS", fontsize=12)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_histogram(scores, save_path, bins, title="Frame score histogram"):
    """Write one score histogram PNG, annotating frames without valid scores."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scores = flattened_scores(scores)
    scores = scores[np.isfinite(scores)]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if len(scores) == 0:
        ax.text(
            0.5,
            0.5,
            "No valid scores",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(scores, bins=bins)
    ax.set_xlabel("NMS-filtered score")
    ax.set_ylabel("Count")
    ax.set_title(title)
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


def process_object(object_dir, args):
    """Create score confidence debug visualizations for one object directory."""
    object_dir = Path(object_dir)
    candidates_path = object_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates", []

    try:
        all_poses, all_scores = load_pose_candidates(candidates_path)
        frames = prepare_pose_score_frames(all_poses, all_scores)
    except (KeyError, ValueError) as exc:
        tqdm.tqdm.write(f"Invalid {candidates_path}: {exc}; skipping")
        return "skipped_invalid_candidates", []

    frame_count = len(frames)
    records = frame_score_records(
        object_dir,
        all_poses,
        all_scores,
        temperature=args.temperature,
        nms_threshold=args.nms_threshold,
    )

    paths = output_paths(
        object_dir,
        frame_count,
        save_score_histograms=args.save_score_histograms,
    )
    plot_confidence_curves(records, paths["curves_path"])
    plot_confidence_distributions(records, paths["distributions_path"], args.hist_bins)

    for (poses, scores), histogram_path in zip(frames, paths["histogram_paths"]):
        valid_poses, valid_scores = valid_pose_score_candidates(poses, scores)
        _, kept_scores, _ = pose_nms_scores(
            valid_poses,
            valid_scores,
            threshold_degrees=args.nms_threshold,
        )
        plot_score_histogram(
            kept_scores,
            histogram_path,
            args.hist_bins,
            title="Frame NMS-filtered score histogram",
        )

    return "processed", records


def main():
    """Run score confidence analysis for every processed object directory."""
    args = parse_args()
    data_root = Path(args.data_root)
    roots = sorted(data_root.glob("**/video"))
    roots = [root.parent for root in roots]
    records = []

    for root in tqdm.tqdm(roots, dynamic_ncols=True):
        try:
            objects_root = root / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                _, object_records = process_object(object_dir, args)
                records.extend(object_records)

        except Exception:
            tqdm.tqdm.write(f"Failed to process {root.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())

    debug_dir = Path(__file__).resolve().parent / DEBUG_DIRNAME
    write_frame_csv(records, debug_dir / FRAME_CSV_FILENAME)
    write_summary_csv(
        aggregate_dataset_summary(records),
        debug_dir / SUMMARY_CSV_FILENAME,
    )


if __name__ == "__main__":
    main()
