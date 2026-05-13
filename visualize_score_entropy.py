"""Visualize per-frame score distribution for precomputed pose candidates."""

import argparse
import io
import pickle
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"
CANDIDATES_FILENAME = "all_poses&scores.pkl"
DEBUG_DIRNAME = "debug"
ENTROPY_FILENAME = "score_entropy.png"
HISTOGRAM_DIRNAME = "score_histograms"
HISTOGRAM_FILENAME = "frame_{frame_idx:06d}.png"


def parse_args(argv=None):
    """Parse command-line arguments for score entropy visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize score entropy for precomputed pose candidates."
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
        help="Number of bins to use for per-frame score histograms.",
    )
    return parser.parse_args(argv)


def load_scores(candidates_path):
    """Load per-frame candidate scores from an ``all_poses&scores.pkl`` file."""
    with open(candidates_path, "rb") as f:
        candidates = pickle.load(f)
    if "scores" not in candidates:
        raise KeyError(f"{candidates_path} must contain 'scores'")
    return candidates["scores"]


def finite_scores(scores):
    """Return one frame's finite scores as a flat float64 array."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return scores[np.isfinite(scores)]


def stable_softmax(scores):
    """Compute softmax probabilities from finite raw scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) == 0:
        return np.empty((0,), dtype=np.float64)
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def score_entropy(scores):
    """Compute Shannon entropy in nats for one frame's finite raw scores."""
    scores = finite_scores(scores)
    if len(scores) == 0:
        return np.nan
    probabilities = stable_softmax(scores)
    return float(-np.sum(probabilities * np.log(probabilities)))


def entropy_per_frame(all_scores):
    """Compute one entropy value per frame, using NaN for empty finite scores."""
    return np.array([score_entropy(scores) for scores in all_scores], dtype=np.float64)


def plot_entropy_curve(entropies, save_path):
    """Write a full-sequence entropy curve PNG."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    frame_indices = np.arange(len(entropies))
    ax.plot(frame_indices, entropies, linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Score entropy")
    ax.grid(True, alpha=0.3)
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


def output_paths(object_dir, frame_count):
    """Return the expected entropy and histogram output paths for an object."""
    debug_dir = Path(object_dir) / DEBUG_DIRNAME
    histogram_dir = debug_dir / HISTOGRAM_DIRNAME
    entropy_path = debug_dir / ENTROPY_FILENAME
    histogram_paths = [
        histogram_dir / HISTOGRAM_FILENAME.format(frame_idx=frame_idx)
        for frame_idx in range(frame_count)
    ]
    return entropy_path, histogram_paths


def expected_outputs_exist(object_dir, frame_count):
    """Return whether all expected debug outputs already exist."""
    entropy_path, histogram_paths = output_paths(object_dir, frame_count)
    return entropy_path.exists() and all(path.exists() for path in histogram_paths)


def process_object(object_dir, args):
    """Create score entropy debug visualizations for one object directory."""
    object_dir = Path(object_dir)
    candidates_path = object_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates"

    all_scores = load_scores(candidates_path)
    frame_count = len(all_scores)

    if not args.overwrite and expected_outputs_exist(object_dir, frame_count):
        return "skipped_existing"

    entropies = entropy_per_frame(all_scores)
    entropy_path, histogram_paths = output_paths(object_dir, frame_count)
    plot_entropy_curve(entropies, entropy_path)
    for scores, histogram_path in zip(all_scores, histogram_paths):
        plot_score_histogram(scores, histogram_path, args.hist_bins)

    return "processed"


def main():
    """Run score entropy visualization for every processed object directory."""
    args = parse_args()

    data_root = Path(args.data_root)
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                process_object(object_dir, args)

        except Exception:
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())


if __name__ == "__main__":
    main()
