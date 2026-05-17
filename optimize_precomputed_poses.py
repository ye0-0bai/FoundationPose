"""Optimize artifact pose candidates with IoU-weighted confidence.

The optimizer consumes ``all_pose_candidates_artifacts.npz`` plus each
object's ``masks.npz`` visible masks. For every valid frame it min-max
normalizes candidate scores, warps the full-frame visible mask into each
FoundationPose crop with the exported ``tf_to_crops`` transform, computes IoU
against the rendered candidate mask, and scores candidates as
``minmax(score) * mask_iou``.

Dynamic programming then selects one globally consistent candidate trajectory,
penalizing translation and rotation jumps between adjacent valid frames. Short
invalid gaps may be interpolated and contiguous valid runs are smoothed with
Savitzky-Golay filtering. Missing artifact or visible-mask files are reported
and skipped without using any alternate input source.

Outputs are written per run as ``poses_optimized_{uid}.npy`` and
``poses_optimized_{uid}.mp4``. Optional trajectory-selection debug data is
written as ``poses_optimized_{uid}_debug.npz``.
"""

import argparse
import datetime as dt
import io
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import kornia
import numpy as np
import torch
import tqdm
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from Utils import compute_mesh_diameter, draw_posed_3d_box, draw_xyz_axis
from process_data import configure_quiet_logging


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"
ARTIFACTS_FILENAME = "all_pose_candidates_artifacts.npz"
MASKS_FILENAME = "masks.npz"
POSES_FILENAME_TEMPLATE = "poses_optimized_{run_id}.npy"
VIDEO_FILENAME_TEMPLATE = "poses_optimized_{run_id}.mp4"
DEBUG_FILENAME_TEMPLATE = "poses_optimized_{run_id}_debug.npz"
EXP_DIRNAME = "exp"
POSE_OPTIMIZATION_RUNS_DIRNAME = "pose_optimization_runs"


# ---------------------------------------------------------------------------
# Configuration and CLI


@dataclass(frozen=True)
class OptimizationConfig:
    """Trajectory-selection and smoothing parameters."""

    max_invalid_gap: int = 5
    smooth_window: int = 7
    smooth_polyorder: int = 2
    trans_lambda: float = 1.0
    rot_lambda: float = 1.0


@dataclass(frozen=True)
class ObjectPaths:
    """Filesystem paths for one processed object directory."""

    object_dir: Path
    mesh: Path
    artifacts: Path
    masks: Path
    optimized_poses: Path
    optimized_video: Path
    debug_data: Path

    @classmethod
    def from_object_dir(cls, object_dir, run_id):
        object_dir = Path(object_dir)
        return cls(
            object_dir=object_dir,
            mesh=object_dir / "mesh.glb",
            artifacts=object_dir / ARTIFACTS_FILENAME,
            masks=object_dir / MASKS_FILENAME,
            optimized_poses=object_dir / POSES_FILENAME_TEMPLATE.format(run_id=run_id),
            optimized_video=object_dir / VIDEO_FILENAME_TEMPLATE.format(run_id=run_id),
            debug_data=object_dir / DEBUG_FILENAME_TEMPLATE.format(run_id=run_id),
        )


def parse_args():
    """Parse command-line arguments for pose candidate optimization.

    Returns:
        argparse.Namespace: Parsed arguments controlling dataset location,
        overwrite behavior, trajectory selection, and smoothing settings.
    """
    # Keep all tunable parameters in the CLI so batch jobs can reuse the same
    # script across datasets without editing constants in the source file.
    parser = argparse.ArgumentParser(
        description="Optimize precomputed pose candidates for processed DexYCB objects."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root of the processed DexYCB dataset.",
    )
    # Existing optimized poses are skipped by default because rendering every
    # object video is expensive; --overwrite makes reruns explicit.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate optimized outputs even when the current run-id output already exists.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run UID used in optimized output filenames. Defaults to script start timestamp.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write per-object trajectory-selection debug data for the current run.",
    )
    # The smoothing options are forwarded to smooth_pose_trajectory(), which
    # fills short invalid gaps before applying Savitzky-Golay filtering.
    parser.add_argument(
        "--max-invalid-gap",
        type=int,
        default=5,
        help="Maximum invalid trajectory gap to interpolate before smoothing.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Savitzky-Golay smoothing window length.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay smoothing polynomial order.",
    )
    # The transition penalties are forwarded to select_pose_trajectory(); they
    # balance per-frame candidate confidence against temporal consistency.
    parser.add_argument(
        "--trans-lambda",
        type=float,
        default=1.0,
        help="Translation transition penalty weight for trajectory selection.",
    )
    parser.add_argument(
        "--rot-lambda",
        type=float,
        default=1.0,
        help="Rotation transition penalty weight for trajectory selection.",
    )
    return parser.parse_args()


def optimization_config_from_args(args):
    """Build an explicit optimization config from a parsed CLI namespace."""
    return OptimizationConfig(
        max_invalid_gap=args.max_invalid_gap,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        trans_lambda=args.trans_lambda,
        rot_lambda=args.rot_lambda,
    )


def ensure_optimization_config(config_or_args):
    """Accept a config or legacy argparse namespace and return config."""
    if isinstance(config_or_args, OptimizationConfig):
        return config_or_args
    return optimization_config_from_args(config_or_args)


def generate_run_id(now=None):
    """Generate the default per-process run UID."""
    if now is None:
        now = dt.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def experiment_record_path(repo_root, run_id):
    """Return the Markdown experiment record path for ``run_id``."""
    return (
        Path(repo_root)
        / EXP_DIRNAME
        / POSE_OPTIMIZATION_RUNS_DIRNAME
        / f"{run_id}.md"
    )


def output_naming_for_run(run_id):
    """Return output filename templates rendered for ``run_id``."""
    return {
        "poses": POSES_FILENAME_TEMPLATE.format(run_id=run_id),
        "video": VIDEO_FILENAME_TEMPLATE.format(run_id=run_id),
        "debug": DEBUG_FILENAME_TEMPLATE.format(run_id=run_id),
    }


def write_experiment_record(
    repo_root,
    run_id,
    generated_at,
    argv,
    args,
    stats,
    run_status="completed",
):
    """Write a Markdown experiment record with an embedded JSON config block."""
    record_path = experiment_record_path(repo_root, run_id)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_status": run_status,
        "generated_at": generated_at,
        "argv": list(argv),
        "data_root": str(args.data_root),
        "output_naming": output_naming_for_run(run_id),
        "optimization_objective": "score - trans_lambda * trans_cost - rot_lambda * rot_cost",
        "optimization_parameters": {
            "max_invalid_gap": args.max_invalid_gap,
            "smooth_window": args.smooth_window,
            "smooth_polyorder": args.smooth_polyorder,
            "trans_lambda": args.trans_lambda,
            "rot_lambda": args.rot_lambda,
        },
        "debug_enabled": bool(args.debug),
        "debug_file_rule": (
            DEBUG_FILENAME_TEMPLATE.format(run_id=run_id)
            if args.debug
            else "not written unless --debug is enabled"
        ),
        "processing_stats": dict(stats),
    }
    record_path.write_text(
        "\n".join(
            [
                f"# Pose Optimization Run {run_id}",
                "",
                "```json",
                json.dumps(payload, indent=2, sort_keys=True),
                "```",
                "",
                "Objective: score - trans_lambda * trans_cost - rot_lambda * rot_cost",
                "",
            ]
        )
    )
    return record_path


# ---------------------------------------------------------------------------
# Artifact data loading, validation, and IoU weighting


def artifact_key(frame_idx, name):
    """Return the zero-padded per-frame artifact key used by the NPZ export."""
    return f"{name}_{frame_idx:04d}"


def normalize_scores(scores, method="minmax"):
    """Scale raw candidate scores while preserving invalid entries.

    Args:
        scores (array-like): One frame's candidate confidence scores.
        method (str): Normalization method. Only ``"minmax"`` is supported.

    Returns:
        np.ndarray: Score array with the same shape. Finite entries are scaled
        into ``[0, 1]``; non-finite entries remain non-finite.
    """
    if method != "minmax":
        raise ValueError(f"unsupported score normalization method: {method}")

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    normalized = np.full_like(scores, np.nan, dtype=np.float64)
    if not finite.any():
        return normalized

    finite_scores = scores[finite]
    score_min = finite_scores.min()
    score_max = finite_scores.max()
    if np.isclose(score_max, score_min):
        normalized[finite] = 1.0
    else:
        normalized[finite] = (finite_scores - score_min) / (score_max - score_min)
    return normalized


def validate_frame_artifacts(artifacts, frame_idx):
    """Load and validate one frame's pose, score, mask, and crop artifacts."""
    poses_key = artifact_key(frame_idx, "poses")
    scores_key = artifact_key(frame_idx, "scores")
    render_masks_key = artifact_key(frame_idx, "render_masks")
    tf_to_crops_key = artifact_key(frame_idx, "tf_to_crops")
    for key in [poses_key, scores_key, render_masks_key, tf_to_crops_key]:
        if key not in artifacts:
            raise KeyError(f"artifact is missing required key: {key}")

    poses = np.asarray(artifacts[poses_key], dtype=np.float64).reshape(-1, 4, 4)
    scores = np.asarray(artifacts[scores_key], dtype=np.float64).reshape(-1)
    render_masks = np.asarray(artifacts[render_masks_key])
    tf_to_crops = np.asarray(artifacts[tf_to_crops_key], dtype=np.float32).reshape(-1, 3, 3)

    if render_masks.ndim != 3:
        raise ValueError(f"{render_masks_key} must have shape (N, H, W)")
    candidate_count = len(poses)
    if not (
        len(scores) == candidate_count
        and len(render_masks) == candidate_count
        and len(tf_to_crops) == candidate_count
    ):
        raise ValueError(f"frame {frame_idx} artifact candidate counts do not match")
    return poses, scores, render_masks, tf_to_crops


def load_pose_candidate_artifacts(artifacts_path):
    """Load validated pose candidate artifacts from an NPZ file."""
    with np.load(artifacts_path) as data:
        artifacts = {key: data[key] for key in data.files}

    if "valid" not in artifacts:
        raise KeyError(f"{artifacts_path} must contain 'valid'")
    valid = np.asarray(artifacts["valid"], dtype=bool).reshape(-1)
    artifacts["valid"] = valid
    for frame_idx, is_valid in enumerate(valid):
        if is_valid:
            validate_frame_artifacts(artifacts, frame_idx)
    return artifacts


def load_visible_masks(masks_path):
    """Load full-frame visible masks used for crop-space IoU confidence."""
    with np.load(masks_path) as data:
        if "masks_visible" not in data:
            raise KeyError(f"{masks_path} must contain 'masks_visible'")
        masks_visible = np.asarray(data["masks_visible"])
    if masks_visible.ndim != 3:
        raise ValueError("masks_visible must have shape (T, H, W)")
    return masks_visible


def compute_mask_ious(visible_mask, render_masks, tf_to_crops):
    """Compute crop-space IoU between visible and rendered candidate masks."""
    render_masks = np.asarray(render_masks)
    if render_masks.ndim != 3:
        raise ValueError("render_masks must have shape (N, H, W)")
    tf_to_crops = np.asarray(tf_to_crops, dtype=np.float32).reshape(-1, 3, 3)
    if len(render_masks) != len(tf_to_crops):
        raise ValueError("render_masks and tf_to_crops must have matching counts")
    if len(render_masks) == 0:
        return np.empty((0,), dtype=np.float64)

    visible_mask = np.asarray(visible_mask, dtype=np.float32)
    if visible_mask.ndim != 2:
        raise ValueError("visible_mask must have shape (H, W)")

    crop_h, crop_w = render_masks.shape[-2:]
    batch_size = len(render_masks)
    visible_tensor = torch.as_tensor(visible_mask, dtype=torch.float32)[None, None]
    visible_tensor = visible_tensor.expand(batch_size, -1, -1, -1)
    tf_tensor = torch.as_tensor(tf_to_crops, dtype=torch.float32)
    cropped_visible = kornia.geometry.transform.warp_perspective(
        visible_tensor,
        tf_tensor,
        dsize=(crop_h, crop_w),
        mode="nearest",
        align_corners=False,
    )
    visible_np = cropped_visible[:, 0].detach().cpu().numpy() > 0.5
    render_np = render_masks > 0.5

    intersection = np.logical_and(visible_np, render_np).sum(axis=(1, 2))
    union = np.logical_or(visible_np, render_np).sum(axis=(1, 2))
    return np.divide(
        intersection,
        union,
        out=np.zeros(batch_size, dtype=np.float64),
        where=union > 0,
    )


def prepare_iou_weighted_candidates(artifacts, masks_visible):
    """Build per-frame pose candidates scored by min-max score times mask IoU."""
    if "valid" not in artifacts:
        raise KeyError("artifacts must contain 'valid'")

    valid = np.asarray(artifacts["valid"], dtype=bool).reshape(-1)
    masks_visible = np.asarray(masks_visible)
    if len(masks_visible) < len(valid):
        raise ValueError("masks_visible has fewer frames than artifacts")

    all_poses = []
    all_scores = []
    for frame_idx, is_valid in enumerate(valid):
        if not is_valid:
            all_poses.append(np.empty((0, 4, 4), dtype=np.float64))
            all_scores.append(np.empty((0,), dtype=np.float64))
            continue

        poses, scores, render_masks, tf_to_crops = validate_frame_artifacts(
            artifacts,
            frame_idx,
        )
        normalized_scores = normalize_scores(scores, method="minmax")
        ious = compute_mask_ious(
            masks_visible[frame_idx],
            render_masks,
            tf_to_crops,
        )
        finite = np.isfinite(normalized_scores) & np.isfinite(ious)
        adjusted_scores = np.full_like(normalized_scores, np.nan, dtype=np.float64)
        adjusted_scores[finite] = normalized_scores[finite] * ious[finite]

        all_poses.append(poses)
        all_scores.append(adjusted_scores)
    return all_poses, all_scores


def finite_candidate_mask(poses, scores):
    """Return candidates with finite score and finite pose matrix entries."""
    finite = np.isfinite(scores)
    if len(poses) > 0:
        finite = finite & np.isfinite(poses).all(axis=(1, 2))
    return finite


def filter_pose_candidates(all_poses, all_scores):
    """Filter invalid candidates and collect frame indices used by DP."""
    if len(all_poses) != len(all_scores):
        raise ValueError("all_poses and all_scores must have the same number of frames")

    poses_per_frame = []
    scores_per_frame = []
    valid_frame_indices = []
    for frame_idx, (poses, scores) in enumerate(zip(all_poses, all_scores)):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(poses) != len(scores):
            raise ValueError(f"frame {frame_idx} pose and score counts do not match")

        finite = finite_candidate_mask(poses, scores)
        if not finite.any():
            continue

        poses_per_frame.append(poses[finite])
        scores_per_frame.append(scores[finite])
        valid_frame_indices.append(frame_idx)

    return poses_per_frame, scores_per_frame, valid_frame_indices


def transition_cost_matrix(prev_poses, cur_poses, mesh_diameter):
    """Compute pairwise normalized translation and rotation transition costs."""
    diameter = max(float(mesh_diameter), 1e-12)

    prev_t = prev_poses[:, :3, 3]
    cur_t = cur_poses[:, :3, 3]
    trans_cost = np.linalg.norm(cur_t[None] - prev_t[:, None], axis=-1) / diameter

    prev_R = prev_poses[:, :3, :3]
    cur_R = cur_poses[:, :3, :3]
    traces = np.einsum("aij,bij->ab", prev_R, cur_R)
    rot_cost = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)) / np.pi
    return trans_cost, rot_cost


def build_trajectory(all_frame_count, valid_frame_indices, selected_poses):
    """Place selected poses back into a full-length zero-initialized trajectory."""
    trajectory = np.zeros((all_frame_count, 4, 4), dtype=np.float64)
    trajectory[valid_frame_indices] = selected_poses
    return trajectory


# ---------------------------------------------------------------------------
# Trajectory post-processing helpers


def pose_valid_mask(trajectory):
    """Return a frame mask for finite, non-zero poses in a trajectory."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    finite = np.isfinite(trajectory).all(axis=(1, 2))
    nonzero = np.any(trajectory != 0.0, axis=(1, 2))
    return finite & nonzero


def continuous_true_segments(mask):
    """Extract half-open ``[start, end)`` segments of contiguous True values."""
    segments = []
    start = None
    for idx, valid in enumerate(mask):
        if valid and start is None:
            start = idx
        elif not valid and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def interpolate_short_invalid_gaps(trajectory, valid_mask, max_invalid_gap):
    """Fill short invalid pose gaps with translation lerp and rotation Slerp."""
    smoothed = trajectory.copy()
    filled_mask = valid_mask.copy()
    n_frames = len(smoothed)
    idx = 0
    while idx < n_frames:
        if valid_mask[idx]:
            idx += 1
            continue

        gap_start = idx
        while idx < n_frames and not valid_mask[idx]:
            idx += 1
        gap_end = idx
        gap_len = gap_end - gap_start
        prev_idx = gap_start - 1
        next_idx = gap_end

        # Only bridge short interior gaps; leading, trailing, or long missing
        # spans stay invalid so the optimizer does not invent unsupported poses.
        if prev_idx < 0 or next_idx >= n_frames or gap_len > max_invalid_gap:
            continue

        gap_indices = np.arange(gap_start, gap_end)
        alpha = ((gap_indices - prev_idx) / (next_idx - prev_idx)).reshape(-1, 1)
        prev_pose = smoothed[prev_idx]
        next_pose = smoothed[next_idx]
        smoothed[gap_indices, :3, 3] = (
            (1.0 - alpha) * prev_pose[:3, 3] + alpha * next_pose[:3, 3]
        )

        key_rots = Rotation.from_matrix(
            np.stack([prev_pose[:3, :3], next_pose[:3, :3]], axis=0)
        )
        interp_rots = Slerp([prev_idx, next_idx], key_rots)(gap_indices)
        smoothed[gap_indices, :3, :3] = interp_rots.as_matrix()
        smoothed[gap_indices, 3, :] = [0.0, 0.0, 0.0, 1.0]
        filled_mask[gap_indices] = True

    return smoothed, filled_mask


def savgol_window_for_length(length, requested_window, polyorder):
    """Choose a valid odd Savitzky-Golay window for a segment length."""
    window = min(int(requested_window), int(length))
    if window % 2 == 0:
        window -= 1
    if window <= int(polyorder):
        return None
    return window


def smooth_pose_segment(trajectory, start, end, smooth_window, smooth_polyorder):
    """Smooth one contiguous valid trajectory segment in-place."""
    segment_len = end - start
    window = savgol_window_for_length(segment_len, smooth_window, smooth_polyorder)
    # Very short segments cannot support the requested polynomial fit, so they
    # are left unchanged instead of forcing a numerically fragile fallback.
    if window is None:
        return
    smooth_polyorder = int(smooth_polyorder)

    segment = trajectory[start:end]
    trajectory[start:end, :3, 3] = savgol_filter(
        segment[:, :3, 3],
        window_length=window,
        polyorder=smooth_polyorder,
        axis=0,
        mode="interp",
    )

    rotations = Rotation.from_matrix(segment[:, :3, :3])
    quats = rotations.as_quat()
    # Quaternion q and -q represent the same rotation; enforcing sign
    # continuity avoids artificial jumps before Savitzky-Golay smoothing.
    for idx in range(1, len(quats)):
        if np.dot(quats[idx - 1], quats[idx]) < 0:
            quats[idx] *= -1.0

    quats = savgol_filter(
        quats,
        window_length=window,
        polyorder=smooth_polyorder,
        axis=0,
        mode="interp",
    )
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    bad_quat = quat_norms[:, 0] < 1e-12
    # Degenerate filtered quaternions are replaced with the original rotation
    # samples so normalization never amplifies near-zero numerical noise.
    if bad_quat.any():
        quats[bad_quat] = rotations.as_quat()[bad_quat]
        quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / quat_norms
    trajectory[start:end, :3, :3] = Rotation.from_quat(quats).as_matrix()
    trajectory[start:end, 3, :] = [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Trajectory selection and optimization


def select_pose_trajectory(
    all_poses,
    all_scores,
    mesh_diameter,
    trans_lambda=1.0,
    rot_lambda=1.0,
    return_debug=False,
):
    """Select one globally consistent pose candidate per valid frame.

    Args:
        all_poses (Sequence[np.ndarray]): Per-frame pose candidates shaped like
            ``(N, 4, 4)``.
        all_scores (Sequence[np.ndarray]): Per-frame candidate scores aligned
            with ``all_poses``.
        mesh_diameter (float): Object diameter used to normalize translation
            jumps between frames.
        trans_lambda (float): Weight for translation transition cost.
        rot_lambda (float): Weight for rotation transition cost.

        return_debug (bool): When true, also return arrays describing the
            selected path's scores and transition penalties.

    Returns:
        np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]: A ``(T, 4, 4)``
        pose trajectory. Frames with no valid candidates remain all zeros. With
        ``return_debug=True``, the second element contains per-selected-frame
        debug arrays.

    Raises:
        ValueError: If inputs are misaligned or no valid candidates exist.
    """
    if len(all_poses) != len(all_scores):
        raise ValueError("all_poses and all_scores must have the same number of frames")
    if len(all_poses) == 0:
        trajectory = np.empty((0, 4, 4), dtype=np.float64)
        if return_debug:
            return trajectory, empty_selection_debug()
        return trajectory

    poses_per_frame, scores_per_frame, valid_frame_indices = filter_pose_candidates(
        all_poses,
        all_scores,
    )

    if len(valid_frame_indices) == 0:
        raise ValueError("no valid pose candidates")

    dp = [scores_per_frame[0].copy()]
    backptr = []
    for frame_idx in range(1, len(poses_per_frame)):
        prev_poses = poses_per_frame[frame_idx - 1]
        cur_poses = poses_per_frame[frame_idx]
        trans_cost, rot_cost = transition_cost_matrix(
            prev_poses,
            cur_poses,
            mesh_diameter,
        )

        values = (
            dp[-1][:, None]
            + scores_per_frame[frame_idx][None, :]
            - trans_lambda * trans_cost
            - rot_lambda * rot_cost
        )
        best_prev = values.argmax(axis=0)
        dp.append(values[best_prev, np.arange(len(cur_poses))])
        backptr.append(best_prev)

    best_idx = int(dp[-1].argmax())
    selected_indices = [best_idx]
    # Backtracking reconstructs the globally best path instead of choosing
    # locally best transitions frame by frame.
    for prev_indices in reversed(backptr):
        best_idx = int(prev_indices[best_idx])
        selected_indices.append(best_idx)
    selected_indices.reverse()

    selected_poses = [
        poses_per_frame[frame_idx][candidate_idx]
        for frame_idx, candidate_idx in enumerate(selected_indices)
    ]
    selected_poses = np.stack(selected_poses, axis=0)
    trajectory = build_trajectory(len(all_poses), valid_frame_indices, selected_poses)
    if return_debug:
        debug = build_selection_debug(
            poses_per_frame,
            scores_per_frame,
            valid_frame_indices,
            selected_indices,
            mesh_diameter=mesh_diameter,
            trans_lambda=trans_lambda,
            rot_lambda=rot_lambda,
        )
        return trajectory, debug
    return trajectory


def empty_selection_debug():
    """Return an empty debug payload with stable array names and dtypes."""
    return {
        "frame_index": np.empty((0,), dtype=np.int64),
        "selected_candidate_index": np.empty((0,), dtype=np.int64),
        "selected_adjusted_score": np.empty((0,), dtype=np.float64),
        "score_rank": np.empty((0,), dtype=np.int64),
        "translation_cost": np.empty((0,), dtype=np.float64),
        "rotation_cost": np.empty((0,), dtype=np.float64),
        "weighted_translation_penalty": np.empty((0,), dtype=np.float64),
        "weighted_rotation_penalty": np.empty((0,), dtype=np.float64),
        "net_contribution": np.empty((0,), dtype=np.float64),
        "cumulative_score": np.empty((0,), dtype=np.float64),
    }


def selected_score_ranks(scores, selected_indices):
    """Compute one-based descending score ranks for selected candidates."""
    ranks = []
    for frame_scores, selected_idx in zip(scores, selected_indices):
        finite_scores = frame_scores[np.isfinite(frame_scores)]
        selected_score = frame_scores[selected_idx]
        rank = 1 + int(np.sum(finite_scores > selected_score))
        ranks.append(rank)
    return np.asarray(ranks, dtype=np.int64)


def build_selection_debug(
    poses_per_frame,
    scores_per_frame,
    valid_frame_indices,
    selected_indices,
    mesh_diameter,
    trans_lambda,
    rot_lambda,
):
    """Build debug arrays for the final selected dynamic-programming path."""
    debug = empty_selection_debug()
    if len(selected_indices) == 0:
        return debug

    selected_scores = np.asarray(
        [
            scores_per_frame[frame_idx][candidate_idx]
            for frame_idx, candidate_idx in enumerate(selected_indices)
        ],
        dtype=np.float64,
    )
    trans_costs = np.zeros(len(selected_indices), dtype=np.float64)
    rot_costs = np.zeros(len(selected_indices), dtype=np.float64)
    for frame_idx in range(1, len(selected_indices)):
        prev_candidate_idx = selected_indices[frame_idx - 1]
        cur_candidate_idx = selected_indices[frame_idx]
        trans_cost, rot_cost = transition_cost_matrix(
            poses_per_frame[frame_idx - 1][prev_candidate_idx : prev_candidate_idx + 1],
            poses_per_frame[frame_idx][cur_candidate_idx : cur_candidate_idx + 1],
            mesh_diameter,
        )
        trans_costs[frame_idx] = trans_cost[0, 0]
        rot_costs[frame_idx] = rot_cost[0, 0]

    weighted_trans = float(trans_lambda) * trans_costs
    weighted_rot = float(rot_lambda) * rot_costs
    net = selected_scores - weighted_trans - weighted_rot
    debug.update(
        {
            "frame_index": np.asarray(valid_frame_indices, dtype=np.int64),
            "selected_candidate_index": np.asarray(selected_indices, dtype=np.int64),
            "selected_adjusted_score": selected_scores,
            "score_rank": selected_score_ranks(scores_per_frame, selected_indices),
            "translation_cost": trans_costs,
            "rotation_cost": rot_costs,
            "weighted_translation_penalty": weighted_trans,
            "weighted_rotation_penalty": weighted_rot,
            "net_contribution": net,
            "cumulative_score": np.cumsum(net),
        }
    )
    return debug


def smooth_pose_trajectory(
    trajectory,
    max_invalid_gap=5,
    smooth_window=7,
    smooth_polyorder=2,
):
    """Interpolate short gaps and smooth contiguous valid pose segments.

    Args:
        trajectory (np.ndarray): Pose trajectory with shape ``(T, 4, 4)``.
        max_invalid_gap (int): Longest invalid run that may be interpolated.
        smooth_window (int): Requested Savitzky-Golay window length.
        smooth_polyorder (int): Savitzky-Golay polynomial order.

    Returns:
        np.ndarray: Smoothed trajectory with long invalid gaps preserved as
        all-zero poses.

    Raises:
        ValueError: If shape or smoothing arguments are invalid.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[-2:] != (4, 4):
        raise ValueError("trajectory must have shape (T, 4, 4)")
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape (T, 4, 4)")
    if max_invalid_gap < 0:
        raise ValueError("max_invalid_gap must be non-negative")
    if smooth_polyorder < 0:
        raise ValueError("smooth_polyorder must be non-negative")

    smoothed = trajectory.copy()
    if len(smoothed) == 0:
        return smoothed

    valid_mask = pose_valid_mask(smoothed)
    # Interpolating before smoothing lets short dropouts participate in a
    # longer continuous segment while still leaving long gaps untouched.
    smoothed, valid_mask = interpolate_short_invalid_gaps(
        smoothed,
        valid_mask,
        max_invalid_gap,
    )

    for start, end in continuous_true_segments(valid_mask):
        smooth_pose_segment(smoothed, start, end, smooth_window, smooth_polyorder)

    # Frames that never became valid are forced back to zero so downstream code
    # can keep using the same invalid-pose convention.
    smoothed[~valid_mask] = 0.0
    return smoothed


def optimize_pose_candidates(all_poses, all_scores, mesh, config, return_debug=False):
    """Select and smooth the best IoU-weighted pose trajectory for one object.

    Args:
        all_poses (Sequence[np.ndarray]): Per-frame candidate poses with shape
            ``(N, 4, 4)`` for each frame.
        all_scores (Sequence[np.ndarray]): Per-frame IoU-weighted scores
            aligned with ``all_poses``.
        mesh (trimesh.Trimesh): Object mesh used to normalize translation
            transition costs.
        config (OptimizationConfig | argparse.Namespace): Optimization and
            smoothing parameters. argparse namespaces are accepted for
            compatibility with the original call site.

    Returns:
        np.ndarray: Optimized pose trajectory with shape ``(T, 4, 4)``.
    """
    config = ensure_optimization_config(config)
    # The mesh diameter puts translation jumps on a comparable scale across
    # objects, so one penalty setting can be reused dataset-wide.
    mesh_diameter = compute_mesh_diameter(
        model_pts=np.asarray(mesh.vertices),
        n_sample=10000,
    )
    selected = select_pose_trajectory(
        all_poses,
        all_scores,
        mesh_diameter=mesh_diameter,
        trans_lambda=config.trans_lambda,
        rot_lambda=config.rot_lambda,
        return_debug=return_debug,
    )
    if return_debug:
        trajectory, debug = selected
    else:
        trajectory = selected
    smoothed = smooth_pose_trajectory(
        trajectory,
        max_invalid_gap=config.max_invalid_gap,
        smooth_window=config.smooth_window,
        smooth_polyorder=config.smooth_polyorder,
    )
    if return_debug:
        return smoothed, debug
    return smoothed


# ---------------------------------------------------------------------------
# Rendering


def render_optimized_poses(seq_dir, object_paths, mesh, trajectory):
    """Render an RGB video overlay for an optimized object trajectory.

    Args:
        seq_dir (Path): Sequence directory containing the ``video`` folder.
        object_paths (ObjectPaths): Object paths including rendered video path.
        mesh (trimesh.Trimesh): Object mesh used to derive the 3D bounding box.
        trajectory (np.ndarray): Optimized pose trajectory with shape
            ``(T, 4, 4)``. All-zero poses are treated as invalid frames.
    """
    # The optimized overlay uses the original sequence video and camera
    # intrinsics so the output can be inspected directly against RGB frames.
    intrinsics_path = seq_dir / "video" / "intrinsics.npy"
    images_path = seq_dir / "video" / "images.mp4"

    # Use float64 intrinsics to match the pose math and projection helpers.
    intrinsics = np.load(intrinsics_path).astype(np.float64)
    # imageio returns the whole RGB video as a frame-major array.
    images = iio.imread(images_path)

    # Compute an oriented bounding box in a centered frame, then draw that box
    # after transforming each object pose into the same centered frame.
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    to_origin_inv = np.linalg.inv(to_origin)

    video = []
    for frame_idx, pose in enumerate(trajectory):
        # All-zero poses mark invalid or intentionally unsmoothed frames; keep
        # the raw image so the rendered video preserves sequence length.
        if not pose.any():
            video.append(images[frame_idx])
            continue

        # Convert the object pose from mesh coordinates to the centered box
        # frame expected by the bounding-box visualization helper.
        center_pose = pose @ to_origin_inv
        # Draw the optimized 3D box first, then add XYZ axes as an orientation
        # cue for quick visual inspection.
        vis = draw_posed_3d_box(
            intrinsics,
            img=images[frame_idx],
            ob_in_cam=center_pose,
            bbox=bbox,
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=intrinsics,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        video.append(vis)

    # Stack the per-frame overlays back into a video tensor before writing MP4.
    iio.imwrite(object_paths.optimized_video, np.stack(video, axis=0))


# ---------------------------------------------------------------------------
# Object and batch processing


def process_object(seq_dir, object_dir, args):
    """Optimize and render artifact pose candidates for one object.

    Args:
        seq_dir (Path): Sequence directory containing RGB frames and camera
            intrinsics.
        object_dir (Path): Directory containing mesh, artifacts, and masks.
        args (argparse.Namespace): Command-line arguments controlling the
            optimization run.

    Returns:
        str: Processing status. One of ``skipped_existing``,
        ``skipped_missing_artifacts``, ``skipped_missing_masks``, or
        ``processed``.
    """
    object_paths = ObjectPaths.from_object_dir(object_dir, run_id=args.run_id)

    save_path = object_paths.optimized_poses
    if not args.overwrite and save_path.exists():
        return "skipped_existing"

    artifacts_path = object_paths.artifacts
    if not artifacts_path.exists():
        tqdm.tqdm.write(f"Missing {artifacts_path}; skipping")
        return "skipped_missing_artifacts"

    masks_path = object_paths.masks
    if not masks_path.exists():
        tqdm.tqdm.write(f"Missing {masks_path}; skipping")
        return "skipped_missing_masks"

    config = ensure_optimization_config(args)
    artifacts = load_pose_candidate_artifacts(artifacts_path)
    masks_visible = load_visible_masks(masks_path)
    all_poses, all_scores = prepare_iou_weighted_candidates(
        artifacts,
        masks_visible,
    )
    mesh = trimesh.load(object_paths.mesh, force="mesh")
    optimized = optimize_pose_candidates(
        all_poses,
        all_scores,
        mesh,
        config,
        return_debug=args.debug,
    )
    if args.debug:
        trajectory, debug = optimized
    else:
        trajectory = optimized

    np.save(save_path, trajectory)
    if args.debug:
        np.savez(object_paths.debug_data, **debug)
    render_optimized_poses(seq_dir, object_paths, mesh, trajectory)
    return "processed"


def main():
    """Run pose optimization for every processed DexYCB object directory."""
    args = parse_args()
    run_id = args.run_id or generate_run_id()
    args.run_id = run_id
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    # Suppress noisy third-party logs so progress and failure messages remain
    # readable during long dataset-wide batch runs.
    configure_quiet_logging()

    # Each sequence contains a video folder; process the sequence parent so the
    # matching objects/gpt directory can be resolved consistently.
    data_root = Path(args.data_root)
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]
    stats = {
        "processed": 0,
        "skipped_existing": 0,
        "skipped_missing_artifacts": 0,
        "skipped_missing_masks": 0,
        "failed": 0,
    }
    record_path = write_experiment_record(
        Path(__file__).resolve().parent,
        run_id=run_id,
        generated_at=generated_at,
        argv=sys.argv,
        args=args,
        stats=stats,
        run_status="running",
    )
    tqdm.tqdm.write(f"Experiment record: {record_path}")

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            # Process every detected object directory in this sequence.
            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                status = process_object(seq_dir, object_dir, args)
                stats[status] = stats.get(status, 0) + 1

        except Exception:
            # Keep batch processing alive after a sequence-level failure while
            # preserving the full traceback for later debugging.
            stats["failed"] = stats.get("failed", 0) + 1
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())

    write_experiment_record(
        Path(__file__).resolve().parent,
        run_id=run_id,
        generated_at=generated_at,
        argv=sys.argv,
        args=args,
        stats=stats,
        run_status="completed",
    )


if __name__ == "__main__":
    main()
