"""Optimize precomputed pose candidates for processed DexYCB objects.

当前优化逻辑说明:

本脚本不重新估计单帧 pose，而是在预计算得到的每帧候选 pose 集合
``all_poses&scores.pkl`` 上做后处理优化。整体流程分为两级:

1. 轨迹选择:
   - 对每一帧保留有限且数值有效的候选 pose 与 score。
   - 将单帧 score 归一化为 log-probability，作为候选本身的置信度项。
   - 使用动态规划从所有有效帧中选择一条全局最优候选轨迹，而不是逐帧
     贪心选择最高分候选。
   - 相邻帧转移代价由两部分组成:
       translation_cost = ||t_cur - t_prev|| / mesh_diameter
       rotation_cost = angle(R_prev, R_cur) / pi
     其中 mesh_diameter 用于把不同尺寸物体的平移跳变归一到可比较尺度。
   - 当前目标相当于最大化:
       sum(normalized_score)
       - trans_lambda * sum(translation_cost)
       - rot_lambda * sum(rotation_cost)
     因此 trans_lambda 与 rot_lambda 越大，轨迹越偏向时间连续；越小，越
     偏向逐帧候选分数。
   - 没有有效候选的帧不参与动态规划，输出轨迹中这些帧先保留为全零 pose。

2. 轨迹后处理:
   - 仅对长度不超过 max_invalid_gap 的无效短缺口做插值: 平移线性插值，
     旋转使用 Slerp。
   - 对连续有效片段做 Savitzky-Golay 平滑: 平移直接平滑，旋转先转为四元数
     并处理符号连续性后再平滑和归一化。
   - 超过 max_invalid_gap 的长缺失区间仍保持全零 pose，避免在缺少可靠候选
     的区域生成过度猜测的轨迹。

脚本最终保存 ``poses_optimized.npy`` 供下游使用，并渲染
``poses_optimized.mp4`` 作为人工检查当前优化方案效果的可视化结果。
"""

import argparse
import io
import pickle
import traceback
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tqdm
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from Utils import compute_mesh_diameter, draw_posed_3d_box, draw_xyz_axis
from process_data import configure_quiet_logging


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"
CANDIDATES_FILENAME = "all_poses&scores.pkl"
POSES_FILENAME = "poses_optimized.npy"
VIDEO_FILENAME = "poses_optimized.mp4"


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
    candidates: Path
    optimized_poses: Path
    optimized_video: Path

    @classmethod
    def from_object_dir(cls, object_dir):
        object_dir = Path(object_dir)
        return cls(
            object_dir=object_dir,
            mesh=object_dir / "mesh.glb",
            candidates=object_dir / CANDIDATES_FILENAME,
            optimized_poses=object_dir / POSES_FILENAME,
            optimized_video=object_dir / VIDEO_FILENAME,
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
        help="Regenerate optimized outputs even when poses_optimized.npy already exists.",
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


# ---------------------------------------------------------------------------
# Candidate data loading and validation


def load_pose_candidates(candidates_path):
    """Load pose candidates and confidence scores from a pickle file.

    Args:
        candidates_path (Path): Path to ``all_poses&scores.pkl``.

    Returns:
        tuple: A pair ``(poses, scores)`` where each item contains per-frame
        pose candidates and their corresponding scores.

    Raises:
        KeyError: If the pickle file does not contain ``poses`` and ``scores``.
    """
    # Candidate files are produced by the precomputation stage and are expected
    # to be dictionaries with parallel per-frame pose and score lists.
    with open(candidates_path, "rb") as f:
        candidates = pickle.load(f)

    # Fail early if the file format is wrong; downstream trajectory selection
    # assumes both keys exist and are aligned frame by frame.
    if "poses" not in candidates or "scores" not in candidates:
        raise KeyError(f"{candidates_path} must contain 'poses' and 'scores'")
    return candidates["poses"], candidates["scores"]


def normalize_scores(scores, temperature=1.0):
    """Convert raw candidate scores into comparable log-probabilities.

    Args:
        scores (array-like): One frame's candidate confidence scores.
        temperature (float): Optional softmax temperature applied before
            normalization.

    Returns:
        np.ndarray: Score array with the same shape, where finite entries are
        normalized log-probabilities.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)

    # Invalid scores are kept at a very low value so accidental reuse is still
    # dominated by any finite candidate after filtering.
    normalized = np.full_like(scores, -1e6, dtype=np.float64)
    if not finite.any():
        return np.zeros_like(scores, dtype=np.float64)

    x = scores[finite] / temperature
    x = x - x.max()
    log_probs = x - np.log(np.exp(x).sum())

    normalized[finite] = log_probs
    return normalized


def finite_candidate_mask(poses, scores):
    """Return candidates with finite score and finite pose matrix entries."""
    finite = np.isfinite(scores)
    if len(poses) > 0:
        finite = finite & np.isfinite(poses).all(axis=(1, 2))
    return finite


def clean_pose_candidates(all_poses, all_scores):
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
        scores_per_frame.append(normalize_scores(scores[finite]))
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

    Returns:
        np.ndarray: A ``(T, 4, 4)`` pose trajectory. Frames with no valid
        candidates remain all zeros.

    Raises:
        ValueError: If inputs are misaligned or no valid candidates exist.
    """
    if len(all_poses) != len(all_scores):
        raise ValueError("all_poses and all_scores must have the same number of frames")
    if len(all_poses) == 0:
        return np.empty((0, 4, 4), dtype=np.float64)

    poses_per_frame, scores_per_frame, valid_frame_indices = clean_pose_candidates(
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
    return build_trajectory(len(all_poses), valid_frame_indices, selected_poses)


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


def optimize_pose_candidates(all_poses, all_scores, mesh, config):
    """Select and smooth the best pose trajectory for one object.

    Args:
        all_poses (Sequence[np.ndarray]): Per-frame candidate poses with shape
            ``(N, 4, 4)`` for each frame.
        all_scores (Sequence[np.ndarray]): Per-frame candidate scores aligned
            with ``all_poses``.
        mesh (trimesh.Trimesh): Object mesh used to normalize translation
            transition costs.
        config (OptimizationConfig | argparse.Namespace): Optimization and
            smoothing parameters. argparse namespaces are accepted for
            compatibility with the original call site.

    Returns:
        np.ndarray: Optimized pose trajectory with shape ``(T, 4, 4)``.
    """
    config = ensure_optimization_config(config)
    # The mesh diameter puts translation jumps on a scale comparable across
    # objects, so large and small meshes can use the same penalty weights.
    mesh_diameter = compute_mesh_diameter(
        model_pts=np.asarray(mesh.vertices),
        n_sample=10000,
    )
    # First select one candidate per valid frame using score and transition
    # costs, leaving invalid frames as all-zero poses.
    trajectory = select_pose_trajectory(
        all_poses,
        all_scores,
        mesh_diameter=mesh_diameter,
        trans_lambda=config.trans_lambda,
        rot_lambda=config.rot_lambda,
    )
    # Then interpolate only short invalid gaps and smooth continuous valid
    # segments; long missing regions remain invalid to avoid hallucinated poses.
    return smooth_pose_trajectory(
        trajectory,
        max_invalid_gap=config.max_invalid_gap,
        smooth_window=config.smooth_window,
        smooth_polyorder=config.smooth_polyorder,
    )


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
    if not isinstance(object_paths, ObjectPaths):
        object_paths = ObjectPaths.from_object_dir(object_paths)

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
    """Optimize and render precomputed pose candidates for one object.

    Args:
        seq_dir (Path): Sequence directory containing RGB frames and camera
            intrinsics.
        object_dir (Path): Directory containing object mesh and precomputed
            pose candidates.
        args (argparse.Namespace): Command-line arguments controlling the
            optimization run.

    Returns:
        str: Processing status. One of ``skipped_existing``,
        ``skipped_missing_candidates``, or ``processed``.
    """
    object_paths = ObjectPaths.from_object_dir(object_dir)

    # Skip completed objects unless the caller explicitly requests regeneration.
    save_path = object_paths.optimized_poses
    if not args.overwrite and save_path.exists():
        return "skipped_existing"

    # Some object directories may not have candidate files, especially during
    # interrupted preprocessing runs; report and continue with other objects.
    candidates_path = object_paths.candidates
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates"

    # Load all object-specific inputs before running the optimization pipeline.
    config = ensure_optimization_config(args)
    all_poses, all_scores = load_pose_candidates(candidates_path)
    mesh = trimesh.load(object_paths.mesh, force="mesh")
    trajectory = optimize_pose_candidates(all_poses, all_scores, mesh, config)

    # Save the numeric trajectory for downstream use and the rendered video for
    # human quality control.
    np.save(save_path, trajectory)
    render_optimized_poses(seq_dir, object_paths, mesh, trajectory)
    return "processed"


def main():
    """Run pose optimization for every processed DexYCB object directory."""
    args = parse_args()
    # Suppress noisy third-party logs so progress and failure messages remain
    # readable during long dataset-wide batch runs.
    configure_quiet_logging()

    # Each sequence contains a video folder; process the sequence parent so the
    # matching objects/gpt directory can be resolved consistently.
    data_root = Path(args.data_root)
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            # Process every detected object directory in this sequence.
            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                process_object(seq_dir, object_dir, args)

        except Exception:
            # Keep batch processing alive after a sequence-level failure while
            # preserving the full traceback for later debugging.
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())


if __name__ == "__main__":
    main()
