"""Optimize precomputed pose candidates for processed DexYCB objects."""

import argparse
import io
import pickle
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tqdm
import trimesh

from Utils import compute_mesh_diameter, draw_posed_3d_box, draw_xyz_axis
from process_data import (
    configure_quiet_logging,
    select_pose_trajectory,
    smooth_pose_trajectory,
)


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"
CANDIDATES_FILENAME = "all_poses&scores.pkl"
POSES_FILENAME = "poses_optimized.npy"
VIDEO_FILENAME = "poses_optimized.mp4"


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


def optimize_pose_candidates(all_poses, all_scores, mesh, args):
    """Select and smooth the best pose trajectory for one object.

    Args:
        all_poses (Sequence[np.ndarray]): Per-frame candidate poses with shape
            ``(N, 4, 4)`` for each frame.
        all_scores (Sequence[np.ndarray]): Per-frame candidate scores aligned
            with ``all_poses``.
        mesh (trimesh.Trimesh): Object mesh used to normalize translation
            transition costs.
        args (argparse.Namespace): Optimization and smoothing parameters.

    Returns:
        np.ndarray: Optimized pose trajectory with shape ``(T, 4, 4)``.
    """
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
        trans_lambda=args.trans_lambda,
        rot_lambda=args.rot_lambda,
    )
    # Then interpolate only short invalid gaps and smooth continuous valid
    # segments; long missing regions remain invalid to avoid hallucinated poses.
    return smooth_pose_trajectory(
        trajectory,
        max_invalid_gap=args.max_invalid_gap,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )


def render_optimized_poses(seq_dir, object_dir, mesh, trajectory):
    """Render an RGB video overlay for an optimized object trajectory.

    Args:
        seq_dir (Path): Sequence directory containing the ``video`` folder.
        object_dir (Path): Object output directory where the rendered video is
            written.
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
    iio.imwrite(object_dir / VIDEO_FILENAME, np.stack(video, axis=0))


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
    # Skip completed objects unless the caller explicitly requests regeneration.
    save_path = object_dir / POSES_FILENAME
    if not args.overwrite and save_path.exists():
        return "skipped_existing"

    # Some object directories may not have candidate files, especially during
    # interrupted preprocessing runs; report and continue with other objects.
    candidates_path = object_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates"

    # Load all object-specific inputs before running the optimization pipeline.
    mesh_path = object_dir / "mesh.glb"
    all_poses, all_scores = load_pose_candidates(candidates_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    trajectory = optimize_pose_candidates(all_poses, all_scores, mesh, args)

    # Save the numeric trajectory for downstream use and the rendered video for
    # human quality control.
    np.save(save_path, trajectory)
    render_optimized_poses(seq_dir, object_dir, mesh, trajectory)
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
