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
    parser = argparse.ArgumentParser(
        description="Optimize precomputed pose candidates for processed DexYCB objects."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root of the processed DexYCB dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate optimized outputs even when poses_optimized.npy already exists.",
    )
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
    with open(candidates_path, "rb") as f:
        candidates = pickle.load(f)

    if "poses" not in candidates or "scores" not in candidates:
        raise KeyError(f"{candidates_path} must contain 'poses' and 'scores'")
    return candidates["poses"], candidates["scores"]


def optimize_pose_candidates(all_poses, all_scores, mesh, args):
    mesh_diameter = compute_mesh_diameter(
        model_pts=np.asarray(mesh.vertices),
        n_sample=10000,
    )
    trajectory = select_pose_trajectory(
        all_poses,
        all_scores,
        mesh_diameter=mesh_diameter,
        trans_lambda=args.trans_lambda,
        rot_lambda=args.rot_lambda,
    )
    return smooth_pose_trajectory(
        trajectory,
        max_invalid_gap=args.max_invalid_gap,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )


def render_optimized_poses(seq_dir, object_dir, mesh, trajectory):
    intrinsics_path = seq_dir / "video" / "intrinsics.npy"
    images_path = seq_dir / "video" / "images.mp4"

    intrinsics = np.load(intrinsics_path).astype(np.float64)
    images = iio.imread(images_path)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    to_origin_inv = np.linalg.inv(to_origin)

    video = []
    for frame_idx, pose in enumerate(trajectory):
        if not pose.any():
            video.append(images[frame_idx])
            continue

        center_pose = pose @ to_origin_inv
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

    iio.imwrite(object_dir / VIDEO_FILENAME, np.stack(video, axis=0))


def process_object(seq_dir, object_dir, args):
    save_path = object_dir / POSES_FILENAME
    if not args.overwrite and save_path.exists():
        return "skipped_existing"

    candidates_path = object_dir / CANDIDATES_FILENAME
    if not candidates_path.exists():
        tqdm.tqdm.write(f"Missing {candidates_path}; skipping")
        return "skipped_missing_candidates"

    mesh_path = object_dir / "mesh.glb"
    all_poses, all_scores = load_pose_candidates(candidates_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    trajectory = optimize_pose_candidates(all_poses, all_scores, mesh, args)

    np.save(save_path, trajectory)
    render_optimized_poses(seq_dir, object_dir, mesh, trajectory)
    return "processed"


def main():
    args = parse_args()
    configure_quiet_logging()

    data_root = Path(args.data_root)
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                process_object(seq_dir, object_dir, args)

        except Exception:
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())


if __name__ == "__main__":
    main()
