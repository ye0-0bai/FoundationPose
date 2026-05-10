"""Estimate and track object poses for processed DexYCB sequences.

This script scans the processed DexYCB dataset, runs FoundationPose on every
predicted object mesh and visible mask sequence, saves per-frame object poses,
and writes an MP4 visualization with the estimated 3D bounding box and axes.
"""

import os
import io
import logging
from pathlib import Path
import argparse
import traceback

import numpy as np
import imageio.v3 as iio
import trimesh
import pickle
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from estimater import *
from datareader import *


DEFAULT_SCFLOW2_CONFIG = "third_party/SCFlow2/configs/flow_refine/scflow2.py"
DEFAULT_SCFLOW2_CHECKPOINT = "weights/scflow2_pretrained.pth"


def configure_quiet_logging():
    """Suppress low-priority logging from dependencies during batch processing."""

    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate and track object poses for processed DexYCB sequences."
    )
    parser.add_argument("--use_scflow2", action="store_true", help="Refine each tracked pose with SCFlow2.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run pose estimation even when the output pose file already exists.",
    )
    parser.add_argument("--scflow2-config", default=DEFAULT_SCFLOW2_CONFIG, help="SCFlow2 config path.")
    parser.add_argument(
        "--scflow2-checkpoint",
        default=DEFAULT_SCFLOW2_CHECKPOINT,
        help="SCFlow2 checkpoint path.",
    )
    return parser.parse_args()


def main():
    """Run pose registration, tracking, and visualization for each object.

    The input directory is expected to contain processed sequences with
    video frames, depths, camera intrinsics, and predicted object assets. For
    each object, the first frame is registered from its mask, subsequent frames
    are tracked from the previous estimate, optionally refined with SCFlow2,
    and results are skipped when the mode-specific pose file is found.
    """

    args = parse_args()
    configure_quiet_logging()
    scflow2_refiner_cls = None
    if args.use_scflow2:
        from scflow2_online_refiner import SCFlow2OnlineRefiner
        scflow2_refiner_cls = SCFlow2OnlineRefiner

    # Each processed sequence is identified by its video folder; the parent
    # directory contains the matching objects, masks, and output locations.
    data_root = Path("/data/datasets/DexYCB/processed")
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = f"{code_dir}/debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Build the expensive FoundationPose components on first real object use,
    # then switch object geometry via reset_object for the rest of this run.
    est = None
    scflow2_refiner = None

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            intrinsics_path = seq_dir / "video" / "intrinsics.npy"
            images_path = seq_dir / "video" / "images.mp4"
            depths_path = seq_dir / "video" / "depths.npy"

            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))

            for object_dir in object_dirs:
                # Every object is processed independently with its own mesh,
                # visible masks, pose array, and rendered visualization video.
                masks_path = object_dir / "masks.npz"
                mesh_path = object_dir / "mesh.glb"

                # Keep SCFlow2 outputs separate so the baseline tracking result
                # can coexist with the refined result in the same object folder.
                pose_filename = "poses_scflow2.npy" if args.use_scflow2 else "poses.npy"
                video_filename = "poses_scflow2.mp4" if args.use_scflow2 else "poses.mp4"
                save_path = object_dir / pose_filename
                if not args.overwrite and save_path.exists():
                    continue

                # FoundationPose expects floating-point camera intrinsics.
                intrinsics = np.load(intrinsics_path)
                intrinsics = intrinsics.astype(np.float64)

                # RGB frames are loaded as a single T x H x W x 3 array.
                images = iio.imread(images_path)

                # Convert invalid depth values to 0 and scale millimeters to meters.
                depths = np.load(depths_path)
                depths[depths==65535] = 0
                depths = (depths.astype(np.float64)) / 1000.0
                depths[(depths<0.001) | (depths>=np.inf)] = 0

                # Use the visible object masks predicted during preprocessing.
                masks = np.load(masks_path)
                masks = masks["masks_visible"]

                # The oriented bounds provide a centered box for drawing the
                # object after converting poses from mesh coordinates.
                mesh = trimesh.load(mesh_path, force="mesh")
                if est is None:
                    scorer = ScorePredictor()
                    refiner = PoseRefinePredictor()
                    glctx = dr.RasterizeCudaContext()
                    est = FoundationPose(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=debug_dir,
                        debug=0,
                        glctx=glctx,
                    )
                else:
                    est.reset_object(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                    )
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

                if args.use_scflow2 and scflow2_refiner is None:
                    try:
                        scflow2_refiner = scflow2_refiner_cls(
                            config_path=args.scflow2_config,
                            checkpoint_path=args.scflow2_checkpoint,
                            device="cuda",
                        )
                    except Exception as e:
                        tqdm.tqdm.write(f"Failed to initialize SCFlow2 for {object_dir.relative_to(data_root)}")
                        io_string = io.StringIO()
                        traceback.print_exc(file=io_string)
                        tqdm.tqdm.write(io_string.getvalue())
                        raise SystemExit(1) from e

                poses = []
                T = images.shape[0]
                for frame_idx in range(T):
                    if frame_idx == 0:
                        # Initialize the pose from the first visible object mask.
                        pose = est.register(
                            K=intrinsics,
                            rgb=images[frame_idx],
                            depth=depths[frame_idx],
                            ob_mask=masks[frame_idx],
                            iteration=5,
                        )
                    else:
                        # Track from the previous frame's pose estimate.
                        pose = est.track_one(
                            K=intrinsics,
                            rgb=images[frame_idx],
                            depth=depths[frame_idx],
                            iteration=3
                        )

                    if scflow2_refiner is not None:
                        # Refine the current FoundationPose estimate with the
                        # same frame inputs before using it as tracking state.
                        try:
                            refined_pose = scflow2_refiner.refine(
                                rgb=images[frame_idx],
                                depth_m=depths[frame_idx],
                                K=intrinsics,
                                mask=masks[frame_idx],
                                pose_m=pose,
                                mesh=mesh,
                            )
                        except Exception:
                            tqdm.tqdm.write(
                                "SCFlow2 refine failed for "
                                f"{object_dir.relative_to(data_root)}, frame {frame_idx}; "
                                "keeping FoundationPose pose"
                            )
                            io_string = io.StringIO()
                            traceback.print_exc(file=io_string)
                            tqdm.tqdm.write(io_string.getvalue())
                            refined_pose = None
                        if refined_pose is not None:
                            pose = refined_pose
                            # Keep FoundationPose's internal tracker aligned
                            # with the refined pose for the next frame.
                            est.set_pose_last_from_model_pose(pose)

                    poses.append(pose)

                poses = np.stack(poses, axis=0)

                # Save the raw 4x4 object-in-camera pose for each frame.
                save_path = object_dir / pose_filename
                np.save(save_path, poses)

                video = []
                for frame_idx, pose in enumerate(poses):
                    # Visualization helpers draw boxes around a centered mesh,
                    # so convert from the original mesh pose first.
                    center_pose = pose@np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(intrinsics, img=images[frame_idx], ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(images[frame_idx], ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

                    video.append(vis)

                # Write a compact MP4 next to the pose array for quick review.
                video = np.stack(video, axis=0)
                save_path = object_dir / video_filename
                iio.imwrite(save_path, video)
                
        except Exception as e:
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())

if __name__ == "__main__":
    main()
