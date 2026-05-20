"""Estimate and track object poses for processed DexYCB sequences.

This script scans the processed DexYCB dataset, runs FoundationPose on every
predicted object mesh and visible mask sequence, saves per-frame object poses,
and writes an MP4 visualization with the estimated 3D bounding box and axes.
"""

import os
import io
import logging
import time
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
from tracking_registration import invalid_pose, is_invalid_pose, registration_inputs_are_valid


DEFAULT_SCFLOW2_CONFIG = "third_party/SCFlow2/configs/flow_refine/scflow2.py"
DEFAULT_SCFLOW2_CHECKPOINT = "weights/scflow2_pretrained.pth"


def add_timing(timing_stats, stage, elapsed):
    """Accumulate elapsed seconds for a named processing stage."""

    stats = timing_stats.setdefault(stage, {"count": 0, "total": 0.0})
    stats["count"] += 1
    stats["total"] += elapsed


def merge_timing_stats(total_timing_stats, object_timing_stats):
    """Merge one successfully processed object's timings into run totals."""

    for stage, stats in object_timing_stats.items():
        total_stats = total_timing_stats.setdefault(stage, {"count": 0, "total": 0.0})
        total_stats["count"] += stats["count"]
        total_stats["total"] += stats["total"]


def print_timing_summary(timing_stats, processed_objects, skipped_objects, failed_objects):
    """Print timing and object-count summaries without disturbing tqdm bars."""

    tqdm.tqdm.write("")
    tqdm.tqdm.write("Timing summary")
    tqdm.tqdm.write(
        f"processed_objects={processed_objects} "
        f"skipped_objects={skipped_objects} "
        f"failed_objects={failed_objects}"
    )

    if processed_objects == 0:
        tqdm.tqdm.write("No processed objects; timing averages are unavailable.")
        return

    header = f"{'stage':<30} {'count':>10} {'total_seconds':>15} {'avg_seconds':>15}"
    tqdm.tqdm.write(header)
    tqdm.tqdm.write("-" * len(header))

    for stage, stats in timing_stats.items():
        count = stats["count"]
        total = stats["total"]
        avg = total / count if count else 0.0
        tqdm.tqdm.write(f"{stage:<30} {count:>10d} {total:>15.6f} {avg:>15.6f}")

    total_stats = timing_stats.get("total_processed_object")
    overall_avg = total_stats["total"] / processed_objects if total_stats else 0.0
    tqdm.tqdm.write(f"Overall average seconds per processed object: {overall_avg:.6f}")


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
    data_root = Path("/data/datasets/DexYCB_1080P_15fps_30s")
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = f"{code_dir}/debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Build the expensive FoundationPose components on first real object use,
    # then switch object geometry via reset_object for the rest of this run.
    est = None
    scflow2_refiner = None
    timing_stats = {
        "input_loading": {"count": 0, "total": 0.0},
        "mesh_setup": {"count": 0, "total": 0.0},
        "scflow2_initialization": {"count": 0, "total": 0.0},
        "register": {"count": 0, "total": 0.0},
        "track_one": {"count": 0, "total": 0.0},
        "scflow2_refinement": {"count": 0, "total": 0.0},
        "pose_saving": {"count": 0, "total": 0.0},
        "visualization_rendering": {"count": 0, "total": 0.0},
        "video_writing": {"count": 0, "total": 0.0},
        "total_processed_object": {"count": 0, "total": 0.0},
    }
    processed_objects = 0
    skipped_objects = 0
    failed_objects = 0

    try:
        for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
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
                    skipped_objects += 1
                    continue

                object_start = time.perf_counter()
                object_timing_stats = {}

                try:
                    # FoundationPose expects floating-point camera intrinsics.
                    stage_start = time.perf_counter()
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
                    add_timing(object_timing_stats, "input_loading", time.perf_counter() - stage_start)

                    # The oriented bounds provide a centered box for drawing the
                    # object after converting poses from mesh coordinates.
                    stage_start = time.perf_counter()
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
                    add_timing(object_timing_stats, "mesh_setup", time.perf_counter() - stage_start)

                    if args.use_scflow2 and scflow2_refiner is None:
                        stage_start = time.perf_counter()
                        try:
                            scflow2_refiner = scflow2_refiner_cls(
                                config_path=args.scflow2_config,
                                checkpoint_path=args.scflow2_checkpoint,
                                device="cuda",
                            )
                        except Exception as e:
                            failed_objects += 1
                            tqdm.tqdm.write(f"Failed to initialize SCFlow2 for {object_dir.relative_to(data_root)}")
                            io_string = io.StringIO()
                            traceback.print_exc(file=io_string)
                            tqdm.tqdm.write(io_string.getvalue())
                            raise SystemExit(1) from e
                        finally:
                            add_timing(
                                object_timing_stats,
                                "scflow2_initialization",
                                time.perf_counter() - stage_start,
                            )

                    poses = []
                    tracking_initialized = False
                    T = images.shape[0]
                    for frame_idx in range(T):
                        if not tracking_initialized:
                            # Keep trying to initialize until a frame has enough
                            # masked valid-depth pixels for FoundationPose.
                            if not registration_inputs_are_valid(masks[frame_idx], depths[frame_idx]):
                                poses.append(invalid_pose())
                                continue

                            stage_start = time.perf_counter()
                            pose = est.register(
                                K=intrinsics,
                                rgb=images[frame_idx],
                                depth=depths[frame_idx],
                                ob_mask=masks[frame_idx],
                                iteration=5,
                            )
                            add_timing(object_timing_stats, "register", time.perf_counter() - stage_start)
                            if pose is None or getattr(est, "pose_last", None) is None:
                                poses.append(invalid_pose())
                                continue
                            tracking_initialized = True
                        else:
                            # Track from the previous frame's pose estimate.
                            stage_start = time.perf_counter()
                            pose = est.track_one(
                                K=intrinsics,
                                rgb=images[frame_idx],
                                depth=depths[frame_idx],
                                iteration=3
                            )
                            add_timing(object_timing_stats, "track_one", time.perf_counter() - stage_start)

                        if scflow2_refiner is not None and not is_invalid_pose(pose):
                            # Refine the current FoundationPose estimate with the
                            # same frame inputs before using it as tracking state.
                            stage_start = time.perf_counter()
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
                            finally:
                                add_timing(
                                    object_timing_stats,
                                    "scflow2_refinement",
                                    time.perf_counter() - stage_start,
                                )
                            if refined_pose is not None:
                                pose = refined_pose
                                # Keep FoundationPose's internal tracker aligned
                                # with the refined pose for the next frame.
                                est.set_pose_last_from_model_pose(pose)

                        poses.append(pose)

                    poses = np.stack(poses, axis=0)

                    # Save the raw 4x4 object-in-camera pose for each frame.
                    save_path = object_dir / pose_filename
                    stage_start = time.perf_counter()
                    np.save(save_path, poses)
                    add_timing(object_timing_stats, "pose_saving", time.perf_counter() - stage_start)

                    video = []
                    for frame_idx, pose in enumerate(poses):
                        # Visualization helpers draw boxes around a centered mesh,
                        # so convert from the original mesh pose first.
                        stage_start = time.perf_counter()
                        if is_invalid_pose(pose):
                            video.append(images[frame_idx])
                            add_timing(
                                object_timing_stats,
                                "visualization_rendering",
                                time.perf_counter() - stage_start,
                            )
                            continue

                        center_pose = pose@np.linalg.inv(to_origin)
                        vis = draw_posed_3d_box(intrinsics, img=images[frame_idx], ob_in_cam=center_pose, bbox=bbox)
                        vis = draw_xyz_axis(images[frame_idx], ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
                        add_timing(
                            object_timing_stats,
                            "visualization_rendering",
                            time.perf_counter() - stage_start,
                        )

                        video.append(vis)

                    # Write a compact MP4 next to the pose array for quick review.
                    video = np.stack(video, axis=0)
                    save_path = object_dir / video_filename
                    stage_start = time.perf_counter()
                    iio.imwrite(save_path, video)
                    add_timing(object_timing_stats, "video_writing", time.perf_counter() - stage_start)
                    add_timing(
                        object_timing_stats,
                        "total_processed_object",
                        time.perf_counter() - object_start,
                    )
                    merge_timing_stats(timing_stats, object_timing_stats)
                    processed_objects += 1
                except Exception:
                    failed_objects += 1
                    tqdm.tqdm.write(f"Failed to process {object_dir.relative_to(data_root)}")
                    io_string = io.StringIO()
                    traceback.print_exc(file=io_string)
                    tqdm.tqdm.write(io_string.getvalue())
    finally:
        print_timing_summary(timing_stats, processed_objects, skipped_objects, failed_objects)

if __name__ == "__main__":
    main()
