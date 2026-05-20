import io
import logging
import os
import time
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import tqdm
import trimesh


def configure_quiet_logging():
    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    elif hasattr(value, "data"):
        value = value.data
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def pose_data_to_artifacts(pose_data):
    render_rgbs = tensor_to_numpy(pose_data.rgbAs)
    render_rgbs = np.moveaxis(render_rgbs, 1, -1)
    render_rgbs = np.clip(render_rgbs * 255.0, 0, 255).astype(np.uint8)

    render_masks = tensor_to_numpy(pose_data.maskAs)
    if render_masks.ndim == 4 and render_masks.shape[1] == 1:
        render_masks = render_masks[:, 0]
    render_masks = (render_masks > 0.5).astype(np.uint8)

    tf_to_crops = tensor_to_numpy(pose_data.tf_to_crops)

    return {
        "render_rgbs": render_rgbs,
        "render_masks": render_masks,
        "tf_to_crops": tf_to_crops,
    }


def artifact_key(frame_idx, name):
    return f"{name}_{frame_idx:04d}"


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def add_timing(timing_stats, stage, elapsed, keep_samples=False):
    stats = timing_stats.setdefault(stage, {"count": 0, "total": 0.0})
    stats["count"] += 1
    stats["total"] += elapsed
    if keep_samples:
        stats.setdefault("samples", []).append(elapsed)


def start_timing(sync_cuda=False):
    if sync_cuda:
        cuda_synchronize()
    return time.perf_counter()


def stop_timing(start, sync_cuda=False):
    if sync_cuda:
        cuda_synchronize()
    return time.perf_counter() - start


def print_timing_summary(
    timing_stats,
    processed_objects,
    skipped_objects,
    failed_sequences,
    valid_frames,
    invalid_frames,
    total_elapsed,
):
    tqdm.tqdm.write("")
    tqdm.tqdm.write("Run summary")
    tqdm.tqdm.write("--------------------------")
    tqdm.tqdm.write(f"processed_objects    {processed_objects}")
    tqdm.tqdm.write(f"skipped_objects      {skipped_objects}")
    tqdm.tqdm.write(f"failed_sequences     {failed_sequences}")
    tqdm.tqdm.write(f"valid_frames         {valid_frames}")
    tqdm.tqdm.write(f"invalid_frames       {invalid_frames}")
    tqdm.tqdm.write(f"total_elapsed_s      {total_elapsed:.3f}")

    processed_total = timing_stats.get("total_processed_object", {}).get("total", 0.0)
    stage_rows = [
        (stage, stats)
        for stage, stats in timing_stats.items()
        if stage != "total_processed_object"
    ]
    stage_rows.sort(key=lambda item: item[1]["total"], reverse=True)

    tqdm.tqdm.write("")
    tqdm.tqdm.write("Stage timing summary")
    tqdm.tqdm.write("--------------------------------------------------------------------------------")
    if processed_total > 0:
        tqdm.tqdm.write(
            f"{'stage':30s} {'count':>8s} {'total_s':>12s} {'avg_s':>12s} "
            f"{'pct_processed_object_time':>26s}"
        )
    else:
        tqdm.tqdm.write(
            "No processed object timing denominator is available; "
            "pct_processed_object_time is omitted."
        )
        tqdm.tqdm.write(f"{'stage':30s} {'count':>8s} {'total_s':>12s} {'avg_s':>12s}")

    for stage, stats in stage_rows:
        count = stats["count"]
        total = stats["total"]
        avg = total / count if count else 0.0
        if processed_total > 0:
            pct = 100.0 * total / processed_total
            tqdm.tqdm.write(f"{stage:30s} {count:8d} {total:12.3f} {avg:12.3f} {pct:26.1f}")
        else:
            tqdm.tqdm.write(f"{stage:30s} {count:8d} {total:12.3f} {avg:12.3f}")

    tqdm.tqdm.write("")
    tqdm.tqdm.write("Frame timing summary")
    tqdm.tqdm.write("--------------------------------------------------------------------------")
    tqdm.tqdm.write(
        f"{'stage':30s} {'count':>8s} {'avg_s':>12s} {'p50_s':>12s} "
        f"{'p90_s':>12s} {'max_s':>12s}"
    )
    for stage in ("register_all", "pose_data_to_artifacts", "artifact_store"):
        samples = timing_stats.get(stage, {}).get("samples", [])
        if samples:
            values = np.asarray(samples, dtype=np.float64)
            avg = float(values.mean())
            p50 = float(np.percentile(values, 50))
            p90 = float(np.percentile(values, 90))
            max_value = float(values.max())
            count = int(values.size)
        else:
            avg = p50 = p90 = max_value = 0.0
            count = 0
        tqdm.tqdm.write(
            f"{stage:30s} {count:8d} {avg:12.3f} {p50:12.3f} "
            f"{p90:12.3f} {max_value:12.3f}"
        )


def main():
    run_start = time.perf_counter()
    timing_stats = {}
    processed_objects = 0
    skipped_objects = 0
    failed_sequences = 0
    valid_frames = 0
    invalid_frames = 0

    configure_quiet_logging()

    data_root = Path("/data/datasets/DexYCB/processed")
    timing_start = start_timing()
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]
    add_timing(timing_stats, "sequence_discovery", stop_timing(timing_start))

    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = f"{code_dir}/debug"
    os.makedirs(debug_dir, exist_ok=True)

    est = None

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            intrinsics_path = seq_dir / "video" / "intrinsics.npy"
            images_path = seq_dir / "video" / "images.mp4"
            depths_path = seq_dir / "video" / "depths.npy"

            objects_root = seq_dir / "objects" / "gpt"
            timing_start = start_timing()
            object_dirs = sorted(objects_root.glob("object_*"))
            add_timing(timing_stats, "object_discovery", stop_timing(timing_start))
            for object_dir in object_dirs:
                masks_path = object_dir / "masks.npz"
                mesh_path = object_dir / "mesh.glb"

                save_path = object_dir / "all_pose_candidates_artifacts.npz"
                tmp_save_path = object_dir / "all_pose_candidates_artifacts.tmp.npz"
                if save_path.exists():
                    skipped_objects += 1
                    continue

                processed_object_start = start_timing()

                timing_start = start_timing()
                intrinsics = np.load(intrinsics_path)
                intrinsics = intrinsics.astype(np.float64)
                add_timing(timing_stats, "load_intrinsics", stop_timing(timing_start))

                timing_start = start_timing()
                images = iio.imread(images_path)
                add_timing(timing_stats, "load_images", stop_timing(timing_start))

                timing_start = start_timing()
                depths = np.load(depths_path)
                depths[depths==65535] = 0
                depths = (depths.astype(np.float64)) / 1000.0
                depths[(depths<0.001) | (depths>=np.inf)] = 0
                add_timing(timing_stats, "load_depths", stop_timing(timing_start))

                timing_start = start_timing()
                masks = np.load(masks_path)
                masks = masks["masks_visible"]
                add_timing(timing_stats, "load_masks", stop_timing(timing_start))

                timing_start = start_timing()
                mesh = trimesh.load(mesh_path, force="mesh")
                add_timing(timing_stats, "mesh_load", stop_timing(timing_start))

                if est is None:
                    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr

                    timing_start = start_timing(sync_cuda=True)
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
                    add_timing(
                        timing_stats,
                        "estimator_first_init",
                        stop_timing(timing_start, sync_cuda=True),
                    )
                else:
                    timing_start = start_timing(sync_cuda=True)
                    est.reset_object(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                    )
                    add_timing(
                        timing_stats,
                        "estimator_reset_object",
                        stop_timing(timing_start, sync_cuda=True),
                    )

                T = images.shape[0]
                valid = np.zeros(T, dtype=bool)
                result = {"valid": valid}
                for frame_idx in range(T):
                    timing_start = start_timing(sync_cuda=True)
                    register_result = est.register_all(
                        K=intrinsics,
                        rgb=images[frame_idx],
                        depth=depths[frame_idx],
                        ob_mask=masks[frame_idx],
                        iteration=5,
                        return_pose_data=True,
                    )
                    add_timing(
                        timing_stats,
                        "register_all",
                        stop_timing(timing_start, sync_cuda=True),
                        keep_samples=True,
                    )
                    if register_result is None:
                        valid[frame_idx] = False
                        invalid_frames += 1
                        continue

                    poses, scores, pose_data = register_result
                    timing_start = start_timing(sync_cuda=True)
                    artifacts = pose_data_to_artifacts(pose_data)
                    add_timing(
                        timing_stats,
                        "pose_data_to_artifacts",
                        stop_timing(timing_start, sync_cuda=True),
                        keep_samples=True,
                    )

                    timing_start = start_timing()
                    valid[frame_idx] = True
                    result[artifact_key(frame_idx, "poses")] = poses
                    result[artifact_key(frame_idx, "scores")] = scores
                    result[artifact_key(frame_idx, "render_rgbs")] = artifacts["render_rgbs"]
                    result[artifact_key(frame_idx, "render_masks")] = artifacts["render_masks"]
                    result[artifact_key(frame_idx, "tf_to_crops")] = artifacts["tf_to_crops"]
                    add_timing(
                        timing_stats,
                        "artifact_store",
                        stop_timing(timing_start),
                        keep_samples=True,
                    )
                    valid_frames += 1

                timing_start = start_timing()
                np.savez_compressed(tmp_save_path, **result)
                add_timing(timing_stats, "npz_compress_write", stop_timing(timing_start))

                timing_start = start_timing()
                tmp_save_path.replace(save_path)
                add_timing(timing_stats, "atomic_replace", stop_timing(timing_start))

                processed_objects += 1
                add_timing(
                    timing_stats,
                    "total_processed_object",
                    stop_timing(processed_object_start),
                )

        except Exception:
            failed_sequences += 1
            tqdm.tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.tqdm.write(io_string.getvalue())

    print_timing_summary(
        timing_stats=timing_stats,
        processed_objects=processed_objects,
        skipped_objects=skipped_objects,
        failed_sequences=failed_sequences,
        valid_frames=valid_frames,
        invalid_frames=invalid_frames,
        total_elapsed=time.perf_counter() - run_start,
    )


if __name__ == "__main__":
    main()
