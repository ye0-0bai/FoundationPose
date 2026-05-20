"""Multi-GPU pose estimation and tracking for processed DexYCB sequences."""

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import traceback
import warnings

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB_1080P_15fps_30s"


def parse_gpu_ids(value):
    if value is None:
        return None
    gpu_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    return gpu_ids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate and track object poses for processed DexYCB sequences across multiple GPUs."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Processed DexYCB dataset root.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated physical GPU IDs. Defaults to all visible CUDA devices.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to the number of selected GPUs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run pose estimation even when object poses.npy already exists.",
    )
    parser.add_argument(
        "--debug-root",
        default=None,
        help="Directory for worker debug outputs. Defaults to <repo>/debug_multigpu.",
    )
    return parser.parse_args(argv)


def resolve_runtime_args(args):
    gpu_ids = parse_gpu_ids(args.gpus)
    if gpu_ids is None:
        import torch

        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        raise ValueError("No GPUs selected. Pass --gpus or run on a host with visible CUDA devices.")

    num_workers = args.num_workers if args.num_workers is not None else len(gpu_ids)
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")
    if num_workers > len(gpu_ids):
        raise ValueError("--num-workers cannot be greater than the number of selected GPUs.")

    code_dir = Path(__file__).resolve().parent
    debug_root = Path(args.debug_root) if args.debug_root else code_dir / "debug_multigpu"

    return {
        "data_root": str(Path(args.data_root)),
        "gpu_ids": gpu_ids[:num_workers],
        "num_workers": num_workers,
        "overwrite": args.overwrite,
        "debug_root": str(debug_root),
    }


def discover_seq_dirs(data_root):
    data_root = Path(data_root)
    return [seq_video_dir.parent for seq_video_dir in sorted(data_root.glob("**/video"))]


def configure_worker_reporting(worker_id, gpu_id, result_queue):
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    class QueueLogHandler(logging.Handler):
        def emit(self, record):
            try:
                result_queue.put(
                    {
                        "type": "log",
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "level": record.levelname,
                        "message": self.format(record),
                    }
                )
            except Exception:
                pass

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = QueueLogHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(handler)

    def showwarning(message, category, filename, lineno, file=None, line=None):
        result_queue.put(
            {
                "type": "warning",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "message": warnings.formatwarning(message, category, filename, lineno, line).strip(),
            }
        )

    warnings.showwarning = showwarning


def make_worker_state(worker_id, gpu_id, debug_root):
    import datareader as fp_datareader
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    debug_dir = Path(debug_root) / f"worker_{worker_id}_gpu_{gpu_id}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "debug_dir": str(debug_dir),
        "est": None,
        "FoundationPose": FoundationPose,
        "PoseRefinePredictor": PoseRefinePredictor,
        "ScorePredictor": ScorePredictor,
        "dr": fp_datareader.dr,
        "draw_posed_3d_box": fp_datareader.draw_posed_3d_box,
        "draw_xyz_axis": fp_datareader.draw_xyz_axis,
    }


def preprocess_depths(depths):
    depths = depths.copy()
    depths[depths == 65535] = 0
    depths = depths.astype(np.float64) / 1000.0
    depths[(depths < 0.001) | (depths >= np.inf)] = 0
    return depths


def send_object_error(result_queue, worker_state, data_root, object_dir):
    result_queue.put(
        {
            "type": "error",
            "worker_id": worker_state["worker_id"],
            "gpu_id": worker_state["gpu_id"],
            "path": relative_path(object_dir, data_root),
            "message": "Failed to process object",
            "traceback": traceback.format_exc(),
        }
    )


def relative_path(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def setup_estimator_for_mesh(mesh, worker_state):
    est = worker_state["est"]
    FoundationPose = worker_state["FoundationPose"]
    PoseRefinePredictor = worker_state["PoseRefinePredictor"]
    ScorePredictor = worker_state["ScorePredictor"]
    dr = worker_state["dr"]
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
            debug_dir=worker_state["debug_dir"],
            debug=0,
            glctx=glctx,
        )
        worker_state["est"] = est
    else:
        est.reset_object(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
        )
    return est


def render_pose_video(images, poses, intrinsics, mesh, worker_state):
    import trimesh

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    frames = []
    for frame_idx, pose in enumerate(poses):
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = worker_state["draw_posed_3d_box"](
            intrinsics,
            img=images[frame_idx],
            ob_in_cam=center_pose,
            bbox=bbox,
        )
        vis = worker_state["draw_xyz_axis"](
            images[frame_idx],
            ob_in_cam=center_pose,
            scale=0.1,
            K=intrinsics,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        frames.append(vis)
    return np.stack(frames, axis=0)


def process_object(object_dir, intrinsics, images, depths, worker_state):
    import trimesh

    masks_path = object_dir / "masks.npz"
    mesh_path = object_dir / "mesh.glb"

    masks = np.load(masks_path)["masks_visible"]
    mesh = trimesh.load(mesh_path, force="mesh")
    est = setup_estimator_for_mesh(mesh, worker_state)

    poses = []
    frame_count = images.shape[0]
    for frame_idx in range(frame_count):
        if frame_idx == 0:
            pose = est.register(
                K=intrinsics,
                rgb=images[frame_idx],
                depth=depths[frame_idx],
                ob_mask=masks[frame_idx],
                iteration=5,
            )
        else:
            pose = est.track_one(
                K=intrinsics,
                rgb=images[frame_idx],
                depth=depths[frame_idx],
                iteration=3,
            )
        poses.append(pose)

    poses = np.stack(poses, axis=0)
    np.save(object_dir / "poses.npy", poses)
    video = render_pose_video(images, poses, intrinsics, mesh, worker_state)
    iio.imwrite(object_dir / "poses.mp4", video)


def process_seq_dir(seq_dir, data_root, overwrite, worker_state, result_queue):
    seq_dir = Path(seq_dir)
    data_root = Path(data_root)
    intrinsics_path = seq_dir / "video" / "intrinsics.npy"
    images_path = seq_dir / "video" / "images.mp4"
    depths_path = seq_dir / "video" / "depths.npy"
    objects_root = seq_dir / "objects" / "gpt"

    try:
        intrinsics = np.load(intrinsics_path).astype(np.float64)
        images = iio.imread(images_path)
        depths = preprocess_depths(np.load(depths_path))
        object_dirs = sorted(objects_root.glob("object_*"))
    except Exception:
        result_queue.put(
            {
                "type": "error",
                "worker_id": worker_state["worker_id"],
                "gpu_id": worker_state["gpu_id"],
                "path": relative_path(seq_dir, data_root),
                "message": "Failed to load sequence inputs",
                "traceback": traceback.format_exc(),
            }
        )
        return {"processed": 0, "skipped": 0, "failed": 1}

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    for object_dir in object_dirs:
        save_path = object_dir / "poses.npy"
        if not overwrite and save_path.exists():
            stats["skipped"] += 1
            continue

        try:
            process_object(object_dir, intrinsics, images, depths, worker_state)
            stats["processed"] += 1
        except Exception:
            stats["failed"] += 1
            send_object_error(result_queue, worker_state, data_root, object_dir)

    return stats


def worker_main(worker_id, gpu_id, task_queue, result_queue, args_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from datareader import dr, draw_posed_3d_box, draw_xyz_axis

    del torch, trimesh, FoundationPose, PoseRefinePredictor, ScorePredictor
    del dr, draw_posed_3d_box, draw_xyz_axis

    configure_worker_reporting(worker_id, gpu_id, result_queue)
    worker_state = make_worker_state(worker_id, gpu_id, args_dict["debug_root"])
    result_queue.put(
        {
            "type": "log",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "level": "INFO",
            "message": f"worker started on physical GPU {gpu_id}",
        }
    )

    while True:
        seq_dir = task_queue.get()
        if seq_dir is None:
            break

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        try:
            stats = process_seq_dir(
                Path(seq_dir),
                Path(args_dict["data_root"]),
                args_dict["overwrite"],
                worker_state,
                result_queue,
            )
        except Exception:
            stats = {"processed": 0, "skipped": 0, "failed": 1}
            result_queue.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "path": relative_path(seq_dir, args_dict["data_root"]),
                    "message": "Unhandled sequence failure",
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            result_queue.put(
                {
                    "type": "seq_done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "path": relative_path(seq_dir, args_dict["data_root"]),
                    "processed": stats["processed"],
                    "skipped": stats["skipped"],
                    "failed": stats["failed"],
                }
            )


def format_worker_message(result):
    prefix = f"[worker {result.get('worker_id')} gpu {result.get('gpu_id')}]"
    path = result.get("path")
    path_text = f" {path}" if path else ""
    if result["type"] == "log":
        return f"{prefix} {result.get('message', '')}"
    if result["type"] == "warning":
        return f"{prefix} warning:{path_text} {result.get('message', '')}"
    if result["type"] == "error":
        message = f"{prefix} error:{path_text} {result.get('message', '')}"
        tb = result.get("traceback")
        if tb:
            message = f"{message}\n{tb}"
        return message
    return f"{prefix}{path_text} {result}"


def enqueue_tasks(task_queue, seq_dirs, num_workers):
    for seq_dir in seq_dirs:
        task_queue.put(str(seq_dir))
    for _ in range(num_workers):
        task_queue.put(None)


def start_workers(ctx, args_dict, task_queue, result_queue):
    workers = []
    for worker_id, gpu_id in enumerate(args_dict["gpu_ids"]):
        process = ctx.Process(
            target=worker_main,
            args=(worker_id, gpu_id, task_queue, result_queue, args_dict),
        )
        process.start()
        workers.append(process)
    return workers


def collect_results(result_queue, workers, total_sequences):
    summary = {
        "sequences_done": 0,
        "objects_processed": 0,
        "objects_skipped": 0,
        "objects_failed": 0,
    }

    with tqdm(total=total_sequences, dynamic_ncols=True) as progress:
        while summary["sequences_done"] < total_sequences:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if all(process.exitcode is not None for process in workers):
                    remaining = total_sequences - summary["sequences_done"]
                    if remaining:
                        tqdm.write(
                            f"all workers exited before {remaining} sequence(s) reported completion"
                        )
                    break
                for process in workers:
                    if process.exitcode not in (None, 0):
                        tqdm.write(f"worker process {process.pid} exited with code {process.exitcode}")
                continue

            if result["type"] == "seq_done":
                summary["sequences_done"] += 1
                summary["objects_processed"] += result["processed"]
                summary["objects_skipped"] += result["skipped"]
                summary["objects_failed"] += result["failed"]
                progress.update(1)
            elif result["type"] in {"log", "warning", "error"}:
                tqdm.write(format_worker_message(result))

    return summary


def print_summary(sequences_total, summary, workers_failed):
    print("")
    print("Summary")
    print(f"sequences_total={sequences_total}")
    print(f"sequences_done={summary['sequences_done']}")
    print(f"objects_processed={summary['objects_processed']}")
    print(f"objects_skipped={summary['objects_skipped']}")
    print(f"objects_failed={summary['objects_failed']}")
    print(f"workers_failed={workers_failed}")


def main(argv=None):
    args = parse_args(argv)
    try:
        args_dict = resolve_runtime_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    data_root = Path(args_dict["data_root"])
    if not data_root.exists():
        print(f"error: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    seq_dirs = discover_seq_dirs(data_root)
    if not seq_dirs:
        print(f"error: no sequence video directories found under {data_root}", file=sys.stderr)
        return 2

    debug_root = Path(args_dict["debug_root"])
    debug_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    enqueue_tasks(task_queue, seq_dirs, args_dict["num_workers"])
    workers = start_workers(ctx, args_dict, task_queue, result_queue)

    summary = collect_results(result_queue, workers, len(seq_dirs))

    workers_failed = 0
    for process in workers:
        process.join()
        if process.exitcode != 0:
            workers_failed += 1
            print(f"worker process {process.pid} exited with code {process.exitcode}", file=sys.stderr)

    print_summary(len(seq_dirs), summary, workers_failed)
    if workers_failed or summary["objects_failed"] or summary["sequences_done"] != len(seq_dirs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
