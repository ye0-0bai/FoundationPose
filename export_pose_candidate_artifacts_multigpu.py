"""Export pose candidate artifacts for processed DexYCB sequences across GPUs."""

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


DEFAULT_DATA_ROOT = "/data/datasets/DexYCB/processed"


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


def parse_gpu_ids(value):
    if value is None:
        return []
    gpu_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    return gpu_ids


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export FoundationPose pose candidate artifacts across multiple GPUs."
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="Processed DexYCB dataset root.",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical GPU IDs, for example 0,1,2,3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate artifacts even when all_pose_candidates_artifacts.npz exists.",
    )
    return parser.parse_args(argv)


def resolve_runtime_args(args):
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integer GPU IDs.") from exc
    if not gpu_ids:
        raise ValueError("--gpus must select at least one GPU.")

    return {
        "data_root": str(Path(args.data_root)),
        "gpu_ids": gpu_ids,
        "overwrite": args.overwrite,
    }


def discover_seq_dirs(data_root):
    data_root = Path(data_root)
    return [seq_video_dir.parent for seq_video_dir in sorted(data_root.glob("**/video"))]


def relative_path(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def preprocess_depths(depths):
    depths = depths.copy()
    depths[depths == 65535] = 0
    depths = depths.astype(np.float64) / 1000.0
    depths[(depths < 0.001) | (depths >= np.inf)] = 0
    return depths


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


def make_worker_state(worker_id, gpu_id, FoundationPose, PoseRefinePredictor, ScorePredictor, dr):
    code_dir = Path(__file__).resolve().parent
    debug_dir = code_dir / "debug_pose_candidate_artifacts_multigpu" / (
        f"worker_{worker_id}_gpu_{gpu_id}"
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "debug_dir": str(debug_dir),
        "est": None,
        "FoundationPose": FoundationPose,
        "PoseRefinePredictor": PoseRefinePredictor,
        "ScorePredictor": ScorePredictor,
        "dr": dr,
    }


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


def send_error(result_queue, worker_state, data_root, path, message):
    result_queue.put(
        {
            "type": "error",
            "worker_id": worker_state["worker_id"],
            "gpu_id": worker_state["gpu_id"],
            "path": relative_path(path, data_root),
            "message": message,
            "traceback": traceback.format_exc(),
        }
    )


def process_object(object_dir, intrinsics, images, depths, overwrite, worker_state):
    import trimesh

    masks_path = object_dir / "masks.npz"
    mesh_path = object_dir / "mesh.glb"
    save_path = object_dir / "all_pose_candidates_artifacts.npz"
    tmp_save_path = object_dir / "all_pose_candidates_artifacts.tmp.npz"

    if not overwrite and save_path.exists():
        return {
            "processed": 0,
            "skipped": 1,
            "failed": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
        }

    masks = np.load(masks_path)["masks_visible"]
    mesh = trimesh.load(mesh_path, force="mesh")
    est = setup_estimator_for_mesh(mesh, worker_state)

    T = images.shape[0]
    valid = np.zeros(T, dtype=bool)
    result = {"valid": valid}
    stats = {
        "processed": 1,
        "skipped": 0,
        "failed": 0,
        "valid_frames": 0,
        "invalid_frames": 0,
    }

    for frame_idx in range(T):
        register_result = est.register_all(
            K=intrinsics,
            rgb=images[frame_idx],
            depth=depths[frame_idx],
            ob_mask=masks[frame_idx],
            iteration=5,
            return_pose_data=True,
        )
        if register_result is None:
            valid[frame_idx] = False
            stats["invalid_frames"] += 1
            continue

        poses, scores, pose_data = register_result
        artifacts = pose_data_to_artifacts(pose_data)
        valid[frame_idx] = True
        result[artifact_key(frame_idx, "poses")] = poses
        result[artifact_key(frame_idx, "scores")] = scores
        result[artifact_key(frame_idx, "render_rgbs")] = artifacts["render_rgbs"]
        result[artifact_key(frame_idx, "render_masks")] = artifacts["render_masks"]
        result[artifact_key(frame_idx, "tf_to_crops")] = artifacts["tf_to_crops"]
        stats["valid_frames"] += 1

    np.savez_compressed(tmp_save_path, **result)
    tmp_save_path.replace(save_path)
    return stats


def process_seq_dir(seq_dir, data_root, overwrite, worker_state, result_queue):
    seq_dir = Path(seq_dir)
    data_root = Path(data_root)
    intrinsics_path = seq_dir / "video" / "intrinsics.npy"
    images_path = seq_dir / "video" / "images.mp4"
    depths_path = seq_dir / "video" / "depths.npy"
    objects_root = seq_dir / "objects" / "gpt"

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "valid_frames": 0,
        "invalid_frames": 0,
    }

    try:
        intrinsics = np.load(intrinsics_path).astype(np.float64)
        images = iio.imread(images_path)
        depths = preprocess_depths(np.load(depths_path))
        object_dirs = sorted(objects_root.glob("object_*"))
    except Exception:
        stats["failed"] += 1
        send_error(result_queue, worker_state, data_root, seq_dir, "Failed to load sequence inputs")
        return stats

    for object_dir in object_dirs:
        try:
            object_stats = process_object(
                object_dir,
                intrinsics,
                images,
                depths,
                overwrite,
                worker_state,
            )
            stats["processed"] += object_stats["processed"]
            stats["skipped"] += object_stats["skipped"]
            stats["failed"] += object_stats["failed"]
            stats["valid_frames"] += object_stats["valid_frames"]
            stats["invalid_frames"] += object_stats["invalid_frames"]
        except Exception:
            stats["failed"] += 1
            send_error(result_queue, worker_state, data_root, object_dir, "Failed to process object")

    return stats


def worker_main(worker_id, gpu_id, task_queue, result_queue, args_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr

    del torch, trimesh

    configure_worker_reporting(worker_id, gpu_id, result_queue)
    worker_state = make_worker_state(
        worker_id,
        gpu_id,
        FoundationPose,
        PoseRefinePredictor,
        ScorePredictor,
        dr,
    )
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

        stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
        }
        try:
            stats = process_seq_dir(
                Path(seq_dir),
                Path(args_dict["data_root"]),
                args_dict["overwrite"],
                worker_state,
                result_queue,
            )
        except Exception:
            stats["failed"] += 1
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
                    "valid_frames": stats["valid_frames"],
                    "invalid_frames": stats["invalid_frames"],
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
        "valid_frames": 0,
        "invalid_frames": 0,
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
                summary["valid_frames"] += result["valid_frames"]
                summary["invalid_frames"] += result["invalid_frames"]
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
    print(f"valid_frames={summary['valid_frames']}")
    print(f"invalid_frames={summary['invalid_frames']}")
    print(f"workers_failed={workers_failed}")


def main(argv=None):
    configure_quiet_logging()
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

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(args_dict["gpu_ids"])
    enqueue_tasks(task_queue, seq_dirs, num_workers)
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
