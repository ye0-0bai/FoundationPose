"""Parallel sequence scheduler for precomputed pose optimization.

This script keeps the object optimization algorithm in
``optimize_precomputed_poses.py`` and only changes dataset-wide scheduling:
the main process discovers sequences, worker processes claim one sequence at a
time, and the main process owns the global progress bar and experiment record.
"""

import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import traceback
from argparse import Namespace

import tqdm
# In this environment scipy.signal can bind the system libstdc++ unless a SciPy
# extension has already loaded the conda C++ runtime. Keep direct script imports
# consistent with the existing optimizer tests.
from scipy.spatial.transform import Rotation as _ScipyNativePreload

from optimize_precomputed_poses import (
    DEFAULT_DATA_ROOT,
    ObjectPaths,
    OptimizationConfig,
    configure_quiet_logging,
    generate_run_id,
    process_object,
    write_experiment_record,
)

del _ScipyNativePreload


OBJECT_STAT_KEYS = (
    "processed",
    "skipped_existing",
    "skipped_missing_artifacts",
    "skipped_missing_masks",
    "failed",
)


def parse_args(argv=None):
    """Parse command-line arguments for parallel pose optimization."""
    parser = argparse.ArgumentParser(
        description="Parallel sequence scheduler for precomputed pose optimization."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root of the processed DexYCB dataset.",
    )
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
    parser.add_argument(
        "--num_workers",
        "--num-workers",
        dest="num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Each worker processes one sequence at a time.",
    )
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num_workers must be >= 1")
    return args


def empty_stats():
    """Return a fresh object-status stats mapping."""
    return {key: 0 for key in OBJECT_STAT_KEYS}


def discover_seq_dirs(data_root):
    """Return sequence directories discovered by their ``video`` subdirectory."""
    data_root = Path(data_root)
    return [video_dir.parent for video_dir in sorted(data_root.glob("**/video"))]


def relative_path(path, root):
    """Return ``path`` relative to ``root`` when possible, else the full path."""
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def args_to_worker_namespace(args):
    """Return a picklable namespace containing fields required by workers."""
    return Namespace(
        data_root=str(args.data_root),
        overwrite=bool(args.overwrite),
        run_id=str(args.run_id),
        debug=bool(args.debug),
        max_invalid_gap=args.max_invalid_gap,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        trans_lambda=args.trans_lambda,
        rot_lambda=args.rot_lambda,
    )


def process_seq_dir(seq_dir, args):
    """Process all GPT object directories in one sequence and aggregate stats."""
    seq_dir = Path(seq_dir)
    stats = empty_stats()
    objects_root = seq_dir / "objects" / "gpt"
    object_dirs = sorted(objects_root.glob("object_*"))
    for object_dir in object_dirs:
        try:
            status = process_object(seq_dir, object_dir, args)
        except Exception:
            stats["failed"] += 1
            raise
        stats[status] = stats.get(status, 0) + 1
    return stats


def worker_main(worker_id, task_queue, result_queue, args):
    """Claim sequence tasks until receiving a ``None`` sentinel."""
    configure_quiet_logging()
    data_root = Path(args.data_root)
    while True:
        seq_dir = task_queue.get()
        if seq_dir is None:
            return
        seq_dir = Path(seq_dir)
        try:
            stats = process_seq_dir(seq_dir, args)
        except Exception as exc:
            result_queue.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "path": relative_path(seq_dir, data_root),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            stats = empty_stats()
            stats["failed"] = 1
        message = {
            "type": "seq_done",
            "worker_id": worker_id,
            "path": relative_path(seq_dir, data_root),
        }
        message.update(stats)
        result_queue.put(message)


def enqueue_tasks(task_queue, seq_dirs, num_workers):
    """Enqueue all sequence paths followed by one stop sentinel per worker."""
    for seq_dir in seq_dirs:
        task_queue.put(str(seq_dir))
    for _ in range(num_workers):
        task_queue.put(None)


def start_workers(ctx, task_queue, result_queue, args, num_workers):
    """Start worker processes in the provided multiprocessing context."""
    workers = []
    worker_args = args_to_worker_namespace(args)
    for worker_id in range(num_workers):
        process = ctx.Process(
            target=worker_main,
            args=(worker_id, task_queue, result_queue, worker_args),
        )
        process.start()
        workers.append(process)
    return workers


def collect_results(result_queue, total_sequences, workers=None):
    """Collect worker messages and update the single global sequence bar."""
    stats = empty_stats()
    errors = []
    completed_sequences = 0
    with tqdm.tqdm(total=total_sequences, dynamic_ncols=True) as progress:
        while completed_sequences < total_sequences:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if workers is not None and all(not worker.is_alive() for worker in workers):
                    break
                continue
            if result["type"] == "seq_done":
                for key in OBJECT_STAT_KEYS:
                    stats[key] = stats.get(key, 0) + int(result.get(key, 0))
                completed_sequences += 1
                progress.update(1)
            elif result["type"] == "error":
                errors.append(result)
                tqdm.tqdm.write(
                    f"Worker {result['worker_id']} failed on {result['path']}: {result['message']}"
                )
                tqdm.tqdm.write(result["traceback"])
            else:
                errors.append(
                    {
                        "type": "error",
                        "worker_id": result.get("worker_id", "unknown"),
                        "path": result.get("path", "unknown"),
                        "message": f"unknown result message type: {result.get('type')}",
                        "traceback": "",
                    }
                )
    if completed_sequences < total_sequences:
        missing_count = total_sequences - completed_sequences
        errors.append(
            {
                "type": "error",
                "worker_id": "unknown",
                "path": "unknown",
                "message": f"{missing_count} sequence(s) did not report completion",
                "traceback": "",
            }
        )
    return stats, errors


def print_summary(stats, sequence_count, errors):
    """Print a compact object and sequence processing summary."""
    tqdm.tqdm.write(
        "Pose optimization summary: "
        f"sequences={sequence_count}, "
        f"processed={stats.get('processed', 0)}, "
        f"skipped_existing={stats.get('skipped_existing', 0)}, "
        f"skipped_missing_artifacts={stats.get('skipped_missing_artifacts', 0)}, "
        f"skipped_missing_masks={stats.get('skipped_missing_masks', 0)}, "
        f"failed={stats.get('failed', 0)}, "
        f"sequence_errors={len(errors)}"
    )


def main(argv=None):
    """Run pose optimization across sequences with worker processes."""
    args = parse_args(argv)
    run_id = args.run_id or generate_run_id()
    args.run_id = run_id
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    configure_quiet_logging()

    data_root = Path(args.data_root)
    seq_dirs = discover_seq_dirs(data_root)
    stats = empty_stats()
    repo_root = Path(__file__).resolve().parent
    argv_for_record = sys.argv if argv is None else [sys.argv[0], *argv]
    record_path = write_experiment_record(
        repo_root,
        run_id=run_id,
        generated_at=generated_at,
        argv=argv_for_record,
        args=args,
        stats=stats,
        run_status="running",
    )
    tqdm.tqdm.write(f"Experiment record: {record_path}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    enqueue_tasks(task_queue, seq_dirs, args.num_workers)
    workers = start_workers(ctx, task_queue, result_queue, args, args.num_workers)
    stats, errors = collect_results(
        result_queue,
        total_sequences=len(seq_dirs),
        workers=workers,
    )

    for worker in workers:
        worker.join()
    bad_exitcodes = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if bad_exitcodes or len(errors) > 0 or stats.get("failed", 0) > 0:
        run_status = "completed_with_errors"
    else:
        run_status = "completed"

    write_experiment_record(
        repo_root,
        run_id=run_id,
        generated_at=generated_at,
        argv=argv_for_record,
        args=args,
        stats=stats,
        run_status=run_status,
    )
    print_summary(stats, len(seq_dirs), errors)
    return 1 if run_status == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
