import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _write_run_record(path, data_root, run_id="testrun"):
    payload = {
        "run_id": run_id,
        "data_root": str(data_root),
        "output_naming": {
            "debug": f"poses_optimized_{run_id}_debug.npz",
            "poses": f"poses_optimized_{run_id}.npy",
            "video": f"poses_optimized_{run_id}.mp4",
        },
        "optimization_parameters": {
            "trans_lambda": 0.25,
            "rot_lambda": 0.5,
            "max_invalid_gap": 5,
            "smooth_window": 7,
            "smooth_polyorder": 2,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# Pose Optimization Run {run_id}",
                "",
                "```json",
                json.dumps(payload, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    )
    return payload


def _debug_payload(
    scores,
    trans_penalties,
    rot_penalties,
    score_ranks=None,
    frame_indices=None,
):
    scores = np.asarray(scores, dtype=np.float64)
    trans_penalties = np.asarray(trans_penalties, dtype=np.float64)
    rot_penalties = np.asarray(rot_penalties, dtype=np.float64)
    count = len(scores)
    if score_ranks is None:
        score_ranks = np.ones(count, dtype=np.int64)
    if frame_indices is None:
        frame_indices = np.arange(count, dtype=np.int64)
    net = scores - trans_penalties - rot_penalties
    return {
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "selected_candidate_index": np.arange(count, dtype=np.int64),
        "selected_adjusted_score": scores,
        "score_rank": np.asarray(score_ranks, dtype=np.int64),
        "translation_cost": trans_penalties * 10.0,
        "rotation_cost": rot_penalties * 20.0,
        "weighted_translation_penalty": trans_penalties,
        "weighted_rotation_penalty": rot_penalties,
        "net_contribution": net,
        "cumulative_score": np.cumsum(net),
    }


def _write_debug_npz(object_dir, run_id, **payload):
    object_dir.mkdir(parents=True, exist_ok=True)
    path = object_dir / f"poses_optimized_{run_id}_debug.npz"
    np.savez(path, **payload)
    return path


class AnalyzePoseOptimizationObjectiveTests(unittest.TestCase):
    def test_default_outputs_summary_and_object_csv_only(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)
            _write_debug_npz(
                data_root / "seq_a" / "objects" / "gpt" / "object_0001",
                "testrun",
                **_debug_payload([10.0, 5.0], [1.0, 0.5], [0.25, 0.25]),
            )
            _write_debug_npz(
                data_root / "seq_b" / "objects" / "gpt" / "object_0002",
                "testrun",
                **_debug_payload([4.0], [0.1], [0.2]),
            )
            out_dir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([str(run_log), "--output-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "summary.md").exists())
            self.assertTrue((out_dir / "object_summary.csv").exists())
            self.assertFalse((out_dir / "frame_details.csv").exists())

    def test_write_frame_details_generates_frame_csv(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)
            _write_debug_npz(
                data_root / "object_a",
                "testrun",
                **_debug_payload([1.0, 2.0], [0.1, 0.2], [0.0, 0.1]),
            )
            _write_debug_npz(
                data_root / "object_b",
                "testrun",
                **_debug_payload([3.0, 4.0, 5.0], [0.1, 0.2, 0.3], [0.1, 0.1, 0.1]),
            )
            out_dir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        str(run_log),
                        "--output-dir",
                        str(out_dir),
                        "--write-frame-details",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with open(out_dir / "frame_details.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)

    def test_object_summary_computes_objective_ratios(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)
            _write_debug_npz(
                data_root / "object_a",
                "testrun",
                **_debug_payload(
                    [10.0, 5.0, 1.0],
                    [1.0, 0.5, 2.0],
                    [0.25, 0.25, 0.0],
                    score_ranks=[1, 2, 1],
                ),
            )
            out_dir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([str(run_log), "--output-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            with open(out_dir / "object_summary.csv", newline="") as f:
                rows = list(csv.DictReader(f))

            row = rows[0]
            self.assertAlmostEqual(float(row["score_sum"]), 16.0)
            self.assertAlmostEqual(float(row["translation_penalty_sum"]), 3.5)
            self.assertAlmostEqual(float(row["rotation_penalty_sum"]), 0.5)
            self.assertAlmostEqual(float(row["total_penalty_to_score_ratio"]), 0.25)
            self.assertAlmostEqual(float(row["negative_net_frame_ratio"]), 1.0 / 3.0)
            self.assertAlmostEqual(float(row["non_rank1_frame_ratio"]), 1.0 / 3.0)

    def test_zero_score_ratio_is_nan_not_crash(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)
            _write_debug_npz(
                data_root / "object_a",
                "testrun",
                **_debug_payload([0.0, 0.0], [1.0, 2.0], [0.5, 0.5]),
            )
            out_dir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([str(run_log), "--output-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            with open(out_dir / "object_summary.csv", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertIn(row["total_penalty_to_score_ratio"], {"", "nan"})

    def test_missing_debug_files_returns_failure(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            data_root.mkdir()
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                exit_code = main([str(run_log), "--output-dir", str(root / "out")])

            self.assertEqual(exit_code, 1)

    def test_malformed_debug_file_is_reported(self):
        from analyze_pose_optimization_objective import main

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            run_log = root / "runs" / "testrun.md"
            _write_run_record(run_log, data_root)
            _write_debug_npz(
                data_root / "valid",
                "testrun",
                **_debug_payload([1.0], [0.1], [0.2]),
            )
            malformed_dir = data_root / "malformed"
            malformed_dir.mkdir(parents=True)
            malformed_path = malformed_dir / "poses_optimized_testrun_debug.npz"
            np.savez(malformed_path, frame_index=np.array([0]))
            out_dir = root / "out"

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([str(run_log), "--output-dir", str(out_dir)])

            self.assertEqual(exit_code, 0)
            summary = (out_dir / "summary.md").read_text()
            self.assertIn("Warnings", summary)
            self.assertIn(str(malformed_path), summary)


if __name__ == "__main__":
    unittest.main()
