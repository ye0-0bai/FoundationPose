import argparse
import ast
import contextlib
import csv
import io
import math
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _parse(path):
    return ast.parse((ROOT / path).read_text())


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def _parser_add_argument_flags(function_node):
    flags = set()
    for child in ast.walk(function_node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "add_argument"
        ):
            continue
        for arg in child.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.add(arg.value)
    return flags


def _constants_under(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _pose(x=0.0, angle_degrees=0.0):
    pose = np.eye(4, dtype=np.float64)
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    pose[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose[0, 3] = x
    return pose


def _args(**overrides):
    values = {
        "hist_bins": 5,
        "temperature": 1.0,
        "save_score_histograms": False,
        "nms_threshold": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_candidates(object_dir, candidates):
    with open(object_dir / "all_poses&scores.pkl", "wb") as f:
        pickle.dump(candidates, f)


class AnalyzePoseScoreConfidenceAlgorithmTests(unittest.TestCase):
    def test_stable_softmax_is_stable_for_large_scores_and_temperature(self):
        from analyze_pose_score_confidence import stable_softmax

        probabilities = stable_softmax(np.array([1000.0, 1000.0]), temperature=2.0)

        np.testing.assert_allclose(probabilities, np.array([0.5, 0.5]))

    def test_valid_candidate_mask_requires_finite_score_and_pose(self):
        from analyze_pose_score_confidence import valid_candidate_mask

        poses = np.stack([_pose(), _pose(), _pose(), _pose()])
        poses[1, 0, 0] = np.nan
        poses[3, 1, 2] = np.inf
        scores = np.array([4.0, 3.0, np.nan, 1.0])

        mask = valid_candidate_mask(poses, scores)

        np.testing.assert_array_equal(mask, np.array([True, False, False, False]))

    def test_rotation_distance_degrees_ignores_translation(self):
        from analyze_pose_score_confidence import rotation_distance_degrees

        self.assertAlmostEqual(
            rotation_distance_degrees(_pose(0.0, 0.0), _pose(100.0, 0.0)),
            0.0,
        )
        self.assertAlmostEqual(
            rotation_distance_degrees(_pose(0.0, 0.0), _pose(0.0, 90.0)),
            90.0,
        )

    def test_pose_nms_scores_uses_rotation_degrees_only(self):
        from analyze_pose_score_confidence import pose_nms_scores

        poses = np.stack([
            _pose(0.0, 0.0),
            _pose(100.0, 0.0),
            _pose(0.0, 3.0),
            _pose(0.0, 10.0),
        ])
        scores = np.array([10.0, 9.0, 8.0, 7.0])

        kept_poses, kept_scores, kept_indices = pose_nms_scores(
            poses, scores, threshold_degrees=5.0
        )

        self.assertEqual(len(kept_poses), 2)
        np.testing.assert_allclose(kept_scores, np.array([10.0, 7.0]))
        np.testing.assert_array_equal(kept_indices, np.array([0, 3]))

    def test_zero_degree_nms_still_suppresses_identical_rotations(self):
        from analyze_pose_score_confidence import pose_nms_scores

        poses = np.stack([
            _pose(0.0, 0.0),
            _pose(50.0, 0.0),
            _pose(0.0, 1.0),
        ])
        scores = np.array([3.0, 2.0, 1.0])

        _, kept_scores, kept_indices = pose_nms_scores(
            poses, scores, threshold_degrees=0.0
        )

        np.testing.assert_allclose(kept_scores, np.array([3.0, 1.0]))
        np.testing.assert_array_equal(kept_indices, np.array([0, 2]))

    def test_compute_frame_score_metrics_reports_nms_only_entropy_norm(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        poses = np.stack([_pose(0.0, 0.0), _pose(0.0, 20.0), _pose(0.0, 40.0)])
        poses[2, 0, 0] = np.nan
        scores = np.array([4.0, 4.0, 100.0])

        metrics = compute_frame_score_metrics(
            poses, scores, temperature=1.0, nms_threshold=5.0
        )

        self.assertEqual(metrics["valid_candidate_count"], 2)
        self.assertEqual(metrics["post_nms_candidate_count"], 2)
        self.assertEqual(metrics["top1_score"], 4.0)
        self.assertEqual(metrics["top1_top2_gap"], 0.0)
        self.assertAlmostEqual(metrics["top1_prob"], 0.5)
        self.assertAlmostEqual(metrics["entropy"], 1.0)
        self.assertNotIn("entropy_norm", metrics)
        self.assertNotIn("confidence_entropy", metrics)

    def test_compute_frame_score_metrics_keeps_empty_frames_as_nan_metrics(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        metrics = compute_frame_score_metrics(
            np.empty((0, 4, 4), dtype=np.float64), [], nms_threshold=0.0
        )

        self.assertEqual(metrics["valid_candidate_count"], 0)
        self.assertEqual(metrics["post_nms_candidate_count"], 0)
        self.assertTrue(np.isnan(metrics["top1_score"]))
        self.assertTrue(np.isnan(metrics["top1_prob"]))
        self.assertTrue(np.isnan(metrics["entropy"]))

    def test_compute_frame_score_metrics_rejects_nonpositive_temperature(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        with self.assertRaises(ValueError):
            compute_frame_score_metrics(
                np.stack([_pose(), _pose()]),
                [1.0, 2.0],
                temperature=0.0,
            )


class AnalyzePoseScoreConfidenceCsvTests(unittest.TestCase):
    def test_process_object_missing_mesh_still_processes_and_writes_nms_only_histograms(self):
        from analyze_pose_score_confidence import process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = Path(tmpdir) / "seq_a" / "objects" / "gpt" / "object_0001"
            object_dir.mkdir(parents=True)
            _write_candidates(
                object_dir,
                {
                    "poses": [
                        np.stack([_pose(0.0, 0.0), _pose(10.0, 0.0), _pose(0.0, 20.0)]),
                        np.stack([_pose(), _pose()]),
                    ],
                    "scores": [[3.0, 2.0, 1.0], [np.nan, np.inf]],
                },
            )

            status, records = process_object(
                object_dir,
                _args(nms_threshold=0.0, save_score_histograms=True),
            )
            curves_exists = (object_dir / "debug" / "score_confidence_curves.png").exists()
            distributions_exists = (
                object_dir / "debug" / "score_confidence_distributions.png"
            ).exists()
            histogram_exists = (
                object_dir / "debug" / "score_histograms" / "frame_000000.png"
            ).exists()
            nms_histogram_dir_exists = (
                object_dir / "debug" / "nms_score_histograms"
            ).exists()

        self.assertEqual(status, "processed")
        self.assertEqual(len(records), 2)
        self.assertTrue(curves_exists)
        self.assertTrue(distributions_exists)
        self.assertTrue(histogram_exists)
        self.assertFalse(nms_histogram_dir_exists)
        self.assertEqual(records[0]["object_dir"], str(object_dir))
        self.assertEqual(records[0]["frame_idx"], 0)
        self.assertEqual(records[0]["valid_candidate_count"], 3)
        self.assertEqual(records[0]["post_nms_candidate_count"], 2)
        self.assertEqual(records[0]["top1_score"], 3.0)
        self.assertEqual(records[0]["top1_top2_gap"], 2.0)
        self.assertEqual(records[1]["valid_candidate_count"], 0)
        self.assertEqual(records[1]["post_nms_candidate_count"], 0)
        self.assertTrue(np.isnan(records[1]["entropy"]))
        self.assertNotIn("seq_dir", records[0])
        self.assertNotIn("object_name", records[0])
        self.assertNotIn("pre_nms_candidate_count", records[0])
        self.assertNotIn("nms_top1_score", records[0])

    def test_process_object_score_only_pickle_skips_entire_object_and_warns(self):
        from analyze_pose_score_confidence import process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = Path(tmpdir) / "seq_b" / "objects" / "gpt" / "object_0002"
            object_dir.mkdir(parents=True)
            _write_candidates(object_dir, {"scores": [[1.0, 2.0]]})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status, records = process_object(object_dir, _args())

        self.assertEqual(status, "skipped_invalid_candidates")
        self.assertEqual(records, [])
        self.assertIn("poses", stdout.getvalue())
        self.assertIn("scores", stdout.getvalue())

    def test_process_object_mismatched_pose_score_counts_skips_without_partial_records(self):
        from analyze_pose_score_confidence import process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = Path(tmpdir) / "seq_c" / "objects" / "gpt" / "object_0003"
            object_dir.mkdir(parents=True)
            _write_candidates(
                object_dir,
                {
                    "poses": [np.stack([_pose(), _pose()]), np.stack([_pose()])],
                    "scores": [[1.0, 2.0], [3.0, 4.0]],
                },
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status, records = process_object(object_dir, _args())

        self.assertEqual(status, "skipped_invalid_candidates")
        self.assertEqual(records, [])
        self.assertIn("pose and score counts do not match", stdout.getvalue())
        self.assertFalse((object_dir / "debug" / "score_confidence_curves.png").exists())

    def test_write_frame_csv_uses_fixed_nms_only_fields(self):
        from analyze_pose_score_confidence import FRAME_CSV_FIELDNAMES, write_frame_csv

        record = {
            "object_dir": "seq/objects/gpt/object_0001",
            "frame_idx": 3,
            "valid_candidate_count": 4,
            "post_nms_candidate_count": 2,
            "top1_score": 8.0,
            "top1_top2_gap": 1.5,
            "top1_prob": 0.7,
            "entropy": 0.4,
            "ignored": "x",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "debug" / "frames.csv"
            write_frame_csv([record], csv_path)

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(
            reader.fieldnames,
            [
                "object_dir",
                "frame_idx",
                "valid_candidate_count",
                "post_nms_candidate_count",
                "top1_score",
                "top1_top2_gap",
                "top1_prob",
                "entropy",
            ],
        )
        self.assertEqual(reader.fieldnames, FRAME_CSV_FIELDNAMES)
        self.assertEqual(rows[0]["frame_idx"], "3")
        self.assertNotIn("ignored", rows[0])

    def test_dataset_summary_aggregates_all_frames_and_finite_score_metrics(self):
        from analyze_pose_score_confidence import (
            SUMMARY_CSV_FIELDNAMES,
            aggregate_dataset_summary,
            write_summary_csv,
        )

        records = [
            {
                "object_dir": "object_a",
                "frame_idx": 0,
                "valid_candidate_count": 3,
                "post_nms_candidate_count": 2,
                "top1_score": 10.0,
                "top1_top2_gap": 2.0,
                "top1_prob": 0.8,
                "entropy": 0.2,
            },
            {
                "object_dir": "object_a",
                "frame_idx": 1,
                "valid_candidate_count": 0,
                "post_nms_candidate_count": 0,
                "top1_score": np.nan,
                "top1_top2_gap": np.nan,
                "top1_prob": np.nan,
                "entropy": np.nan,
            },
            {
                "object_dir": "object_b",
                "frame_idx": 0,
                "valid_candidate_count": 1,
                "post_nms_candidate_count": 1,
                "top1_score": 5.0,
                "top1_top2_gap": np.nan,
                "top1_prob": 1.0,
                "entropy": 0.0,
            },
        ]

        summary = aggregate_dataset_summary(records)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["object_count"], 2)
        self.assertEqual(row["frame_count"], 3)
        self.assertEqual(row["valid_frame_count"], 2)
        self.assertAlmostEqual(row["valid_candidate_count_mean"], 4.0 / 3.0)
        self.assertAlmostEqual(row["post_nms_candidate_count_mean"], 1.0)
        self.assertAlmostEqual(row["top1_score_mean"], 7.5)
        self.assertAlmostEqual(row["top1_top2_gap_mean"], 2.0)
        self.assertAlmostEqual(row["top1_prob_p50"], 0.9)
        self.assertAlmostEqual(row["entropy_p90"], 0.18)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "debug" / "summary.csv"
            write_summary_csv(summary, csv_path)
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, SUMMARY_CSV_FIELDNAMES)
        self.assertEqual(rows[0]["object_count"], "2")
        self.assertNotIn("dataset_id", reader.fieldnames)
        self.assertNotIn("object_name", reader.fieldnames)


class AnalyzePoseScoreConfidenceStaticTests(unittest.TestCase):
    def test_registers_expected_cli_flags(self):
        tree = _parse("analyze_pose_score_confidence.py")
        parse_args = _find_function(tree, "parse_args")

        self.assertEqual(
            _parser_add_argument_flags(parse_args),
            {
                "--data-root",
                "--hist-bins",
                "--temperature",
                "--save-score-histograms",
                "--nms-threshold",
            },
        )

    def test_contains_expected_debug_output_paths_and_no_mesh_or_object_csv(self):
        tree = _parse("analyze_pose_score_confidence.py")
        constants = _constants_under(tree)

        self.assertIn("**/video", constants)
        self.assertIn("objects", constants)
        self.assertIn("gpt", constants)
        self.assertIn("object_*", constants)
        self.assertIn("all_poses&scores.pkl", constants)
        self.assertIn("debug", constants)
        self.assertIn("score_confidence_by_frame.csv", constants)
        self.assertIn("score_confidence_summary.csv", constants)
        self.assertIn("object_dir", constants)
        self.assertIn("frame_idx", constants)
        self.assertIn("valid_candidate_count", constants)
        self.assertIn("post_nms_candidate_count", constants)
        self.assertIn("entropy", constants)
        self.assertIn("score_confidence_curves.png", constants)
        self.assertIn("score_confidence_distributions.png", constants)
        self.assertIn("score_histograms", constants)
        self.assertIn("frame_{frame_idx:06d}.png", constants)
        self.assertNotIn("mesh.glb", constants)
        self.assertNotIn("score_confidence_by_object.csv", constants)
        self.assertNotIn("seq_dir", constants)
        self.assertNotIn("object_name", constants)
        self.assertNotIn("pre_nms_candidate_count", constants)
        self.assertNotIn("nms_score_histograms", constants)
        self.assertNotIn("confidence_entropy", constants)

    def test_parse_args_returns_new_defaults(self):
        from analyze_pose_score_confidence import parse_args

        args = parse_args([])

        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.hist_bins, 30)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.nms_threshold, 5.0)
        self.assertFalse(args.save_score_histograms)
        self.assertFalse(hasattr(args, "overwrite"))

    def test_parse_args_accepts_zero_nms_threshold_and_rejects_negative(self):
        from analyze_pose_score_confidence import parse_args

        self.assertEqual(parse_args(["--nms-threshold", "0"]).nms_threshold, 0.0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--nms-threshold", "-0.1"])

    def test_parse_args_rejects_nonpositive_temperature(self):
        from analyze_pose_score_confidence import parse_args

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--temperature", "0"])


if __name__ == "__main__":
    unittest.main()
