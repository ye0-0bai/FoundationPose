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


class AnalyzePoseScoreConfidenceAlgorithmTests(unittest.TestCase):
    def test_stable_softmax_is_stable_for_large_scores_and_temperature(self):
        from analyze_pose_score_confidence import stable_softmax

        probabilities = stable_softmax(np.array([1000.0, 1000.0]), temperature=2.0)

        np.testing.assert_allclose(probabilities, np.array([0.5, 0.5]))

    def test_compute_frame_score_metrics_for_multiple_candidates(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        scores = np.array([4.0, 2.0, 1.0, -1.0])
        metrics = compute_frame_score_metrics(scores, temperature=1.0)
        sorted_scores = np.array([4.0, 2.0, 1.0, -1.0])
        probs = np.exp(sorted_scores - sorted_scores.max())
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs))

        self.assertEqual(metrics["candidate_count"], 4)
        self.assertEqual(metrics["finite_score_count"], 4)
        self.assertEqual(metrics["nonfinite_score_count"], 0)
        self.assertEqual(metrics["top1_score"], 4.0)
        self.assertEqual(metrics["top2_score"], 2.0)
        self.assertAlmostEqual(metrics["score_mean"], np.mean(sorted_scores))
        self.assertAlmostEqual(metrics["score_std"], np.std(sorted_scores))
        self.assertEqual(metrics["score_min"], -1.0)
        self.assertEqual(metrics["score_max"], 4.0)
        self.assertEqual(metrics["score_range"], 5.0)
        self.assertEqual(metrics["top1_top2_gap"], 2.0)
        self.assertAlmostEqual(metrics["top1_mean_gap"], 4.0 - np.mean(sorted_scores))
        self.assertAlmostEqual(metrics["top1_median_gap"], 4.0 - np.median(sorted_scores))
        self.assertAlmostEqual(
            metrics["top1_zscore"],
            (4.0 - np.mean(sorted_scores)) / np.std(sorted_scores),
        )
        self.assertAlmostEqual(metrics["top1_prob"], probs[0])
        self.assertAlmostEqual(metrics["top2_prob"], probs[1])
        self.assertAlmostEqual(metrics["prob_gap"], probs[0] - probs[1])
        self.assertAlmostEqual(metrics["top5_prob_mass"], 1.0)
        self.assertAlmostEqual(metrics["entropy_nats"], entropy)
        self.assertAlmostEqual(metrics["entropy_norm"], entropy / math.log(4.0))
        self.assertAlmostEqual(metrics["confidence_entropy"], 1.0 - entropy / math.log(4.0))
        self.assertAlmostEqual(metrics["effective_candidate_count"], math.exp(entropy))

    def test_entropy_handles_softmax_underflow_as_zero_probability(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        metrics = compute_frame_score_metrics([1000.0, -1000.0])

        self.assertEqual(metrics["top1_prob"], 1.0)
        self.assertEqual(metrics["top2_prob"], 0.0)
        self.assertEqual(metrics["entropy_nats"], 0.0)
        self.assertEqual(metrics["entropy_norm"], 0.0)
        self.assertEqual(metrics["confidence_entropy"], 1.0)
        self.assertEqual(metrics["effective_candidate_count"], 1.0)

    def test_compute_frame_score_metrics_ignores_nonfinite_scores(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        metrics = compute_frame_score_metrics([2.0, np.inf, 1.0, np.nan, -np.inf])

        self.assertEqual(metrics["candidate_count"], 5)
        self.assertEqual(metrics["finite_score_count"], 2)
        self.assertEqual(metrics["nonfinite_score_count"], 3)
        self.assertEqual(metrics["top1_score"], 2.0)
        self.assertEqual(metrics["top2_score"], 1.0)
        self.assertEqual(metrics["top1_top2_gap"], 1.0)

    def test_compute_frame_score_metrics_for_single_candidate(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        metrics = compute_frame_score_metrics([7.0])

        self.assertEqual(metrics["candidate_count"], 1)
        self.assertEqual(metrics["finite_score_count"], 1)
        self.assertEqual(metrics["top1_score"], 7.0)
        self.assertTrue(np.isnan(metrics["top2_score"]))
        self.assertTrue(np.isnan(metrics["top1_top2_gap"]))
        self.assertEqual(metrics["top1_prob"], 1.0)
        self.assertTrue(np.isnan(metrics["top2_prob"]))
        self.assertTrue(np.isnan(metrics["prob_gap"]))
        self.assertEqual(metrics["top5_prob_mass"], 1.0)
        self.assertEqual(metrics["entropy_nats"], 0.0)
        self.assertEqual(metrics["entropy_norm"], 0.0)
        self.assertEqual(metrics["confidence_entropy"], 1.0)
        self.assertEqual(metrics["effective_candidate_count"], 1.0)

    def test_compute_frame_score_metrics_for_empty_or_nonfinite_candidates(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        for scores in ([], [np.nan, np.inf, -np.inf]):
            metrics = compute_frame_score_metrics(scores)

            self.assertEqual(metrics["finite_score_count"], 0)
            self.assertTrue(np.isnan(metrics["top1_score"]))
            self.assertTrue(np.isnan(metrics["top1_prob"]))
            self.assertTrue(np.isnan(metrics["entropy_nats"]))
            self.assertTrue(np.isnan(metrics["effective_candidate_count"]))

    def test_compute_frame_score_metrics_rejects_nonpositive_temperature(self):
        from analyze_pose_score_confidence import compute_frame_score_metrics

        with self.assertRaises(ValueError):
            compute_frame_score_metrics([1.0, 2.0], temperature=0.0)


class AnalyzePoseScoreConfidenceCsvTests(unittest.TestCase):
    def test_process_object_returns_frame_records_and_generates_object_debug_plots(self):
        from analyze_pose_score_confidence import process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_dir = Path(tmpdir) / "seq_a"
            object_dir = seq_dir / "objects" / "gpt" / "object_0001"
            object_dir.mkdir(parents=True)
            with open(object_dir / "all_poses&scores.pkl", "wb") as f:
                pickle.dump({"scores": [[1.0, 1.0], [2.0, np.nan]]}, f)

            status, records = process_object(
                object_dir,
                SimpleNamespace(
                    overwrite=False,
                    hist_bins=5,
                    temperature=1.0,
                    save_score_histograms=False,
                ),
            )
            curves_path = object_dir / "debug" / "score_confidence_curves.png"
            distributions_path = (
                object_dir / "debug" / "score_confidence_distributions.png"
            )
            curves_exists = curves_path.exists()
            distributions_exists = distributions_path.exists()
            histogram_dir_exists = (
                object_dir / "debug" / "score_histograms"
            ).exists()

        self.assertEqual(status, "processed")
        self.assertEqual(len(records), 2)
        self.assertTrue(curves_exists)
        self.assertTrue(distributions_exists)
        self.assertEqual(records[0]["seq_dir"], str(seq_dir))
        self.assertEqual(records[0]["object_dir"], str(object_dir))
        self.assertEqual(records[0]["object_name"], "object_0001")
        self.assertEqual(records[0]["frame_idx"], 0)
        self.assertEqual(records[0]["candidate_count"], 2)
        self.assertEqual(records[0]["finite_score_count"], 2)
        self.assertAlmostEqual(records[0]["entropy_nats"], math.log(2.0))
        self.assertAlmostEqual(records[0]["entropy_norm"], 1.0)
        self.assertAlmostEqual(records[0]["confidence_entropy"], 0.0)
        self.assertEqual(records[1]["frame_idx"], 1)
        self.assertAlmostEqual(records[1]["entropy_nats"], 0.0)
        self.assertFalse(histogram_dir_exists)

    def test_process_object_keeps_records_when_visualizations_already_exist(self):
        from analyze_pose_score_confidence import output_paths, process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = (
                Path(tmpdir)
                / "seq_b"
                / "objects"
                / "gpt"
                / "object_0002"
            )
            object_dir.mkdir(parents=True)
            with open(object_dir / "all_poses&scores.pkl", "wb") as f:
                pickle.dump({"scores": [[0.0, 0.0]]}, f)
            paths = output_paths(object_dir, 1, save_score_histograms=False)
            paths["curves_path"].parent.mkdir(parents=True)
            paths["curves_path"].write_text("existing")
            paths["distributions_path"].write_text("existing")

            status, records = process_object(
                object_dir,
                SimpleNamespace(
                    overwrite=False,
                    hist_bins=5,
                    temperature=1.0,
                    save_score_histograms=False,
                ),
            )

        self.assertEqual(status, "skipped_existing")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame_idx"], 0)
        self.assertAlmostEqual(records[0]["entropy_nats"], math.log(2.0))

    def test_save_score_histograms_is_optional(self):
        from analyze_pose_score_confidence import process_object

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = (
                Path(tmpdir)
                / "seq_c"
                / "objects"
                / "gpt"
                / "object_0003"
            )
            object_dir.mkdir(parents=True)
            with open(object_dir / "all_poses&scores.pkl", "wb") as f:
                pickle.dump({"scores": [[0.0, 1.0]]}, f)

            process_object(
                object_dir,
                SimpleNamespace(
                    overwrite=True,
                    hist_bins=5,
                    temperature=1.0,
                    save_score_histograms=True,
                ),
            )

            histogram_path = (
                object_dir / "debug" / "score_histograms" / "frame_000000.png"
            )
            histogram_exists = histogram_path.exists()

        self.assertTrue(histogram_exists)

    def test_write_frame_csv_uses_expected_fields_and_creates_debug_dir(self):
        from analyze_pose_score_confidence import FRAME_CSV_FIELDNAMES, write_frame_csv

        record = {field: "" for field in FRAME_CSV_FIELDNAMES}
        record.update(
            {
                "seq_dir": "seq",
                "object_dir": "seq/objects/gpt/object_0001",
                "object_name": "object_0001",
                "frame_idx": 3,
                "entropy_nats": 1.25,
                "confidence_entropy": 0.5,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "debug" / "score_confidence_by_frame.csv"

            write_frame_csv([record], csv_path)

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, FRAME_CSV_FIELDNAMES)
        self.assertEqual(rows[0]["frame_idx"], "3")
        self.assertEqual(rows[0]["entropy_nats"], "1.25")
        self.assertEqual(rows[0]["confidence_entropy"], "0.5")

    def test_object_aggregation_outputs_mean_std_and_quantiles(self):
        from analyze_pose_score_confidence import (
            OBJECT_AGGREGATE_FIELDNAMES,
            aggregate_object_records,
            write_object_csv,
        )

        records = [
            {
                "seq_dir": "seq",
                "object_dir": "object",
                "object_name": "object_0001",
                "top1_top2_gap": 1.0,
                "entropy_norm": 0.25,
                "confidence_entropy": 0.75,
                "top1_prob": 0.7,
                "effective_candidate_count": 1.5,
                "finite_score_count": 2,
            },
            {
                "seq_dir": "seq",
                "object_dir": "object",
                "object_name": "object_0001",
                "top1_top2_gap": 3.0,
                "entropy_norm": 0.75,
                "confidence_entropy": 0.25,
                "top1_prob": 0.9,
                "effective_candidate_count": 3.5,
                "finite_score_count": 4,
            },
        ]

        aggregated = aggregate_object_records(records)

        self.assertEqual(len(aggregated), 1)
        row = aggregated[0]
        self.assertEqual(row["seq_dir"], "seq")
        self.assertEqual(row["object_dir"], "object")
        self.assertEqual(row["object_name"], "object_0001")
        self.assertAlmostEqual(row["top1_top2_gap_mean"], 2.0)
        self.assertAlmostEqual(row["top1_top2_gap_std"], 1.0)
        self.assertAlmostEqual(row["top1_top2_gap_p10"], 1.2)
        self.assertAlmostEqual(row["top1_top2_gap_p50"], 2.0)
        self.assertAlmostEqual(row["top1_top2_gap_p90"], 2.8)
        self.assertAlmostEqual(row["finite_score_count_mean"], 3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "debug" / "score_confidence_by_object.csv"

            write_object_csv(aggregated, csv_path)

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, OBJECT_AGGREGATE_FIELDNAMES)
        self.assertEqual(rows[0]["object_name"], "object_0001")
        self.assertEqual(rows[0]["top1_top2_gap_mean"], "2.0")


class AnalyzePoseScoreConfidenceStaticTests(unittest.TestCase):
    def test_registers_expected_cli_flags(self):
        tree = _parse("analyze_pose_score_confidence.py")
        parse_args = _find_function(tree, "parse_args")

        self.assertEqual(
            _parser_add_argument_flags(parse_args),
            {
                "--data-root",
                "--overwrite",
                "--hist-bins",
                "--temperature",
                "--save-score-histograms",
            },
        )

    def test_contains_expected_debug_output_paths_and_traversal(self):
        tree = _parse("analyze_pose_score_confidence.py")
        constants = _constants_under(tree)

        self.assertIn("**/video", constants)
        self.assertIn("objects", constants)
        self.assertIn("gpt", constants)
        self.assertIn("object_*", constants)
        self.assertIn("all_poses&scores.pkl", constants)
        self.assertIn("debug", constants)
        self.assertIn("score_confidence_by_frame.csv", constants)
        self.assertIn("score_confidence_by_object.csv", constants)
        self.assertIn("seq_dir", constants)
        self.assertIn("object_dir", constants)
        self.assertIn("object_name", constants)
        self.assertIn("frame_idx", constants)
        self.assertIn("entropy_nats", constants)
        self.assertIn("score_confidence_curves.png", constants)
        self.assertIn("score_confidence_distributions.png", constants)
        self.assertIn("score_histograms", constants)
        self.assertIn("frame_{frame_idx:06d}.png", constants)

    def test_parse_args_returns_new_defaults(self):
        from analyze_pose_score_confidence import parse_args

        args = parse_args([])

        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.hist_bins, 30)
        self.assertEqual(args.temperature, 1.0)
        self.assertFalse(args.overwrite)
        self.assertFalse(args.save_score_histograms)

    def test_parse_args_rejects_nonpositive_temperature(self):
        from analyze_pose_score_confidence import parse_args

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--temperature", "0"])


if __name__ == "__main__":
    unittest.main()
