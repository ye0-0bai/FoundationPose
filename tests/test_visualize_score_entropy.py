import argparse
import ast
from pathlib import Path
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


class VisualizeScoreEntropyAlgorithmTests(unittest.TestCase):
    def test_score_entropy_uses_stable_softmax_probabilities(self):
        from visualize_score_entropy import score_entropy

        scores = np.array([1000.0, 1000.0])

        self.assertAlmostEqual(score_entropy(scores), np.log(2.0))

    def test_score_entropy_ignores_nonfinite_scores(self):
        from visualize_score_entropy import score_entropy

        scores = np.array([2.0, np.inf, 1.0, np.nan, -np.inf])
        finite_scores = np.array([2.0, 1.0])
        shifted = finite_scores - finite_scores.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        expected = -np.sum(probs * np.log(probs))

        self.assertAlmostEqual(score_entropy(scores), expected)

    def test_score_entropy_returns_nan_for_frame_without_finite_scores(self):
        from visualize_score_entropy import score_entropy

        self.assertTrue(np.isnan(score_entropy([np.nan, np.inf, -np.inf])))
        self.assertTrue(np.isnan(score_entropy([])))


class VisualizeScoreEntropyStaticTests(unittest.TestCase):
    def test_registers_only_expected_cli_flags(self):
        tree = _parse("visualize_score_entropy.py")
        parse_args = _find_function(tree, "parse_args")

        self.assertEqual(
            _parser_add_argument_flags(parse_args),
            {"--data-root", "--overwrite", "--hist-bins"},
        )

    def test_contains_expected_debug_output_paths_and_traversal(self):
        tree = _parse("visualize_score_entropy.py")
        constants = _constants_under(tree)

        self.assertIn("**/video", constants)
        self.assertIn("objects", constants)
        self.assertIn("gpt", constants)
        self.assertIn("object_*", constants)
        self.assertIn("all_poses&scores.pkl", constants)
        self.assertIn("debug", constants)
        self.assertIn("score_entropy.png", constants)
        self.assertIn("score_histograms", constants)
        self.assertIn("frame_{frame_idx:06d}.png", constants)

    def test_parse_args_returns_hist_bins_default(self):
        from visualize_score_entropy import parse_args

        args = parse_args([])

        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.hist_bins, 30)
        self.assertFalse(args.overwrite)


if __name__ == "__main__":
    unittest.main()
