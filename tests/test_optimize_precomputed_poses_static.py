import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _parse(path):
    return ast.parse((ROOT / path).read_text())


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name} not found")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _calls_under(node):
    return {
        _call_name(child.func)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    }


def _constants_under(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _for_loop_by_target(node, target_name):
    for child in ast.walk(node):
        if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
            if child.target.id == target_name:
                return child
    raise AssertionError(f"for loop over {target_name} not found")


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


def _is_overwrite_guarded_save_exists_test(node):
    return (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and len(node.values) == 2
        and isinstance(node.values[0], ast.UnaryOp)
        and isinstance(node.values[0].op, ast.Not)
        and isinstance(node.values[0].operand, ast.Attribute)
        and isinstance(node.values[0].operand.value, ast.Name)
        and node.values[0].operand.value.id == "args"
        and node.values[0].operand.attr == "overwrite"
        and isinstance(node.values[1], ast.Call)
        and isinstance(node.values[1].func, ast.Attribute)
        and node.values[1].func.attr == "exists"
        and isinstance(node.values[1].func.value, ast.Name)
        and node.values[1].func.value.id == "save_path"
    )


def _if_by_test(node, predicate):
    for child in ast.walk(node):
        if isinstance(child, ast.If) and predicate(child.test):
            return child
    raise AssertionError("matching if statement not found")


class OptimizePrecomputedPosesStaticTests(unittest.TestCase):
    def test_registers_expected_cli_flags(self):
        tree = _parse("optimize_precomputed_poses.py")
        parse_args = _find_function(tree, "parse_args")

        self.assertTrue(
            {
                "--data-root",
                "--overwrite",
                "--max-invalid-gap",
                "--smooth-window",
                "--smooth-polyorder",
                "--trans-lambda",
                "--rot-lambda",
            }.issubset(_parser_add_argument_flags(parse_args))
        )

    def test_uses_precomputed_candidates_without_foundationpose(self):
        tree = _parse("optimize_precomputed_poses.py")
        calls = _calls_under(tree)

        self.assertIn("select_pose_trajectory", calls)
        self.assertIn("smooth_pose_trajectory", calls)
        self.assertIn("compute_mesh_diameter", calls)
        self.assertNotIn("FoundationPose", calls)
        self.assertNotIn("ScorePredictor", calls)
        self.assertNotIn("PoseRefinePredictor", calls)
        self.assertNotIn("dr.RasterizeCudaContext", calls)
        self.assertNotIn("est.register_all", calls)

    def test_traverses_gpt_objects_and_writes_expected_outputs(self):
        tree = _parse("optimize_precomputed_poses.py")
        constants = _constants_under(tree)

        self.assertIn("**/video", constants)
        self.assertIn("objects", constants)
        self.assertIn("gpt", constants)
        self.assertIn("object_*", constants)
        self.assertIn("all_poses&scores.pkl", constants)
        self.assertIn("poses_optimized.npy", constants)
        self.assertIn("poses_optimized.mp4", constants)

    def test_default_skip_guard_respects_overwrite(self):
        tree = _parse("optimize_precomputed_poses.py")
        process_object = _find_function(tree, "process_object")
        skip_if = _if_by_test(process_object, _is_overwrite_guarded_save_exists_test)

        self.assertTrue(
            any(
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == "skipped_existing"
                for stmt in skip_if.body
            ),
            "overwrite-aware skip guard should return before processing existing outputs",
        )


if __name__ == "__main__":
    unittest.main()
