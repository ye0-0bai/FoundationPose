import ast
import re
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


def _import_from_module(tree, module_name):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module_name
    ]


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


def _imported_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


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
                "--run-id",
                "--debug",
            }.issubset(_parser_add_argument_flags(parse_args))
        )

        self.assertTrue(
            all(
                "dp" not in flag.lower()
                for flag in _parser_add_argument_flags(parse_args)
            )
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

    def test_inlines_trajectory_optimization_helpers(self):
        tree = _parse("optimize_precomputed_poses.py")
        process_data_imports = _import_from_module(tree, "process_data")

        imported_names = {
            alias.name
            for node in process_data_imports
            for alias in node.names
        }

        self.assertIn("configure_quiet_logging", imported_names)
        self.assertNotIn("select_pose_trajectory", imported_names)
        self.assertNotIn("smooth_pose_trajectory", imported_names)
        _find_function(tree, "select_pose_trajectory")
        _find_function(tree, "smooth_pose_trajectory")
        _find_function(tree, "prepare_iou_weighted_candidates")
        _find_function(tree, "compute_mask_ious")
        _find_function(tree, "transition_cost_matrix")

    def test_keeps_single_file_config_and_object_path_helpers(self):
        tree = _parse("optimize_precomputed_poses.py")
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }

        self.assertIn("OptimizationConfig", class_names)
        self.assertIn("ObjectPaths", class_names)
        _find_function(tree, "optimization_config_from_args")

    def test_traverses_gpt_objects_and_writes_expected_outputs(self):
        tree = _parse("optimize_precomputed_poses.py")
        constants = _constants_under(tree)

        self.assertIn("**/video", constants)
        self.assertIn("objects", constants)
        self.assertIn("gpt", constants)
        self.assertIn("object_*", constants)
        self.assertIn("all_pose_candidates_artifacts.npz", constants)
        self.assertIn("poses_optimized_{run_id}.npy", constants)
        self.assertIn("poses_optimized_{run_id}.mp4", constants)
        self.assertIn("poses_optimized_{run_id}_debug.npz", constants)
        self.assertIn("exp", constants)
        self.assertIn("pose_optimization_runs", constants)
        self.assertNotIn("poses_optimized_iou.npy", constants)
        self.assertNotIn("poses_optimized_iou.mp4", constants)
        self.assertNotIn("all_poses&scores.pkl", constants)
        self.assertNotIn("poses_optimized.npy", constants)
        self.assertNotIn("poses_optimized.mp4", constants)

    def test_run_id_default_uses_single_timestamp_format(self):
        source = (ROOT / "optimize_precomputed_poses.py").read_text()

        self.assertIn("%Y%m%d-%H%M%S", source)
        self.assertRegex(source, r"run_id\s*=\s*args\.run_id\s+or\s+generate_run_id\(\)")

    def test_static_experiment_record_helpers_exist(self):
        tree = _parse("optimize_precomputed_poses.py")

        _find_function(tree, "experiment_record_path")
        _find_function(tree, "write_experiment_record")
        _find_function(tree, "generate_run_id")

    def test_main_writes_running_record_before_processing_loop(self):
        tree = _parse("optimize_precomputed_poses.py")
        main = _find_function(tree, "main")
        seq_loop = _for_loop_by_target(main, "seq_dir")
        running_record_calls = []

        for node in ast.walk(main):
            if not (
                isinstance(node, ast.Call)
                and _call_name(node.func) == "write_experiment_record"
            ):
                continue
            if any(
                keyword.arg == "run_status"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "running"
                for keyword in node.keywords
            ):
                running_record_calls.append(node)

        self.assertTrue(running_record_calls)
        self.assertLess(running_record_calls[0].lineno, seq_loop.lineno)

    def test_main_writes_running_and_completed_record_statuses(self):
        source = (ROOT / "optimize_precomputed_poses.py").read_text()

        self.assertIn('run_status="running"', source)
        self.assertIn('run_status="completed"', source)

    def test_gitignore_allows_exp_tracking(self):
        gitignore = (ROOT / ".gitignore").read_text()

        self.assertIsNone(re.search(r"(?m)^exp/$", gitignore))

    def test_uses_kornia_warp_perspective_for_mask_crop_iou(self):
        tree = _parse("optimize_precomputed_poses.py")
        calls = _calls_under(tree)

        self.assertIn("kornia.geometry.transform.warp_perspective", calls)

    def test_legacy_pickle_loader_and_comments_are_removed(self):
        source = (ROOT / "optimize_precomputed_poses.py").read_text()
        tree = ast.parse(source)

        self.assertNotIn("load_pose_candidates", {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)})
        self.assertNotIn("CANDIDATES_FILENAME", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
        self.assertNotIn("pickle", _imported_names(tree))
        self.assertNotIn("log-probability", source)
        self.assertNotIn("log-softmax", source)
        self.assertNotIn("all_poses&scores.pkl", source)

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
