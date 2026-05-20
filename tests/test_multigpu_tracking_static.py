import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "process_data_estimate+tracking_multigpu.py"


def _parse_script():
    return ast.parse(SCRIPT.read_text())


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


def _parser_flags(function_node):
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


def _for_loop_by_target(node, target_name):
    for child in ast.walk(node):
        if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
            if child.target.id == target_name:
                return child
    raise AssertionError(f"for loop over {target_name} not found")


def _imports_at_module_level(tree):
    imported = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            imported.update(alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            imported.add(stmt.module or "")
    return imported


class MultiGpuTrackingStaticTests(unittest.TestCase):
    def test_script_uses_only_light_module_level_imports(self):
        tree = _parse_script()
        imports = _imports_at_module_level(tree)

        self.assertFalse({"torch", "trimesh", "estimater", "datareader"} & imports)
        self.assertTrue({"argparse", "os", "queue", "traceback", "multiprocessing"}.issubset(imports))

    def test_parser_exposes_multigpu_interface_without_scflow2_flags(self):
        tree = _parse_script()
        flags = _parser_flags(_find_function(tree, "parse_args"))

        self.assertTrue(
            {"--data-root", "--gpus", "--num-workers", "--overwrite", "--debug-root"}.issubset(flags)
        )
        self.assertFalse({"--use_scflow2", "--scflow2-config", "--scflow2-checkpoint"} & flags)

    def test_worker_binds_gpu_before_heavy_imports(self):
        tree = _parse_script()
        worker = _find_function(tree, "worker_main")
        first_statements = worker.body[:5]
        first_text = ast.unparse(ast.Module(body=first_statements, type_ignores=[]))

        self.assertIn("os.environ['CUDA_VISIBLE_DEVICES']", first_text)
        self.assertLess(first_text.index("CUDA_VISIBLE_DEVICES"), first_text.index("import torch"))
        self.assertIn("from estimater import", first_text)
        self.assertIn("from datareader import", first_text)

    def test_process_seq_loads_sequence_inputs_once_and_reuses_estimator(self):
        tree = _parse_script()
        process_seq = _find_function(tree, "process_seq_dir")
        setup_estimator = _find_function(tree, "setup_estimator_for_mesh")
        object_loop = _for_loop_by_target(process_seq, "object_dir")
        before_object_loop = ast.Module(
            body=process_seq.body[: process_seq.body.index(object_loop)],
            type_ignores=[],
        )
        calls = _calls_under(setup_estimator)
        constants = _constants_under(process_seq) | _constants_under(_find_function(tree, "process_object"))

        self.assertGreaterEqual(ast.unparse(before_object_loop).count("np.load"), 2)
        self.assertIn("iio.imread", _calls_under(before_object_loop))
        self.assertIn("FoundationPose", calls)
        self.assertIn("est.reset_object", calls)
        self.assertIn("poses.npy", constants)
        self.assertIn("poses.mp4", constants)
        self.assertNotIn("poses_scflow2.npy", constants)

    def test_main_uses_spawn_context_and_global_sequence_progress(self):
        tree = _parse_script()
        main = _find_function(tree, "main")
        start_workers = _find_function(tree, "start_workers")
        collect_results = _find_function(tree, "collect_results")
        calls = _calls_under(main) | _calls_under(start_workers)
        constants = _constants_under(main) | _constants_under(_find_function(tree, "worker_main"))

        self.assertIn("mp.get_context", calls)
        self.assertIn("ctx.Queue", calls)
        self.assertIn("ctx.Process", calls)
        self.assertIn("tqdm", _calls_under(collect_results))
        self.assertIn("spawn", constants)
        self.assertIn("seq_done", constants)

    def test_no_timing_or_scflow2_support_in_multigpu_script(self):
        tree = _parse_script()
        source_constants = _constants_under(tree)
        calls = _calls_under(tree)

        self.assertFalse(any("scflow2" in value.lower() for value in source_constants))
        self.assertFalse(any("timing" in value.lower() for value in source_constants))
        self.assertNotIn("time.perf_counter", calls)


if __name__ == "__main__":
    unittest.main()
