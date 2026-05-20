import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "export_pose_candidate_artifacts_multigpu.py"


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


def _imports_at_module_level(tree):
    imported = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            imported.update(alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            imported.add(stmt.module or "")
    return imported


class PoseCandidateArtifactsMultiGpuStaticTests(unittest.TestCase):
    def test_script_exists_and_parser_exposes_requested_interface(self):
        self.assertTrue(SCRIPT.exists())

        tree = _parse_script()
        flags = _parser_flags(_find_function(tree, "parse_args"))

        self.assertTrue({"--data_root", "--gpus", "--overwrite"}.issubset(flags))
        self.assertFalse({"--num-workers", "--debug-root"} & flags)

    def test_script_uses_only_light_module_level_imports(self):
        tree = _parse_script()
        imports = _imports_at_module_level(tree)

        self.assertFalse({"torch", "trimesh", "estimater", "datareader"} & imports)
        self.assertTrue(
            {
                "argparse",
                "logging",
                "multiprocessing",
                "os",
                "queue",
                "sys",
                "traceback",
                "warnings",
            }.issubset(imports)
        )

    def test_worker_binds_gpu_before_heavy_imports(self):
        tree = _parse_script()
        worker = _find_function(tree, "worker_main")
        first_statements = worker.body[:5]
        first_text = ast.unparse(ast.Module(body=first_statements, type_ignores=[]))

        self.assertIn("os.environ['CUDA_VISIBLE_DEVICES']", first_text)
        self.assertLess(first_text.index("CUDA_VISIBLE_DEVICES"), first_text.index("import torch"))
        self.assertIn("import trimesh", first_text)
        self.assertIn("from estimater import", first_text)

    def test_main_uses_spawn_context_queues_and_processes(self):
        tree = _parse_script()
        main = _find_function(tree, "main")
        start_workers = _find_function(tree, "start_workers")
        calls = _calls_under(main) | _calls_under(start_workers)
        constants = _constants_under(main)

        self.assertIn("mp.get_context", calls)
        self.assertIn("ctx.Queue", calls)
        self.assertIn("ctx.Process", calls)
        self.assertIn("spawn", constants)

    def test_collect_results_has_single_sequence_progress_and_summary_counts(self):
        tree = _parse_script()
        collect_results = _find_function(tree, "collect_results")
        print_summary = _find_function(tree, "print_summary")
        collect_source = ast.unparse(collect_results)
        summary_constants = _constants_under(collect_results) | _constants_under(print_summary)

        self.assertIn("tqdm(total=total_sequences", collect_source)
        self.assertIn("if result['type'] == 'seq_done':", collect_source)
        self.assertIn("progress.update(1)", collect_source)
        self.assertTrue(
            {
                "objects_processed",
                "objects_skipped",
                "objects_failed",
                "valid_frames",
                "invalid_frames",
            }.issubset(summary_constants)
        )

    def test_artifact_contract_is_preserved(self):
        tree = _parse_script()
        constants = _constants_under(tree)
        process_object = _find_function(tree, "process_object")
        process_source = ast.unparse(process_object)
        calls = _calls_under(process_object)

        self.assertIn("all_pose_candidates_artifacts.npz", constants)
        self.assertIn("all_pose_candidates_artifacts.tmp.npz", constants)
        self.assertIn("valid", constants)
        self.assertIn("np.savez_compressed", calls)
        self.assertTrue(any(call.endswith(".replace") for call in calls))
        self.assertIn("return_pose_data=True", process_source)
        self.assertIn("register_result is None", process_source)
        self.assertIn("artifact_key(frame_idx, 'poses')", process_source)
        self.assertIn("artifact_key(frame_idx, 'scores')", process_source)
        self.assertIn("artifact_key(frame_idx, 'render_rgbs')", process_source)
        self.assertIn("artifact_key(frame_idx, 'render_masks')", process_source)
        self.assertIn("artifact_key(frame_idx, 'tf_to_crops')", process_source)


if __name__ == "__main__":
    unittest.main()
