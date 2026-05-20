import ast
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "optimize_precomputed_poses_parallel.py"


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


class FakeQueue:
    def __init__(self, items):
        self.items = list(items)

    def get(self, timeout=None):
        if not self.items:
            raise AssertionError("queue unexpectedly exhausted")
        return self.items.pop(0)


class OptimizePrecomputedPosesParallelStaticTests(unittest.TestCase):
    def test_script_exists_and_uses_expected_imports(self):
        self.assertTrue(SCRIPT.exists())

        tree = _parse_script()
        imports = _imports_at_module_level(tree)

        self.assertTrue({"multiprocessing", "queue", "traceback"}.issubset(imports))
        self.assertIn("tqdm", imports)

    def test_parser_exposes_parallel_and_serial_optimizer_flags(self):
        tree = _parse_script()
        flags = _parser_flags(_find_function(tree, "parse_args"))

        self.assertTrue(
            {
                "--data-root",
                "--overwrite",
                "--run-id",
                "--debug",
                "--max-invalid-gap",
                "--smooth-window",
                "--smooth-polyorder",
                "--trans-lambda",
                "--rot-lambda",
                "--num_workers",
                "--num-workers",
            }.issubset(flags)
        )

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

    def test_collect_results_owns_sequence_progress_bar(self):
        tree = _parse_script()
        collect_results = _find_function(tree, "collect_results")
        collect_source = ast.unparse(collect_results)

        self.assertIn("tqdm(total=total_sequences", collect_source)
        self.assertIn("if result['type'] == 'seq_done':", collect_source)
        self.assertIn("progress.update(1)", collect_source)

    def test_expected_parallel_helpers_exist(self):
        tree = _parse_script()

        for name in [
            "discover_seq_dirs",
            "relative_path",
            "args_to_worker_namespace",
            "process_seq_dir",
            "worker_main",
            "enqueue_tasks",
            "start_workers",
            "collect_results",
            "print_summary",
            "main",
        ]:
            _find_function(tree, name)

    def test_reuses_serial_process_object_without_copying_algorithms(self):
        source = SCRIPT.read_text()

        self.assertIn("process_object", source)
        self.assertNotIn("def select_pose_trajectory", source)
        self.assertNotIn("def smooth_pose_trajectory", source)
        self.assertNotIn("def compute_mask_ious", source)

    def test_process_seq_dir_aggregates_object_statuses(self):
        import optimize_precomputed_poses_parallel as parallel

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_dir = Path(tmpdir) / "seq_a"
            object_root = seq_dir / "objects" / "gpt"
            (seq_dir / "video").mkdir(parents=True)
            (object_root / "object_001").mkdir(parents=True)
            (object_root / "object_002").mkdir(parents=True)
            args = Namespace(run_id="test", overwrite=False)

            def fake_process_object(seq_path, object_path, parsed_args):
                self.assertEqual(seq_path, seq_dir)
                self.assertIs(parsed_args, args)
                if object_path.name == "object_001":
                    return "processed"
                return "skipped_existing"

            with mock.patch.object(parallel, "process_object", side_effect=fake_process_object):
                stats = parallel.process_seq_dir(seq_dir, args)

            self.assertEqual(
                stats,
                {
                    "processed": 1,
                    "skipped_existing": 1,
                    "skipped_missing_artifacts": 0,
                    "skipped_missing_masks": 0,
                    "failed": 0,
                },
            )

    def test_collect_results_aggregates_stats_and_errors(self):
        import optimize_precomputed_poses_parallel as parallel

        result_queue = FakeQueue(
            [
                {
                    "type": "seq_done",
                    "worker_id": 0,
                    "path": "seq_a",
                    "processed": 1,
                    "skipped_existing": 0,
                    "skipped_missing_artifacts": 0,
                    "skipped_missing_masks": 0,
                    "failed": 0,
                },
                {
                    "type": "error",
                    "worker_id": 1,
                    "path": "seq_b",
                    "message": "boom",
                    "traceback": "traceback text",
                },
                {
                    "type": "seq_done",
                    "worker_id": 1,
                    "path": "seq_b",
                    "processed": 0,
                    "skipped_existing": 1,
                    "skipped_missing_artifacts": 0,
                    "skipped_missing_masks": 0,
                    "failed": 1,
                },
            ]
        )

        with mock.patch.object(parallel.tqdm, "tqdm") as tqdm_mock:
            progress = tqdm_mock.return_value.__enter__.return_value
            stats, errors = parallel.collect_results(result_queue, total_sequences=2)

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["skipped_existing"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["message"], "boom")
        self.assertEqual(progress.update.call_count, 2)


if __name__ == "__main__":
    unittest.main()
