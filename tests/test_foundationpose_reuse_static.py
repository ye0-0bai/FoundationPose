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


def _assigned_self_attrs(node):
    attrs = set()
    for child in ast.walk(node):
        targets = []
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        elif isinstance(child, ast.AugAssign):
            targets = [child.target]

        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attrs.add(target.attr)
    return attrs


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


def _for_loop_by_target(node, target_name):
    for child in ast.walk(node):
        if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
            if child.target.id == target_name:
                return child
    raise AssertionError(f"for loop over {target_name} not found")


def _is_est_none_test(node):
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "est"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Is)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _if_by_test(node, predicate):
    for child in ast.walk(node):
        if isinstance(child, ast.If) and predicate(child.test):
            return child
    raise AssertionError("matching if statement not found")


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


def _assigns_name_to_none(node, name):
    for child in node.body:
        if not isinstance(child, ast.Assign):
            continue
        if not (
            len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
            and child.targets[0].id == name
        ):
            continue
        if isinstance(child.value, ast.Constant) and child.value.value is None:
            return True
    return False


def _none_assignment_count(node, name):
    count = 0
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not (isinstance(child.value, ast.Constant) and child.value.value is None):
            continue
        for target in child.targets:
            if isinstance(target, ast.Name) and target.id == name:
                count += 1
    return count


def _is_scflow2_lazy_init_test(node):
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
        return False

    has_use_scflow2 = any(
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "args"
        and value.attr == "use_scflow2"
        for value in node.values
    )
    has_refiner_none = any(
        isinstance(value, ast.Compare)
        and isinstance(value.left, ast.Name)
        and value.left.id == "scflow2_refiner"
        and len(value.ops) == 1
        and isinstance(value.ops[0], ast.Is)
        and len(value.comparators) == 1
        and isinstance(value.comparators[0], ast.Constant)
        and value.comparators[0].value is None
        for value in node.values
    )
    return has_use_scflow2 and has_refiner_none


def _contains_scflow2_constructor(node):
    if isinstance(node, list):
        node = ast.Module(body=node, type_ignores=[])
    return any(
        isinstance(child, ast.Call)
        and _call_name(child.func) in {"SCFlow2OnlineRefiner", "scflow2_refiner_cls"}
        for child in ast.walk(node)
    )


def _raises_system_exit(node):
    return any(
        isinstance(child, ast.Raise)
        and isinstance(child.exc, ast.Call)
        and _call_name(child.exc.func) == "SystemExit"
        for child in ast.walk(node)
    )


def _has_broad_exception_handler(node):
    for child in ast.walk(node):
        if not isinstance(child, ast.ExceptHandler):
            continue
        if child.type is None:
            return True
        if _call_name(child.type) in {"Exception", "BaseException"}:
            return True
    return False


def _try_nodes_wrapping_call(node, call_name):
    matches = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Try):
            continue
        calls = _calls_under(ast.Module(body=child.body, type_ignores=[]))
        if call_name in calls:
            matches.append(child)
    return matches


class FoundationPoseReuseStaticTests(unittest.TestCase):
    def test_reset_object_clears_per_object_tracking_state(self):
        tree = _parse("estimater.py")
        reset_object = _find_function(tree, "reset_object")

        self.assertTrue(
            {
                "pose_last",
                "poses",
                "scores",
                "best_id",
                "ob_id",
                "ob_mask",
                "K",
                "H",
                "W",
            }.issubset(_assigned_self_attrs(reset_object))
        )

    def test_processed_tracking_uses_lazy_first_object_foundationpose(self):
        tree = _parse("process_data_estimate+tracking.py")
        main = _find_function(tree, "main")
        self.assertTrue(_assigns_name_to_none(main, "est"))

        object_loop = _for_loop_by_target(main, "object_dir")
        lazy_init = _if_by_test(object_loop, _is_est_none_test)

        init_calls = _calls_under(ast.Module(body=lazy_init.body, type_ignores=[]))
        reuse_calls = _calls_under(ast.Module(body=lazy_init.orelse, type_ignores=[]))

        self.assertIn("ScorePredictor", init_calls)
        self.assertIn("PoseRefinePredictor", init_calls)
        self.assertIn("dr.RasterizeCudaContext", init_calls)
        self.assertIn("FoundationPose", init_calls)
        self.assertIn("est.reset_object", reuse_calls)

    def test_processed_tracking_does_not_use_temporary_box_mesh(self):
        tree = _parse("process_data_estimate+tracking.py")
        calls = _calls_under(tree)

        self.assertNotIn("trimesh.primitives.Box", calls)

    def test_processed_tracking_registers_overwrite_flag(self):
        tree = _parse("process_data_estimate+tracking.py")
        parse_args = _find_function(tree, "parse_args")

        self.assertIn("--overwrite", _parser_add_argument_flags(parse_args))

    def test_processed_tracking_skip_guard_respects_overwrite(self):
        tree = _parse("process_data_estimate+tracking.py")
        main = _find_function(tree, "main")
        object_loop = _for_loop_by_target(main, "object_dir")
        skip_if = _if_by_test(object_loop, _is_overwrite_guarded_save_exists_test)

        self.assertTrue(
            any(isinstance(stmt, ast.Continue) for stmt in skip_if.body),
            "overwrite-aware skip guard should continue when output exists",
        )

    def test_processed_tracking_uses_lazy_first_object_scflow2(self):
        tree = _parse("process_data_estimate+tracking.py")
        main = _find_function(tree, "main")
        self.assertTrue(_assigns_name_to_none(main, "scflow2_refiner"))
        self.assertEqual(_none_assignment_count(main, "scflow2_refiner"), 1)

        object_loop = _for_loop_by_target(main, "object_dir")
        lazy_init = _if_by_test(object_loop, _is_scflow2_lazy_init_test)

        self.assertTrue(_contains_scflow2_constructor(lazy_init.body))

        for child in ast.walk(object_loop):
            if isinstance(child, ast.Try) and _contains_scflow2_constructor(child):
                self.assertTrue(
                    any(_raises_system_exit(handler) for handler in child.handlers),
                    "SCFlow2 initialization failures should terminate instead of falling back",
                )

    def test_scflow2_refine_raises_failures_for_contextual_callers(self):
        tree = _parse("scflow2_online_refiner.py")
        refine = _find_function(tree, "refine")

        self.assertFalse(
            _has_broad_exception_handler(refine),
            "SCFlow2OnlineRefiner.refine should let callers add sequence/object/frame context",
        )

    def test_processed_tracking_logs_scflow2_frame_failures_and_continues(self):
        tree = _parse("process_data_estimate+tracking.py")
        main = _find_function(tree, "main")
        frame_loop = _for_loop_by_target(main, "frame_idx")
        refine_try_nodes = _try_nodes_wrapping_call(frame_loop, "scflow2_refiner.refine")

        self.assertEqual(
            len(refine_try_nodes),
            1,
            "tracking should catch SCFlow2 refine failures at the frame level",
        )

        handler_calls = set()
        handler_raises = []
        for handler in refine_try_nodes[0].handlers:
            handler_calls.update(_calls_under(handler))
            handler_raises.extend(
                child for child in ast.walk(handler) if isinstance(child, ast.Raise)
            )

        self.assertIn("traceback.print_exc", handler_calls)
        self.assertIn("tqdm.tqdm.write", handler_calls)
        self.assertEqual(
            handler_raises,
            [],
            "frame-level SCFlow2 failures should keep the FoundationPose pose and continue",
        )


if __name__ == "__main__":
    unittest.main()
