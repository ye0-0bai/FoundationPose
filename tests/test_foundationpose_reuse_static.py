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


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found")


def _class_level_none_assignments(class_node):
    names = set()
    for stmt in class_node.body:
        targets = []
        value = None
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
            value = stmt.value

        if not (isinstance(value, ast.Constant) and value.value is None):
            continue

        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


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


def _constants_under(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _name_loads_under(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
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


def _subscript_constant_key(node):
    if not isinstance(node, ast.Subscript):
        return None
    key = node.slice
    if isinstance(key, ast.Constant):
        return key.value
    return None


def _assigns_subscript_key(node, target_name, container_name, key):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in child.targets):
            continue
        if (
            isinstance(child.value, ast.Subscript)
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id == container_name
            and _subscript_constant_key(child.value) == key
        ):
            return True
    return False


def _appends_subscript_key(node, list_name, container_name, key):
    for child in ast.walk(node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "append"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == list_name
            and len(child.args) == 1
        ):
            continue
        arg = child.args[0]
        if (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.value, ast.Name)
            and arg.value.id == container_name
            and _subscript_constant_key(arg) == key
        ):
            return True
    return False


def _name_initialized_to_empty_list(node, name):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in child.targets):
            continue
        if isinstance(child.value, ast.List) and child.value.elts == []:
            return True
    return False


def _call_has_keyword(call_node, keyword_name, value_name=None, none_value=False):
    for keyword in call_node.keywords:
        if keyword.arg != keyword_name:
            continue
        if value_name is not None:
            return isinstance(keyword.value, ast.Name) and keyword.value.id == value_name
        if none_value:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is None
        return True
    return False


def _batch_pose_data_calls(node):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child.func) == "BatchPoseData"
    ]


def _assignment_values(node, target_name):
    values = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == target_name for target in child.targets):
            values.append(child.value)
    return values


def _is_torch_cat_permute_to_nchw(value, source_name):
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "permute"
    ):
        return False
    if [getattr(arg, "value", None) for arg in value.args] != [0, 3, 1, 2]:
        return False
    cat_call = value.func.value
    return (
        isinstance(cat_call, ast.Call)
        and _call_name(cat_call.func) == "torch.cat"
        and len(cat_call.args) >= 1
        and isinstance(cat_call.args[0], ast.Name)
        and cat_call.args[0].id == source_name
    )


def _is_flip_dims_one(value):
    if not (isinstance(value, ast.Call) and _call_name(value.func) == "torch.flip"):
        return False
    for keyword in value.keywords:
        if keyword.arg != "dims":
            continue
        return (
            isinstance(keyword.value, ast.List)
            and len(keyword.value.elts) == 1
            and isinstance(keyword.value.elts[0], ast.Constant)
            and keyword.value.elts[0].value == 1
        )
    return False


def _is_rast_out_alpha_slice(node):
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "rast_out"
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[0], ast.Constant)
        and node.slice.elts[0].value is Ellipsis
        and isinstance(node.slice.elts[1], ast.Slice)
        and isinstance(node.slice.elts[1].lower, ast.UnaryOp)
        and isinstance(node.slice.elts[1].lower.op, ast.USub)
        and isinstance(node.slice.elts[1].lower.operand, ast.Constant)
    ):
        return False
    return node.slice.elts[1].lower.operand.value == 1


def _function_assigns_mask_from_clamped_alpha(function_node):
    for child in ast.walk(function_node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "mask" for target in child.targets):
            continue
        if not (
            isinstance(child.value, ast.Call)
            and _call_name(child.value.func) == "torch.clamp"
            and len(child.value.args) == 3
            and _is_rast_out_alpha_slice(child.value.args[0])
            and isinstance(child.value.args[1], ast.Constant)
            and child.value.args[1].value == 0
            and isinstance(child.value.args[2], ast.Constant)
            and child.value.args[2].value == 1
        ):
            continue
        return True
    return False


def _function_assigns_extra_mask_with_flipped_mask(function_node):
    for child in ast.walk(function_node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "extra"
            and _subscript_constant_key(target) == "mask"
            for target in child.targets
        ):
            continue
        if _is_flip_dims_one(child.value):
            return True
    return False


def _has_mask_warp_branch(function_node):
    for child in ast.walk(function_node):
        if not isinstance(child, ast.If):
            continue
        if not (
            isinstance(child.test, ast.Compare)
            and isinstance(child.test.left, ast.Subscript)
            and isinstance(child.test.left.value, ast.Attribute)
            and isinstance(child.test.left.value.value, ast.Name)
            and child.test.left.value.value.id == "mask_rs"
            and child.test.left.value.attr == "shape"
            and isinstance(child.test.ops[0], ast.NotEq)
        ):
            continue
        body_text = ast.unparse(ast.Module(body=child.body, type_ignores=[]))
        orelse_text = ast.unparse(ast.Module(body=child.orelse, type_ignores=[]))
        if (
            "kornia.geometry.transform.warp_perspective" in body_text
            and "mode='nearest'" in body_text
            and "maskAs = mask_rs" in orelse_text
        ):
            return True
    return False


def _crop_builder_collects_and_passes_mask(path):
    tree = _parse(path)
    make_crop_data_batch = _find_function(tree, "make_crop_data_batch")

    self = unittest.TestCase()
    self.assertTrue(_name_initialized_to_empty_list(make_crop_data_batch, "mask_rs"))
    self.assertTrue(_appends_subscript_key(make_crop_data_batch, "mask_rs", "extra", "mask"))
    self.assertTrue(
        any(
            _is_torch_cat_permute_to_nchw(value, "mask_rs")
            for value in _assignment_values(make_crop_data_batch, "mask_rs")
        )
    )
    self.assertTrue(_has_mask_warp_branch(make_crop_data_batch))

    batch_calls = _batch_pose_data_calls(make_crop_data_batch)
    self.assertTrue(batch_calls, "make_crop_data_batch should construct BatchPoseData")
    self.assertTrue(any(_call_has_keyword(call, "maskAs", value_name="maskAs") for call in batch_calls))
    self.assertTrue(any(_call_has_keyword(call, "maskBs", none_value=True) for call in batch_calls))


class FoundationPoseReuseStaticTests(unittest.TestCase):
    def test_batch_pose_data_exposes_batch_masks(self):
        tree = _parse("learning/datasets/pose_dataset.py")
        batch_pose_data = _find_class(tree, "BatchPoseData")

        self.assertTrue(
            {"maskAs", "maskBs"}.issubset(_class_level_none_assignments(batch_pose_data))
        )

    def test_batch_pose_data_select_by_indices_indexes_all_present_tensors(self):
        tree = _parse("learning/datasets/pose_dataset.py")
        select_by_indices = _find_function(tree, "select_by_indices")

        self.assertIn("self.__dict__", ast.unparse(select_by_indices))
        self.assertIn("out.__dict__[k]", ast.unparse(select_by_indices))
        self.assertIn("ids.to", _calls_under(select_by_indices))

    def test_nvdiffrast_render_exports_flipped_silhouette_mask(self):
        tree = _parse("Utils.py")
        nvdiffrast_render = _find_function(tree, "nvdiffrast_render")

        self.assertTrue(_function_assigns_mask_from_clamped_alpha(nvdiffrast_render))
        self.assertTrue(_function_assigns_extra_mask_with_flipped_mask(nvdiffrast_render))

    def test_score_crop_builder_collects_render_mask_as_batch_maskA(self):
        _crop_builder_collects_and_passes_mask("learning/training/predict_score.py")

    def test_refine_crop_builder_collects_render_mask_as_batch_maskA(self):
        _crop_builder_collects_and_passes_mask("learning/training/predict_pose_refine.py")

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

    def test_pose_candidate_script_uses_lazy_first_object_foundationpose(self):
        tree = _parse("export_pose_candidate_artifacts.py")
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

        constructor_try_nodes = [
            child
            for child in ast.walk(object_loop)
            if isinstance(child, ast.Try) and _contains_scflow2_constructor(child)
        ]
        self.assertTrue(
            any(
                any(_raises_system_exit(handler) for handler in child.handlers)
                for child in constructor_try_nodes
            ),
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

    def test_processed_tracking_defines_timing_summary_helper(self):
        tree = _parse("process_data_estimate+tracking.py")
        summary = _find_function(tree, "print_timing_summary")
        calls = _calls_under(summary)
        constants = _constants_under(summary)

        self.assertIn("tqdm.tqdm.write", calls)
        self.assertIn("avg_seconds", constants)
        self.assertIn("No processed objects; timing averages are unavailable.", constants)
        self.assertTrue(
            any(
                value.startswith("Overall average seconds per processed object:")
                for value in constants
            )
        )

    def test_processed_tracking_records_expected_timing_stages(self):
        tree = _parse("process_data_estimate+tracking.py")
        main = _find_function(tree, "main")
        constants = _constants_under(main)
        loaded_names = _name_loads_under(main)

        self.assertTrue(
            {
                "input_loading",
                "mesh_setup",
                "scflow2_initialization",
                "register",
                "track_one",
                "scflow2_refinement",
                "pose_saving",
                "visualization_rendering",
                "video_writing",
                "total_processed_object",
            }.issubset(constants)
        )
        self.assertIn("processed_objects", loaded_names)
        self.assertIn("skipped_objects", loaded_names)
        self.assertIn("failed_objects", loaded_names)
        self.assertIn("time.perf_counter", _calls_under(main))


if __name__ == "__main__":
    unittest.main()
