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


def _keyword_value(call_node, keyword_name):
    for keyword in call_node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    return None


def _function_has_arg(function_node, arg_name):
    return any(arg.arg == arg_name for arg in function_node.args.args)


def _has_keyword_constant(call_node, keyword_name, value):
    keyword_value = _keyword_value(call_node, keyword_name)
    return isinstance(keyword_value, ast.Constant) and keyword_value.value == value


def _assigns_name_from_call_with_keyword(function_node, name, call_name, keyword_name, keyword_value):
    for child in ast.walk(function_node):
        if not isinstance(child, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in child.targets):
            continue
        if not isinstance(child.value, ast.Call):
            continue
        if _call_name(child.value.func) != call_name:
            continue
        if _has_keyword_constant(child.value, keyword_name, keyword_value):
            return True
    return False


def _format_spec_names(function_node):
    names = set()
    for child in ast.walk(function_node):
        if isinstance(child, ast.JoinedStr):
            for value in child.values:
                if isinstance(value, ast.FormattedValue) and value.format_spec is not None:
                    names.add(ast.unparse(value.format_spec))
    return names


def _np_savez_calls(function_node):
    return [
        child
        for child in ast.walk(function_node)
        if isinstance(child, ast.Call) and _call_name(child.func) in {"np.savez_compressed", "np.savez"}
    ]


class PoseCandidateArtifactsStaticTests(unittest.TestCase):
    def test_export_script_and_output_names_match_artifact_contract(self):
        self.assertTrue((ROOT / "export_pose_candidate_artifacts.py").exists())
        self.assertFalse((ROOT / "get_all_pose_candidates_and_scores.py").exists())

        tree = _parse("export_pose_candidate_artifacts.py")
        constants = _constants_under(tree)

        self.assertIn("all_pose_candidates_artifacts.npz", constants)
        self.assertIn("all_pose_candidates_artifacts.tmp.npz", constants)
        self.assertIn("valid", constants)
        self.assertNotIn("all_poses&scores.pkl", constants)
        self.assertNotIn("num_frames", constants)
        self.assertNotIn("frame_indices", constants)
        self.assertNotIn("candidate_counts", constants)
        self.assertNotIn("image_crops", constants)
        self.assertNotIn("crop_to_oris", constants)
        self.assertNotIn("crop_boxes", constants)

    def test_export_script_requests_pose_data_and_handles_invalid_frames(self):
        tree = _parse("export_pose_candidate_artifacts.py")
        main = _find_function(tree, "main")
        source = ast.unparse(main)

        self.assertIn("valid = np.zeros(T, dtype=bool)", source)
        self.assertIn("register_result is None", source)
        self.assertIn("valid[frame_idx] = False", source)
        self.assertIn("valid[frame_idx] = True", source)
        self.assertIn("return_pose_data=True", source)
        self.assertIn("artifact_key(frame_idx, 'poses')", source)
        self.assertIn("artifact_key(frame_idx, 'scores')", source)
        self.assertIn("artifact_key(frame_idx, 'render_rgbs')", source)
        self.assertIn("artifact_key(frame_idx, 'render_masks')", source)
        self.assertIn("artifact_key(frame_idx, 'tf_to_crops')", source)

    def test_frame_keys_are_four_digit_zero_padded(self):
        tree = _parse("export_pose_candidate_artifacts.py")
        artifact_key = _find_function(tree, "artifact_key")
        self.assertIn("04d", "".join(_format_spec_names(artifact_key)))

    def test_export_uses_atomic_npz_replacement(self):
        tree = _parse("export_pose_candidate_artifacts.py")
        main = _find_function(tree, "main")
        calls = _calls_under(main)

        self.assertIn("np.savez_compressed", calls)
        self.assertTrue(any(call.endswith(".replace") for call in calls))

    def test_score_predictor_can_return_pose_data_without_changing_default(self):
        tree = _parse("learning/training/predict_score.py")
        predict = _find_function(tree, "predict")

        self.assertTrue(_function_has_arg(predict, "return_pose_data"))
        source = ast.unparse(predict)
        self.assertIn("if return_pose_data:", source)
        self.assertIn("return (scores, canvas, pose_data)", source)
        self.assertIn("return (scores, None, pose_data)", source)
        self.assertIn("return (scores, canvas)", source)
        self.assertIn("return (scores, None)", source)

    def test_foundationpose_register_all_returns_none_for_too_few_valid_pixels(self):
        tree = _parse("estimater.py")
        register_all = _find_function(tree, "register_all")
        source = ast.unparse(register_all)

        self.assertTrue(_function_has_arg(register_all, "return_pose_data"))
        self.assertIn("if valid.sum() < 4:", source)
        self.assertIn("return None", source)
        self.assertNotIn("np.array([np.nan])", source)

    def test_foundationpose_sorts_pose_data_with_scores(self):
        tree = _parse("estimater.py")
        register_all = _find_function(tree, "register_all")
        source = ast.unparse(register_all)

        self.assertIn("return_pose_data=return_pose_data", source)
        self.assertIn("pose_data = pose_data.select_by_indices(ids)", source)
        self.assertIn("return (poses_out.data.cpu().numpy(), scores.data.cpu().numpy(), pose_data)", source)

    def test_current_register_all_callers_handle_none_results(self):
        for path in ["process_data.py", "export_pose_candidate_artifacts.py"]:
            tree = _parse(path)
            source = ast.unparse(tree)

            self.assertIn("register_result = est.register_all", source)
            self.assertIn("register_result is None", source)

    def test_artifact_tensor_conversion_helper_shapes_and_types(self):
        from export_pose_candidate_artifacts import pose_data_to_artifacts

        class TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value)

            @property
            def data(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class PoseData:
            pass

        pose_data = PoseData()
        pose_data.rgbAs = TensorLike(
            np.array(
                [
                    [
                        [[0.0, 0.5], [1.0, 1.5]],
                        [[0.25, 0.75], [1.25, -0.5]],
                        [[1.0, 0.0], [0.5, 0.25]],
                    ]
                ],
                dtype=np.float32,
            )
        )
        pose_data.maskAs = TensorLike(np.array([[[[0.0, 0.2], [0.51, 1.0]]]], dtype=np.float32))
        pose_data.tf_to_crops = TensorLike(np.eye(3, dtype=np.float32)[None])

        artifacts = pose_data_to_artifacts(pose_data)

        self.assertEqual(artifacts["render_rgbs"].shape, (1, 2, 2, 3))
        self.assertEqual(artifacts["render_rgbs"].dtype, np.uint8)
        self.assertEqual(artifacts["render_masks"].shape, (1, 2, 2))
        self.assertEqual(artifacts["render_masks"].dtype, np.uint8)
        self.assertTrue(np.isin(artifacts["render_masks"], [0, 1]).all())
        self.assertEqual(artifacts["tf_to_crops"].shape, (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
