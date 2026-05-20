import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "process_data_estimate+tracking.py",
    ROOT / "process_data_estimate+tracking_multigpu.py",
]


def _parse(path):
    return ast.parse(path.read_text())


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


def _assigned_names(node):
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _source(path):
    return path.read_text()


class DelayedRegistrationStaticTests(unittest.TestCase):
    def test_tracking_scripts_do_not_register_only_on_frame_zero(self):
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                source = _source(path)

                self.assertNotIn("frame_idx == 0", source)

    def test_tracking_scripts_use_tracking_initialized_state(self):
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                tree = _parse(path)
                function = _find_function(tree, "main" if path.name.endswith("+tracking.py") else "process_object")

                self.assertIn("tracking_initialized", _assigned_names(function))
                self.assertIn("registration_inputs_are_valid", _calls_under(function))

    def test_tracking_scripts_append_zero_poses_before_registration(self):
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                tree = _parse(path)
                function = _find_function(tree, "main" if path.name.endswith("+tracking.py") else "process_object")
                calls = _calls_under(function)

                self.assertIn("invalid_pose", calls)
                self.assertIn("poses.append", calls)

    def test_tracking_scripts_skip_overlay_for_zero_pose(self):
        for path in SCRIPTS:
            with self.subTest(script=path.name):
                tree = _parse(path)
                function_name = "main" if path.name.endswith("+tracking.py") else "render_pose_video"
                function = _find_function(tree, function_name)
                source = ast.unparse(function)

                self.assertIn("is_invalid_pose", _calls_under(function))
                self.assertRegex(source, r"(video|frames)\.append\(images\[frame_idx\]\)")
                self.assertIn("continue", source)


if __name__ == "__main__":
    unittest.main()
