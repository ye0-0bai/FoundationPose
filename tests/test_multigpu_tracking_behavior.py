import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "process_data_estimate+tracking_multigpu.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tracking_multigpu_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEstimator:
    def __init__(self):
        self.pose_last = None
        self.register_calls = 0
        self.track_calls = 0

    def register(self, **kwargs):
        self.register_calls += 1
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 10.0 + self.register_calls
        self.pose_last = pose
        return pose

    def track_one(self, **kwargs):
        self.track_calls += 1
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 20.0 + self.track_calls
        self.pose_last = pose
        return pose


class MultiGpuTrackingBehaviorTests(unittest.TestCase):
    def test_invalid_leading_frames_delay_registration_until_first_valid_frame(self):
        module = _load_module()
        estimator = FakeEstimator()
        images = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
        depths = np.ones((4, 2, 2), dtype=np.float64) * 0.001
        masks = np.zeros((4, 2, 2), dtype=np.uint8)
        masks[2:] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            object_dir = Path(tmpdir)
            np.savez(object_dir / "masks.npz", masks_visible=masks)
            (object_dir / "mesh.glb").write_bytes(b"unused")
            draw_box = mock.Mock(side_effect=lambda *args, img, **kwargs: img + 1)
            draw_axis = mock.Mock(side_effect=lambda img, **kwargs: img + 1)

            worker_state = {
                "est": estimator,
                "draw_posed_3d_box": draw_box,
                "draw_xyz_axis": draw_axis,
            }
            fake_trimesh = types.SimpleNamespace(
                load=mock.Mock(return_value=object()),
                bounds=types.SimpleNamespace(
                    oriented_bounds=mock.Mock(return_value=(np.eye(4), np.ones(3)))
                ),
            )

            with (
                mock.patch.dict("sys.modules", {"trimesh": fake_trimesh}),
                mock.patch.object(module, "setup_estimator_for_mesh", return_value=estimator),
                mock.patch.object(module.iio, "imwrite") as imwrite,
            ):
                module.process_object(
                    object_dir=object_dir,
                    intrinsics=np.eye(3, dtype=np.float64),
                    images=images,
                    depths=depths,
                    worker_state=worker_state,
                )

            poses = np.load(object_dir / "poses.npy")
            self.assertFalse(poses[0].any())
            self.assertFalse(poses[1].any())
            self.assertEqual(poses[2, 0, 3], 11.0)
            self.assertEqual(poses[3, 0, 3], 21.0)
            self.assertEqual(estimator.register_calls, 1)
            self.assertEqual(estimator.track_calls, 1)

            rendered_video = imwrite.call_args.args[1]
            self.assertTrue(np.array_equal(rendered_video[0], images[0]))
            self.assertTrue(np.array_equal(rendered_video[1], images[1]))
            self.assertEqual(draw_box.call_count, 2)
            self.assertEqual(draw_axis.call_count, 2)
            self.assertTrue(np.array_equal(rendered_video[2], images[2] + 1))


if __name__ == "__main__":
    unittest.main()
