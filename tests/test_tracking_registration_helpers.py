import unittest

import numpy as np

from tracking_registration import invalid_pose, is_invalid_pose, registration_inputs_are_valid


class TrackingRegistrationHelperTests(unittest.TestCase):
    def test_empty_mask_is_invalid(self):
        mask = np.zeros((3, 3), dtype=np.uint8)
        depth = np.ones((3, 3), dtype=np.float64)

        self.assertFalse(registration_inputs_are_valid(mask, depth))

    def test_insufficient_valid_depth_pixels_is_invalid(self):
        mask = np.ones((3, 3), dtype=np.uint8)
        depth = np.zeros((3, 3), dtype=np.float64)
        depth[0, :3] = 0.001

        self.assertFalse(registration_inputs_are_valid(mask, depth))

    def test_four_masked_depth_pixels_is_valid(self):
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask.flat[:4] = 1
        depth = np.ones((3, 3), dtype=np.float64) * 0.001

        self.assertTrue(registration_inputs_are_valid(mask, depth))

    def test_invalid_pose_is_all_zero_4x4_float64(self):
        pose = invalid_pose()

        self.assertEqual(pose.shape, (4, 4))
        self.assertEqual(pose.dtype, np.float64)
        self.assertFalse(pose.any())

    def test_is_invalid_pose_detects_all_zero_pose_only(self):
        zero_pose = np.zeros((4, 4), dtype=np.float64)
        valid_pose = np.eye(4, dtype=np.float64)

        self.assertTrue(is_invalid_pose(zero_pose))
        self.assertFalse(is_invalid_pose(valid_pose))


if __name__ == "__main__":
    unittest.main()
