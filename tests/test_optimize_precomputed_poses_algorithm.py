import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from optimize_precomputed_poses import (
    normalize_scores,
    select_pose_trajectory,
    smooth_pose_trajectory,
)


def _pose(x=0.0, angle=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    pose[0, 3] = x
    return pose


class OptimizePrecomputedPosesAlgorithmTests(unittest.TestCase):
    def test_normalize_scores_preserves_existing_finite_and_nonfinite_behavior(self):
        scores = np.array([2.0, -np.inf, 1.0, np.nan])

        normalized = normalize_scores(scores)

        expected_finite = np.array([2.0, 1.0])
        expected_finite = expected_finite - expected_finite.max()
        expected_finite = expected_finite - np.log(np.exp(expected_finite).sum())
        self.assertTrue(np.allclose(normalized[[0, 2]], expected_finite))
        self.assertEqual(normalized[1], -1e6)
        self.assertEqual(normalized[3], -1e6)

    def test_select_pose_trajectory_prefers_smooth_global_path(self):
        all_poses = [
            np.stack([_pose(0.0), _pose(10.0)]),
            np.stack([_pose(0.2), _pose(10.2)]),
            np.stack([_pose(0.4), _pose(10.4)]),
        ]
        all_scores = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        ]

        trajectory = select_pose_trajectory(
            all_poses,
            all_scores,
            mesh_diameter=1.0,
            trans_lambda=10.0,
            rot_lambda=0.0,
        )

        self.assertTrue(np.allclose(trajectory[:, 0, 3], [0.0, 0.2, 0.4]))

    def test_select_pose_trajectory_keeps_invalid_frame_zero(self):
        all_poses = [
            np.stack([_pose(0.0)]),
            np.empty((0, 4, 4), dtype=np.float64),
            np.stack([_pose(1.0)]),
        ]
        all_scores = [
            np.array([1.0]),
            np.empty((0,), dtype=np.float64),
            np.array([1.0]),
        ]

        trajectory = select_pose_trajectory(all_poses, all_scores, mesh_diameter=1.0)

        self.assertTrue(np.allclose(trajectory[0], _pose(0.0)))
        self.assertFalse(trajectory[1].any())
        self.assertTrue(np.allclose(trajectory[2], _pose(1.0)))

    def test_select_pose_trajectory_rejects_mismatched_frame_counts_before_empty_return(self):
        with self.assertRaisesRegex(ValueError, "same number of frames"):
            select_pose_trajectory([], [np.array([1.0])], mesh_diameter=1.0)

    def test_smooth_pose_trajectory_interpolates_short_gaps_and_preserves_long_gaps(self):
        trajectory = np.zeros((7, 4, 4), dtype=np.float64)
        trajectory[0] = _pose(0.0)
        trajectory[2] = _pose(2.0)
        trajectory[6] = _pose(6.0)

        smoothed = smooth_pose_trajectory(
            trajectory,
            max_invalid_gap=1,
            smooth_window=3,
            smooth_polyorder=1,
        )

        self.assertTrue(smoothed[1].any())
        self.assertTrue(np.allclose(smoothed[1, 0, 3], 1.0))
        self.assertFalse(smoothed[3].any())
        self.assertFalse(smoothed[4].any())
        self.assertFalse(smoothed[5].any())


if __name__ == "__main__":
    unittest.main()
