import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from optimize_precomputed_poses import (
    compute_mask_ious,
    normalize_scores,
    prepare_iou_weighted_candidates,
    select_pose_trajectory,
    smooth_pose_trajectory,
)


def _pose(x=0.0, angle=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    pose[0, 3] = x
    return pose


class OptimizePrecomputedPosesAlgorithmTests(unittest.TestCase):
    def test_normalize_scores_uses_minmax_scaling(self):
        scores = np.array([2.0, 4.0, 6.0])

        normalized = normalize_scores(scores, method="minmax")

        self.assertTrue(np.allclose(normalized, [0.0, 0.5, 1.0]))

    def test_normalize_scores_equal_finite_scores_become_ones(self):
        scores = np.array([3.0, 3.0, 3.0])

        normalized = normalize_scores(scores, method="minmax")

        self.assertTrue(np.allclose(normalized, [1.0, 1.0, 1.0]))

    def test_normalize_scores_ignores_nonfinite_values_for_minmax(self):
        scores = np.array([2.0, -np.inf, 1.0, np.nan])

        normalized = normalize_scores(scores, method="minmax")

        self.assertTrue(np.allclose(normalized[[0, 2]], [1.0, 0.0]))
        self.assertFalse(np.isfinite(normalized[1]))
        self.assertFalse(np.isfinite(normalized[3]))

    def test_compute_mask_ious_handles_perfect_partial_empty_and_disjoint_masks(self):
        visible_mask = np.array(
            [
                [1, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        )
        identity_crop = np.eye(3, dtype=np.float32)
        render_masks = np.array(
            [
                [[1, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        tf_to_crops = np.repeat(identity_crop[None], len(render_masks), axis=0)

        ious = compute_mask_ious(visible_mask, render_masks, tf_to_crops)

        self.assertTrue(np.allclose(ious, [1.0, 0.5, 0.0, 0.0]))

    def test_prepare_iou_weighted_candidates_multiplies_minmax_scores_by_iou(self):
        valid = np.array([True])
        poses = [_pose(0.0), _pose(1.0), _pose(2.0)]
        artifact_data = {
            "valid": valid,
            "poses_0000": np.stack(poses),
            "scores_0000": np.array([2.0, 4.0, 6.0]),
            "render_masks_0000": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [0, 0]],
                    [[0, 0], [0, 1]],
                ],
                dtype=np.uint8,
            ),
            "tf_to_crops_0000": np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
        }
        masks_visible = np.array(
            [
                [
                    [1, 0],
                    [0, 0],
                ]
            ],
            dtype=np.uint8,
        )

        all_poses, adjusted_scores = prepare_iou_weighted_candidates(
            artifact_data,
            masks_visible,
        )

        self.assertEqual(len(all_poses), 1)
        self.assertTrue(np.allclose(adjusted_scores[0], [0.0, 0.25, 0.0]))

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
