import unittest
from argparse import Namespace
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from optimize_precomputed_poses import (
    compute_mask_ious,
    experiment_record_path,
    ObjectPaths,
    normalize_scores,
    prepare_iou_weighted_candidates,
    select_pose_trajectory,
    smooth_pose_trajectory,
    write_experiment_record,
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

    def test_select_pose_trajectory_debug_matches_selected_path_contributions(self):
        all_poses = [
            np.stack([_pose(0.0), _pose(5.0)]),
            np.stack([_pose(0.5), _pose(5.1)]),
            np.empty((0, 4, 4), dtype=np.float64),
            np.stack([_pose(1.0), _pose(7.0)]),
        ]
        all_scores = [
            np.array([0.4, 0.5]),
            np.array([0.8, 0.2]),
            np.empty((0,), dtype=np.float64),
            np.array([0.6, 0.7]),
        ]

        trajectory, debug = select_pose_trajectory(
            all_poses,
            all_scores,
            mesh_diameter=1.0,
            trans_lambda=0.25,
            rot_lambda=0.0,
            return_debug=True,
        )

        self.assertTrue(np.allclose(trajectory[[0, 1, 3], 0, 3], [0.0, 0.5, 1.0]))
        self.assertTrue(np.array_equal(debug["frame_index"], [0, 1, 3]))
        self.assertTrue(np.array_equal(debug["selected_candidate_index"], [0, 0, 0]))
        self.assertTrue(np.allclose(debug["selected_adjusted_score"], [0.4, 0.8, 0.6]))
        self.assertTrue(np.array_equal(debug["score_rank"], [2, 1, 2]))
        self.assertTrue(np.allclose(debug["translation_cost"], [0.0, 0.5, 0.5]))
        self.assertTrue(np.allclose(debug["rotation_cost"], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(debug["weighted_translation_penalty"], [0.0, 0.125, 0.125]))
        self.assertTrue(np.allclose(debug["weighted_rotation_penalty"], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(debug["net_contribution"], [0.4, 0.675, 0.475]))
        self.assertTrue(np.allclose(debug["cumulative_score"], [0.4, 1.075, 1.55]))

    def test_select_pose_trajectory_rejects_mismatched_frame_counts_before_empty_return(self):
        with self.assertRaisesRegex(ValueError, "same number of frames"):
            select_pose_trajectory([], [np.array([1.0])], mesh_diameter=1.0)

    def test_object_paths_uses_run_id_in_output_names(self):
        paths = ObjectPaths.from_object_dir("/tmp/object_001", run_id="test123")

        self.assertEqual(paths.optimized_poses.name, "poses_optimized_test123.npy")
        self.assertEqual(paths.optimized_video.name, "poses_optimized_test123.mp4")
        self.assertEqual(paths.debug_data.name, "poses_optimized_test123_debug.npz")

    def test_write_experiment_record_contains_json_config_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = {
                "processed": 2,
                "skipped_existing": 1,
                "skipped_missing_artifacts": 0,
                "skipped_missing_masks": 1,
                "failed": 0,
            }
            args = Namespace(
                data_root="/data/example",
                overwrite=True,
                max_invalid_gap=5,
                smooth_window=7,
                smooth_polyorder=2,
                trans_lambda=0.25,
                rot_lambda=0.5,
                run_id="test123",
                debug=True,
            )

            record_path = write_experiment_record(
                root,
                run_id="test123",
                generated_at="2026-05-18T12:34:56+08:00",
                argv=["optimize_precomputed_poses.py", "--run-id", "test123", "--debug"],
                args=args,
                stats=stats,
            )

            self.assertEqual(record_path, experiment_record_path(root, "test123"))
            text = record_path.read_text()
            self.assertIn("```json", text)
            self.assertIn("score - trans_lambda * trans_cost - rot_lambda * rot_cost", text)
            payload = text.split("```json", 1)[1].split("```", 1)[0].strip()
            data = json.loads(payload)
            self.assertEqual(data["run_id"], "test123")
            self.assertEqual(data["data_root"], "/data/example")
            self.assertEqual(data["output_naming"]["poses"], "poses_optimized_test123.npy")
            self.assertEqual(data["output_naming"]["debug"], "poses_optimized_test123_debug.npz")
            self.assertTrue(data["debug_enabled"])
            self.assertEqual(data["run_status"], "completed")
            self.assertEqual(data["processing_stats"], stats)

    def test_write_experiment_record_accepts_running_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stats = {
                "processed": 0,
                "skipped_existing": 0,
                "skipped_missing_artifacts": 0,
                "skipped_missing_masks": 0,
                "failed": 0,
            }
            args = Namespace(
                data_root="/data/example",
                overwrite=False,
                max_invalid_gap=5,
                smooth_window=7,
                smooth_polyorder=2,
                trans_lambda=0.25,
                rot_lambda=0.5,
                run_id="test123",
                debug=False,
            )

            record_path = write_experiment_record(
                root,
                run_id="test123",
                generated_at="2026-05-18T12:34:56+08:00",
                argv=["optimize_precomputed_poses.py", "--run-id", "test123"],
                args=args,
                stats=stats,
                run_status="running",
            )

            text = record_path.read_text()
            payload = text.split("```json", 1)[1].split("```", 1)[0].strip()
            data = json.loads(payload)
            self.assertEqual(data["run_status"], "running")
            self.assertEqual(data["run_id"], "test123")
            self.assertEqual(data["data_root"], "/data/example")
            self.assertEqual(data["output_naming"]["poses"], "poses_optimized_test123.npy")
            self.assertEqual(data["optimization_parameters"]["trans_lambda"], 0.25)
            self.assertEqual(data["processing_stats"], stats)

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
