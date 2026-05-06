import os

import numpy as np
import imageio.v3 as iio
import trimesh

from estimater import *
from datareader import *


def normalize_scores(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)

    normalized = np.full_like(scores, -1e6, dtype=np.float64)
    if not finite.any():
        return np.zeros_like(scores, dtype=np.float64)

    x = scores[finite] / temperature
    x = x - x.max()
    log_probs = x - np.log(np.exp(x).sum())

    normalized[finite] = log_probs
    return normalized


def rotation_angle(R1, R2):
    R_delta = R1 @ R2.T
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def transition_cost(prev_pose, cur_pose, mesh_diameter):
    diameter = max(float(mesh_diameter), 1e-12)
    trans_cost = np.linalg.norm(cur_pose[:3, 3] - prev_pose[:3, 3]) / diameter
    rot_cost = rotation_angle(prev_pose[:3, :3], cur_pose[:3, :3]) / np.pi
    return trans_cost + rot_cost


def select_pose_trajectory(all_poses, all_scores, mesh_diameter, trans_lambda=1.0, rot_lambda=1.0):
    if len(all_poses) != len(all_scores):
        raise ValueError("all_poses and all_scores must have the same number of frames")
    if len(all_poses) == 0:
        return np.empty((0, 4, 4), dtype=np.float64)

    poses_per_frame = []
    scores_per_frame = []
    valid_frame_indices = []
    for frame_idx, (poses, scores) in enumerate(zip(all_poses, all_scores)):
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(poses) != len(scores):
            raise ValueError(f"frame {frame_idx} pose and score counts do not match")

        finite = np.isfinite(scores)
        if len(poses) > 0:
            finite = finite & np.isfinite(poses).all(axis=(1, 2))
        if not finite.any():
            continue

        poses_per_frame.append(poses[finite])
        scores_per_frame.append(normalize_scores(scores[finite]))
        valid_frame_indices.append(frame_idx)

    if len(valid_frame_indices) == 0:
        raise ValueError("no valid pose candidates")

    dp = [scores_per_frame[0].copy()]
    backptr = []
    for frame_idx in range(1, len(poses_per_frame)):
        prev_poses = poses_per_frame[frame_idx - 1]
        cur_poses = poses_per_frame[frame_idx]
        diameter = max(float(mesh_diameter), 1e-12)

        prev_t = prev_poses[:, :3, 3]
        cur_t = cur_poses[:, :3, 3]
        trans_cost = np.linalg.norm(cur_t[None] - prev_t[:, None], axis=-1) / diameter

        prev_R = prev_poses[:, :3, :3]
        cur_R = cur_poses[:, :3, :3]
        traces = np.einsum("aij,bij->ab", prev_R, cur_R)
        rot_cost = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)) / np.pi

        values = dp[-1][:, None] + scores_per_frame[frame_idx][None, :] - trans_lambda * trans_cost - rot_lambda * rot_cost
        best_prev = values.argmax(axis=0)
        dp.append(values[best_prev, np.arange(len(cur_poses))])
        backptr.append(best_prev)

    best_idx = int(dp[-1].argmax())
    selected_indices = [best_idx]
    for prev_indices in reversed(backptr):
        best_idx = int(prev_indices[best_idx])
        selected_indices.append(best_idx)
    selected_indices.reverse()

    selected_poses = [
        poses_per_frame[frame_idx][candidate_idx]
        for frame_idx, candidate_idx in enumerate(selected_indices)
    ]
    selected_poses = np.stack(selected_poses, axis=0)

    trajectory = np.zeros((len(all_poses), 4, 4), dtype=np.float64)
    trajectory[valid_frame_indices] = selected_poses
    return trajectory


def main():
    
    intrinsics = np.load("/data/datasets/DexYCB/processed/20200709-subject-01/20200709_141754/841412060263/video/intrinsics.npy")
    intrinsics = intrinsics.astype(np.float64)
    
    images = iio.imread("/data/datasets/DexYCB/processed/20200709-subject-01/20200709_141754/841412060263/video/images.mp4")
    
    depths = np.load("/data/datasets/DexYCB/processed/20200709-subject-01/20200709_141754/841412060263/video/depths.npy")
    depths[depths==65535] = 0
    depths = (depths.astype(np.float64)) / 1000.0
    depths[(depths<0.001) | (depths>=np.inf)] = 0
    
    masks = np.load("/data/datasets/DexYCB/processed/20200709-subject-01/20200709_141754/841412060263/objects/predicted/object_0/masks.npz")
    masks = masks["masks_visible"]
    
    mesh = trimesh.load("/data/datasets/DexYCB/processed/20200709-subject-01/20200709_141754/841412060263/objects/predicted/object_0/mesh.glb", force="mesh")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = f"{code_dir}/debug"
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=0,
        glctx=glctx,
    )
    
    os.makedirs(debug_dir, exist_ok=True)
    
    T = images.shape[0]
    all_poses = []
    all_scores = []
    for frame_idx in range(T):
        poses, scores = est.register_all(
            K=intrinsics,
            rgb=images[frame_idx],
            depth=depths[frame_idx],
            ob_mask=masks[frame_idx],
            iteration=5,
        )
        all_poses.append(poses)
        all_scores.append(scores)

    trajectory = select_pose_trajectory(
        all_poses,
        all_scores,
        mesh_diameter=est.diameter,
        trans_lambda=1.0,
        rot_lambda=1.0,
    )

    video = []
    for frame_idx, pose in enumerate(trajectory):
        if not pose.any():
            video.append(images[frame_idx])
            continue
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(intrinsics, img=images[frame_idx], ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(images[frame_idx], ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

        video.append(vis)
        
    video = np.stack(video, axis=0)
    iio.imwrite(os.path.join(debug_dir, "vis.mp4"), video)

if __name__ == "__main__":
    main()
