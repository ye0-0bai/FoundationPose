import os
import logging
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import trimesh
import pickle
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from estimater import *
from datareader import *


def configure_quiet_logging():
    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)


def main():
    configure_quiet_logging()

    data_root = Path("/data/datasets/DexYCB/processed")
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        intrinsics_path = seq_dir / "video" / "intrinsics.npy"
        images_path = seq_dir / "video" / "images.mp4"
        depths_path = seq_dir / "video" / "depths.npy"

        objects_root = seq_dir / "objects" / "predicted"
        object_dirs = sorted(objects_root.glob("object_*"))
        for object_dir in object_dirs:
            masks_path = object_dir / "masks.npz"
            mesh_path = object_dir / "mesh.glb"
            
            save_path = object_dir / "poses.npy"
            if save_path.exists():
                continue
                
            intrinsics = np.load(intrinsics_path)
            intrinsics = intrinsics.astype(np.float64)

            images = iio.imread(images_path)

            depths = np.load(depths_path)
            depths[depths==65535] = 0
            depths = (depths.astype(np.float64)) / 1000.0
            depths[(depths<0.001) | (depths>=np.inf)] = 0

            masks = np.load(masks_path)
            masks = masks["masks_visible"]

            mesh = trimesh.load(mesh_path, force="mesh")
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

            poses = []
            T = images.shape[0]
            for frame_idx in range(T):
                if frame_idx == 0:
                    pose = est.register(
                        K=intrinsics,
                        rgb=images[frame_idx],
                        depth=depths[frame_idx],
                        ob_mask=masks[frame_idx],
                        iteration=5,
                    )
                else:
                    pose = est.track_one(
                        K=intrinsics,
                        rgb=images[frame_idx],
                        depth=depths[frame_idx],
                        iteration=3
                    )
                    
                poses.append(pose)

            poses = np.stack(poses, axis=0)

            save_path = object_dir / "poses.npy"
            np.save(save_path, poses)

            video = []
            for frame_idx, pose in enumerate(poses):
                center_pose = pose@np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(intrinsics, img=images[frame_idx], ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(images[frame_idx], ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

                video.append(vis)

            video = np.stack(video, axis=0)
            save_path = object_dir / "poses.mp4"
            iio.imwrite(save_path, video)

if __name__ == "__main__":
    main()
