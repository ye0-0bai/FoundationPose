import os

import numpy as np
import imageio.v3 as iio
import trimesh

from estimater import *
from datareader import *


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
    
    video = []
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
                iteration=3,
            )
            
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(intrinsics, img=images[frame_idx], ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(images[frame_idx], ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

        video.append(vis)
        
    video = np.stack(video, axis=0)
    iio.imwrite(os.path.join(debug_dir, "vis.mp4"), video)

if __name__ == "__main__":
    main()
