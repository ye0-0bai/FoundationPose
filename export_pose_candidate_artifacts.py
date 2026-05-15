import io
import logging
import os
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import trimesh

from estimater import *
from datareader import *


def configure_quiet_logging():
    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.ERROR)


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    elif hasattr(value, "data"):
        value = value.data
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def pose_data_to_artifacts(pose_data):
    render_rgbs = tensor_to_numpy(pose_data.rgbAs)
    render_rgbs = np.moveaxis(render_rgbs, 1, -1)
    render_rgbs = np.clip(render_rgbs * 255.0, 0, 255).astype(np.uint8)

    render_masks = tensor_to_numpy(pose_data.maskAs)
    if render_masks.ndim == 4 and render_masks.shape[1] == 1:
        render_masks = render_masks[:, 0]
    render_masks = (render_masks > 0.5).astype(np.uint8)

    tf_to_crops = tensor_to_numpy(pose_data.tf_to_crops)

    return {
        "render_rgbs": render_rgbs,
        "render_masks": render_masks,
        "tf_to_crops": tf_to_crops,
    }


def artifact_key(frame_idx, name):
    return f"{name}_{frame_idx:04d}"


def main():
    configure_quiet_logging()

    data_root = Path("/data/datasets/DexYCB/processed")
    seq_dirs = sorted(data_root.glob("**/video"))
    seq_dirs = [seq_dir.parent for seq_dir in seq_dirs]

    code_dir = os.path.dirname(os.path.realpath(__file__))
    debug_dir = f"{code_dir}/debug"
    os.makedirs(debug_dir, exist_ok=True)

    est = None

    for seq_dir in tqdm.tqdm(seq_dirs, dynamic_ncols=True):
        try:
            intrinsics_path = seq_dir / "video" / "intrinsics.npy"
            images_path = seq_dir / "video" / "images.mp4"
            depths_path = seq_dir / "video" / "depths.npy"

            objects_root = seq_dir / "objects" / "gpt"
            object_dirs = sorted(objects_root.glob("object_*"))
            for object_dir in object_dirs:
                masks_path = object_dir / "masks.npz"
                mesh_path = object_dir / "mesh.glb"

                save_path = object_dir / "all_pose_candidates_artifacts.npz"
                tmp_save_path = object_dir / "all_pose_candidates_artifacts.tmp.npz"
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

                if est is None:
                    scorer = ScorePredictor()
                    refiner = PoseRefinePredictor()
                    glctx = dr.RasterizeCudaContext()
                    est = FoundationPose(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                        scorer=scorer,
                        refiner=refiner,
                        debug_dir=debug_dir,
                        debug=0,
                        glctx=glctx,
                    )
                else:
                    est.reset_object(
                        model_pts=mesh.vertices,
                        model_normals=mesh.vertex_normals,
                        mesh=mesh,
                    )

                T = images.shape[0]
                valid = np.zeros(T, dtype=bool)
                result = {"valid": valid}
                for frame_idx in range(T):
                    register_result = est.register_all(
                        K=intrinsics,
                        rgb=images[frame_idx],
                        depth=depths[frame_idx],
                        ob_mask=masks[frame_idx],
                        iteration=5,
                        return_pose_data=True,
                    )
                    if register_result is None:
                        valid[frame_idx] = False
                        continue

                    poses, scores, pose_data = register_result
                    artifacts = pose_data_to_artifacts(pose_data)
                    valid[frame_idx] = True
                    result[artifact_key(frame_idx, "poses")] = poses
                    result[artifact_key(frame_idx, "scores")] = scores
                    result[artifact_key(frame_idx, "render_rgbs")] = artifacts["render_rgbs"]
                    result[artifact_key(frame_idx, "render_masks")] = artifacts["render_masks"]
                    result[artifact_key(frame_idx, "tf_to_crops")] = artifacts["tf_to_crops"]

                np.savez_compressed(tmp_save_path, **result)
                tmp_save_path.replace(save_path)

        except Exception:
            tqdm.write(f"Failed to process {seq_dir.relative_to(data_root)}")
            io_string = io.StringIO()
            traceback.print_exc(file=io_string)
            tqdm.write(io_string.getvalue())


if __name__ == "__main__":
    main()
