"""Online SCFlow2 refinement for FoundationPose tracking.

The public helper functions in this module are dependency-light so they can be
tested without importing SCFlow2/MMCV. Heavy SCFlow2 imports happen only when
``SCFlow2OnlineRefiner`` is constructed.
"""

import logging
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_SIZE = 256
DEFAULT_NORMALIZE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_NORMALIZE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@contextmanager
def force_cpu_default_tensor_type(torch):
    """Temporarily make implicit torch tensor construction use CPU tensors."""

    previous_type = torch.tensor([0.0]).type()
    torch.set_default_tensor_type(torch.FloatTensor)
    try:
        yield
    finally:
        torch.set_default_tensor_type(previous_type)


def pose_m_to_scflow2_fields(pose_m):
    """Return SCFlow2 rotation and millimeter translation from a meter pose."""

    pose_m = np.asarray(pose_m, dtype=np.float32)
    if pose_m.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {pose_m.shape}")
    return pose_m[:3, :3].astype(np.float32), (pose_m[:3, 3] * 1000.0).astype(np.float32)


def pose_from_scflow2_fields(rotation, translation_mm):
    """Build a FoundationPose-compatible meter pose from SCFlow2 fields."""

    pose_m = np.eye(4, dtype=np.float32)
    pose_m[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    pose_m[:3, 3] = np.asarray(translation_mm, dtype=np.float32).reshape(3) / 1000.0
    return pose_m


def depth_to_xyz_map(depth_m, K):
    """Convert a depth map in meters to an ``H x W x 3`` XYZ map in meters."""

    depth_m = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    height, width = depth_m.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    z = depth_m
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def sample_scene_cloud(depth_m, K, mask, num_points=1024, minimum_points=32, rng=None):
    """Sample scene points in meters, falling back to all valid depth if needed."""

    rng = rng or np.random.default_rng()
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid_depth = depth_m > 0
    mask = np.asarray(mask).astype(bool)
    valid = valid_depth & mask
    if int(valid.sum()) < minimum_points:
        valid = valid_depth
    if int(valid.sum()) < 1:
        raise ValueError("SCFlow2 refine skipped: no valid depth points")

    cloud = depth_to_xyz_map(depth_m, K)[valid].astype(np.float32)
    replace = len(cloud) < num_points
    indices = rng.choice(len(cloud), size=num_points, replace=replace)
    return cloud[indices].astype(np.float32)


def sample_model_points_m(mesh, num_points=1024, rng=None):
    """Sample mesh vertices in meters for SCFlow2 dense point matching."""

    rng = rng or np.random.default_rng()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) < 1:
        raise ValueError("SCFlow2 refine skipped: mesh has no vertices")
    replace = len(vertices) < num_points
    indices = rng.choice(len(vertices), size=num_points, replace=replace)
    return vertices[indices].astype(np.float32)


def project_mesh_bbox(mesh, K, rotation, translation_mm, image_shape):
    """Project mesh vertices and return ``[left, top, right, bottom]`` in pixels."""

    vertices_mm = np.asarray(mesh.vertices, dtype=np.float32) * 1000.0
    pts_cam = (np.asarray(rotation, dtype=np.float32) @ vertices_mm.T).T
    pts_cam += np.asarray(translation_mm, dtype=np.float32).reshape(1, 3)
    valid = pts_cam[:, 2] > 1e-6
    if not np.any(valid):
        raise ValueError("SCFlow2 refine skipped: projected mesh is behind camera")

    pts_cam = pts_cam[valid]
    K = np.asarray(K, dtype=np.float32)
    xs = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    ys = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    height, width = image_shape[:2]
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"SCFlow2 refine skipped: invalid projected bbox {bbox.tolist()}")
    return bbox


def crop_resize_pad(image, mask, depth_mm, bbox, image_size=DEFAULT_IMAGE_SIZE, pad_value=128):
    """Crop, resize with aspect ratio, and center-pad image/mask/depth.

    Returns transformed arrays plus the 3x3 image transform matrix that maps
    original pixels to patch pixels.
    """

    import cv2

    height, width = image.shape[:2]
    left, top, right, bottom = bbox.astype(np.float32)
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    crop_size = max(right - left, bottom - top, 1.0) * 1.1
    x0 = int(np.floor(cx - crop_size * 0.5))
    y0 = int(np.floor(cy - crop_size * 0.5))
    x1 = int(np.ceil(cx + crop_size * 0.5))
    y1 = int(np.ceil(cy + crop_size * 0.5))

    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(width, x1), min(height, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        raise ValueError("SCFlow2 refine skipped: crop is outside image")

    crop_h, crop_w = y1 - y0, x1 - x0
    crop_img = np.full((crop_h, crop_w, 3), pad_value, dtype=image.dtype)
    crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    crop_depth = np.zeros((crop_h, crop_w), dtype=depth_mm.dtype)

    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)
    crop_img[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    crop_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    crop_depth[dst_y0:dst_y1, dst_x0:dst_x1] = depth_mm[src_y0:src_y1, src_x0:src_x1]

    scale = float(image_size) / float(max(crop_h, crop_w))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized_img = cv2.resize(crop_img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    resized_depth = cv2.resize(crop_depth, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)

    pad_x = (image_size - resized_w) // 2
    pad_y = (image_size - resized_h) // 2
    patch_img = np.full((image_size, image_size, 3), pad_value, dtype=image.dtype)
    patch_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    patch_depth = np.zeros((image_size, image_size), dtype=depth_mm.dtype)
    patch_img[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized_img
    patch_mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized_mask
    patch_depth[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized_depth

    transform = np.array(
        [[scale, 0.0, pad_x - scale * x0], [0.0, scale, pad_y - scale * y0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return patch_img, patch_mask, patch_depth, transform


def normalize_rgb_patch(rgb_patch):
    """Normalize an RGB patch like SCFlow2's test pipeline."""

    patch = rgb_patch.astype(np.float32)
    patch = (patch - DEFAULT_NORMALIZE_MEAN.reshape(1, 1, 3)) / DEFAULT_NORMALIZE_STD.reshape(1, 1, 3)
    return np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32)


def extract_first_pose_m(outputs):
    """Extract the first refined pose from SCFlow2 forward outputs."""

    rotations = outputs["rotations"]
    translations = outputs["translations"]
    if isinstance(rotations, (list, tuple)):
        rotations = rotations[0]
    if isinstance(translations, (list, tuple)):
        translations = translations[0]

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(rotations):
        rotations = rotations.detach().cpu().numpy()
    if torch is not None and torch.is_tensor(translations):
        translations = translations.detach().cpu().numpy()

    rotations = np.asarray(rotations)
    translations = np.asarray(translations)
    if rotations.ndim == 3:
        rotations = rotations[0]
    if translations.ndim == 2:
        translations = translations[0]
    return pose_from_scflow2_fields(rotations, translations)


class SCFlow2OnlineRefiner:
    """Single-object online SCFlow2 refiner.

    The wrapper constructs the same high-level batch fields that SCFlow2's
    refiner uses at test time, without writing BOP-style intermediate files.
    """

    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.rng = np.random.default_rng(0)
        self.mesh_cache = {}
        self.placeholder_mesh_dir = Path(tempfile.mkdtemp(prefix="scflow2_placeholder_mesh_"))

        if not self.config_path.exists():
            raise FileNotFoundError(f"SCFlow2 config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SCFlow2 checkpoint not found: {self.checkpoint_path}")

        self.scflow2_root = self.config_path.resolve().parents[2]
        if str(self.scflow2_root) not in sys.path:
            sys.path.insert(0, str(self.scflow2_root))

        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from models import build_refiner

        self.torch = torch
        self.cfg = Config.fromfile(str(self.config_path))
        self._prepare_placeholder_renderer_mesh()
        with force_cpu_default_tensor_type(self.torch):
            self.model = build_refiner(self.cfg.model)
            fp16_cfg = self.cfg.get("fp16", None)
            if fp16_cfg is not None:
                wrap_fp16_model(self.model)
            if hasattr(self.model, "load_checkpoint"):
                self.model.load_checkpoint([str(self.checkpoint_path)])
            else:
                load_checkpoint(self.model, str(self.checkpoint_path), map_location=device)
        self.model.to(device)
        self.model.eval()

    def _prepare_placeholder_renderer_mesh(self):
        """Give SCFlow2's renderer a valid mesh path until the real mesh is known."""

        mesh_path = self.placeholder_mesh_dir / "obj_000001.ply"
        mesh_path.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "element face 1",
                    "property list uchar int vertex_indices",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "3 0 1 2",
                    "",
                ]
            ),
            encoding="ascii",
        )
        if "renderer" in self.cfg.model:
            self.cfg.model.renderer.mesh_dir = str(self.placeholder_mesh_dir)

    def _ensure_single_mesh_renderer(self, mesh):
        mesh_key = id(mesh)
        if mesh_key in self.mesh_cache:
            self.model.renderer.meshes = self.mesh_cache[mesh_key]["meshes"]
            return

        import tempfile
        from pytorch3d.io import load_objs_as_meshes

        mesh_dir = Path(tempfile.mkdtemp(prefix="scflow2_online_mesh_"))
        mesh_path = mesh_dir / "obj_000001.obj"
        render_mesh = mesh.copy()
        render_mesh.vertices = np.asarray(render_mesh.vertices, dtype=np.float32) * 1000.0
        render_mesh.export(mesh_path)
        with force_cpu_default_tensor_type(self.torch):
            pytorch3d_mesh = load_objs_as_meshes([str(mesh_path)], device="cpu")
        meshes = {0: pytorch3d_mesh.to(self.device)}
        self.model.renderer.meshes = meshes
        self.mesh_cache[mesh_key] = {"mesh_dir": mesh_dir, "meshes": meshes}

    def _make_batch(self, rgb, depth_m, K, mask, pose_m, mesh):
        import torch

        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        mask = np.asarray(mask).astype(bool)
        if int(mask.sum()) < 1:
            raise ValueError("SCFlow2 refine skipped: empty mask")

        rotation, translation_mm = pose_m_to_scflow2_fields(pose_m)
        bbox = project_mesh_bbox(mesh, K, rotation, translation_mm, rgb.shape)
        depth_mm = (depth_m * 1000.0).astype(np.float32)
        patch_rgb, patch_mask, patch_depth_mm, transform = crop_resize_pad(rgb, mask, depth_mm, bbox)
        patch_K = transform @ K
        model_points_m = sample_model_points_m(mesh, rng=self.rng)
        scene_cloud_m = sample_scene_cloud(depth_m, K, mask, rng=self.rng)

        img_tensor = torch.from_numpy(normalize_rgb_patch(patch_rgb)).to(self.device)
        depth_tensor = torch.from_numpy(patch_depth_mm.astype(np.float32)).to(self.device)
        rotation_tensor = torch.from_numpy(rotation[None]).to(self.device)
        translation_tensor = torch.from_numpy(translation_mm[None]).to(self.device)
        label_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        k_tensor = torch.from_numpy(patch_K[None].astype(np.float32)).to(self.device)
        ori_k_tensor = torch.from_numpy(K.astype(np.float32)).to(self.device)
        transform_tensor = torch.from_numpy(transform[None].astype(np.float32)).to(self.device)
        model_tensor = torch.from_numpy(model_points_m[None]).to(self.device)
        cloud_tensor = torch.from_numpy(scene_cloud_m[None]).to(self.device)

        img_meta = {
            "img_path": "",
            "ori_shape": rgb.shape,
            "img_shape": patch_rgb.shape,
            "img_norm_cfg": {
                "mean": DEFAULT_NORMALIZE_MEAN,
                "std": DEFAULT_NORMALIZE_STD,
                "to_rgb": True,
            },
            "scale_factor": np.ones((1, 4), dtype=np.float32),
            "keypoints_3d": [np.asarray(mesh.bounding_box.vertices, dtype=np.float32) * 1000.0],
            "geometry_transform_mode": "adapt_intrinsic",
            "transform_matrix": transform[None],
            "ori_k": K,
        }

        return {
            "img": [img_tensor[None]],
            "annots": {
                "ref_rotations": [rotation_tensor],
                "ref_translations": [translation_tensor],
                "labels": [label_tensor],
                "k": [k_tensor],
                "ori_k": [ori_k_tensor],
                "transform_matrix": [transform_tensor],
                "depths": [depth_tensor[None]],
                "model_list": [model_tensor],
                "cloud_list": [cloud_tensor],
            },
            "img_metas": [img_meta],
        }

    def refine(self, rgb, depth_m, K, mask, pose_m, mesh):
        """Return a refined meter pose.

        Callers handle failures so they can log sequence/object/frame context.
        """

        self._ensure_single_mesh_renderer(mesh)
        batch = self._make_batch(rgb, depth_m, K, mask, pose_m, mesh)
        with self.torch.no_grad():
            outputs = self.model(batch, return_loss=False)
        return extract_first_pose_m(outputs)
