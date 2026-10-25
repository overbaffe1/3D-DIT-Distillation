import io
import os
import sys
from typing import Tuple

import numpy as np
import torch
import trimesh
from torchvision.utils import make_grid

from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ops import mesh as ops_mesh


def fov2focal_torch(fov: torch.Tensor, pixels: int) -> torch.Tensor:
    return pixels / (2 * torch.tan(fov / 2))


def vecrot(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    bsz = vec.shape[0]
    c0, c1, c2 = rot.view(bsz, 1, 3, 3).unbind(dim=-1)
    x, y, z = vec.view(bsz, -1, 3, 1).unbind(dim=-2)
    return (c0 * x + c1 * y + c2 * z).view(*vec.shape)

def vecrot4(vec: torch.Tensor, rot: torch.Tensor):
    B = vec.shape[0]
    c0, c1, c2, c3 = rot.view(B, 1, 4, 4).unbind(dim=-1)
    x, y, z, w = vec.view(B, -1, 4, 1).unbind(dim=-2)
    return (c0 * x + c1 * y + c2 * z + c3 * w).view(*vec.shape)

def get_camera(radius: torch.Tensor, azim: torch.Tensor, elev: torch.Tensor) -> torch.Tensor:
    camera_positions = torch.stack(
        [
            radius * torch.cos(elev) * torch.cos(azim),
            radius * torch.cos(elev) * torch.sin(azim),
            radius * torch.sin(elev),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions, device=camera_positions.device)
    lookat = torch.nn.functional.normalize(center - camera_positions, dim=-1)

    up = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=camera_positions.device)[None, :].repeat(
        len(radius), 1
    )
    parallel = (lookat * up).sum(dim=-1).abs() > 0.98
    if parallel.any():
        up[parallel] = torch.as_tensor([0, 1, 0], dtype=torch.float32, device=camera_positions.device)

    right = torch.nn.functional.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = torch.nn.functional.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def get_rays(c2ws: torch.Tensor, focal: torch.Tensor, res: int) -> Tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        indexing="ij",
    )
    y, x = y.unsqueeze(0).repeat(len(c2ws), 1, 1), x.unsqueeze(0).repeat(len(c2ws), 1, 1)
    x, y = x + 0.5, y + 0.5
    dirs_x = (x - 0.5 * res) / focal
    dirs_y = -(y - 0.5 * res) / focal
    dirs_z = -torch.ones(len(c2ws), res, res, dtype=torch.float32, device=c2ws.device)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)
    ray_dir = vecrot(ray_dir_cam, c2ws[:, :3, :3])
    ray_orig = c2ws[:, None, None, :3, 3].expand(ray_dir.shape)
    return ray_orig, ray_dir

def get_ortho_rays(c2ws, focal, res):
    y, x = torch.meshgrid(
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        indexing="ij",
    )
    y, x = y.unsqueeze(0).repeat(len(c2ws), 1, 1), x.unsqueeze(0).repeat(len(c2ws), 1, 1)
    
    if not isinstance(focal, torch.Tensor):
        focal = torch.tensor(focal, dtype=torch.float32, device=c2ws.device)
    focal = focal.view(-1, 1, 1)
    
    x, y = x + 0.5, y + 0.5
    dirs_x = (x - 0.5 * res) / focal
    dirs_y = -(y - 0.5 * res) / focal
    dirs_z = -torch.ones(len(c2ws), res, res, dtype=torch.float32, device=c2ws.device)
    
    ray_dir_cam = torch.stack([torch.zeros_like(dirs_x), torch.zeros_like(dirs_y), dirs_z], dim=-1)
    ray_dir = vecrot(ray_dir_cam, c2ws[:, :3, :3])
    ray_ori_cam = torch.stack([dirs_x, dirs_y, torch.zeros_like(dirs_z)], dim=-1)
    ray_ori = vecrot(ray_ori_cam, c2ws[:, :3, :3]) + c2ws[:, None, None, :3, 3].expand(ray_dir.shape)
    return ray_ori, ray_dir

def load_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    if os.path.splitext(mesh_path)[-1].lower() == ".npz":
        with np.load(mesh_path) as data:
            vertices, faces = data["vertices"], data["faces"]
    else:
        mesh = trimesh.load(mesh_path, force="mesh", process=False, validate=False)
        trimesh.grouping.merge_vertices(mesh, digits_vertex=8)
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if os.path.splitext(mesh_path)[-1].lower() in (".glb", ".obj"):
            vertices = vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=vertices.dtype)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) < 4 or len(faces) < 4:
        raise FileNotFoundError(mesh_path)
    v = torch.tensor(vertices, dtype=torch.float32)
    bbox_min, bbox_max = v.min(dim=0).values, v.max(dim=0).values
    shift = (bbox_max + bbox_min) / 2
    scale = 1.0 / (bbox_max - bbox_min).max().clamp(min=1e-6)
    v = (v - shift) * scale
    return v.numpy().astype(np.float32), faces.astype(np.int32)


def render_rays(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    c2w: torch.Tensor,
    fov: torch.Tensor,
    res: int = 512,
) -> dict:
    # Manual inverse for rigid transform to avoid CUDA linalg issues.
    rot = c2w[:, :3, :3]
    trans = c2w[:, :3, 3]
    rot_t = rot.transpose(-1, -2)
    w2c = torch.eye(4, device=c2w.device, dtype=c2w.dtype).unsqueeze(0).repeat(len(c2w), 1, 1)
    w2c[:, :3, :3] = rot_t
    w2c[:, :3, 3] = -(rot_t @ trans.unsqueeze(-1)).squeeze(-1)
    focal = fov2focal_torch(fov[:, None, None], res)
    ray_ori, ray_dir = get_rays(c2w, focal, res)
    ray_crd = torch.stack(
        torch.meshgrid(
            torch.arange(ray_ori.shape[-3], device=ray_ori.device, dtype=torch.int32),
            torch.arange(ray_ori.shape[-2], device=ray_ori.device, dtype=torch.int32),
            indexing="ij",
        ),
        dim=-1,
    )[None,].repeat(len(c2w), 1, 1, 1).view(-1, 2)
    ray_crd = torch.cat(
        [
            torch.arange(len(c2w), device=ray_crd.device, dtype=ray_crd.dtype)
            .repeat_interleave(len(ray_crd) // len(c2w), dim=0)[:, None],
            ray_crd,
        ],
        dim=-1,
    )
    ray_ori, ray_dir = ray_ori.reshape(-1, 3), ray_dir.reshape(-1, 3)

    orig_dist = c2w[ray_crd[:, 0], :3, 3].norm(dim=-1)
    min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5

    tri = vertices[faces].float()
    tri_normal = torch.nn.functional.normalize(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )

    with torch.no_grad():
        tri_idx, tri_depth, tri_uv = ops_mesh.cast_rays(tri, ray_ori, ray_dir, ray_crd, 1)

    layer_idx, layer_depth, layer_uv = tri_idx[0], tri_depth[0], tri_uv[0]
    layer_mask = layer_idx >= 0
    valid_idx = layer_mask.nonzero().view(-1)

    wuv = torch.cat([1 - layer_uv.sum(dim=-1, keepdim=True), layer_uv], dim=-1)

    depth = torch.where(layer_mask, layer_depth, torch.ones_like(layer_depth) * max_dist)
    depth = depth.clamp(min=min_dist, max=max_dist)
    disparity = (max_dist - depth) / (max_dist - min_dist)

    ray_rot, ray_shift = w2c[ray_crd[:, 0], :3, :4].split([3, 1], dim=-1)
    ray_shift = ray_shift.view(-1, 3)

    pnt = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
        dim=0,
        index=valid_idx.unsqueeze(-1).expand(-1, 3),
        src=(tri[layer_idx[valid_idx]] * wuv[valid_idx][:, :, None]).sum(dim=1),
    )
    cam_pnt = vecrot(pnt, ray_rot) + ray_shift

    normal = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
        dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=tri_normal[layer_idx[valid_idx]]
    )
    normal = torch.where((-ray_dir * normal).sum(dim=-1, keepdim=True) >= 0, normal, -normal)

    cam_normal = vecrot(normal, ray_rot)
    cam_normal = torch.where(
        layer_mask[:, None],
        cam_normal,
        torch.tensor([0, 0, 1], device=cam_normal.device, dtype=cam_normal.dtype).view(1, 3),
    )

    return {
        "mask": layer_mask.view(len(c2w), res, res),
        "depth": depth,
        "disparity": disparity,
        "pnt": pnt,
        "cam_pnt": cam_pnt,
        "normal": normal,
        "cam_normal": cam_normal,
    }

def render_ortho_rays(v, f, c2w, scale, res=1024):
    
    w2c = torch.linalg.inv(c2w)
    
    ray_ori, ray_dir = get_ortho_rays(c2w, res / scale, res)
    
    ray_crd = torch.stack(torch.meshgrid(
        torch.arange(ray_ori.shape[-3], device=ray_ori.device, dtype=torch.int32),
        torch.arange(ray_ori.shape[-2], device=ray_ori.device, dtype=torch.int32),
        indexing="ij"
    ), dim=-1)[None,].repeat(len(c2w), 1, 1, 1).view(-1, 2)
    ray_crd = torch.cat([
        torch.arange(len(c2w), device=ray_crd.device, dtype=ray_crd.dtype)
        .repeat_interleave(len(ray_crd) // len(c2w), dim=0)[:, None],
        ray_crd
    ], dim=-1)
    ray_ori, ray_dir = ray_ori.reshape(-1, 3), ray_dir.reshape(-1, 3)
    
    orig_dist = c2w[ray_crd[:, 0], :3, 3].norm(dim=-1)
    min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5
    
    tri = v[f].float()
    tri_normal = torch.nn.functional.normalize(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    
    with torch.no_grad():
        tri_idx, tri_depth, tri_uv = ops_mesh.cast_rays(tri, ray_ori, ray_dir, ray_crd, 1)
    
    layer_idx, layer_depth, layer_uv = tri_idx[0], tri_depth[0], tri_uv[0]
    layer_mask = layer_idx >= 0
    valid_idx = layer_mask.nonzero().view(-1)
    
    wuv = torch.cat([1 - layer_uv.sum(dim=-1, keepdim=True), layer_uv], dim=-1)
    
    
    depth = torch.where(layer_mask, layer_depth, torch.ones_like(layer_depth) * max_dist)
    depth = depth.clamp(min=min_dist, max=max_dist)
    disparity = (max_dist - depth) / (max_dist - min_dist)
    
    ray_rot, ray_shift = w2c[ray_crd[:, 0], :3, :4].split([3, 1], dim=-1)
    ray_shift = ray_shift.view(-1, 3)
    
    pnt = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
        dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3),
        src=(tri[layer_idx[valid_idx]] * wuv[valid_idx][:, :, None]).sum(dim=1)
    )
    cam_pnt = vecrot(pnt, ray_rot) + ray_shift
    
    normal = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
        dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=tri_normal[layer_idx[valid_idx]]
    )
    normal = torch.where((-ray_dir * normal).sum(dim=-1, keepdim=True) >= 0, normal, -normal)
    
    cam_normal = vecrot(normal, ray_rot)
    cam_normal = torch.where(layer_mask[:, None], cam_normal,
                             torch.tensor([0, 0, 1], device=cam_normal.device, dtype=cam_normal.dtype).view(1, 3))
    
    return {
        "mask": layer_mask.view(len(c2w), res, res),
        "depth": depth,
        "disparity": disparity,
        "pnt": pnt,
        "cam_pnt": cam_pnt,
        "normal": normal,
        "cam_normal": cam_normal,
    }

def render_mesh_grid(mesh_path: str, res: int = 2048) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vertices, faces = load_mesh(mesh_path)
    v = torch.tensor(vertices, device=device, dtype=torch.float32)
    f = torch.tensor(faces, device=device, dtype=torch.int32)

    radius = torch.ones(4, device=device, dtype=torch.float32) * 1.5
    azim = torch.arange(4, device=device, dtype=torch.float32) * 2 * torch.pi / 4
    elev = torch.ones(4, device=device, dtype=torch.float32) * 30 / 180 * torch.pi
    c2w = get_camera(radius, azim, elev)
    fov = torch.tensor(np.arctan(32 / 2 / 35) * 2, dtype=torch.float32, device=device).view(-1)
    
    obj_min, obj_max = v.amin(0), v.amax(0)
    obj_corners = torch.stack(torch.meshgrid(
        torch.tensor([obj_min[0], obj_max[0]], device=v.device),
        torch.tensor([obj_min[1], obj_max[1]], device=v.device),
        torch.tensor([obj_min[2], obj_max[2]], device=v.device),
        indexing="ij"
    ), dim=-1).reshape(-1, 3)
    
    w2c = torch.linalg.inv(c2w)
    obj_corners = torch.nn.functional.pad(obj_corners, pad=(0, 1), value=1.0)
    cam_corners = vecrot4(obj_corners.view(1, 8, 4).repeat_interleave(len(c2w), dim=0), w2c)
    ortho_scale = cam_corners[..., :2].abs().amax(dim=[1, 2]) * 2
    
    # out = render_rays(v, f, c2w, fov, res=res)
    out = render_ortho_rays(v, f, c2w, ortho_scale / 0.95, res=res)
    normals = out["cam_normal"].view(len(c2w), res, res, 3).clamp(-1, 1)
    mask = out["mask"].view(len(c2w), res, res).float()
    img = normals.mul(0.5).add(0.5)
    img = img * mask[..., None] + (1 - mask[..., None])

    grid = make_grid(img.permute(0, 3, 1, 2), nrow=2, padding=4, pad_value=1.0)
    grid = (grid.clamp(0, 1) * 255).byte().cpu().numpy()
    grid = np.transpose(grid, (1, 2, 0))


    im = Image.fromarray(grid).resize((res, res))
    return im 


def render_mesh_preview(mesh_path: str, res: int = 256) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vertices, faces = load_mesh(mesh_path)
    v = torch.tensor(vertices, device=device, dtype=torch.float32)
    f = torch.tensor(faces, device=device, dtype=torch.int32)

    radius = torch.ones(1, device=device, dtype=torch.float32) * 1.5
    azim = torch.zeros(1, device=device, dtype=torch.float32)
    elev = torch.ones(1, device=device, dtype=torch.float32) * 20 / 180 * torch.pi
    c2w = get_camera(radius, azim, elev)

    fov = torch.tensor(np.arctan(32 / 2 / 35) * 2, dtype=torch.float32, device=device).view(-1)
    
    obj_min, obj_max = v.amin(0), v.amax(0)
    obj_corners = torch.stack(torch.meshgrid(
        torch.tensor([obj_min[0], obj_max[0]], device=v.device),
        torch.tensor([obj_min[1], obj_max[1]], device=v.device),
        torch.tensor([obj_min[2], obj_max[2]], device=v.device),
        indexing="ij"
    ), dim=-1).reshape(-1, 3)
    
    w2c = torch.linalg.inv(c2w)
    obj_corners = torch.nn.functional.pad(obj_corners, pad=(0, 1), value=1.0)
    cam_corners = vecrot4(obj_corners.view(1, 8, 4).repeat_interleave(len(c2w), dim=0), w2c)
    ortho_scale = cam_corners[..., :2].abs().amax(dim=[1, 2]) * 2
    
    # out = render_rays(v, f, c2w, fov, res=res)
    out = render_ortho_rays(v, f, c2w, ortho_scale / 0.95, res=res)
    normals = out["cam_normal"].view(1, res, res, 3).clamp(-1, 1)
    mask = out["mask"].view(1, res, res).float()
    img = normals.mul(0.5).add(0.5)
    img = img * mask[..., None] + (1 - mask[..., None])

    grid = (img[0].permute(2, 0, 1).clamp(0, 1) * 255).byte().cpu().numpy()
    grid = np.transpose(grid, (1, 2, 0))


    im = Image.fromarray(grid)
    return im 


def render_mesh_previews(mesh_path: str, res: int = 224) -> bytes:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vertices, faces = load_mesh(mesh_path)
    v = torch.tensor(vertices, device=device, dtype=torch.float32)
    f = torch.tensor(faces, device=device, dtype=torch.int32)

    radius = torch.ones(4, device=device, dtype=torch.float32) * 1.5
    azim = torch.arange(4, device=device, dtype=torch.float32) * 2 * torch.pi / 4
    elev = torch.ones(1, device=device, dtype=torch.float32) * 20 / 180 * torch.pi
    c2w = get_camera(radius, azim, elev)
    fov = torch.tensor(np.arctan(32 / 2 / 35) * 2, dtype=torch.float32, device=device).view(-1)
    
    obj_min, obj_max = v.amin(0), v.amax(0)
    obj_corners = torch.stack(torch.meshgrid(
        torch.tensor([obj_min[0], obj_max[0]], device=v.device),
        torch.tensor([obj_min[1], obj_max[1]], device=v.device),
        torch.tensor([obj_min[2], obj_max[2]], device=v.device),
        indexing="ij"
    ), dim=-1).reshape(-1, 3)
    
    w2c = torch.linalg.inv(c2w)
    obj_corners = torch.nn.functional.pad(obj_corners, pad=(0, 1), value=1.0)
    cam_corners = vecrot4(obj_corners.view(1, 8, 4).repeat_interleave(len(c2w), dim=0), w2c)
    ortho_scale = cam_corners[..., :2].abs().amax(dim=[1, 2]) * 2
    
    out = render_ortho_rays(v, f, c2w, ortho_scale / 0.95, res=res)
    # out = render_rays(v, f, c2w, fov, res=res)
    normals = out["cam_normal"].view(len(c2w), res, res, 3).clamp(-1, 1)
    mask = out["mask"].view(len(c2w), res, res).float()
    img = normals.mul(0.5).add(0.5)
    img = img * mask[..., None] + (1 - mask[..., None])

    grid = make_grid(img.permute(0, 3, 1, 2), nrow=4, padding=4, pad_value=1.0)
    grid = (grid.clamp(0, 1) * 255).byte().cpu().numpy()
    grid = np.transpose(grid, (1, 2, 0))

    im = Image.fromarray(grid)
    return im 
