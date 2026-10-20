import os
import math
import numpy as np
import torch

try:
    from . import custom_ops # usual app run
    from . import sparse as sparse_ops
except:
    import custom_ops # run in current directory
    import sparse as sparse_ops

VERBOSE = True
_libmesh = None


def _init():
    global _libmesh
    if _libmesh is None:
        cuda_flags = []
        try:
            import libmesh as _libmesh
        except ImportError:
            _libmesh = custom_ops.load_ext(
                "libmesh",
                headers=["cutil_math.h", "bvh.h", "mesh.h"],
                sources=[
                    "dd.cu",
                    "bvh.cu",
                    "rays.cu",
                    "fmc.cu",
                    "simplify.cu",
                    "repair.cu",
                    "geo.cu",
                    "sdf.cu",
                    "libmesh.cpp",
                ],
                source_root=os.path.join(os.path.dirname(__file__), "libmesh"),
                verbose=VERBOSE,
                optimize=False,
                cuda_flags=cuda_flags,
            )
    return True


class FlexibleMC_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vpos, vsdf, cube_vidx, iso: float = 0.0):
        
        vpos = vpos.float().contiguous()
        vsdf = vsdf.float().contiguous()
        cube_vidx = cube_vidx.int().contiguous()
        
        verts, faces, edge_idx = _libmesh.fmc_forward(vpos, vsdf, cube_vidx, iso)
        
        ctx.save_for_backward(vpos, vsdf, edge_idx)
        ctx.iso = float(iso)
        
        return verts, faces
    
    @staticmethod
    def backward(ctx, grad_verts, gF=None):
        
        grad_verts = grad_verts.float().contiguous()
        vpos, vsdf, edge_idx = ctx.saved_tensors
        grad_vpos, grad_vsdf = _libmesh.fmc_backward(grad_verts, edge_idx, vpos, vsdf, ctx.iso)
        
        return grad_vpos, grad_vsdf, None, None

def fmc(vpos, vsdf, cube2vidx, iso: float = 0.0):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    return FlexibleMC_func.apply(vpos, vsdf, cube2vidx, iso)


class RayCast_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tri: torch.Tensor, ray_ori: torch.Tensor, ray_dir: torch.Tensor, ray_crd: torch.Tensor, N_LAYERS: int):
        
        tri = (tri.view(-1, 3, 3).float() / 2.0 + 0.5).contiguous()
        ray_ori = (ray_ori.view(-1, 3).float() / 2.0 + 0.5).contiguous()
        ray_dir = (ray_dir.view(-1, 3).float() / 2).contiguous()
        ray_crd = ray_crd.view(-1, 3).int().contiguous()
        
        tri_idx, tri_depth, tri_uv = _libmesh.cast_rays(
            tri, ray_ori, ray_dir, ray_crd, N_LAYERS
        )
        
        ctx.save_for_backward(tri, ray_ori, ray_dir, tri_idx)
        ctx.N_LAYERS = N_LAYERS
        
        return tri_idx, tri_depth, tri_uv
    
    @staticmethod
    def backward(ctx, grad_idx, grad_depth, grad_uv):
        
        tri, ray_ori, ray_dir, tri_idx = ctx.saved_tensors
        
        grad_tri = _libmesh.cast_rays_bwd(
            tri,
            ray_ori, ray_dir, ctx.N_LAYERS,
            tri_idx,
            grad_depth.float().contiguous(), grad_uv.float().contiguous()
        )
        grad_tri.div_(2.0)
        
        return grad_tri, None, None, None, None


def cast_rays(tri: torch.Tensor, # [-1, 1]
              ray_ori: torch.Tensor,
              ray_dir: torch.Tensor,
              ray_crd: torch.Tensor,
              N_LAYERS: int = 1):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    assert len(tri) > 0, 'empty mesh for rendering'
    
    return RayCast_func.apply(tri, ray_ori, ray_dir, ray_crd, N_LAYERS)


class RayCastRadius_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tri, ray_ori, ray_dir, ray_crd, K, sigma):
        
        sigma = sigma / 2
        radius = sigma
        # radius = math.sqrt(-math.log(1e-4) * sigma * sigma)
        
        tri = (tri.view(-1, 3, 3) / 2 + 0.5).contiguous()
        ray_ori = (ray_ori.view(-1, 3) / 2 + 0.5).contiguous()
        ray_dir = (ray_dir.view(-1, 3) / 2).contiguous()
        ray_crd = ray_crd.view(-1, 3).int().contiguous()
        idx, depth, uv, alpha, hard_idx = _libmesh.cast_radius_rays(tri, ray_ori, ray_dir, ray_crd, K, radius, sigma)
        ctx.save_for_backward(tri, ray_ori, ray_dir, idx, hard_idx)
        ctx.sigma = sigma
        return idx, hard_idx, alpha, depth, uv
    
    @staticmethod
    def backward(ctx, grad_idx, grad_hard_idx, grad_alpha, grad_depth, grad_uv):
        tri, ray_ori, ray_dir, idx, hard_idx = ctx.saved_tensors
        grad_tri = _libmesh.cast_radius_rays_bwd(tri, ray_ori, ray_dir, idx, hard_idx, grad_depth.float().contiguous(), grad_uv.float().contiguous(), grad_alpha.float().contiguous(), ctx.sigma)
        grad_tri.div_(2.0)
        return grad_tri, None, None, None, None, None, None


def cast_radius_rays(
        tri: torch.Tensor, # [-1, 1]
        ray_ori: torch.Tensor,
        ray_dir: torch.Tensor,
        ray_crd: torch.Tensor,
        K: int = 1, sigma: float = 0.01
):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    assert len(tri) > 0, 'empty mesh for rendering'
    
    return RayCastRadius_func.apply(tri, ray_ori, ray_dir, ray_crd, K, sigma)


def pcd_attn(
        keypoints,
        points,
        head_dim: int,
        key_feat=None,
        point_kfeat=None,
        point_vfeat=None,
        K: int = 256,
        radius: float = 2 * np.sqrt(3),
        scale: float = None,
        weights=None,
        bias=None,
        key_extras=None,
        point_extras=None,
        coord_scale: float = 1.0,
        rel_clamp=None,
        use_wmma: bool = False,
        split_k_bwd: int = 1,
        split_q_bwd: int = 1,
        proj_share_group: int = 1,
        nn_idx=None,
        nn_count=None,
):
    return sparse_ops.pcd_attn(
        keypoints=keypoints,
        points=points,
        head_dim=head_dim,
        key_feat=key_feat,
        point_kfeat=point_kfeat,
        point_vfeat=point_vfeat,
        K=K,
        radius=radius,
        scale=scale,
        weights=weights,
        bias=bias,
        key_extras=key_extras,
        point_extras=point_extras,
        coord_scale=coord_scale,
        rel_clamp=rel_clamp,
        use_wmma=use_wmma,
        split_k_bwd=split_k_bwd,
        split_q_bwd=split_q_bwd,
        proj_share_group=proj_share_group,
        nn_idx=nn_idx,
        nn_count=nn_count,
    )


def pcd_knn(keypoints, points, K: int, radius: float = 0.0):
    return sparse_ops.pcd_knn(keypoints, points, K, radius)


def pcd_vox_knn(keypoints, points, grid_res: int, dilation: int):
    return sparse_ops.pcd_vox_knn(keypoints, points, grid_res, dilation)


def get_sdf(points, v, f, watertight=True, return_attr=False, rand_rays=100, ext_rays=5):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    # assume points/vertices in [-1, 1]
    scale = 2 * v.abs().max().item()
    
    tri = v[f.view(-1).int()].view(len(f), 3, 3)
    sdf, tri_idx, tri_uv = _libmesh.sdf(points.float() / scale + 0.5, tri.float() / scale + 0.5, watertight, rand_rays, ext_rays)
    if return_attr:
        return sdf * scale, tri_idx, tri_uv
    return sdf * scale


def get_flow_sdf(v, f, grid_res):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    # assume points/vertices in [-1, 1]
    # scale = 2 * v.abs().max().item()
    assert v.abs().max().item() <= 1.0
    scale = 2
    tri = v[f.view(-1).int()].view(len(f), 3, 3) / scale + 0.5
    sdf = _libmesh.flow_sdf(tri.float(), grid_res)
    return sdf * scale


def up_tsdf(sdf, threshold=0.99):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.up_tsdf(sdf.float().clamp(-1, 1).view(-1).contiguous(), threshold)


def get_dist(points, v, f, efficient=True, watertight=True, return_attr=False):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    scale = 2 * v.abs().max().item()
    tri = v[f.view(-1).int()].view(len(f), 3, 3)
    dist, tri_idx, tri_uv = _libmesh.dist(points.float() / scale + 0.5, tri.float() / scale + 0.5, efficient)
    if return_attr:
        return dist * scale, tri_idx, tri_uv
    return dist * scale

def get_sign(points, v, f, efficient=True, watertight=True, rand_rays=100, ext_rays=5):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    scale = 2 * v.abs().max().item()
    tri = v[f.view(-1).int()].view(len(f), 3, 3)
    return _libmesh.sign(points.float() / scale + 0.5, tri.float() / scale + 0.5, efficient, watertight, rand_rays, ext_rays)


def get_render_sign(points, depth, w2cs, fovs):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    # points (n 3)
    # depth (b h w)
    # w2cs (b 4 4)
    # fovs (b,)
    return _libmesh.render_sign(points, depth, w2cs, fovs)


def project_features(coords, features, w2cs, fov):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    # coords (b n 3)
    # features (b h w n c)
    # w2cs (b n 4 4)
    # fov (b, n)
    return _libmesh.project_features(coords, features, w2cs, fov)

def local_features(v, f, n_base=256, n_surface=200000):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    # v (n 3)
    # f (m 3)
    
    # surface coverage
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    farea = torch.norm(face_normals, dim=-1).clamp(min=1e-8) * 0.5
    cdf = farea.cumsum(dim=0)
    cdf = cdf / cdf[-1]
    
    face_indices = torch.searchsorted(cdf, torch.rand(n_surface, device=cdf.device, dtype=cdf.dtype)).clamp(max=len(f) - 1)
    
    uu, vv = torch.rand(n_surface, device=v.device, dtype=v.dtype), torch.rand(n_surface, device=v.device, dtype=v.dtype)
    pcd = (
            (1 - torch.sqrt(uu))[..., None] * v0[face_indices]
            + (torch.sqrt(uu) * (1 - vv))[..., None] * v1[face_indices]
            + (torch.sqrt(uu) * vv)[..., None] * v2[face_indices]
    )
    nrm = torch.nn.functional.normalize(face_normals[face_indices], dim=-1)
    
    # base points
    from torch_cluster import fps
    fps_idx = fps(
        pcd,
        ratio=float(n_base / len(pcd))
    )[:n_base]
    base_pcd, base_nrm = pcd[fps_idx], nrm[fps_idx]
    
    # cover distance
    dist = (base_pcd[:, None] - base_pcd[None, :]).norm(dim=-1).sort(dim=-1).values[:, 1].max().item() / 2
    
    return _libmesh.local_features(base_pcd.contiguous(), base_nrm.contiguous(), dist, v.contiguous(), f.contiguous(), farea.quantile(0.95).item())


def get_components(faces):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    labels = torch.arange(len(faces), dtype=torch.int32, device=faces.device)
    edges = torch.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=-2).view(-1, 2).sort(dim=-1).values
    edges, face2edge, cnt = torch.unique(edges, dim=0, return_inverse=True, return_counts=True)
    edge2face = labels.repeat_interleave(3)[torch.argsort(face2edge)]
    estart = torch.cat([torch.zeros_like(cnt[:1]), cnt.cumsum(0)[:-1]])
    edge2face = torch.stack([
        edge2face[estart],
        edge2face[estart + cnt.clamp(max=2) - 1]
    ], dim=-1)
    face2edge, edge2face = face2edge.view(-1, 3), edge2face.view(-1, 2)
    face2face = edge2face[face2edge]
    face2face = torch.where(face2face[:, :, 0] != labels[:, None], face2face[:, :, 0], face2face[:, :, 1])
    
    _libmesh.face_propagate(labels, face2face.contiguous())
    
    sorted_labels, indices = torch.sort(labels)
    peaks = torch.cat([torch.tensor([True], dtype=torch.bool, device=labels.device), sorted_labels[1:] != sorted_labels[:-1]])
    comps = torch.split(indices, torch.bincount(peaks.int().cumsum(dim=0) - 1).tolist())
    
    return comps

def get_oriented_components(verts, faces, thr=math.radians(180) - 1e-6):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    face_label = _libmesh.get_oriented_components(verts.float().contiguous(), faces.int().contiguous(), thr)
    face_idxs = torch.argsort(face_label)
    labels, labels_cnt = torch.unique_consecutive(face_label[face_idxs], return_counts=True)
    comps = torch.split(face_idxs, labels_cnt.tolist())
    
    return comps


def get_mesh_intersections(verts, faces, tol=1e-8):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    scale = 2 * verts.abs().max().item()
    intersections = _libmesh.mesh_intersection(verts.float().contiguous() / scale + 0.5, faces.int().contiguous(), tol)
    return intersections


def tri2vox(verts, faces, grid_res=256):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    # assume verts in [-1, 1]
    verts = (verts / 2.0 + 0.5).clamp(0, 1)
    
    return _libmesh.voxels_from_triangles(verts[faces].float().contiguous(), grid_res)


def pcd2vox(pcd, grid_res=256):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    # assume pcd in [-1, 1]
    pcd = (pcd / 2.0 + 0.5).clamp(0, 1)
    
    return _libmesh.voxels_from_points(pcd.float().contiguous(), grid_res)

def active_voxels(sdf):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    return _libmesh.active_voxels(sdf.float().contiguous())

def near_voxels(dist, thr):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    return _libmesh.near_voxels(dist.float().contiguous(), thr)


@torch._dynamo.disable
def dilate_coords(coords, batch_size: int = 1, dilate: int = 1, grid_res: int = 1024, return_old: bool = True):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    coords = coords.int().contiguous()
    vox = _libmesh.dilate_coords(coords, batch_size, dilate, grid_res)
    if return_old:
        return vox.nonzero()
    bi, i0, i1, i2 = coords.unbind(dim=-1)
    vox[bi, i0, i1, i2] = False
    return vox.nonzero()

def exterior_voxels(cube_idx: torch.Tensor, batch_size: int = 1, grid_res: int = 1024):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    cube_idx = cube_idx.int().contiguous()
    ext_vox = _libmesh.exterior_voxels(cube_idx, batch_size, grid_res)
    return ext_vox

def find_corner_points(vidx, cube_idx, batch_size: int = 1, grid_res: int = 1024):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    vidx = vidx.int().contiguous()
    cube_idx = cube_idx.int().contiguous()
    corner_label = _libmesh.find_corner_points(vidx, cube_idx, batch_size, grid_res)
    return corner_label


def mark_corners_from_exterior_voxels(vidx: torch.Tensor, ext_vox: torch.Tensor, grid_res: int = 1024):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    vidx = vidx.int().contiguous()
    ext_mask = _libmesh.mark_corners_from_exterior_voxels(vidx, ext_vox, grid_res)
    return ext_mask

def simplify(verts, faces, n_faces=200000, max_iters=10000):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    if len(faces) < 4 or len(verts) < 3:
        return verts, faces
    
    verts, faces = _libmesh.simplify(verts.float().clone().contiguous(), faces.int().clone().contiguous(), n_faces, max_iters)
    return verts, faces


def fill_holes(verts, faces):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    verts, faces = _libmesh.fill_holes(verts.float().contiguous(), faces.int().contiguous())
    return verts, faces

def orient_mesh(verts, faces, n_rays=32, n_bounce=8, return_rays=False):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    
    scale = 2 * verts.abs().max().item()
    tri = verts[faces] / scale + 0.5
    tri = tri.float().contiguous()
    nrm_cnt, inv_cnt = _libmesh.orient_rays(tri, tri, n_rays, n_bounce).unbind(dim=-1)
    if return_rays:
        return nrm_cnt, inv_cnt # return only count
    
    comps = get_oriented_components(verts, faces, thr=math.radians(15))
    inv_comps = [c for c in comps if (nrm_cnt[c] - inv_cnt[c]).sum(dtype=torch.int32).item() < 0]
    for c in inv_comps:
        faces[c] = faces[c][:, [0, 2, 1]]
    
    return verts, faces

def vertex_faces(faces):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.vertex_faces(faces.int().contiguous())

def add_first_item_pair(item2item, off):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.add_first_item(item2item.int().contiguous(), off.int().contiguous())

def geo_neighbours(verts, faces, radius: float = 0.0):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.geo_neighbours(verts.float().contiguous(), faces.int().contiguous(), radius)

def topo_neighbours(faces):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.topo_neighbours(faces.int().contiguous())

def guidance_normals(verts, faces, face2face, foff):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.guidance_normals(verts.float().contiguous(), faces.int().contiguous(), face2face.int().contiguous(), foff.int().contiguous())


def neighbour_faces_ori_scores(verts, faces, face_normals, face2face, foff, sigma_dist: float = 1e-3, sigma_nrm: float = math.radians(5)):
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.neighbour_faces_ori_scores(verts.float().contiguous(), faces.int().contiguous(),
                                               face_normals.float().contiguous(),
                                               face2face.int().contiguous(), foff.int().contiguous(), sigma_dist, sigma_nrm)

def vert2cube(v_idx: torch.Tensor, grid_res: int = 1024):
    
    if not _init():
        raise RuntimeError(
            f"Cannot load libmesh in {__file__} set VERBOSE=True for more information!"
        )
    return _libmesh.vert2cube(v_idx.int().contiguous(), grid_res)


def gather_features(pcd, feat_idx, features, grid_res):
    # pcd (n 3)
    # feat_idx (n 3)
    # features (b h w c)
    
    i, j, k = ((pcd/2+0.5) * grid_res).floor().clamp(max=grid_res-1).long().unbind(dim=-1)
    codes = i * grid_res * grid_res + j * grid_res + k
    codes, inv_codes, cnts = codes.unique(return_counts=True, return_inverse=True)
    indexes = torch.stack([
        codes.div(grid_res * grid_res, rounding_mode='trunc'),
        (codes % (grid_res * grid_res)).div(grid_res, rounding_mode='trunc'),
        codes % grid_res
    ], dim=-1)
    
    coord_features = torch.zeros(len(indexes), features.shape[-1], device=features.device, dtype=features.dtype)
    # memory intensive in case of large feature dim
    coord_features.index_put_((inv_codes,), features[feat_idx[:, 0], feat_idx[:, 1], feat_idx[:, 2]], accumulate=True)
    coord_features /= cnts.float()[:, None]
    
    return indexes, coord_features


for _name in (
    "fmc",
    "cast_rays",
    "cast_radius_rays",
    "pcd_attn",
    "pcd_knn",
    "pcd_vox_knn",
    "get_sdf",
    "get_flow_sdf",
    "up_tsdf",
    "get_dist",
    "get_sign",
    "get_render_sign",
    "project_features",
    "local_features",
    "get_components",
    "get_oriented_components",
    "get_mesh_intersections",
    "tri2vox",
    "pcd2vox",
    "active_voxels",
    "near_voxels",
    "dilate_coords",
    "exterior_voxels",
    "find_corner_points",
    "mark_corners_from_exterior_voxels",
    "simplify",
    "fill_holes",
    "orient_mesh",
    "vertex_faces",
    "add_first_item_pair",
    "geo_neighbours",
    "topo_neighbours",
    "guidance_normals",
    "neighbour_faces_ori_scores",
    "vert2cube",
):
    globals()[_name] = torch._dynamo.disable(globals()[_name])

 
