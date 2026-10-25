import math
import os
from collections import defaultdict
from typing import Literal
import numpy as np
import torch
import torchvision
import trimesh
from nvdiffrast import torch as nvt
from ops.mesh import (
    get_sdf,
    get_dist,
    get_sign,
    get_components,
    get_oriented_components,
    simplify,
    tri2vox,
    pcd2vox,
    orient_mesh,
    cast_rays,
    cast_radius_rays,
)
from sklearn.neighbors import KDTree
from torch_scatter import scatter_mean, scatter_add


def normalize_vector(a):
    return a / (a.norm(dim=-1, keepdim=True) + 1e-8)


def dot(x, y):
    return (normalize_vector(x) * normalize_vector(y)).sum(dim=-1)


def get_mesh_edges(faces, unique=False):
    n = torch.max(faces) + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    if unique:
        codes = torch.cat(
            [
                torch.minimum(v0, v1) * n + torch.maximum(v0, v1),
                torch.minimum(v1, v2) * n + torch.maximum(v1, v2),
                torch.minimum(v0, v2) * n + torch.maximum(v0, v2),
                ],
            dim=0,
        ).unique()
    else:
        codes = torch.cat(
            [
                v0 * n + v1,
                v1 * n + v0,
                v1 * n + v2,
                v2 * n + v1,
                v2 * n + v0,
                v0 * n + v2,
                ],
            dim=0,
        ).unique()
    
    edges = torch.stack([codes.div(n, rounding_mode="trunc"), codes % n], dim=0)
    return edges


def get_tetrahedras_edges(tets, unique=False):
    n = torch.max(tets) + 1 if len(tets) > 0 else torch.ones(1).type_as(tets)
    v0, v1, v2, v3 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
    if unique:
        codes = torch.cat(
            [
                torch.minimum(v0, v1) * n + torch.maximum(v0, v1),
                torch.minimum(v0, v2) * n + torch.maximum(v0, v2),
                torch.minimum(v0, v3) * n + torch.maximum(v0, v3),
                torch.minimum(v1, v2) * n + torch.maximum(v1, v2),
                torch.minimum(v1, v3) * n + torch.maximum(v1, v3),
                torch.minimum(v2, v3) * n + torch.maximum(v2, v3),
                ],
            dim=0,
        ).unique()
    else:
        codes = torch.cat(
            [
                v0 * n + v1,
                v1 * n + v0,
                v0 * n + v2,
                v2 * n + v0,
                v0 * n + v3,
                v3 * n + v0,
                v1 * n + v2,
                v2 * n + v1,
                v1 * n + v3,
                v3 * n + v1,
                v2 * n + v3,
                v3 * n + v2,
                ],
            dim=0,
        ).unique()
    
    edges = torch.stack([codes.div(n, rounding_mode="trunc"), codes % n], dim=0)
    return edges


def laplace_smoothing(vertices, edges):
    values = torch.zeros_like(vertices)
    norms = torch.zeros_like(vertices)
    
    v0, v1 = edges[0], edges[1]
    neighbour_vertices = vertices[v1]
    values = values.index_put_((v0,), neighbour_vertices, accumulate=True)
    norms = norms.index_put_(
        (v0,), torch.ones_like(neighbour_vertices), accumulate=True
    )
    values = values / torch.clamp(norms, min=1.0)
    return values


def laplace_loss_mesh(vertices, faces):
    edges = get_mesh_edges(faces)
    values = laplace_smoothing(vertices, edges)
    loss = torch.mean(torch.sum((vertices - values) ** 2, dim=-1))
    return loss


def laplace_loss_tets(vertices, tets):
    edges = get_tetrahedras_edges(tets)
    values = laplace_smoothing(vertices, edges)
    loss = torch.mean(torch.sum((vertices - values) ** 2, dim=-1))
    return loss


def get_mesh_edges_with_extra_vertex(faces):
    n = torch.max(faces) + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    v0, v1, v2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    codes = torch.cat(
        [
            torch.minimum(v0, v1) * n * n + torch.maximum(v0, v1) * n + v2,
            torch.minimum(v1, v2) * n * n + torch.maximum(v1, v2) * n + v0,
            torch.minimum(v0, v2) * n * n + torch.maximum(v0, v2) * n + v1,
            ],
        dim=0,
    ).unique(sorted=True)
    
    edges = torch.stack(
        [
            codes.div(n * n, rounding_mode="trunc"),
            (codes % (n * n)).div(n, rounding_mode="trunc"),
            codes % n,
            ],
        dim=0,
    )
    return edges


def get_mesh_edges_with_face_index(faces):
    
    n = torch.max(faces).long() + 1 if len(faces) > 0 else torch.ones(1).type_as(faces)
    m = torch.ones_like(n).long() * len(faces)
    
    v0, v1, v2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    face_index = torch.arange(len(faces), device=faces.device, dtype=torch.long)
    
    codes = (
        torch.cat(
            [
                torch.minimum(v0, v1) * n * m + torch.maximum(v0, v1) * m + face_index,
                torch.minimum(v1, v2) * n * m + torch.maximum(v1, v2) * m + face_index,
                torch.minimum(v0, v2) * n * m + torch.maximum(v0, v2) * m + face_index,
                ],
            dim=0,
        )
        .clamp(min=0)
        .unique(sorted=True)
    )
    
    edges = torch.stack(
        [
            codes.div(n * m, rounding_mode="trunc") % n,
            (codes % (n * m)).div(m, rounding_mode="trunc"),
            codes % m,
            ],
        dim=0,
    )
    return edges


def smoothness_loss(vertices, faces):
    edges = get_mesh_edges_with_extra_vertex(faces).transpose(0, 1)
    indexes = torch.arange(len(edges) - 1).type_as(edges)[
        torch.all(torch.eq(edges[:-1, :2], edges[1:, :2]), dim=1)
    ]
    indexes = torch.stack([indexes, indexes + 1], dim=1).view(-1)
    edges = edges[indexes].view(
        -1, 2, 3
    )  # two consequent edges it's same edge from different faces
    
    v0, v1 = vertices[edges[:, 0, 0]], vertices[edges[:, 0, 1]]
    v2, v3 = vertices[edges[:, 0, 2]], vertices[edges[:, 1, 2]]
    
    first_normal = torch.cross(v1 - v0, v2 - v0, dim=1)
    first_normal = first_normal / (first_normal.norm(dim=-1, keepdim=True) + 1e-15)
    
    # change sign as we want to keep same orientation
    second_normal = -torch.cross(v1 - v0, v3 - v0, dim=1)
    second_normal = second_normal / (second_normal.norm(dim=-1, keepdim=True) + 1e-15)
    
    cos = torch.sum(first_normal * second_normal, dim=-1)
    loss = torch.mean((1 - cos) ** 2)
    return loss


@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
def get_grads(
        normalized_points,
        forward_density,
        sdf=None,
        approx_type="finite_difference_laplacian",
        eps=0.01,
        second_order=False,
        num_samples="full",
):
    
    b, total_points, _ = normalized_points.shape
    if approx_type == "finite_difference":
        offsets = torch.as_tensor(
            [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
        ).to(normalized_points)
        points_offset = normalized_points[..., None, :] + offsets
        offset = forward_density(points_offset.view(b, total_points * 3, 3)).view(
            b, total_points, 3
        )
        grad = (offset - sdf.view(b, total_points, 1)) / eps
    elif approx_type == "finite_difference4":
        eps = eps  # / 3 ** 0.5
        x = normalized_points
        k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)
        k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)
        k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)
        k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)
        
        if num_samples == "single":
            sdf1 = forward_density(x + k1 * eps).view(b, total_points, 1).float()
            sdf2 = forward_density(x + k2 * eps).view(b, total_points, 1).float()
            sdf3 = forward_density(x + k3 * eps).view(b, total_points, 1).float()
            sdf4 = forward_density(x + k4 * eps).view(b, total_points, 1).float()
        else:
            sdf1, sdf2, sdf3, sdf4 = (
                forward_density(
                    torch.stack(
                        [x + k1 * eps, x + k2 * eps, x + k3 * eps, x + k4 * eps], dim=-2
                    ).view(b, total_points * 4, 3)
                )
                .view(b, total_points, 4, 1)
                .float()
                .unbind(dim=-2)
            )
        
        grad = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
        if second_order:
            grad2 = (
                            (sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf.view(b, total_points, 1)
                    ) / eps**2
            grad2 = torch.cat([grad2] * 3, dim=-1) / 3
            return grad, grad2
    else:  # approx_type=="finite_difference_laplacian":
        x = normalized_points
        offsets = torch.as_tensor(
            [
                [eps, 0.0, 0.0],
                [-eps, 0.0, 0.0],
                [0.0, eps, 0.0],
                [0.0, -eps, 0.0],
                [0.0, 0.0, eps],
                [0.0, 0.0, -eps],
            ]
        ).to(normalized_points)
        if num_samples == "single":
            sdf1 = forward_density(x + offsets[0]).view(b, total_points, 1)
            sdf2 = forward_density(x + offsets[1]).view(b, total_points, 1)
            sdf3 = forward_density(x + offsets[2]).view(b, total_points, 1)
            sdf4 = forward_density(x + offsets[3]).view(b, total_points, 1)
            sdf5 = forward_density(x + offsets[4]).view(b, total_points, 1)
            sdf6 = forward_density(x + offsets[5]).view(b, total_points, 1)
            grad = torch.cat(
                [
                    (sdf1 - sdf2) / (2 * eps),
                    (sdf3 - sdf4) / (2 * eps),
                    (sdf5 - sdf6) / (2 * eps),
                    ],
                dim=-1,
            )
            if second_order:
                grad2 = torch.cat(
                    [
                        (sdf1 + sdf2 - 2 * sdf.view(b, total_points, 1)) / eps**2,
                        (sdf3 + sdf4 - 2 * sdf.view(b, total_points, 1)) / eps**2,
                        (sdf5 + sdf6 - 2 * sdf.view(b, total_points, 1)) / eps**2,
                        ]
                )
                return grad, grad2
        else:
            points_offset = normalized_points[..., None, :] + offsets
            offset = forward_density(points_offset.view(b, total_points * 6, 3)).view(
                b, total_points, 3, 2
            )
            grad = (offset[..., 0] - offset[..., 1]) / (2 * eps)
            if second_order:
                grad2 = (
                                offset[..., 0] + offset[..., 1] - 2 * sdf.view(b, total_points, 1)
                        ) / eps**2
                return grad, grad2
    
    return grad


def grad_on_tets(f, x, tets):
    b, n, _ = x.shape
    
    f, x = f.view(-1), x.view(-1, 3)
    
    n_verts = tets.shape[-1]
    vs = (
        (
                tets.unsqueeze(0).repeat(b, 1, 1)
                + torch.arange(b).view(b, 1, 1).type_as(tets) * n
        )
        .view(-1, n_verts)
        .unbind(dim=-1)
    )
    grads, denom = torch.zeros_like(x), torch.zeros_like(x)
    for idx, v1 in enumerate(vs):
        # other vertices
        v2, v3, v4 = (vs[(idx + shift) % n_verts] for shift in range(1, n_verts))
        # function diff for this vertex
        diff = torch.stack([f[v2] - f[v1], f[v3] - f[v1], f[v4] - f[v1]], dim=-1)  # n 3
        # point diff for this vertex
        delta = torch.stack(
            [x[v2] - x[v1], x[v3] - x[v1], x[v4] - x[v1]], dim=-2
        )  # n 3 3
        grad = torch.bmm(torch.inverse(delta), diff.unsqueeze(-1)).squeeze(-1)  # n 3
        
        grads.index_put_((v1,), grad, accumulate=True)
        denom.index_put_((v1,), torch.ones_like(grad), accumulate=True)
    grads /= denom.clamp(min=1)
    return grads.view(b, n, 3)


def grad2_on_tets(f, x, tets, df=None):
    if df is None:
        df = grad_on_tets(f, x, tets)
    ddf = torch.stack(
        [
            grad_on_tets(df[..., 0], f, tets)[..., 0],
            grad_on_tets(df[..., 1], f, tets)[..., 1],
            grad_on_tets(df[..., 2], f, tets)[..., 2],
        ],
        dim=-1,
    )
    return ddf


def grad_on_grid(f, x, grid_res, second_order=True):
    b, n, _ = x.shape
    
    f, x = f.view(b, grid_res + 1, grid_res + 1, grid_res + 1), x.view(
        b, grid_res + 1, grid_res + 1, grid_res + 1, 3
    )
    
    grad = torch.stack(
        [
            (f[:, 2:, 1:-1, 1:-1] - f[:, :-2, 1:-1, 1:-1])
            / (x[:, 2:, 1:-1, 1:-1] - x[:, :-2, 1:-1, 1:-1]).norm(dim=-1),
            (f[:, 1:-1, 2:, 1:-1] - f[:, 1:-1, :-2, 1:-1])
            / (x[:, 1:-1, 2:, 1:-1] - x[:, 1:-1, :-2, 1:-1]).norm(dim=-1),
            (f[:, 1:-1, 1:-1, 2:] - f[:, 1:-1, 1:-1, :-2])
            / (x[:, 1:-1, 1:-1, 2:] - x[:, 1:-1, 1:-1, :-2]).norm(dim=-1),
            ],
        dim=-1,
    )
    # borders have no grad with such method (maybe add finite difference?)
    grad = torch.nn.functional.pad(grad, (0, 0, 1, 1, 1, 1, 1, 1), value=0)
    grad = grad.view(b, n, 3)
    
    if second_order:
        grad2 = torch.stack(
            [
                (
                        f[:, 2:, 1:-1, 1:-1]
                        + f[:, :-2, 1:-1, 1:-1]
                        - 2 * f[:, 1:-1, 1:-1, 1:-1]
                )
                / (x[:, 2:, 1:-1, 1:-1] - x[:, :-2, 1:-1, 1:-1])
                .norm(dim=-1)
                .div(2)
                .pow(2),
                (
                        f[:, 1:-1, 2:, 1:-1]
                        + f[:, 1:-1, :-2, 1:-1]
                        - 2 * f[:, 1:-1, 1:-1, 1:-1]
                )
                / (x[:, 1:-1, 2:, 1:-1] - x[:, 1:-1, :-2, 1:-1])
                .norm(dim=-1)
                .div(2)
                .pow(2),
                (
                        f[:, 1:-1, 1:-1, 2:]
                        + f[:, 1:-1, 1:-1, :-2]
                        - 2 * f[:, 1:-1, 1:-1, 1:-1]
                )
                / (x[:, 1:-1, 1:-1, 2:] - x[:, 1:-1, 1:-1, :-2])
                .norm(dim=-1)
                .div(2)
                .pow(2),
                ],
            dim=-1,
                )
        grad2 = torch.nn.functional.pad(grad2, (0, 0, 1, 1, 1, 1, 1, 1), value=0)
        grad2 = grad2.view(b, n, 3)
        return grad, grad2
    
    return grad


def sdf_from_grad(f, x, grad, tets, anneal_ratio=1.0):
    b, n, _ = x.shape
    
    f, x, grad = f.view(-1), x.view(-1, 3), grad.view(-1, 3)
    
    v1, v2, v3, v4 = (
        (
                tets.unsqueeze(0).repeat(b, 1, 1)
                + torch.arange(b).view(b, 1, 1).type_as(tets) * n
        )
        .view(-1, 4)
        .unbind(dim=-1)
    )
    values, denom = torch.zeros_like(f), torch.zeros_like(f)
    
    base_value = f[v1]
    for v_next in (v2, v3, v4):
        vector = x[v_next] - x[v1]
        
        f_next = base_value + (grad[v1] * vector).sum(dim=-1) * anneal_ratio
        if anneal_ratio < 1.0:
            # use sdf of sphere on initialization
            f_next = f_next + (1.0 - anneal_ratio) * (
                    x[v_next].norm(dim=-1, p=2) - 0.5 - f_next
            )
        values.index_put_((v_next,), f_next, accumulate=True)
        denom.index_put_((v_next,), torch.ones_like(f_next), accumulate=True)
    
    values /= denom.clamp(min=1)
    return values.view(b, n, 1)


def viscosity_sdf_reg(sdf, sdf_points=None, tets=None, eps=1e-2, second_order=True):
    if tets is not None:
        df = grad_on_tets(sdf, sdf_points, tets)
    else:
        if not sdf.requires_grad:
            return torch.tensor(0).type_as(sdf)
        df = torch.autograd.grad(
            sdf.sum(),
            sdf_points,
            retain_graph=True,
            allow_unused=True,
            create_graph=second_order,
        )[0]
    
    if second_order:
        if tets is not None:
            ddf = grad2_on_tets(sdf, sdf_points, tets, df=df)
        else:
            ddf = torch.stack(
                [
                    torch.autograd.grad(
                        df[..., 0].sum(),
                        sdf_points,
                        retain_graph=True,
                        allow_unused=True,
                    )[0][:, 0],
                    torch.autograd.grad(
                        df[..., 1].sum(),
                        sdf_points,
                        retain_graph=True,
                        allow_unused=True,
                    )[0][:, 1],
                    torch.autograd.grad(
                        df[..., 2].sum(),
                        sdf_points,
                        retain_graph=True,
                        allow_unused=True,
                    )[0][:, 2],
                ],
                dim=-1,
            )
        loss = torch.mean(
            (
                    (torch.norm(df.view(-1, 3), dim=-1) - 1) * torch.sign(sdf.view(-1))
                    - eps * torch.sum(ddf.view(-1, 3), dim=-1)
            )
            ** 2
        )
    else:
        # simple eikonal loss \norm{\grad f}} = 1
        loss = torch.mean((torch.norm(df.view(-1, 3), dim=-1) - 1) ** 2)
    
    return loss


def coarea_sdf_reg(sdf, sdf_points=None, tets=None, beta=1e-4):
    if tets is not None:
        df = grad_on_tets(sdf, sdf_points, tets)
    else:
        if not sdf.requires_grad:
            return torch.tensor(0).type_as(sdf)
        df = torch.autograd.grad(
            sdf.sum(), sdf_points, retain_graph=True, allow_unused=True
        )[0]
    loss = torch.mean(
        torch.exp(-torch.abs(sdf.view(-1)) / beta - np.log(2 * beta))
        * torch.norm(df.view(-1, 3), dim=-1)
    )
    return loss


def get_center_boundary_index(verts):
    length_ = torch.sum(verts**2, dim=-1)
    center_idx = torch.argmin(length_)
    boundary_neg = verts == verts.max()
    boundary_pos = verts == verts.min()
    boundary = torch.bitwise_or(boundary_pos, boundary_neg)
    boundary = torch.sum(boundary.float(), dim=-1)
    boundary_idx = torch.nonzero(boundary)
    return center_idx, boundary_idx.squeeze(dim=-1)


def load_mesh(obj_path, device):
    if os.path.splitext(obj_path)[-1] == ".npz":
        data = np.load(obj_path)
        v, f = (
            torch.from_numpy(data["vertices"]).float().to(device).contiguous(),
            torch.from_numpy(data["faces"]).int().to(device).contiguous(),
        )
    else:
        mesh = trimesh.load(obj_path, force="mesh", process=False, validate=False)
        v, f = torch.tensor(
            mesh.vertices, device=device, dtype=torch.float32
        ), torch.tensor(mesh.faces, device=device, dtype=torch.int32)
    bbox_min, bbox_max = v.min(dim=0).values, v.max(dim=0).values
    v -= (bbox_max + bbox_min) / 2
    bbox_min, bbox_max = v.min(dim=0).values, v.max(dim=0).values
    v = v / (bbox_max - bbox_min).max(dim=0).values
    if os.path.splitext(obj_path)[-1] == ".glb":
        v = torch.matmul(
            v,
            torch.tensor(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=v.dtype, device=v.device
            ),
        )
    return v, f


def get_diffmc_grid(grid_res, device, bbox=1.01, return_indices=True):
    verts = torch.stack(
        torch.meshgrid(
            torch.linspace(-bbox, bbox, grid_res + 1, device=device),
            torch.linspace(-bbox, bbox, grid_res + 1, device=device),
            torch.linspace(-bbox, bbox, grid_res + 1, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).view(-1, 3)
    if not return_indices:
        return verts
    # enumerate all vertices to cubes
    indexes = torch.stack(
        torch.meshgrid(
            torch.arange(grid_res, device=device),
            torch.arange(grid_res, device=device),
            torch.arange(grid_res, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).view(-1, 3)
    indices = torch.stack(
        [
            (indexes[:, 0] + int(idx & 1 > 0))
            + (indexes[:, 1] + int(idx & 2 > 0)) * (grid_res + 1)
            + (indexes[:, 2] + int(idx & 4 > 0)) * (grid_res + 1) ** 2
            for idx in range(8)
        ],
        dim=-1,
    ).long()
    return verts, indices


@torch.no_grad()
# @torch.compile()
def sample_surface(
        verts: torch.Tensor, f: torch.Tensor, n_samples: int, use_edges=False, salient_rate=1, use_verts=False, thr=20
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if len(f) < 1:
        points = torch.nn.functional.normalize(
            torch.randn(n_samples, 3, dtype=verts.dtype, device=verts.device)
        )
        return points, points
    
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    weight = torch.norm(face_normals, dim=-1) * 0.5
    cdf = weight.cumsum(dim=0)
    cdf = cdf / cdf[-1]
    
    samples = torch.rand(n_samples, device=weight.device, dtype=weight.dtype)
    face_indices = torch.searchsorted(cdf, samples).clamp(max=len(f) - 1)
    
    u, v = torch.rand_like(samples), torch.rand_like(samples)
    points = (
            (1 - torch.sqrt(u))[..., None] * v0[face_indices]
            + (torch.sqrt(u) * (1 - v))[..., None] * v1[face_indices]
            + (torch.sqrt(u) * v)[..., None] * v2[face_indices]
    )
    
    normals = torch.nn.functional.normalize(face_normals[face_indices], dim=-1)
    
    if use_edges:
        # find valid edges which have opposite
        # edges = get_mesh_edges_with_face_index(f)
        # first_index = torch.arange(
        #     edges.shape[1] - 1, device=edges.device, dtype=torch.long
        # )[(edges[0, :-1] == edges[0, 1:]) & (edges[1, :-1] == edges[1, 1:])]
        # if len(first_index) < 1:
        #     return points, normals
        # edges = torch.stack(
        #     [
        #         edges[0, first_index],
        #         edges[1, first_index],
        #         edges[2, first_index],
        #         edges[2, first_index + 1],
        #     ]
        # )
        
        labels = torch.arange(len(f), dtype=torch.int32, device=f.device)
        edges = torch.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=-2).view(-1, 2).sort(dim=-1).values
        edges, face2edge, cnt = torch.unique(edges, dim=0, return_inverse=True, return_counts=True)
        edge2face = labels.repeat_interleave(3)[torch.argsort(face2edge)]
        estart = torch.cat([torch.zeros_like(cnt[:1]), cnt.cumsum(0)[:-1]])
        edge2face = torch.stack([
            edge2face[estart],
            edge2face[estart + cnt.clamp(max=2) - 1]
        ], dim=-1)
        
        # filter only high curvature edges
        angle = (
                torch.acos(
                    torch.nn.functional.cosine_similarity(
                        face_normals[edge2face[:, 0]], face_normals[edge2face[:, 1]], dim=-1
                    ).clamp(min=-1 + 1e-6, max=1 - 1e-6)
                )
                / torch.pi
                * 180
        )
        angle = angle.nan_to_num().clamp(min=0, max=135)
        # at least 15 degrees angle and top 25% edges by curvature
        curve_mask = angle >= thr
        if not curve_mask.any().item():
            return points, normals
        # if len(angle) > n_samples:
        #     curve_mask = curve_mask & (angle >= -(-angle).kthvalue(n_samples).values)
        edges, edge2face, angle = edges[curve_mask], edge2face[curve_mask], angle[curve_mask]
        
        edge_weight = (torch.rand(len(edges), device=edges.device, dtype=torch.float32) < 0.5).float()[:, None]
        edge_normals = torch.nn.functional.normalize(
            # (
            #     torch.nn.functional.normalize(face_normals[edges[2]], dim=-1)
            #     + torch.nn.functional.normalize(face_normals[edges[3]], dim=-1)
            # )
            # / 2,
            face_normals[edge2face[:, 0]] * edge_weight + face_normals[edge2face[:, 1]] * (1 - edge_weight),
            dim=-1,
            )
        
        n_salient = n_samples // (1 + salient_rate) * salient_rate
        if use_verts:
            curve_verts, inverse_edges = torch.unique(
                torch.cat([edges[0], edges[1]]), return_inverse=True
            )
            verts_pcd = verts[curve_verts]
            verts_normals = torch.zeros_like(verts_pcd)
            verts_normals.index_put_(
                (inverse_edges,), torch.cat([edge_normals] * 2), accumulate=True
            )
            verts_normals = torch.where(
                verts_normals.norm(dim=-1, keepdim=True) > 1e-6,
                verts_normals,
                torch.tensor(
                    [0, 0, 1], dtype=verts_normals.dtype, device=verts_normals.device
                )
                .reshape(1, 3)
                .expand(*verts_normals.shape),
                )
            verts_normals = torch.nn.functional.normalize(verts_normals, dim=-1)
            
            if len(verts_pcd) >= n_salient // 2:
                # from torch_cluster import fps
                # fps_idx = fps(
                #     verts_pcd,
                #     ratio=float(verts_cnt / len(verts_pcd))
                # )[:verts_cnt]
                fps_idx = torch.randperm(
                    len(verts_pcd), device=verts_pcd.device, dtype=torch.long
                )[: n_salient // 2]
                verts_pcd, verts_normals = verts_pcd[fps_idx], verts_normals[fps_idx]
            
            n_salient -= len(verts_pcd)
        
        # sample by edge length
        esize = (verts[edges[:, 0]] - verts[edges[:, 1]]).norm(dim=-1)
        # eweight = (angle / thr) ** 0.3
        eweight = torch.ones_like(esize)
        cdf = (esize * eweight).cumsum(dim=0)
        cdf = cdf / cdf[-1]
        edge_indices = torch.searchsorted(
            cdf, torch.rand(n_salient, device=cdf.device, dtype=cdf.dtype)
        ).clamp(max=len(edges) - 1)
        edges, edge2face = edges[edge_indices], edge2face[edge_indices]
        
        u = torch.rand(len(edge_indices), device=verts.device, dtype=verts.dtype)[
            ..., None
        ]
        if use_verts:
            verts_pcd = torch.cat(
                [verts[edges[:, 0]] * u + (1 - u) * verts[edges[:, 1]], verts_pcd]
            )
            verts_normals = torch.cat([edge_normals[edge_indices], verts_normals])
        else:
            verts_pcd, verts_normals = (
                verts[edges[:, 0]] * u + (1 - u) * verts[edges[:, 1]],
                edge_normals[edge_indices],
            )
        
        # interleave points to take same ratio
        points = torch.cat(
            [
                points[: n_samples // (1 + salient_rate)].reshape(-1, 1, 3),
                verts_pcd.reshape(-1, salient_rate, 3),
            ],
            dim=1,
        ).reshape(-1, 3)
        normals = torch.cat(
            [
                normals[: n_samples // (1 + salient_rate)].reshape(-1, 1, 3),
                verts_normals.reshape(-1, salient_rate, 3),
            ],
            dim=1,
        ).reshape(-1, 3)
    
    return points, normals


def sample_surface_edges(
        v,
        f,
        n_samples,
        edge_radius,
        edge_dotproduct_threshold,
        edge_ratio_sample,
        edge_ratio,
):
    
    if len(f) < 1:
        points = torch.nn.functional.normalize(
            torch.randn(n_samples, 3, dtype=v.dtype, device=v.device)
        )
        return points, points
    
    mesh = trimesh.base.Trimesh(v.cpu().numpy(), f.cpu().numpy())
    
    points, indexs = mesh.sample(n_samples * edge_ratio_sample, return_index=True)
    normals = mesh.face_normals[indexs]
    
    tree = KDTree(points)
    
    indslist = tree.query_radius(points, edge_radius)
    flags = np.zeros([len(points)], np.bool_)
    for p in range(len(points)):
        inds = indslist[p]
        if len(inds) > 0:
            this_normals = normals[p : p + 1]
            neighbor_normals = normals[inds]
            dotproduct = np.abs(np.sum(this_normals * neighbor_normals, axis=1))
            if np.any(dotproduct < edge_dotproduct_threshold):
                flags[p] = True
    
    edge_points = np.ascontiguousarray(points[flags])
    edge_normals = np.ascontiguousarray(normals[flags])
    
    remain_num_samples = n_samples - len(edge_points)
    
    if remain_num_samples < 0:
        
        extra = np.random.uniform(0, len(edge_points), size=int(n_samples * edge_ratio))
        edge_points = edge_points[np.isin(np.arange(len(edge_points)), extra)]
        edge_normals = edge_normals[np.isin(np.arange(len(edge_normals)), extra)]
        
        remain_num_samples = n_samples - len(edge_points)
    
    remain_points, remain_indexs = mesh.sample(remain_num_samples, return_index=True)
    remain_normals = mesh.face_normals[remain_indexs]
    
    edge_points = np.concatenate((edge_points, remain_points), axis=0)
    edge_normals = np.concatenate((edge_normals, remain_normals), axis=0)
    
    edge_points = torch.tensor(edge_points, dtype=v.dtype, device=v.device)
    edge_normals = torch.tensor(edge_normals, dtype=v.dtype, device=v.device)
    
    return edge_points, edge_normals


def fov2focal_torch(fov, pixels):
    return pixels / (2 * torch.tan(fov / 2))


def focal2fov_torch(focal, pixels):
    return 2 * torch.atan(pixels / (2 * focal))


def get_camera(radius, azim, elev, center=None):
    camera_positions = torch.stack(
        [
            radius * torch.cos(elev) * torch.cos(azim),
            radius * torch.cos(elev) * torch.sin(azim),
            radius * torch.sin(elev),
            ],
        dim=-1,
    )
    if center is None:
        center = torch.zeros_like(camera_positions, device=camera_positions.device)
    
    lookat = torch.nn.functional.normalize(center - camera_positions, dim=-1)
    up = torch.as_tensor(
        [0, 0, 1], dtype=torch.float32, device=camera_positions.device
    )[None, :].repeat(len(radius), 1)
    # up = up + torch.randn_like(up) * 1e-3
    up = torch.where(1 - lookat[:, 2:].abs() > 1e-3, up, torch.as_tensor(
        [-1, 0, 0], dtype=torch.float32, device=camera_positions.device
    )[None, :].repeat(len(radius), 1))
    
    right = torch.nn.functional.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = torch.nn.functional.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def projection_matrix(
        fovx,
        fovy,
        near=0.1,
        far=1000,
        n=1.0,
):
    focal_x = torch.tan(fovx / 2)
    focal_y = torch.tan(fovy / 2)
    z, e = torch.zeros_like(focal_x), torch.ones_like(focal_x)
    row1 = torch.stack([n / focal_x, z, z, z], dim=-1)
    row2 = torch.stack([z, n / focal_y * (-1), z, z], dim=-1)
    row3 = torch.stack(
        [z, z, -(near + far) / (far - near) * e, -2 * far * near / (far - near) * e],
        dim=-1,
    )
    row4 = torch.stack([z, z, -e, z], dim=-1)
    matrix = torch.stack([row1, row2, row3, row4], dim=-2)
    return matrix


@torch.compile()
def vecrot(vec: torch.Tensor, rot: torch.Tensor):
    B = vec.shape[0]
    c0, c1, c2 = rot.view(B, 1, 3, 3).unbind(dim=-1)
    x, y, z = vec.view(B, -1, 3, 1).unbind(dim=-2)
    return (c0 * x + c1 * y + c2 * z).view(*vec.shape)

@torch.compile()
def vecrot4(vec: torch.Tensor, rot: torch.Tensor):
    B = vec.shape[0]
    c0, c1, c2, c3 = rot.view(B, 1, 4, 4).unbind(dim=-1)
    x, y, z, w = vec.view(B, -1, 4, 1).unbind(dim=-2)
    return (c0 * x + c1 * y + c2 * z + c3 * w).view(*vec.shape)


def get_rays(c2ws, focal, res):
    y, x = torch.meshgrid(
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        torch.linspace(0, res - 1, res, dtype=torch.float32, device=c2ws.device),
        indexing="ij",
    )
    y, x = y.unsqueeze(0).repeat(len(c2ws), 1, 1), x.unsqueeze(0).repeat(len(c2ws), 1, 1)
    # x, y = x + torch.rand_like(x), y + torch.rand_like(y)
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

@torch.no_grad()
def sample_camera(device, n_renders=4, radius=2.0, n_base=4):
    # sample camera
    dist = radius + torch.rand(
        max(n_renders - 2 * n_base, 1), device=device, dtype=torch.float32
    ) * radius / 2
    azim = torch.rand(
        max(n_renders - 2 * n_base, 1), device=device, dtype=torch.float32
    ) * 2 * torch.pi
    elev = torch.rand(
        max(n_renders - 2 * n_base, 1), device=device, dtype=torch.float32
    ) * torch.pi - torch.pi / 2
    if n_base > 0:
        dist = torch.cat([
            torch.ones(n_base, device=device, dtype=torch.float32) * radius,
            dist,
            ])[:n_renders]
        base_azim = torch.arange(n_base, device=device, dtype=torch.float32) * 2 * torch.pi / n_base
        base_elev = torch.ones(n_base, device=device, dtype=torch.float32) * 30 / 180 * torch.pi
        if n_base > 1:
            base_azim = torch.cat([
                base_azim, torch.pi / n_base + torch.arange(n_base, device=device, dtype=torch.float32) * 2 * torch.pi / n_base
            ])
            base_elev = torch.cat([
                base_elev, torch.ones(n_base, device=device, dtype=torch.float32) * -30 / 180 * torch.pi
            ])
        azim = torch.cat([base_azim[:2 * n_base], azim])[:n_renders]
        elev = torch.cat([base_elev[:2 * n_base], elev])[:n_renders]
    c2ws = get_camera(dist, azim, elev)
    
    return c2ws


def sample_render(
        v,
        f,
        ctx,
        res=512,
        n_renders=8,
        radius=2.0,
        fov=0.8575560548920328,
        n_base=4,
        split_size=32,
        extra_cameras=None,
        rot=None,
        scale=None,
):
    
    c2ws = sample_camera(v.device, n_renders, radius, n_base)
    if rot is not None:
        c2ws = rot[None,] @ c2ws
    if scale is not None:
        c2ws[..., :3, 3] *= scale
    
    if extra_cameras is not None:
        c2ws = torch.cat([c2ws, extra_cameras])
    w2cs = torch.linalg.inv(c2ws)
    
    # project
    fov = torch.tensor(fov, dtype=torch.float32, device=v.device)
    proj_mtx = projection_matrix(fov, fov)
    
    # face normals
    v0, v1, v2 = v[f[:, 0], :], v[f[:, 1], :], v[f[:, 2], :]
    face_normal = torch.nn.functional.normalize(
        torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
    )
    face_normal_indices = torch.arange(
        0, face_normal.shape[0], dtype=torch.int32, device=face_normal.device
    )[:, None].repeat(1, 3)
    
    out = defaultdict(list)
    for __w2cs, __c2ws in zip(
            w2cs.split(split_size, dim=0), c2ws.split(split_size, dim=0)
    ):
        v_pos = torch.matmul(
            torch.nn.functional.pad(v[None,], pad=(0, 1), mode="constant", value=1.0),
            torch.transpose(__w2cs, 1, 2),
        )
        v_pos_clip = torch.matmul(v_pos, proj_mtx.transpose(0, 1)[None,])
        out["v_pos_clip"].append(v_pos_clip)
        
        # rasterize
        rast, _ = nvt.rasterize(ctx, v_pos_clip, f, (res, res), grad_db=False)
        hard_mask = rast[..., -1] > 0
        out["rast"].append(rast), out["hard_mask"].append(hard_mask)
        
        __face_normal, _ = nvt.interpolate(
            face_normal[None, ...], rast, face_normal_indices
        )
        out["normal"].append(__face_normal)
        
        # world_pos, _ = nvt.interpolate(v, rast, f)
        # view_dir = c2ws[:, None, None, :3, 3] - world_pos
        # face_normal = torch.where((face_normal * view_dir).sum(dim=-1, keepdim=True) >= 0, face_normal, -face_normal)
        
        # make point cloud by depth back-projection
        orig_dist = __c2ws[:, :3, 3].norm(dim=-1).view(-1, 1, 1)
        min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5
        out["radius"].append(orig_dist.view(-1))
        
        camera_pos, _ = nvt.interpolate(v_pos[..., :3].contiguous(), rast, f)
        depth = -camera_pos[..., 2]
        depth = torch.where(hard_mask, depth, torch.ones_like(depth) * max_dist)
        depth = depth.clamp(min=min_dist, max=max_dist)
        out["depth"].append(depth)
    out = {k: torch.cat(items, dim=0) for k, items in out.items()}
    out["c2ws"] = c2ws
    
    return out


def flyover_render(v, f, ctx, res=512, n_renders=8, radius=2.0, fov=0.8575560548920328):
    
    # sample camera
    radius = torch.ones(n_renders, device=v.device, dtype=torch.float32) * radius
    azim = (
            torch.arange(n_renders, device=v.device, dtype=torch.float32)
            * 2
            * torch.pi
            / n_renders
    )
    elev = (
            torch.ones(n_renders, device=v.device, dtype=torch.float32) * 0 / 180 * torch.pi
    )
    c2ws = get_camera(radius, azim, elev)
    w2cs = torch.linalg.inv(c2ws)
    
    # project
    v_pos = torch.matmul(
        torch.nn.functional.pad(v[None,], pad=(0, 1), mode="constant", value=1.0),
        torch.transpose(w2cs, 1, 2),
    )
    fov = torch.tensor(fov, dtype=torch.float32, device=v.device)
    proj_mtx = projection_matrix(fov, fov)
    v_pos_clip = torch.matmul(v_pos, proj_mtx.transpose(0, 1)[None,])
    
    # rasterize
    rast, _ = nvt.rasterize(ctx, v_pos_clip, f, (res, res), grad_db=False)
    hard_mask = rast[..., -1] > 0
    
    # face normals
    v0, v1, v2 = v[f[:, 0], :], v[f[:, 1], :], v[f[:, 2], :]
    face_normal = torch.nn.functional.normalize(
        torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
    )
    face_normal_indices = torch.arange(
        0, face_normal.shape[0], dtype=torch.int32, device=face_normal.device
    )[:, None].repeat(1, 3)
    face_normal, _ = nvt.interpolate(face_normal[None, ...], rast, face_normal_indices)
    
    # world_pos, _ = nvt.interpolate(v, rast, f)
    # view_dir = c2ws[:, None, None, :3, 3] - world_pos
    # face_normal = torch.where((face_normal * view_dir).sum(dim=-1, keepdim=True) >= 0, face_normal, -face_normal)
    
    # make point cloud by depth back-projection
    camera_pos, _ = nvt.interpolate(v_pos[..., :3].contiguous(), rast, f)
    depth = -camera_pos[..., 2]
    orig_dist = c2ws[:, :3, 3].norm(dim=-1).view(-1, 1, 1)
    min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5
    depth = torch.where(hard_mask, depth, torch.ones_like(depth) * max_dist)
    depth = depth.clamp(min=min_dist, max=max_dist)
    
    return {
        "rast": rast,
        "c2ws": c2ws,
        "hard_mask": hard_mask,
        "depth": depth,
        "normal": face_normal,
        "radius": orig_dist.view(-1),
    }


def sample_surface_render(
        v,
        f,
        n_samples,
        ctx,
        render=None,
        res=512,
        n_renders=16,
        radius=2.625,
        fov=0.8575560548920328,
):
    
    focal = fov2focal_torch(
        torch.tensor(fov, dtype=torch.float32, device=v.device), res
    )
    if render is None:
        render = sample_render(
            v, f, ctx, res=res, n_renders=n_renders, radius=radius, fov=fov
        )
    hard_mask = render["hard_mask"]
    
    visible_faces = torch.zeros(len(f), device=f.device, dtype=torch.int32)
    if torch.sum(hard_mask).item() > 0:
        visible_faces[render["rast"][..., -1][hard_mask].long() - 1] = 1
    else:
        visible_faces = torch.ones_like(visible_faces)
    
    ray_ori, ray_dir = get_rays(render["c2ws"], focal, res)
    points = ray_ori + ray_dir * render["depth"][..., None]
    
    # sample from surface point cloud
    pcd, nrm = points[hard_mask], render["normal"][hard_mask]
    if len(pcd) == 0:
        pcd = torch.nn.functional.normalize(
            torch.randn(n_samples, 3, dtype=v.dtype, device=v.device)
        )
        nrm = pcd
    
    indexes = torch.randperm(len(pcd), device=pcd.device, dtype=torch.int64)[:n_samples]
    if len(indexes) < n_samples:
        indexes = indexes.repeat((n_samples + len(indexes) - 1) // len(indexes))[
            :n_samples
        ]
    
    return (
        pcd[indexes],
        torch.nn.functional.normalize(nrm[indexes], dim=-1),
        f[visible_faces > 0],
    )

@torch.compile()
def charbonier(diff, eps=1e-3):
    return torch.sqrt(diff.square() + eps ** 2).mean()

@torch.compile()
def huber(diff, eps=1.0):
    with torch.no_grad():
        mask = (diff.abs() < eps).float()
    return (
            mask * (0.5 * diff.square()) +
            (1 - mask) * eps * (diff.abs() - 0.5 * eps)
    ).mean()

def surface_loss(verts, faces, gt_verts, gt_faces, n_samples=65534, use_faces=True, use_edges=False, use_verts=False, tp="l2"):
    
    tri = gt_verts[gt_faces].float().contiguous()
    
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fnorm = torch.linalg.vector_norm(face_normals, dim=-1, dtype=torch.float32).clamp(min=1e-8)
    face_normals = (face_normals / fnorm[:, None]).type_as(face_normals)
    
    pcd, nrm = [], []
    
    if use_faces:
        with torch.no_grad():
            cdf = fnorm.cumsum(dim=0)
            face_indices = torch.searchsorted(
                cdf, torch.rand(n_samples // 2, device=cdf.device, dtype=cdf.dtype) * cdf[-1]
            ).clamp(max=len(faces) - 1)
            uu = torch.sqrt(torch.rand(len(face_indices), device=verts.device, dtype=torch.float32))
            vv = torch.rand(len(face_indices), device=verts.device, dtype=torch.float32)
            w0, w1, w2 = 1 - uu, uu * (1 - vv), uu * vv
        
        pcd_face = (
                w0.type_as(verts)[:, None] * v0[face_indices]
                + w1.type_as(verts)[:, None] * v1[face_indices]
                + w2.type_as(verts)[:, None] * v2[face_indices]
        )
        nrm_face = face_normals[face_indices]
        pcd.append(pcd_face), nrm.append(nrm_face)
    
    if use_verts:
        nrm_vert = scatter_mean(face_normals.repeat(3, 1), torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]]).long(), dim=0,
                                out=torch.zeros(len(verts), 3, device=face_normals.device, dtype=face_normals.dtype))
        nrm_vert = torch.nn.functional.normalize(nrm_vert, dim=-1)
        pcd.append(verts), nrm.append(nrm_vert)
    
    if use_edges:
        with torch.no_grad():
            e0, e1 = torch.cat([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]).sort(dim=-1).values.unique(dim=0).unbind(dim=-1)
            cdf = (verts[e1] - verts[e0]).norm(dim=-1).cumsum(0)
            edge_indices = torch.searchsorted(
                cdf, torch.rand(n_samples // 4, device=cdf.device, dtype=cdf.dtype) * cdf[-1]
            ).clamp(max=len(e0) - 1)
            w = torch.rand(len(edge_indices), device=verts.device, dtype=torch.float32)
        
        pcd_edge = verts[e0[edge_indices]] * w[:, None].type_as(verts) + verts[e1[edge_indices]] * (1 - w[:, None]).type_as(verts)
        nrm_edge = torch.nn.functional.normalize(nrm_vert[e0[edge_indices]] + nrm_vert[e1[edge_indices]], dim=-1)
        
        pcd.append(pcd_edge), nrm.append(nrm_edge)
    
    pcd = torch.cat(pcd)
    nrm = torch.cat(nrm)
    
    # _, tri_idx, tri_uv = _libmesh.dist(pcd.float() / 2 + 0.5, tri.float() / 2 + 0.5, True)
    tri_dist, tri_idx, tri_uv = get_dist(pcd, gt_verts, gt_faces, return_attr=True)
    tri_wuv = torch.cat([1 - tri_uv.sum(dim=-1, keepdim=True), tri_uv], dim=-1)
    pcd_surf = (tri[tri_idx] * tri_wuv[:, :, None]).sum(dim=1)
    
    if tp == "l2":
        pcd_loss = (pcd - pcd_surf).square().sum(dim=-1).mean()
    else:
        pcd_loss = (pcd - pcd_surf).abs().sum(dim=-1).mean()
    
    with torch.no_grad():
        tri_normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        tri_normals = torch.nn.functional.normalize(tri_normals, dim=-1)[tri_idx]
        tri_normals = torch.where((nrm * tri_normals).sum(dim=-1, keepdim=True) >= 0, tri_normals, -tri_normals)
    nrm_loss = (1 - (nrm * tri_normals).sum(dim=-1)).square().mean()
    
    
    return {
        "pcd_loss": pcd_loss,
        "nrm_loss": nrm_loss,
    }

def mesh_developable_reg(verts, faces):
    
    
    device = verts.device
    V = verts.shape[0]
    F = faces.shape[0]
    
    POS_EPS = 1e-6
    REL_EPS = 1e-6
    
    def normalize(vecs):
        return vecs / (torch.linalg.norm(vecs, dim=-1, keepdim=True) + POS_EPS)
    
    tri_pos = verts[faces]
    
    vert_normal_covariance_sum = torch.zeros((V, 9), device=device)
    vert_area = torch.zeros(V, device=device)
    vert_degree = torch.zeros(V, dtype=torch.int32, device=device)
    
    for iC in range(3):  # loop over three corners of each triangle
        
        # gather tri verts
        pRoot = tri_pos[:, iC, :]
        pA = tri_pos[:, (iC + 1) % 3, :]
        pB = tri_pos[:, (iC + 2) % 3, :]
        
        # compute the corner angle & normal
        vA = pA - pRoot
        vAn = normalize(vA)
        vB = pB - pRoot
        vBn = normalize(vB)
        area_normal = torch.linalg.cross(vA, vB, dim=-1)
        face_area = 0.5 * torch.linalg.norm(area_normal, dim=-1)
        normal = normalize(area_normal)
        corner_angle = torch.acos(torch.clamp(torch.sum(vAn * vBn, dim=-1), min=-1., max=1.))
        
        # add up the contribution to the covariance matrix
        outer = normal[:, :, None] @ normal[:, None, :]
        contrib = corner_angle[:, None] * outer.reshape(-1, 9)
        
        # scatter the result to the appropriate matrices
        idx = faces[:, iC].long()
        vert_normal_covariance_sum = scatter_add(src=contrib,
                                                 index=idx,
                                                 dim=-2,
                                                 out=vert_normal_covariance_sum)
        
        vert_area = scatter_add(src=face_area / 3.,
                                index=idx,
                                dim=-1,
                                out=vert_area)
        
        vert_degree = scatter_add(src=torch.ones(F, dtype=torch.int32, device=device),
                                  index=idx,
                                  dim=-1,
                                  out=vert_degree)
    
    # The energy is the smallest eigenvalue of the outer-product matrix
    vert_normal_covariance_sum = vert_normal_covariance_sum.reshape(
        -1, 3, 3)  # reshape to a batch of matrices
    vert_normal_covariance_sum = vert_normal_covariance_sum + torch.eye(
        3, device=device)[None, :, :] * REL_EPS
    
    min_eigvals = torch.min(torch.linalg.eigvals(vert_normal_covariance_sum).abs(), dim=-1).values
    
    # Mask out degree-3 vertices
    vert_area = torch.where(vert_degree == 3, torch.tensor(0, dtype=vert_area.dtype,device=vert_area.device), vert_area)
    
    # Adjust the vertex area weighting so it is unit-less, and 1 on average
    vert_area = vert_area * (V / torch.sum(vert_area, dim=-1, keepdim=True))
    
    return vert_area * min_eigvals




def render_mesh(
        v,
        f,
        c2ws,
        w2cs,
        proj_mtx,
        ctx,
        res=512,
        split_size=32,
        antialias=False,
        fix_normals=False,
        use_depth=True,
        use_pos=True,
        use_normal=True,
        use_cam_normal=True,
        use_smooth_normal=True,
):
    
    # face normals
    v0, v1, v2 = v[f[:, 0], :], v[f[:, 1], :], v[f[:, 2], :]
    face_normal = torch.nn.functional.normalize(
        torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
    )
    face_normal_indices = torch.arange(
        0, face_normal.shape[0], dtype=torch.int32, device=face_normal.device
    )[:, None].repeat(1, 3)
    
    if use_smooth_normal:
        # valid only if mesh orientation defined
        smooth_normal = scatter_mean(face_normal.repeat(3, 1), torch.cat([f[:, 0], f[:, 1], f[:, 2]]).long(), dim=0,
                                     out=torch.zeros(len(v), 3, device=face_normal.device, dtype=face_normal.dtype))
        smooth_normal = torch.nn.functional.normalize(smooth_normal, dim=-1)
    
    out = defaultdict(list)
    for __w2cs, __c2ws in zip(
            w2cs.split(split_size, dim=0), c2ws.split(split_size, dim=0)
    ):
        v_pos = torch.matmul(
            torch.nn.functional.pad(v[None,], pad=(0, 1), mode="constant", value=1.0),
            torch.transpose(__w2cs, 1, 2),
        )
        v_pos_clip = torch.matmul(v_pos, proj_mtx.transpose(0, 1)[None,])
        
        # rasterize
        rast, _ = nvt.rasterize(ctx, v_pos_clip, f, (res, res), grad_db=False)
        hard_mask = rast[..., -1] > 0
        out["hard_mask"].append(hard_mask)
        
        # smooth mask
        mask = nvt.antialias(hard_mask.float()[..., None], rast, v_pos_clip, f)[..., 0]
        out["mask"].append(mask)
        
        # render depth
        cam_pos, _ = nvt.interpolate(v_pos[..., :3].contiguous(), rast, f)
        depth = -cam_pos[..., 2]
        
        if use_pos:
            bg_pos = torch.zeros_like(cam_pos)
            bg_pos[..., 2] = -(__c2ws[:, :3, 3].norm(dim=-1).view(-1, 1, 1) + np.sqrt(3))
            __cam_pos = cam_pos * hard_mask[..., None].float() + bg_pos * (1 - hard_mask[..., None].float())
            if antialias:
                __cam_pos = nvt.antialias(__cam_pos, rast, v_pos_clip, f) # may be remove for stability?
            out["cpos"].append(__cam_pos)
        
        if use_depth:
            orig_dist = __c2ws[:, :3, 3].norm(dim=-1).view(-1, 1, 1)
            min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5
            
            depth = torch.where(hard_mask, depth, torch.ones_like(depth) * max_dist)
            depth = depth.clamp(min=min_dist, max=max_dist)
            __depth = depth
            if antialias:
                __depth = nvt.antialias(__depth[..., None], rast, v_pos_clip, f)[..., 0] # may be remove for stability?
            out["depth"].append(__depth)
            
            out["disparity"].append((max_dist - __depth) / (max_dist - min_dist))
        
        # render normals
        if use_normal:
            __face_normal, _ = nvt.interpolate(face_normal[None, ...], rast, face_normal_indices)
            __face_normal = torch.nn.functional.normalize(__face_normal, dim=-1)
            __face_normal = __face_normal * hard_mask[..., None].float() + torch.zeros_like(__face_normal) * (1 - hard_mask[..., None].float())
            if fix_normals:
                w_pos = torch.matmul(
                    torch.nn.functional.pad(cam_pos, pad=(0, 1), mode="constant", value=1.0),
                    torch.transpose(__c2ws, 1, 2).view(-1, 1, 4, 4),
                )[..., :3]
                __face_normal = torch.where(torch.sum(__face_normal * (__c2ws[:, :3, 3].view(-1, 1, 1, 3) - w_pos), dim=-1, keepdim=True) >= 0, __face_normal, -__face_normal)
            if antialias:
                __face_normal = nvt.antialias(__face_normal, rast, v_pos_clip, f) # may be remove for stability??
            out["normal"].append(__face_normal)
        
        if use_cam_normal:
            __cam_normal = torch.matmul(face_normal[None,], __w2cs[:, :3, :3].transpose(1, 2))
            __cam_normal, _ = nvt.interpolate(__cam_normal, rast, face_normal_indices)
            __cam_normal = torch.nn.functional.normalize(__cam_normal, dim=-1)
            cam_bg = torch.tensor([0, 0, 1], dtype=__cam_normal.dtype, device=__cam_normal.device).view(1, 1, 1, 3)
            __cam_normal = __cam_normal * hard_mask[..., None].float() + cam_bg * (1 - hard_mask[..., None].float())
            if fix_normals:
                __cam_normal = torch.where(torch.sum(__cam_normal * (0-cam_pos), dim=-1, keepdim=True) >= 0, __cam_normal, -__cam_normal)
            if antialias:
                __cam_normal = nvt.antialias(__cam_normal, rast, v_pos_clip, f) # may be remove for stability?
            out["cam_normal"].append(__cam_normal)
        
        if use_smooth_normal:
            __smooth_normal = torch.matmul(smooth_normal[None,], __w2cs[:, :3, :3].transpose(1, 2))
            __smooth_normal, _ = nvt.interpolate(__smooth_normal, rast, f)
            # make all normals point to camera
            # cam_normal = torch.where(__smooth_normal[..., -1:] >= 0, __smooth_normal,
            # __smooth_normal * torch.tensor([1, 1, -1], dtype=__smooth_normal.dtype, device=__smooth_normal.device).view(1, 1, 1, 3))
            __smooth_normal = torch.nn.functional.normalize(__smooth_normal, dim=-1)
            cam_bg = torch.tensor([0, 0, 1], dtype=__smooth_normal.dtype, device=__smooth_normal.device).view(1, 1, 1, 3)
            __smooth_normal = __smooth_normal * hard_mask[..., None].float() + cam_bg * (1 - hard_mask[..., None].float())
            if antialias:
                __smooth_normal = nvt.antialias(__smooth_normal, rast, v_pos_clip, f) # may be remove for stability?
            out["smooth_normal"].append(__smooth_normal)
    
    
    out = {k: torch.cat(items, dim=0) if len(items) > 1 else items[0] for k, items in out.items()}
    
    return out

def render_loss(verts, faces, gt_verts, gt_faces, ctx, res=512, n_renders=16, beta=1.0/1024):
    
    # rand radius
    radius = 2.5 + torch.rand(
        n_renders, device=verts.device, dtype=torch.float32
    ) * 0.5
    # rand gaussian vec -> random camera rot
    vec = torch.nn.functional.normalize(torch.randn(n_renders, 3, device=verts.device, dtype=torch.float32), dim=-1)
    azim = (torch.atan2(vec[:, 1], vec[:, 0]) + 2 * torch.pi) % (
            2 * torch.pi
    )
    elev = torch.asin(vec[:, 2] / vec[:, :3].norm(dim=-1))
    c2ws = get_camera(radius, azim, elev)
    
    # can we use random fov?
    w2cs = torch.linalg.inv(c2ws)
    fov = torch.tensor(0.8575560548920328, dtype=torch.float32, device=verts.device)
    proj_mtx = projection_matrix(fov, fov)
    
    with torch.no_grad():
        target = render_mesh(gt_verts, gt_faces, c2ws, w2cs, proj_mtx, ctx, res=res, split_size=n_renders)
        # ray_ori, ray_dir = get_rays(c2ws, fov2focal_torch(fov, res), res)
        # target["normal"] = torch.where((-ray_dir * target["normal"]).sum(dim=-1, keepdim=True) >= 0, target["normal"], -target["normal"])
    pred = render_mesh(verts, faces, c2ws, w2cs, proj_mtx, ctx, res=res, split_size=n_renders)
    
    # we have 2 smoothing components:
    # 1) antialiasing
    # 2) multiply by smooth mask
    # which combination of them is more stable?
    mask_loss = (pred["mask"] - target["mask"]).abs().mean() # l1 mask
    pos_loss = (pred["cpos"] - target["cpos"]).square().sum(-1).add(1e-8).sqrt().mul(target["hard_mask"].float()).mean() # l2 camera pos
    depth_diff = (pred["depth"] - target["depth"]).mul(target["mask"]).abs() # smooth l1, may be use hard mask?
    depth_loss = torch.where(depth_diff < beta, 0.5 * depth_diff ** 2 / beta, depth_diff - 0.5 * beta).mean()
    normal_loss = (pred["cam_normal"] - target["cam_normal"]).mul(target["mask"][..., None]).abs().mean() # l1 for geo_normal
    smooth_loss = (pred["smooth_normal"] - target["smooth_normal"]).mul(target["mask"][..., None]).abs().mean() # l1 for smooth normal
    
    # project deformed verts to gt depth to compare their sdf?
    # bilinear torch interpolate and compare within 3/grid_res delta
    # ssim/lpips 0.2 tsdf 0.01
    # face normal 1.0 smooth normal 0.1
    
    return {
        "mask_loss": mask_loss,
        "pos_loss": pos_loss,
        "normal_loss": normal_loss,
        "smooth_loss": smooth_loss,
        "depth_loss": depth_loss,
        
        "target": target,
        "pred": pred,
    }

def render_rays(
        verts: torch.Tensor, faces: torch.Tensor,
        ray_ori: torch.Tensor, ray_dir: torch.Tensor, ray_crd: torch.Tensor,
        c2ws: torch.Tensor, w2cs: torch.Tensor, n_layers: int = 1,
        fix_normals: bool = False,
):
    
    tri = verts[faces].float()
    tri_idx, tri_depth, tri_uv = cast_rays(tri, ray_ori, ray_dir, ray_crd, n_layers)
    
    orig_dist = c2ws[ray_crd[:, 0], :3, 3].norm(dim=-1)
    min_dist, max_dist = orig_dist - 3**0.5, orig_dist + 3**0.5
    
    tri_normal = torch.nn.functional.normalize(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    # valid only for consistent orientation
    smooth_normal = scatter_mean(tri_normal.repeat(3, 1), torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]]).long(), dim=0,
                                 out=torch.zeros(len(verts), 3, device=tri_normal.device, dtype=tri_normal.dtype))
    smooth_normal = torch.nn.functional.normalize(smooth_normal, dim=-1)
    
    out = defaultdict(list)
    for lidx in range(n_layers):
        
        layer_idx, layer_depth, layer_uv = tri_idx[lidx], tri_depth[lidx], tri_uv[lidx]
        out["idx"].append(layer_idx)
        
        layer_mask = layer_idx >= 0
        valid_idx = layer_mask.nonzero().view(-1)
        out["hard_mask"].append(layer_mask)
        
        wuv = torch.cat([1 - layer_uv.sum(dim=-1, keepdim=True), layer_uv], dim=-1)
        out["wuv"].append(wuv)
        
        depth = torch.where(layer_mask, layer_depth, torch.ones_like(layer_depth) * max_dist)
        depth = depth.clamp(min=min_dist, max=max_dist)
        out["depth"].append(depth)
        
        disparity = (max_dist - depth) / (max_dist - min_dist)
        out["disparity"].append(disparity)
        
        ray_rot, ray_shift = w2cs[ray_crd[:, 0], :3, :4].split([3, 1], dim=-1)
        ray_shift = ray_shift.view(-1, 3)
        
        pnt = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3),
            src=(tri[layer_idx[valid_idx]] * wuv[valid_idx][:, :, None]).sum(dim=1)
        )
        out["cpos"].append(vecrot(pnt, ray_rot) + ray_shift)
        
        normal = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=tri_normal[layer_idx[valid_idx]]
        )
        if fix_normals:
            normal = torch.where((-ray_dir * normal).sum(dim=-1, keepdim=True) >= 0, normal, -normal)
        out["normal"].append(normal)
        
        cam_normal = vecrot(normal, ray_rot)
        cam_normal = torch.where(layer_mask[:, None], cam_normal,
                                 torch.tensor([0, 0, 1], device=cam_normal.device, dtype=cam_normal.dtype).view(1, 3))
        out["cam_normal"].append(cam_normal)
        
        smooth_normal = torch.zeros(len(layer_depth), 3, device=layer_depth.device, dtype=layer_depth.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3),
            src=torch.nn.functional.normalize((smooth_normal[faces[layer_idx[valid_idx]]] * wuv[valid_idx][:, :, None]).sum(dim=1), dim=-1)
        )
        out["smooth_normal"].append(vecrot(smooth_normal, ray_rot))
    
    out = {name: torch.stack(vals) for name, vals in out.items()}
    
    return out


def render_radius_rays(
        verts: torch.Tensor, faces: torch.Tensor,
        ray_ori: torch.Tensor, ray_dir: torch.Tensor, ray_crd: torch.Tensor,
        c2ws: torch.Tensor, w2cs: torch.Tensor, K: int = 1,
        fix_normals: bool = False, sigma=1e-3, gamma=0.0,
        compositing: Literal["transmittance", "sigmoid"] = "transmittance"
):
    
    # import lovely_tensors as lt
    
    tri = verts[faces].float()
    # print(f"tri: {lt.lovely(tri)}")
    
    if compositing == "transmittance": K = 1 # if hard intersection closest triangle would have opacity 1
    assert K == 1
    
    N_RAYS = len(ray_ori)
    N_PIECES = N_RAYS * K
    assert N_PIECES == N_RAYS
    
    tri_idx, hard_idx, alpha, depth, uv = cast_radius_rays(
        tri, ray_ori, ray_dir, ray_crd,
        1, sigma
    )
    tri_idx = tri_idx.view(-1).contiguous()
    hard_mask = hard_idx >= 0
    depth = depth.view(-1).contiguous()
    uv = uv.view(-1, 2).contiguous()
    
    alpha = alpha.view(-1).contiguous()
    # print(f"alpha: {lt.lovely(alpha)}")
    # if alpha.requires_grad:
    #     alpha.register_hook(lambda grad: print(f"alpha_grad: {lt.lovely(grad)}"))
    
    # Mask for valid hits (-1 indicates no triangle found)
    valid_mask = tri_idx >= 0
    valid_idx = valid_mask.view(-1).nonzero().view(-1)
    
    # interpolation weights
    wuv = torch.cat([1 - uv.sum(dim=-1, keepdim=True), uv], dim=-1)
    wuv = torch.where(valid_mask[:, None], wuv, torch.zeros_like(wuv))
    # print(f"wuv: {lt.lovely(wuv)}")
    # if wuv.requires_grad:
    #     wuv.register_hook(lambda grad: print(f"wuv_grad: {lt.lovely(grad)}"))
    
    # depth clamped
    ray_size = ray_ori.norm(dim=-1)
    z_near, z_far = ray_size - np.sqrt(3), ray_size + np.sqrt(3)
    depth = depth * alpha + z_far * (1 - alpha)
    depth = depth.clamp(min=z_near, max=z_far)
    # print(f"depth: {lt.lovely(depth)}")
    # if depth.requires_grad:
    #     depth.register_hook(lambda grad: print(f"depth_grad: {lt.lovely(grad)}"))
    
    # interpolate triangle normals
    tri_normal = torch.nn.functional.normalize(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    
    if len(valid_idx) > 0:
        
        valid_normal = tri_normal[tri_idx[valid_idx]]
        if fix_normals:
            valid_normal = torch.where((-ray_dir.view(-1, 3)[valid_idx] * valid_normal).sum(dim=-1, keepdim=True) >= 0, valid_normal, -valid_normal)
        
        # can we fix selection of backfaces for side edges?
        normal = torch.zeros(N_PIECES, 3, device=valid_normal.device, dtype=valid_normal.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=valid_normal
        ).view(N_RAYS, 3)
        normal = normal * alpha[:, None] + torch.tensor([0, 0, 0], dtype=normal.dtype, device=normal.device).view(1, 3) * (1 - alpha)[:, None]
        # print(f"normal: {lt.lovely(normal)}")
        
        ray_rot, ray_shift = w2cs[ray_crd[:, 0][valid_idx], :3, :4].split([3, 1], dim=-1)
        ray_shift = ray_shift.view(-1, 3)
        
        cam_normal = torch.zeros(N_PIECES, 3, device=normal.device, dtype=normal.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=vecrot(valid_normal, ray_rot)
        ).view(N_RAYS, 3)
        cam_normal = cam_normal * alpha[:, None] + torch.tensor([0, 0, 1], dtype=normal.dtype, device=normal.device).view(1, 3) * (1 - alpha)[:, None]
        # print(f"cam_normal: {lt.lovely(cam_normal)}")
        
        # valid only for consistent orientation
        vert_normal = scatter_mean(tri_normal.repeat(3, 1), torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]]).long(), dim=0,
                                   out=torch.zeros(len(verts), 3, device=tri_normal.device, dtype=tri_normal.dtype))
        vert_normal = torch.nn.functional.normalize(vert_normal, dim=-1)
        vert_normal = torch.nn.functional.normalize((vert_normal[faces[tri_idx[valid_idx]]] * wuv[valid_idx][:, :, None]).sum(dim=1), dim=-1)
        vert_normal = vecrot(vert_normal, ray_rot)
        smooth_normal = torch.zeros(N_PIECES, 3, device=vert_normal.device, dtype=vert_normal.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=vert_normal
        ).view(N_RAYS, 3)
        smooth_normal = smooth_normal * alpha[:, None] + torch.tensor([0, 0, 1], dtype=normal.dtype, device=normal.device).view(1, 3) * (1 - alpha)[:, None]
        
        # positions in camera space
        pnt = (tri[tri_idx[valid_idx]] * wuv[valid_idx][:, :, None]).sum(dim=1)
        pnt = vecrot(pnt, ray_rot) + ray_shift
        cpos = torch.zeros(N_PIECES, 3, device=pnt.device, dtype=pnt.dtype).scatter_(
            dim=0, index=valid_idx.unsqueeze(-1).expand(-1, 3), src=pnt
        ).view(N_RAYS, 3)
        cpos_bg = torch.zeros_like(cpos)
        cpos_bg[:, 2] = -z_far
        cpos = cpos * alpha[:, None] + cpos_bg * (1 - alpha)[:, None]
        # print(f"cpos: {lt.lovely(cpos)}")
    else:
        normal = torch.zeros(N_RAYS, 3, device=tri.device, dtype=tri.dtype)
        cam_normal = torch.tensor([0, 0, 1], dtype=tri.dtype, device=tri.device).view(1, 3).expand(N_RAYS, 3)
        smooth_normal = torch.tensor([0, 0, 1], dtype=tri.dtype, device=tri.device).view(1, 3).expand(N_RAYS, 3)
        cpos = torch.zeros(N_RAYS, 3, device=tri.device, dtype=tri.dtype)
        cpos[:, 2] = -z_far
    
    return {
        "indices": tri_idx,
        "hard_mask": hard_mask.float(),
        "mask": alpha,
        "depth": depth,
        "wuv": wuv,
        "disparity": (z_far - depth) / (z_far - z_near),
        "normal": normal,
        "cam_normal": cam_normal,
        "smooth_normal": smooth_normal,
        "cpos": cpos,
    }


@torch.no_grad()
def sample_ray_render(
        verts: torch.Tensor, faces: torch.Tensor,
        n_renders: int = 4, res: int = 512,
        fov: float = 0.8575560548920328, radius: float = 2.5,
        extra_cameras: torch.Tensor = None,
):
    
    fov = torch.tensor(fov, dtype=torch.float32, device=verts.device)
    foc = fov2focal_torch(fov, res)
    
    radius = radius * torch.ones(n_renders, device=verts.device, dtype=torch.float32)
    elev = torch.ones(n_renders, device=verts.device, dtype=torch.float32) * math.radians(30)
    azim = torch.arange(n_renders, device=verts.device, dtype=torch.float32) / n_renders * 2 * torch.pi
    
    c2ws = get_camera(radius, azim, elev)
    if extra_cameras is not None:
        c2ws = torch.cat([c2ws, extra_cameras], dim=0)
    w2cs = torch.linalg.inv(c2ws)
    
    
    ray_ori, ray_dir = get_rays(c2ws, foc, res)
    ray_crd = torch.stack(torch.meshgrid(
        torch.arange(ray_ori.shape[-3], device=ray_ori.device, dtype=torch.int32),
        torch.arange(ray_ori.shape[-2], device=ray_ori.device, dtype=torch.int32),
        indexing="ij"
    ), dim=-1)[None,].repeat(len(c2ws), 1, 1, 1).view(-1, 2)
    ray_crd = torch.cat([
        torch.arange(len(c2ws), device=ray_crd.device, dtype=ray_crd.dtype)
        .repeat_interleave(len(ray_crd) // len(c2ws), dim=0)[:, None],
        ray_crd
    ], dim=-1)
    ray_ori, ray_dir = ray_ori.reshape(-1, 3), ray_dir.reshape(-1, 3)
    
    render = render_rays(verts, faces, ray_ori, ray_dir, ray_crd, c2ws, w2cs)
    
    return {name: val[0] for name, val in render.items()}


def ray_loss(
        verts, faces,
        gt_verts, gt_faces,
        res=512, n_renders=16, n_layers=1,
        use_nearest_layer=False, adaptive_target=False
):
    
    with torch.no_grad():
        fov = torch.tensor(0.8575560548920328, dtype=torch.float32, device=verts.device)
        foc = fov2focal_torch(fov, res)
        
        radius = 2.5 + torch.rand(
            n_renders, device=verts.device, dtype=torch.float32
        ) * 0.5
        # rand gaussian vec
        vec = torch.nn.functional.normalize(torch.randn(n_renders, 3, device=verts.device, dtype=torch.float32), dim=-1)
        azim = (torch.atan2(vec[:, 1], vec[:, 0]) + 2 * torch.pi) % (
                2 * torch.pi
        )
        elev = torch.asin(vec[:, 2] / vec[:, :3].norm(dim=-1))
        if adaptive_target:
            origin = torch.stack(
                [
                    radius * torch.cos(elev) * torch.cos(azim),
                    radius * torch.cos(elev) * torch.sin(azim),
                    radius * torch.sin(elev),
                    ],
                dim=-1,
            )
            ori_dist, ori_tri_idx, ori_tri_uv = get_dist(origin, gt_verts, gt_faces, return_attr=True)
            target = (gt_verts[gt_faces[ori_tri_idx]] * torch.cat([1 - ori_tri_uv.sum(dim=-1, keepdim=True), ori_tri_uv], dim=-1)[:, :, None]).sum(dim=1)
        else:
            target = torch.zeros(n_renders, 3, device=radius.device, dtype=radius.dtype)
        c2ws = get_camera(radius, azim, elev, center=target)
        w2cs = torch.linalg.inv(c2ws)
        
        ray_ori, ray_dir = get_rays(c2ws, foc, res)
        ray_crd = torch.stack(torch.meshgrid(
            torch.arange(ray_ori.shape[-3], device=ray_ori.device, dtype=torch.int32),
            torch.arange(ray_ori.shape[-2], device=ray_ori.device, dtype=torch.int32),
            indexing="ij"
        ), dim=-1)[None,].repeat(len(c2ws), 1, 1, 1).view(-1, 2)
        ray_crd = torch.cat([
            torch.arange(len(c2ws), device=ray_crd.device, dtype=ray_crd.dtype)
            .repeat_interleave(len(ray_crd) // len(c2ws), dim=0)[:, None],
            ray_crd
        ], dim=-1)
        ray_ori, ray_dir = ray_ori.reshape(-1, 3), ray_dir.reshape(-1, 3)
    
    with torch.no_grad():
        target = render_rays(gt_verts, gt_faces, ray_ori, ray_dir, ray_crd, c2ws, w2cs, n_layers=n_layers, fix_normals=True)
    
    pred = render_rays(verts, faces, ray_ori, ray_dir, ray_crd, c2ws, w2cs, n_layers=n_layers, fix_normals=True)
    
    if use_nearest_layer:
        with torch.no_grad():
            depth, gt_depth = pred["depth"], target["depth"]
            layer_idx = (depth[None,] - gt_depth[:, None]).abs().argmin(dim=0)
        mask = torch.gather(target["hard_mask"], dim=0, index=layer_idx)
        pos_loss = (pred["cpos"] - torch.gather(target["cpos"], dim=0, index=layer_idx.unsqueeze(-1).expand(*target["pos"].shape))).square().sum(dim=-1).add(1e-8).sqrt().mul(mask).mean()
        depth_loss = (pred["depth"] - torch.gather(target["depth"], dim=0, index=layer_idx)).mul(mask).abs().mean()
        normal_dot = (
                1 - (pred["cam_normal"] * torch.gather(target["cam_normal"], dim=0, index=layer_idx.unsqueeze(-1).expand(*target["cam_normal"].shape))).sum(dim=-1)
        ).mul(mask).square().mean()
        normal_l1 = (
                pred["cam_normal"] - torch.gather(target["cam_normal"], dim=0, index=layer_idx.unsqueeze(-1).expand(*target["cam_normal"].shape))
        ).abs().mul(mask).mean()
    else:
        pos_loss = (pred["cpos"] - target["cpos"]).square().sum(dim=-1).add(1e-8).sqrt().mul(target["hard_mask"]).mean()
        depth_loss = (pred["depth"] - target["depth"]).mul(target["hard_mask"]).abs().mean()
        normal_dot = (
                1 - (pred["cam_normal"] * target["cam_normal"]).sum(dim=-1)
        ).square().mul(target["hard_mask"]).mean()
        normal_l1 = (pred["cam_normal"] - target["cam_normal"]).abs().mul(target["hard_mask"][..., None]).mean()
        smooth_normal_l1 = (pred["smooth_normal"] - target["smooth_normal"]).abs().mul(target["hard_mask"][..., None]).mean()
    
    return {
        "depth_loss": depth_loss,
        "pos_loss": pos_loss,
        "normal_dot": normal_dot,
        "normal_loss": normal_l1,
        "smooth_loss": smooth_normal_l1,
        
        "pred": pred,
        "target": target,
    }

@torch.no_grad()
# @timeit("sample_sdf")
def sample_sdf(
        data_dict,
        device,
        context=None,
        pcd_size=None,
        n_close_samples=100000,
        n_far_samples=100000,
        std=1 / 128,
        calculate_sdf=True,
        watertight=False,
        use_edges=False,
        salient_rate=1,
        use_verts=False,
        salient_angle=20,
        surface_points=0,
        grad_near_surface=False,
        near_same=False,
        near_diff=False,
):
    if pcd_size is None:
        pcd_size = n_close_samples
    vertices, faces = data_dict["vertices"], data_dict["faces"]
    pcd, pcd_normals, pcd_rand, pcd_sdf, pcd_grad = [], [], [], [], []
    if surface_points > 0:
        data_dict["pcd_surface"], data_dict["pcd_surface_normals"] = [], []
        if grad_near_surface:
            data_dict["pcd_surface_sdf"] = []
    
    for v, f in zip(vertices, faces):
        # sample orig surface
        f_valid = f
        if context is None:
            gt_pcd, gt_normals = sample_surface(
                v,
                f,
                pcd_size + n_close_samples,
                use_edges=use_edges,
                salient_rate=salient_rate,
                use_verts=use_verts,
                thr=salient_angle,
                )
        else:
            gt_pcd, gt_normals, f_valid = sample_surface_render(
                v, f, pcd_size + n_close_samples, context
            )
        pcd.append(gt_pcd[:pcd_size]), pcd_normals.append(gt_normals[:pcd_size])
        
        # make random points around surface
        if near_same:
            gt_near = gt_pcd[:n_close_samples]
        else:
            gt_near = gt_pcd[pcd_size : pcd_size + n_close_samples]
        if near_diff:
            shifts = torch.tensor(
                [1.0, 0.5, 0.25, 0.125], device=gt_pcd.device, dtype=gt_pcd.dtype
            )
        else:
            shifts = torch.tensor([1.0], device=gt_pcd.device, dtype=gt_pcd.dtype)
        shifts = shifts[
            torch.randint(
                low=0,
                high=len(shifts),
                size=(len(gt_near),),
                device=gt_near.device,
                dtype=torch.int32,
            )
        ]
        __pcd = torch.cat(
            [
                gt_near + torch.randn_like(gt_near) * shifts[:, None] * std,
                (
                        (
                                torch.rand(n_far_samples, 3, device=device, dtype=torch.float32)
                                - 0.5
                        )
                        * (2 * (3**0.5) + 0.05)
                )[:n_far_samples],
                ]
        )
        pcd_rand.append(__pcd)
        
        if calculate_sdf and surface_points > 0:
            pcd_surface, pcd_surface_normals = sample_surface(
                v, f, surface_points, use_edges=False
            )
            if grad_near_surface:
                rand_radius = (
                        torch.sigmoid(
                            torch.randn(
                                len(pcd_surface),
                                1,
                                device=pcd_surface.device,
                                dtype=pcd_surface.dtype,
                            )
                        )
                        * std
                        * 0.95
                )
                rand_dir = torch.nn.functional.normalize(
                    torch.randn_like(pcd_surface), dim=-1
                )
                pcd_surface = pcd_surface + rand_dir * rand_radius
                
                eps = 1 / 1024
                
                offsets = torch.tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ],
                    device=pcd_surface.device,
                    dtype=pcd_surface.dtype,
                )
                pcd_surface_offset = pcd_surface[..., None, :] + offsets
                __pcd = torch.cat(
                    [
                        __pcd,
                        pcd_surface.view(surface_points, 3),
                        pcd_surface_offset.view(surface_points * 6, 3),
                    ]
                )
                
                # k1 = torch.tensor([1, -1, -1], dtype=pcd_surface.dtype, device=pcd_surface.device)
                # k2 = torch.tensor([-1, -1, 1], dtype=pcd_surface.dtype, device=pcd_surface.device)
                # k3 = torch.tensor([-1, 1, -1], dtype=pcd_surface.dtype, device=pcd_surface.device)
                # k4 = torch.tensor([1, 1, 1], dtype=pcd_surface.dtype, device=pcd_surface.device)
                # __pcd = torch.cat([
                #     __pcd,
                #     pcd_surface.view(surface_points, 3),
                #     pcd_surface + k1 * eps,
                #     pcd_surface + k2 * eps,
                #     pcd_surface + k3 * eps,
                #     pcd_surface + k4 * eps,
                # ])
                #
            
            data_dict["pcd_surface"].append(pcd_surface)
        
        # sample gt sdf from surface
        if calculate_sdf:
            if context is None:
                sdf, points = get_sdf(
                    __pcd, v, f, watertight=watertight, return_points=True
                )
                pcd_sdf.append(sdf[: n_close_samples + n_far_samples])
                pcd_grad.append(
                    torch.nn.functional.normalize(__pcd - points, dim=-1)[
                        : n_close_samples + n_far_samples
                    ]
                )
            else:
                sdf = get_sign(__pcd, v, f, watertight=False) * get_dist(
                    __pcd, v, f_valid, watertight=False
                )
                pcd_sdf.append(sdf[: n_close_samples + n_far_samples])
        else:
            sdf = torch.ones_like(__pcd[: n_close_samples + n_far_samples, 0])
            points = __pcd.clone()
            pcd_sdf.append(sdf)
        
        if calculate_sdf and surface_points > 0:
            if grad_near_surface:
                total_surface_sdf = sdf[-surface_points * (1 + 3 * 2) :]
                # total_surface_sdf = sdf[-surface_points * (1 + 4):]
                data_dict["pcd_surface_sdf"].append(total_surface_sdf[:surface_points])
                # pcd_surface_normals = torch.nn.functional.normalize(pcd_surface - points[-surface_points:], dim=-1) * surface_sdf.sign().unsqueeze(-1)
                
                sdf_offset = total_surface_sdf[
                    surface_points : surface_points * (1 + 3 * 2)
                ].view(surface_points, 3, 2)
                pcd_surface_normals = (sdf_offset[..., 0] - sdf_offset[..., 1]) / (
                        2 * eps
                )
                # sdf1, sdf2, sdf3, sdf4 = total_surface_sdf[surface_points*8:].float().view(surface_points, 4, 1).unbind(dim=-2)
                # pcd_surface_normals2 = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
                # print(torch.nn.functional.mse_loss(pcd_surface_normals, pcd_surface_normals2))
            
            data_dict["pcd_surface_normals"].append(pcd_surface_normals)
    
    data_dict["pcd"], data_dict["pcd_normals"] = torch.stack(pcd), torch.stack(
        pcd_normals
    )
    if calculate_sdf and surface_points > 0:
        data_dict["pcd_surface"], data_dict["pcd_surface_normals"] = torch.stack(
            data_dict["pcd_surface"]
        ), torch.stack(data_dict["pcd_surface_normals"])
        if grad_near_surface:
            data_dict["pcd_surface_sdf"] = torch.stack(data_dict["pcd_surface_sdf"])
    data_dict["pcd_rand"] = torch.stack(pcd_rand)
    data_dict["pcd_sdf"] = torch.stack(pcd_sdf)
    if len(pcd_grad) > 0:
        data_dict["pcd_grad"] = torch.stack(pcd_grad)
    
    return data_dict


def sdf_criteria(gt_pcd, gt_sdf, feat, fn, data_dict, total_iteration, config):
    
    if config["representations"].get("volume_far_reg", 0) > 0:
        dirvec = (
                torch.nn.functional.normalize(torch.randn_like(gt_pcd), dim=-1) * 2 / 256
        )
        gt_pcd = torch.cat([gt_pcd, gt_pcd + dirvec], dim=1)
        gt_grad = data_dict["pcd_grad"]
        gt_sdf = torch.cat(
            [gt_sdf, gt_sdf + torch.sum(dirvec * gt_grad, dim=-1)], dim=1
        )
    
    criterias = {}
    
    fn_out = fn(
        gt_pcd, feat, config.get("representations", {}).get("pcd_grad_loss", 0) > 0
    )
    sdf = fn_out["sdf"].view(*gt_sdf.shape)
    
    close_threshold = config["representations"].get("volume_close_threshold", 2 / 512)
    valid_mask = gt_pcd.abs().max(dim=-1)[0] <= (3**0.5 + 0.05)
    close_mask = (gt_sdf.abs() <= close_threshold) & valid_mask
    far_mask = (~close_mask) & valid_mask
    
    if config["representations"].get("volume_far_reg", 0) > 0:
        gt_sdf, gt_sdf2 = gt_sdf.chunk(2, dim=1)
        far_mask, far_mask2 = far_mask.chunk(2, dim=1)
        close_mask, close_mask2 = close_mask.chunk(2, dim=1)
        sdf, sdf2 = sdf.chunk(2, dim=1)
        both_mask = torch.logical_and(far_mask, far_mask2)
        criterias["volume_reg"] = torch.nn.functional.mse_loss(
            sdf[both_mask], sdf2[both_mask]
        ) * config["representations"].get("volume_far_reg", 0)
    
    sdf_loss_type = config.get("representations", {}).get("sdf_loss_type", "l2")
    if config.get("representations", {}).get("sdf2opacity", False):
        target = torch.sign(gt_sdf)
        if close_threshold > 0:
            target = torch.where(
                close_mask,
                gt_sdf / close_threshold,
                target,
                )
        values = torch.nn.functional.binary_cross_entropy_with_logits(
            sdf.float(), target / 2 + 0.5, reduction="none"
        )
    elif config.get("representations", {}).get("sdf_with_opacity", False):
        
        if config.get("representations", {}).get("smooth_sign", False):
            target = (
                    torch.sigmoid((gt_sdf.float() / close_threshold).clamp(-128, 128)) * 2
                    - 1
            )
        else:
            target = torch.where(
                close_mask,
                gt_sdf / max(close_threshold, 1e-6),
                gt_sdf.sign(),
                )
        
        values_opacity = torch.nn.functional.binary_cross_entropy_with_logits(
            sdf.float(), target / 2 + 0.5, reduction="none"
        )
        
        if config.get("representations", {}).get("tsdf", False):
            target = torch.where(
                close_mask,
                gt_sdf / max(close_threshold, 1e-6),
                gt_sdf.sign(),  # * (2 * (gt_sdf.abs() - close_threshold) + 1),
            )
            sdf_clamp = config.get("representations", {}).get("sdf_clamp", 65536)
            if sdf_loss_type == "l2":
                values_sdf = torch.nn.functional.mse_loss(
                    target,
                    sdf.float().clamp(min=-sdf_clamp, max=sdf_clamp),
                    reduction="none",
                )
            elif sdf_loss_type == "l1l2":
                values_sdf = (
                        torch.nn.functional.mse_loss(
                            target,
                            sdf.float().clamp(min=-sdf_clamp, max=sdf_clamp),
                            reduction="none",
                        )
                        + torch.nn.functional.l1_loss(
                    target,
                    sdf.float().clamp(min=-sdf_clamp, max=sdf_clamp),
                    reduction="none",
                )
                        * 0.1
                )
            else:
                values_sdf = torch.nn.functional.l1_loss(
                    target,
                    sdf.float().clamp(min=-sdf_clamp, max=sdf_clamp),
                    reduction="none",
                )
        else:
            target = gt_sdf.float()
            if sdf_loss_type == "l2":
                values_sdf = torch.nn.functional.mse_loss(
                    target, sdf.float(), reduction="none"
                )
            else:
                values_sdf = torch.nn.functional.l1_loss(
                    target, sdf.float(), reduction="none"
                )
        if config.get("representations", {}).get("sdf_transition", 0) > 0:
            sdf_mul = min(
                1,
                total_iteration
                / config.get("representations", {}).get("sdf_transition", 0),
                )
            values = values_sdf * sdf_mul * config.get("representations", {}).get(
                "sdf_weight", 1
            ) + values_opacity * (1 - sdf_mul) * config.get("representations", {}).get(
                "opacity_weight", 1
            )
        else:
            values = values_sdf * config.get("representations", {}).get(
                "sdf_weight", 1
            ) + values_opacity * config.get("representations", {}).get(
                "opacity_weight", 1
            )
    else:
        if sdf_loss_type == "l2":
            values = torch.square(gt_sdf - sdf.float())
        else:
            values = torch.abs(gt_sdf - sdf.float())
    
    # values_close, values_far = values[close_mask], values[far_mask]
    # denom = values_close.numel() + values_far.numel()
    # criterias["sdf"] = (
    #     (
    #         values_close.sum() * config["representations"].get("volume_close_weight", 1)
    #         + values_far.sum() * config["representations"].get("volume_far_weight", 1)
    #     )
    #     / denom
    #     * config["representations"].get("sdf_loss", 0)
    # )
    criterias["sdf"] = values.mean() * config["representations"].get("sdf_loss", 0)
    
    if config.get("representations", {}).get("surface_points", 0) > 0:
        pcd, pcd_normals = data_dict[
            "pcd_surface"
        ].float(), torch.nn.functional.normalize(
            data_dict["pcd_surface_normals"].float(), dim=-1
        )
        sdf_outs = fn(pcd, feat, True)
        
        if config.get("representations", {}).get("grad_near_surface", False):
            pcd_sdf = data_dict["pcd_surface_sdf"].float()
            sdf = sdf_outs["sdf"].float().view(*pcd_sdf.shape)
            diff = sdf - pcd_sdf / close_threshold
            if sdf_loss_type == "l1":
                diff = diff.abs().mean()
            elif sdf_loss_type == "l2":
                diff = diff.square().mean()
            else:
                diff = (diff.square() + diff.abs()).mean()
            criterias["sdf_surface"] = diff * config.get("representations", {}).get(
                "sdf_weight", 1
            )
        
        grad = sdf_outs["grad"].float()
        grad_norm = grad.norm(dim=-1, keepdim=True)
        criterias["eikonal_surface"] = (
                                               grad_norm * close_threshold - 1
                                       ).abs().mean() * config.get("representations", {}).get(
            "surface_eikonal_loss", 0.1
        )
        dots = (grad / grad_norm.clamp(min=1e-6) * pcd_normals).sum(dim=-1)
        criterias["grad_surface"] = (1 - dots).abs().mean() * config.get(
            "representations", {}
        ).get("surface_grad_loss", 10.0)
    
    if config.get("representations", {}).get("pcd_grad_loss", 0) > 0:
        multiplier = config.get("representations", {}).get("pcd_grad_loss", 0)
        gt_grad = data_dict["pcd_grad"]
        criterias["sdf_grad"] = (
                                        fn_out["grad"].float() - gt_grad
                                ).abs().mean() * multiplier
    
    data_dict["sdf_values"] = values.mean(dim=list(range(1, values.ndim)))
    
    return criterias


def extract_cycles(E):
    
    nbrs = defaultdict(set)
    for a, b in E:
        a = int(a); b = int(b)
        if a == b:
            continue
        nbrs[a].add(b)
        nbrs[b].add(a)

    adj = {u: list(vs) for u, vs in nbrs.items()}

    def ek(u, v):
        return (u, v) if u < v else (v, u)

    used_edges = set()  # undirected edges
    loops = []

    for s in list(adj.keys()):
        if all(ek(s, t) in used_edges for t in adj[s]):
            continue

        path = [s]
        pos = {s: 0}              # active position in path; missing or -1 means inactive
        stack = [[s, 0]]          # [vertex, next_neighbor_index]

        while stack:
            u, i = stack[-1]

            if i >= len(adj[u]):
                stack.pop()
                pu = pos.get(u, -1)
                if pu != -1:
                    if path and path[-1] == u:
                        path.pop()
                    pos[u] = -1
                continue

            v = adj[u][i]
            stack[-1][1] += 1

            e = ek(u, v)
            if e in used_edges:
                continue

            used_edges.add(e)
            pv = pos.get(v, -1)

            # back to active path vertex -> cycle found
            if pv != -1:
                cyc = path[pv:].copy()
                if len(cyc) >= 3:
                    loops.append(cyc)

                # collapse subloop to the vertex v
                while len(path) > pv + 1:
                    w = path.pop()
                    pos[w] = -1

                # keep DFS stack aligned with active path
                while len(stack) > pv + 1:
                    stack.pop()

                continue

            # descend DFS
            pos[v] = len(path)
            path.append(v)
            stack.append([v, 0])

    return loops

def fill_holes(verts, faces):

    import mapbox_earcut as earcut
    import igl

    V = np.asarray(verts.detach().double().cpu().numpy(), np.float64)
    F = np.asarray(faces.detach().long().cpu().numpy(), np.int64)

    BF = igl.boundary_facets(F)
    E = BF[0] if isinstance(BF, tuple) else BF
    E = np.asarray(E, np.int64).reshape(-1, 2)
    E = np.unique(np.sort(E, axis=1), axis=0)

    loops = extract_cycles(E)

    def pca_basis(P):
        c = P.mean(axis=0)
        X = P - c
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        u, v = vt[0], vt[1]
        n = np.cross(u, v)
        n /= (np.linalg.norm(n) + 1e-12)
        return c, u, v, n

    new_tris = []
    for loop in loops:
        loop = np.asarray(loop, np.int64)

        keep = np.ones(len(loop), dtype=bool)
        keep[1:] = loop[1:] != loop[:-1]
        loop = loop[keep]
        if len(loop) < 3:
            continue

        P = V[loop]
        c, u, v, n = pca_basis(P)
        XY = np.stack([(P - c) @ u, (P - c) @ v], axis=1).astype(np.float32)

        rings = np.array([XY.shape[0]], dtype=np.uint32)
        idx = earcut.triangulate_float32(XY, rings)
        idx = np.asarray(idx, np.int64)

        if idx.size > 0:
            idx = idx.reshape(-1, 3)
            T = loop[idx]
            new_tris.append(T)
        elif len(loop) >= 3:
            T = np.array([[loop[0], loop[i], loop[i+1]] for i in range(1, len(loop)-1)], dtype=np.int64)
            new_tris.append(T)

    F_out = np.vstack([F] + new_tris) if new_tris else F
    return (
        torch.tensor(V, device=verts.device, dtype=verts.dtype),
        torch.tensor(F_out, device=faces.device, dtype=faces.dtype),
    )


def postprocess_mesh(verts, faces, n_faces=600000, fix:bool = True):
    
    if len(verts) < 3 or len(faces) < 3:
        return get_default_mesh(verts.device)
    
    verts, faces = verts.double(), faces.int()
    
    # remove inf values
    valid_verts = torch.isfinite(verts).all(dim=-1)
    verts = verts[valid_verts]
    faces = faces[torch.isfinite(faces).all(dim=-1)]
    faces = faces[valid_verts[faces].all(dim=-1)]
    faces = (valid_verts.int().cumsum(dim=0) - 1)[faces]
    
    # floater removing
    comps = get_oriented_components(verts, faces)
    valid_comps = []
    max_comp_size = max([len(c) for c in comps])
    for comp in comps:
        if len(comp) < 0.005 * max_comp_size:
            continue
        valid_comps.append(comp)
    faces = torch.cat([faces[c] for c in valid_comps])
    
    
    # unreferenced vertices
    v_indices, inv_f = torch.unique(faces, return_inverse=True)
    faces = inv_f.int().reshape(-1, 3)
    verts = verts[v_indices]
    
    verts = verts.float()
    
    if fix:
        verts, faces = fill_holes(verts, faces)
    
    if len(faces) > n_faces:
        verts, faces = simplify(verts, faces, n_faces=int(n_faces), max_iters=10000)
        verts, faces = fill_holes(verts, faces)
        
    if fix:
        verts, faces = orient_mesh(verts, faces, n_rays=32, n_bounce=8)
    
    return verts, faces


# @torch.compile()


def sample_voxels(vertices, faces, grid_res=256, n_points=4096000, shift=0, use_edges=False, salient_angle=15):
    
    # assume vertices in [-1, 1]
    # get active voxels for mesh
    avoxels = tri2vox(vertices, faces, grid_res)
    
    # sample dense point cloud
    pcd, nrm = sample_surface(vertices.float(), faces.int(), n_points, use_edges=use_edges, thr=salient_angle)
    # if shift is None:
    # shift = 1 / (grid_res * 4 - 1) * 2.0
    if shift > 0:
        cube_dilate = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, 0],
                [0, 1, 1],
                [0, -1, 1],
                [0, 1, -1],
                [0, -1, -1],
                
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, -1],
                [1, -1, 0],
                [1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [1, -1, -1],
                
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, 0],
                [-1, 0, -1],
                [-1, -1, 0],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
            ], device=pcd.device, dtype=pcd.dtype
        )
        pcd_dilate = (pcd[..., None, :] + cube_dilate * shift).reshape(-1, 3)
    else:
        pcd_dilate = pcd
    avoxels |= pcd2vox(pcd_dilate, grid_res)
    
    voxels_idxs = avoxels.nonzero()
    # voxels_idxs = torch.cat([torch.zeros_like(voxels_idxs[..., :1]), voxels_idxs], dim=-1).int()
    return pcd, nrm, voxels_idxs


def get_default_mesh(device):
    
    v = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)
    
    f = torch.tensor([
        [2, 1, 0],
        [0, 1, 3],
        [3, 2, 0],
        [1, 2, 3],
    ], device=device, dtype=torch.int32)
    
    return v, f