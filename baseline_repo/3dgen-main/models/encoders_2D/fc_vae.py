import gc
from typing import *
from collections import defaultdict
from itertools import product
import random
import torch
from torch_scatter import scatter_mean, scatter_add, scatter_max
import random

# from ops.mesh import (
#     get_dist, get_sdf, dilate_coords, fmc,
#     find_corner_points, exterior_voxels, mark_corners_from_exterior_voxels, pcd_attn
# )

from ops.mesh import (
    get_dist, get_sdf, dilate_coords, fmc,  pcd_attn
)
from ops.rope import rope_nd

import base_models.sparse as sp
from base_models.fourier_utils import get_positional_encoding
# from base_models.mesh_utils import (
#     focal2fov_torch, fov2focal_torch, get_camera, projection_matrix, get_rays, get_ortho_rays, render_rays, render_radius_rays, render_mesh
# )
import numpy as np


def maybe_dummy_weight(x, weight):
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    return weight


def layer_norm(x, weight=None, bias=None, eps=1e-6):
    from flash_attn.ops.layer_norm import layer_norm
    return layer_norm(x, maybe_dummy_weight(x, weight), bias, eps)


def rms_norm(x, weight=None, eps=1e-6):
    from flash_attn.ops.rms_norm import rms_norm
    return rms_norm(x, maybe_dummy_weight(x, weight), eps)


def init_linear(linear, normal=False, n_out=1):
    if normal:
        torch.nn.init.trunc_normal_(
            linear.weight, 0, 1.0 / (linear.weight.data[0].numel() ** 0.5)
        )
    else:
        weight = linear.weight
        fan_out, fan_in = weight.shape[0], weight[0].numel()
        a = np.sqrt(6 / (fan_in + fan_out / n_out))
        torch.nn.init.uniform_(linear.weight, -a, a)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)
    return linear


def zero_module(layer):
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer


class LayerNorm(torch.nn.Module):
    
    def __init__(self, hidden_dim):
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
    
    def forward(self, x):
        return layer_norm(x, 1 + self.weight, self.bias)


class ScaleNorm(torch.nn.Module):
    
    def __init__(self, hidden_dim, learnable=True):
        super(ScaleNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
    
    def forward(self, x):
        return rms_norm(x, 1 + self.weight)

class SparseLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        return super().forward(x)


class SparseScaleNorm(ScaleNorm):
    def forward(self, x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        return super().forward(x)


class SparseGroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int, learnable: bool = True):
        super().__init__()
        
        self.num_groups = num_groups
        
        self.learnable = learnable
        if self.learnable:
            self.weight = torch.nn.Parameter(torch.zeros(num_channels, dtype=torch.float32), requires_grad=True)
            self.bias = torch.nn.Parameter(torch.zeros(num_channels, dtype=torch.float32), requires_grad=True)
        else:
            self.weight = self.bias = None
    
    def forward(self, x: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        
        outs = []
        for h in x.split(sizes):
            h = h.view(len(h), self.num_groups, h.shape[-1] // self.num_groups)
            h = h - h.mean(dim=[0, 2], keepdim=True, dtype=torch.float32).type_as(h)
            h = h * torch.rsqrt(h.square().mean(dim=[0, 2], keepdim=True, dtype=torch.float32) + 1e-5).type_as(h)
            h = h.view(len(h), h.shape[1] * h.shape[2])
            outs.append(h)
        
        h = torch.cat(outs)
        if self.learnable:
            h = h * (1 + self.weight.type_as(h)) + self.bias.type_as(h)
        
        return h



class SparseResBlock3d(torch.nn.Module):
    def __init__(
            self,
            layer_idx: str,
            hidden_dim: int,
            norm_type: Literal["pixel", "layer", "rms", "group"] = "pixel",
            conv_type: Literal["full", "fast", "warp", "flex"] = "full",
            out_dim: int = None,
            in_dim: int = None,
            gc: bool = False,
            save_memory: bool = False,
    ):
        super().__init__()
        self.gc = gc
        self.save_memory = save_memory
        
        if in_dim is None:
            in_dim = hidden_dim
        
        if out_dim is None:
            out_dim = hidden_dim
        
        self.conv_type = conv_type
        if self.conv_type == "full":
            self.conv1 = sp.SparseConv3d(in_dim, hidden_dim, 3, indice_key=f"{layer_idx}")
            self.conv2 = sp.SparseConv3d(hidden_dim, out_dim, 3, indice_key=f"{layer_idx}")
        elif self.conv_type == "warp":
            self.conv1 = sp.WarpConv3d(hidden_dim, kernel_size=3)
            self.conv2 = sp.WarpConv3d(hidden_dim, kernel_size=3)
        elif self.conv_type == "flex":
            self.conv1 = sp.FlexConv3d(hidden_dim, kernel_size=3)
            self.conv2 = sp.FlexConv3d(hidden_dim, kernel_size=3)
        elif self.conv_type == "mixfast":
            self.conv1 = sp.FastSparseConv3d(hidden_dim)
            self.ffn1 = torch.compile(torch.nn.Sequential(
                ScaleNorm(hidden_dim),
                init_linear(torch.nn.Linear(hidden_dim, hidden_dim * 4)),
                torch.nn.SiLU(),
                zero_module(torch.nn.Linear(hidden_dim * 4, hidden_dim))
            ))
            self.conv2 = sp.FastSparseConv3d(hidden_dim)
            self.ffn2 = torch.compile(torch.nn.Sequential(
                ScaleNorm(hidden_dim),
                init_linear(torch.nn.Linear(hidden_dim, hidden_dim * 4)),
                torch.nn.SiLU(),
                zero_module(torch.nn.Linear(hidden_dim * 4, hidden_dim))
            ))
        else:
            self.conv1 = sp.FastSparseConv3d(hidden_dim)
            self.conv2 = sp.FastSparseConv3d(hidden_dim)
        
        if norm_type in ("pixel", "layer"):
            self.norm1 = SparseLayerNorm(in_dim)
            self.norm2 = SparseLayerNorm(hidden_dim)
        elif norm_type == "group":
            self.norm1 = SparseGroupNorm(32, in_dim)
            self.norm2 = SparseGroupNorm(32, hidden_dim)
        elif norm_type == "rms":
            self.norm1 = SparseScaleNorm(in_dim)
            self.norm2 = SparseScaleNorm(hidden_dim)
        
        self.skip = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim != in_dim
            else torch.nn.Identity()
        )
        if in_dim != out_dim:
            init_linear(self.skip)
    
    def _prepare_feat(self, x: torch.Tensor, sizes: List[int], idx: int):
        if idx == 0:
            return torch.nn.functional.silu(self.norm1(x, sizes), inplace=False)
        if idx == 1:
            return torch.nn.functional.silu(self.norm2(x, sizes), inplace=False)
    
    # @torch.compile()
    def prepare_feat(self, x: torch.Tensor, sizes: List[int], idx: int):
        if self.gc and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._prepare_feat, x, sizes, idx, use_reentrant=False
            )
        return self._prepare_feat(x, sizes, idx)
    
    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        
        sizes = [lay.stop - lay.start for lay in x.layout]
        prepare_feat = self._prepare_feat if self.save_memory else self.prepare_feat
        
        if self.conv_type == "full":
            h = self.conv1(x.replace(prepare_feat(x.feats, sizes, 0)))
            h = self.conv2(h.replace(prepare_feat(h.feats, sizes, 1)))
        elif self.conv_type == "warp":
            voxels = x.get_spatial_cache("warp_voxels")
            if voxels is None:
                from warpconvnet.geometry.types.voxels import Voxels
                voxels = Voxels(
                    [x.coords[lay, 1:] for lay in x.layout],
                    [x.feats[lay] for lay in x.layout],
                )
            else:
                voxels = voxels.replace(voxels.batched_coordinates, x.feats)
            voxels = self.conv1(voxels.replace(voxels.batched_coordinates, prepare_feat(voxels.features, sizes, 0)))
            voxels = self.conv2(voxels.replace(voxels.batched_coordinates, prepare_feat(voxels.features, sizes, 1)))
            h = x.replace(voxels.features)
            h.register_spatial_cache("warp_voxels", voxels)
        elif self.conv_type == "flex":
            
            max_coord = x.coords[:, 1:].max().item() + 1
            res = 1
            while res < max_coord:
                res *= 2
            shape = (x.shape[0], x.feats.shape[-1], res, res, res)
            
            
            neighbours = x.get_spatial_cache("flex_neighbours")
            h, neighbours = self.conv1(prepare_feat(x.feats, sizes, 0), x.coords, shape, neighbours)
            h, neighbours = self.conv2(prepare_feat(h, sizes, 1), x.coords, shape, neighbours)
            h = x.replace(h)
            h.register_spatial_cache("flex_neighbours", neighbours)
        
        elif self.conv_type == "mixfast":
            conv_indices = x.get_spatial_cache("conv_indices")
            if conv_indices is None:
                from ops.sparse import get_conv_indices
                conv_indices = get_conv_indices(x.coords)
                x.register_spatial_cache("conv_indices", conv_indices)
            h = x.feats
            h = h + self.ffn1(self.conv1(prepare_feat(h, sizes, 0), x.coords, conv_indices))
            h = h + self.ffn2(self.conv2(prepare_feat(h, sizes, 1), x.coords, conv_indices))
            h = x.replace(h)
        else:
            conv_indices = x.get_spatial_cache("conv_indices")
            if conv_indices is None:
                from ops.sparse import get_conv_indices
                conv_indices = get_conv_indices(x.coords)
                x.register_spatial_cache("conv_indices", conv_indices)
            h = self.conv1(prepare_feat(x.feats, sizes, 0), x.coords, conv_indices)
            h = self.conv2(prepare_feat(h, sizes, 1), x.coords, conv_indices)
            h = x.replace(h)
        # out skip connection
        x = x.replace(self.skip(x.feats) + h.feats)
        
        return x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.gc and self.training and self.save_memory:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)


class SparseFFN(torch.nn.Module):
    def __init__(self, dim: int, out_dim: [int | None] = None, gc: bool = False):
        super().__init__()
        self.gc = gc
        out_dim = out_dim or dim
        self.norm = LayerNorm(dim)
        self.ffn0 = torch.nn.Linear(dim, dim * 4)
        self.ffn1 = torch.nn.Linear(dim * 4, out_dim)
        init_linear(self.ffn0), init_linear(self.ffn1)
    
    def __forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.ffn0(h)
        h = torch.nn.functional.gelu(h, approximate="tanh")
        h = self.ffn1(h)
        return h
    
    # @torch.compile()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__forward(x)
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return x.replace(self._forward(x.feats))


class SparseMHA(torch.nn.Module):
    def __init__(
            self,
            head_dim: int = 64,
            num_heads: int = 12,
            order_idx: int = 0,
            add_context=False,
            rope: bool = False,
            gc: bool = False,
    ):
        super().__init__()
        
        self.gc = gc
        self.rope = rope
        
        self.head_dim = head_dim
        self.num_heads = num_heads
        channels = head_dim * num_heads
        
        self.norm = LayerNorm(channels)
        self.to_qkv = torch.nn.Linear(channels, 3 * channels)
        self.to_out = torch.nn.Linear(channels, channels)
        init_linear(self.to_qkv, n_out=3), init_linear(self.to_out)
        
        self.add_context = add_context
        if self.add_context:
            self.to_qkv_context = torch.nn.Linear(channels, 3 * channels)
            self.to_out_context = torch.nn.Linear(channels, channels)
            init_linear(self.to_qkv_context, n_out=3), init_linear(self.to_out_context)
        
        self.order_idx = order_idx
    
    def __forward(
            self,
            x: torch.Tensor,
            pos: torch.Tensor,
            lens: List[int],
            fwd_indices: torch.Tensor,
            bwd_indices: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_len: int,
    ) -> torch.Tensor:
        
        x_norm = self.norm(x)
        qkv_feats = self.to_qkv(x_norm).view(-1, 3, self.num_heads, self.head_dim)
        
        if self.rope:
            q, k, v = qkv_feats.unbind(dim=-3)
            q, k = rope_nd(q[None,], pos[None,], freq=128)[0], rope_nd(k[None,], pos[None,], freq=128)[0]
            qkv_feats = torch.stack([q, k, v], dim=-3)
        
        attn = sp.sparse_windowed_sdp(
            qkv_feats,
            fwd_indices,
            bwd_indices,
            cu_seqlens,
            max_len,
        )
        attn = self.to_out(attn.reshape(-1, self.num_heads * self.head_dim))
        
        if self.add_context:
            q, k, v = self.to_qkv_context(x_norm).view(
                -1, 3, self.num_heads, self.head_dim
            ).unbind(dim=-3)
            context = sp.sparse_linear_attention(q, k, v, lens).view(q.shape[0], q.shape[1] * q.shape[2])
            attn = attn + self.to_out_context(context)
        
        return attn
    
    # @torch.compile()
    def _forward(
            self,
            x: torch.Tensor,
            pos: torch.Tensor,
            lens: List[int],
            fwd_indices: torch.Tensor,
            bwd_indices: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_len: int
    ) -> torch.Tensor:
        return self.__forward(
            x, pos, lens, fwd_indices, bwd_indices, cu_seqlens, max_len
        )
    
    def forward(self, x: sp.SparseTensor, pos: torch.Tensor) -> sp.SparseTensor:
        
        fwd_indices, bwd_indices, cu_seqlens, max_len = x.get_spatial_cache(f"attn_order{self.order_idx}")
        
        return x.replace(
            self._forward(x.feats, pos, [lay.stop - lay.start for lay in x.layout], fwd_indices, bwd_indices, cu_seqlens, max_len)
        )


class SparseTransformerBlock(torch.nn.Module):
    def __init__(
            self,
            head_dim: int = 64,
            num_heads: int = 12,
            order_idx: int = 0,
            add_context: bool = False,
            rope: bool = False,
            gc: bool = False,
    ):
        super().__init__()
        self.attn = SparseMHA(
            head_dim=head_dim,
            num_heads=num_heads,
            order_idx=order_idx,
            add_context=add_context,
            rope=rope,
            gc=gc,
        )
        self.ffn = SparseFFN(head_dim * num_heads, gc=gc)
    
    def forward(
            self, x: sp.SparseTensor, pos: torch.Tensor) -> sp.SparseTensor:
        h = self.attn(x, pos)
        x = h.replace(x.feats + h.feats)
        h = self.ffn(x)
        x = h.replace(x.feats + h.feats)
        return x


class SparseEncoder(torch.nn.Module):
    def __init__(
            self,
            latent_dim=16,
            res=128,
            grid_res=1024,
            pos_emb: bool = True,
            rope: bool = False,
            encode_attributes: bool = False,
            use_pcd_attn: bool = False,
            pcd_attn_k: int = 256,
            use_edges: bool = False,
            # use_relative_pe=True,
            # use_pointnet=True,
            # pcd_feat_blocks=5,
            # pcd_feat_dim=256,
            pcd_pos_scale=16,
            # pcd_pos_features=24,
            # pcd_norm: Literal["layer", "rms"] = "layer",
            # encode_normal=False,
            # reduce="mean",
            reduce_dim: int = None,
            use_conv=True,
            down_dims=(32, 128, 512),
            down_depths=(1, 1, 1),
            # down_type: Literal["res", "swin"] = "res",
            res_norm: Literal["pixel", "layer", "rms", "group"] = "pixel",
            res_conv: Literal["full", "fast"] = "full",
            orders: List[Literal["full", "swin", "swin_shift", "z", "zt", "h", "ht", "xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"]] = ("swin", "swin_shift"),
            encoder_dim=768,
            encoder_depth=12,
            head_dim: int = 64,
            context_attn: bool = False,
            adj_first: bool = False,
            window_size: int = 8,
            patch_size: int = 1024,
            bucket_size: int = 512,
            global_size: int = 512,
            gc=False,
            save_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.encode_attributes = encode_attributes
        
        self.use_pcd_attn = use_pcd_attn
        self.pcd_attn_k = pcd_attn_k
        self.use_edges = use_edges
        self.pos_scale = pcd_pos_scale
        
        self.pos_emb = pos_emb
        self.rope = rope
        self.res = res * (1 << (len(down_dims) - 1)) if use_conv else res
        self.latres = res
        self.grid_res = grid_res
        
        self.dim = encoder_dim
        self.head_dim = head_dim
        self.reduce_dim = reduce_dim or encoder_dim
        self.use_conv = use_conv
        

        if self.encode_attributes:
            
            self.down_type = "res"
            self.down_in = torch.nn.Linear((1 + 3) * 8, down_dims[0], bias=False)
            init_linear(self.down_in, normal=True)
            
            self.down_blocks, self.downs = torch.nn.ModuleList([]), torch.nn.ModuleList([])
            for idx in range(len(down_dims)):
                blocks = torch.nn.ModuleList([])
                for d in range(down_depths[idx]):
                    blocks.append(
                        SparseResBlock3d(
                            f"layer_{idx}", hidden_dim=down_dims[idx], norm_type=res_norm, conv_type=res_conv, gc=gc, save_memory=save_memory
                        )
                    )
                self.down_blocks.append(blocks)
                if idx < len(down_dims) - 1:
                    self.downs.append(
                        sp.SparseDownsample(
                            down_dims[idx], down_dims[idx + 1], patch_size=2
                        )
                    )
                else:
                    self.downs.append(
                        init_linear(
                            sp.SparseLinear(down_dims[idx], encoder_dim, bias=False),
                            normal=True,
                        )
                    )        

        
        self.orders = orders
        self.order_kwargs = {
            "window_size": window_size,
            "patch_size": patch_size,
            "bucket_size": bucket_size,
            "global_size": global_size,
            "adj_first": adj_first,
        }
        self.blocks = torch.nn.ModuleList(
            [
                SparseTransformerBlock(
                    head_dim=head_dim,
                    num_heads=encoder_dim // head_dim,
                    order_idx=idx % len(self.orders),
                    add_context=context_attn,
                    rope=rope,
                    gc=gc,
                )
                for idx in range(encoder_depth)
            ]
        )
        
        
        self.out_norm = LayerNorm(encoder_dim)
        self.out_mean = torch.nn.Linear(encoder_dim, latent_dim)
        self.out_logvar = torch.nn.Linear(encoder_dim, latent_dim)
    
    def _reduce_pcd(self, vox_idxs, pos_encode, pos_index):
        
        if self.reduce == "attn":
            
            if self.use_relative_pe:
                q_pos = scatter_mean(pos_encode, pos_index, dim=0,
                                     out=torch.zeros(len(vox_idxs), pos_encode.shape[-1], device=pos_encode.device, dtype=pos_encode.dtype))
            else:
                q_pos = (vox_idxs[:, 1:].float() + 0.5) / self.latres * 2 - 1
                if self.pos_features > 0:
                    q_pos = torch.cat(
                        [
                            q_pos,
                            get_positional_encoding(
                                q_pos,
                                self.pos_features,
                                scale=self.pos_scale,
                                logspace=True,
                                include_pi=False,
                            ),
                        ],
                        dim=-1,
                    )
                q_pos = self.qpos_map(q_pos)
            
            q = self.reduce_q(self.reduce_qnorm(q_pos))
            q = q / np.sqrt(q.shape[-1])
            
            k, v = self.reduce_kv(self.reduce_cnorm(pos_encode)).chunk(2, dim=-1)
            
            w = (q[pos_index] * k).sum(dim=-1)
            w_max = scatter_max(w, pos_index, dim=0, out=torch.zeros(len(q), dtype=w.dtype, device=w.device))[0]
            w = (w - w_max[pos_index]).float().exp_()
            denom = scatter_add(w, pos_index, dim=0, out=torch.zeros(len(q), dtype=w.dtype, device=w.device))
            w = w / denom[pos_index].clamp(min=1e-6)
            
            context = v * w.unsqueeze(1).type_as(v)
            context = scatter_add(context, pos_index, dim=0, out=torch.zeros_like(q))
            
            return q_pos + self.reduce_out(context)
        
        if self.reduce == "mean":
            return scatter_mean(
                pos_encode,
                pos_index,
                dim=0,
                out=torch.zeros(
                    len(vox_idxs), pos_encode.shape[-1], device=pos_encode.device, dtype=pos_encode.dtype
                ),
            )
    # @torch.compile()
    def reduce_pcd(self, vox_idxs, pos_encode, pos_index):
        return self._reduce_pcd(vox_idxs, pos_encode, pos_index)
    
    
    def encode_pcd(self, vox_idxs, pcd, pcd_normals):
        
        B, N, _ = pcd.shape
        
        pcd = pcd.div(2).add(0.5).clamp(1e-6, 1 - 1e-6)
        coord = pcd * self.res
        voxel = coord.floor().clamp(max=self.res - 1)
        
        code = torch.cat(
            [
                torch.arange(B, device=voxel.device, dtype=torch.long)
                .view(B, 1)
                .repeat_interleave(N, dim=1)
                .view(B * N, 1),
                voxel.long().view(B * N, 3),
            ],
            dim=-1,
                )
        offset = torch.tensor(
            [self.res * self.res * self.res, self.res * self.res, self.res, 1],
            device=code.device,
            dtype=code.dtype,
        )
        code = (code * offset).sum(dim=-1)
        code, index = code.unique(return_inverse=True)
        
        if self.use_relative_pe:
            pos = coord - (voxel + 0.5)
        else:
            pos = pcd * 2 - 1.0
        if self.pos_features > 0:
            pos_encode = torch.cat(
                [
                    pos,
                    pcd_normals,
                    get_positional_encoding(
                        pos,
                        self.pos_features,
                        scale=self.pos_scale,
                        include_pi=False,
                        logspace=True,
                    ),
                ],
                dim=-1,
            )
            if self.encode_normal:
                pos_encode = torch.cat([pos_encode, get_positional_encoding(pcd_normals, self.pos_features, scale=self.pos_scale, include_pi=False, logspace=True)], dim=-1)
        else:
            pos_encode = torch.cat([pcd, pcd_normals], dim=-1)
        pos_encode = pos_encode.view(B * N, pos_encode.shape[-1])
        
        if self.use_pointnet:
            h = self.pc_in(pos_encode)
            
            for block_idx, block in enumerate(self.pc_blocks):
                
                context = scatter_mean(
                    h,
                    index,
                    dim=0,
                    out=torch.zeros(
                        len(code), h.shape[1], device=h.device, dtype=h.dtype
                    ),
                )
                
                h = block(h, context, index)
            h = self.pc_out(h)
        else:
            h = self.pos_map(pos_encode)
        
        vox_code = vox_idxs.int()
        lat_offset = torch.tensor(
            [
                self.latres * self.latres * self.latres,
                self.latres * self.latres,
                self.latres,
                1,
                ],
            device=vox_code.device,
            dtype=vox_code.dtype,
        )
        vox_code = (vox_code * lat_offset).sum(dim=-1)
        
        if self.use_conv:
            coords = torch.cat(
                [
                    code[:, None] // offset[0],
                    (code[:, None] % offset[:-1]) // offset[1:],
                    ],
                dim=-1,
            ).int()
            feats = self.reduce_pcd(coords, h, index)
            feats = self.down_in(feats)
            
            x = sp.SparseTensor(feats, coords)
            for level_idx, (blocks, ds) in enumerate(zip(self.down_blocks, self.downs)):
                
                if self.down_type != "res":
                    sp.put_attn_orders(x, self.orders, **self.order_kwargs)
                
                pos = (x.coords[:, 1:].float() + 0.5) / (self.res >> level_idx) * 2 - 1
                for block_idx, block in enumerate(blocks):
                    if self.down_type == "res":
                        x = block(x)
                    elif self.down_type == "swin":
                        x = block(
                            x, pos)
                x = ds(x)
            
            xcode = (x.coords.int() * lat_offset).sum(dim=-1, dtype=torch.int32)
            xindex = torch.searchsorted(vox_code, xcode)
            feats = scatter_add(
                x.feats,
                xindex,
                dim=0,
                out=torch.zeros(
                    len(vox_idxs), x.feats.shape[-1], device=x.device, dtype=x.dtype
                ),
            )
        else:
            pos_code = pcd.mul(self.latres).floor().clamp(max=self.latres - 1).long()
            pos_code = torch.cat(
                [
                    torch.arange(B, device=pos_code.device, dtype=pos_code.dtype)
                    .view(B, 1)
                    .repeat_interleave(N, dim=1)
                    .view(B * N, 1),
                    pos_code.view(B * N, 3),
                ],
                dim=-1,
                    )
            pos_code = (pos_code * lat_offset).sum(dim=-1)
            pos_index = torch.searchsorted(vox_code, pos_code)
            feats = self.reduce_pcd(vox_idxs, h, pos_index)
            feats = self.to_feats(feats)
        
        return feats
    
    def encode_cube_feats(self, vox_idxs, cube_idx, cube_feat):
        
        cube_lat = self.down_in(cube_feat)
        x = sp.SparseTensor(cube_lat, cube_idx)
        for level_idx, (blocks, ds) in enumerate(zip(self.down_blocks, self.downs)):
            for block_idx, block in enumerate(blocks):
                x = block(x)
            x = ds(x)
        
        vox_code = vox_idxs.int()
        lat_offset = torch.tensor(
            [
                self.latres * self.latres * self.latres,
                self.latres * self.latres,
                self.latres,
                1,
                ],
            device=vox_code.device,
            dtype=vox_code.dtype,
        )
        vox_code = (vox_code * lat_offset).sum(dim=-1)
        
        xcode = (x.coords.int() * lat_offset).sum(dim=-1, dtype=torch.int32)
        xindex = torch.searchsorted(vox_code, xcode).clamp(min=0, max=len(vox_code) - 1)
        vox_mask = vox_code[xindex] == xcode
        # assert vox_mask.all().item()
        feats = scatter_add(
            x.feats[vox_mask],
            xindex[vox_mask],
            dim=0,
            out=torch.zeros(
                len(vox_idxs), x.feats.shape[-1], device=x.device, dtype=x.dtype
            ),
        )
        
        return feats
    
    
    def pcd_attn(self, vox_idx, cube_idx, pcd, pcd_normals):
        
        B, N, _ = pcd.shape
        
        
        cube_idx = torch.cat([cube_idx[:, :1], cube_idx[:, 1:] // (self.grid_res // self.res)], dim=-1).unique(dim=0)
        seq_len = torch.bincount(cube_idx[:, 0], minlength=B)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [
            slice((offset[i] - seq_len[i]).item(), offset[i].item())
            for i in range(B)
        ]
        
        outs = []
        for bidx, lay in enumerate(layout):
            
            key_pcd = (cube_idx[lay, 1:].float() + 0.5) / self.res * 2 - 1
            key_feat = None
            
            if self.use_edges:
                salient_code = torch.stack([
                    torch.zeros(N // 2, device=pcd.device, dtype=pcd.dtype),
                    torch.ones(N // 2, device=pcd.device, dtype=pcd.dtype)
                ], dim=-1).view(-1, 1)
            else:
                salient_code = torch.zeros(N, device=pcd.device, dtype=pcd.dtype).view(-1, 1)
            
            points = pcd[bidx]
            point_extras = torch.cat([salient_code, pcd_normals[bidx]], dim=-1).float()
            
            for layer_idx in range(self.pcd_attn_layers):
                
                key_q = self.to_q[layer_idx - 1](key_feat) if layer_idx > 0 else key_feat
                
                attn_out = pcd_attn(
                    keypoints=key_pcd, points=points, point_extras=point_extras,
                    key_feat=key_q, #point_kfeat=kfeats, point_vfeat=vfeats,
                    weights=self.pcd_attn_weights[layer_idx], bias=self.pcd_attn_bias[layer_idx],
                    head_dim=self.head_dim, K=self.pcd_attn_k, radius=2 * np.sqrt(3) / self.res,
                    scale=1.0, coord_scale=self.res / 2, use_wmma=False
                )[0]
                key_feat = self.to_out[layer_idx](attn_out) + (key_feat if key_feat is not None else 0)
                key_feat = self.ffns[layer_idx](key_feat) + key_feat
            outs.append(key_feat)
        
        cube_feat = torch.cat(outs) if len(outs) > 1 else outs[0]
        
        
        if self.use_conv:
            
            x = sp.SparseTensor(cube_feat, cube_idx)
            
            for level_idx, (blocks, ds) in enumerate(zip(self.down_blocks, self.downs)):
                for block_idx, block in enumerate(blocks):
                    x = block(x)
                x = ds(x)
        else:
            x = sp.SparseTensor(self.pcd_map(cube_feat), cube_idx)
        
        
        lat_offset = torch.tensor(
            [
                self.latres * self.latres * self.latres,
                self.latres * self.latres,
                self.latres,
                1,
                ],
            device=vox_idx.device,
            dtype=torch.int32,
        )
        vox_code = (vox_idx.int() * lat_offset).sum(dim=-1, dtype=torch.int32)
        
        xcode = (x.coords.int() * lat_offset).sum(dim=-1, dtype=torch.int32)
        xindex = torch.searchsorted(vox_code, xcode).clamp(min=0, max=len(vox_code) - 1)
        vox_mask = vox_code[xindex] == xcode
        
        feats = scatter_add(
            x.feats[vox_mask],
            xindex[vox_mask],
            dim=0,
            out=torch.zeros(
                len(vox_idx), x.feats.shape[-1], device=x.device, dtype=x.dtype
            ),
        )
        
        return feats
    
    
    def encode_cube_points(self, vox_idxs, cube_idx, cube_feat):
        
        cube_lat = self.key_map(cube_feat)
        x = sp.SparseTensor(cube_lat, cube_idx)
        for level_idx, (blocks, ds) in enumerate(zip(self.down_blocks, self.downs)):
            for block_idx, block in enumerate(blocks):
                x = block(x)
            x = ds(x)
        
        vox_code = vox_idxs.int()
        lat_offset = torch.tensor(
            [
                self.latres * self.latres * self.latres,
                self.latres * self.latres,
                self.latres,
                1,
                ],
            device=vox_code.device,
            dtype=vox_code.dtype,
        )
        vox_code = (vox_code * lat_offset).sum(dim=-1)
        
        xcode = (x.coords.int() * lat_offset).sum(dim=-1, dtype=torch.int32)
        xindex = torch.searchsorted(vox_code, xcode).clamp(min=0, max=len(vox_code) - 1)
        vox_mask = vox_code[xindex] == xcode
        feats = scatter_add(
            x.feats[vox_mask],
            xindex[vox_mask],
            dim=0,
            out=torch.zeros(
                len(vox_idxs), x.feats.shape[-1], device=x.device, dtype=x.dtype
            ),
        )
        
        return feats
    
    def forward(
            self,
            vox_idxs: torch.Tensor,
            pcd: torch.Tensor=None,
            pcd_normals: torch.Tensor=None,
            cube_idxs: torch.Tensor=None,
            cube_feat: torch.Tensor=None,
    ):
        
        if self.use_pcd_attn:
            feats = self.pcd_attn(vox_idxs, cube_idxs, pcd, pcd_normals)
        elif self.encode_attributes:
            feats = self.encode_cube_feats(vox_idxs, cube_idxs, cube_feat)
        else:
            feats = self.encode_pcd(vox_idxs, pcd, pcd_normals)
        
        # add global voxel pe
        pos = vox_idxs[:, 1:].float()
        if self.pos_emb:
            pos = (pos + 0.5) / self.latres * 128
            feats = feats + get_positional_encoding(
                pos,
                feats.shape[-1],
                scale=10000 ** (-1.0),
                include_pi=False,
                logspace=True,
            ).type_as(feats)
        elif self.rope:
            pos = (pos + 0.5) / self.latres * 2 - 1
        
        x = sp.SparseTensor(feats, vox_idxs)
        sp.put_attn_orders(x, self.orders, **self.order_kwargs)
        
        for block_idx, block in enumerate(self.blocks):
            x = block(x, pos)
        
        x = x.replace(self.out_norm(x.feats))
        mean = x.replace(self.out_mean(x.feats))
        logvar = x.replace(self.out_logvar(x.feats))
        
        return {
            "mean": mean,
            "logvar": logvar,
        }


class SparseDecoder(torch.nn.Module):
    def __init__(
            self,
            out_dim=1 * 8 + 3 * 8 + 21,  # sdf + deform + weights
            sdf_separate=False,
            latent_dim=16,
            res=128,
            pos_emb: bool = True,
            rope: bool = False,
            up_dims=(512, 256, 128, 64),
            up_depths=(1, 1, 1, 1),
            up_type: Literal["res", "swin"] = "res",
            up_skip: bool = False,
            up_dilation: int = 0,
            up_noise: float = 0.0,
            up_safe_rate: float = 1.0,
            up_partial: bool = False,
            up_chunks: int = 1,
            max_latlen: int = 1 << 17,
            up_padding: bool = True,
            res_norm: Literal["pixel", "layer", "rms", "group"] = "pixel",
            res_conv: Literal["full", "fast"] = "full",
            orders: List[Literal["full", "swin", "swin_shift", "z", "zt", "h", "ht", "xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"]] = ("swin", "swin_shift"),
            decoder_dim=1024,
            decoder_depth=12,
            context_attn: bool = False,
            adj_first: bool = False,
            head_dim=64,
            window_size=8,
            patch_size: int = 1024,
            bucket_size: int = 1 << 9,
            global_size: int = 1 << 9,
            voxel_limits: tuple[int] = (400_000, 1_600_000, 6_400_000),
            gc: bool = False,
            save_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        
        self.latres = res
        self.pos_emb = pos_emb
        self.rope = rope
        
        self.max_latlen = max_latlen
        self.voxel_limits = voxel_limits
        self.up_padding = up_padding
        self.up_partial = up_partial
        
        self.input_map = torch.nn.Linear(latent_dim, decoder_dim)
        
        self.orders = orders
        self.order_kwargs = {
            "window_size": window_size,
            "patch_size": patch_size,
            "bucket_size": bucket_size,
            "global_size": global_size,
            "adj_first": adj_first,
        }
        self.blocks = torch.nn.ModuleList(
            [
                SparseTransformerBlock(
                    head_dim=head_dim,
                    num_heads=decoder_dim // head_dim,
                    order_idx=idx % len(self.orders),
                    add_context=context_attn,
                    rope=rope,
                    gc=gc,
                )
                for idx in range(decoder_depth)
            ]
        )
        
        self.up_dilation = up_dilation
        self.up_noise = up_noise
        self.up_chunks = up_chunks
        self.up_safe_rate = up_safe_rate
        self.up_type = up_type
        self.up_dims = up_dims
        self.up_blocks = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.prunes = torch.nn.ModuleList([])
        for idx in range(len(up_dims)):
            
            if idx == 0:
                self.ups.append(
                    init_linear(
                        sp.SparseLinear(decoder_dim, up_dims[idx], bias=False),
                        normal=True,
                    )
                )
            else:
                self.ups.append(
                    sp.SparseUpsample(up_dims[idx - 1], up_dims[idx], patch_size=2, use_skip=up_skip)
                )
            
            if idx < len(up_dims) - 1:
                self.prunes.append(SparseFFN(up_dims[idx], 8, gc=gc))
            else:
                self.prunes.append(sp.SparseIdentity())
            
            blocks = torch.nn.ModuleList([])
            for d in range(up_depths[idx]):
                if up_type == "res":
                    blocks.append(
                        SparseResBlock3d(f"layer_{idx}", hidden_dim=up_dims[idx], norm_type=res_norm, conv_type=res_conv, gc=gc, save_memory=save_memory)
                    )
                elif up_type == "swin":
                    block_heads = max(up_dims[idx] // head_dim, 4)
                    blocks.append(
                        SparseTransformerBlock(
                            head_dim=up_dims[idx] // block_heads,
                            num_heads=block_heads,
                            order_idx=d % len(orders),
                            rope=rope,
                            gc=gc,
                        )
                    )
            self.up_blocks.append(blocks)
        
        self.sdf_separate = sdf_separate
        self.out_norm = LayerNorm(up_dims[-1])
        self.out_map = torch.nn.Linear(up_dims[-1], out_dim + 8 * int(sdf_separate))
    
    @torch.no_grad()
    @torch.compile()
    def _extend_coords(self, coords, batch_size, dilate, curr_res):
        offset = torch.tensor(
            [curr_res * curr_res * curr_res, curr_res * curr_res, curr_res, 1],
            device=coords.device,
            dtype=torch.int32,
        )
        code = (coords * offset).sum(dim=-1, dtype=torch.int32)
        dcoords = dilate_coords(
            coords,
            batch_size=batch_size,
            dilate=dilate,
            grid_res=curr_res
        ).int().contiguous()
        dcode = (dcoords * offset).sum(dim=-1, dtype=torch.int32)
        orig_idx = torch.searchsorted(dcode, code)
        orig_msk = torch.zeros(len(dcode), device=dcode.device, dtype=torch.bool).index_fill_(0, orig_idx, True)
        return dcoords, orig_idx, orig_msk
    
    @torch.no_grad()
    @torch.compile()
    def _get_gt_occ_mask(self, coords, gt_coords, res, factor):
        
        offset = torch.tensor(
            [res * res * res, res * res, res, 1],
            device=gt_coords.device,
            dtype=torch.int32,
        )
        ws = torch.tensor(
            [1, 2, 2, 2], device=gt_coords.device, dtype=torch.int32
        )
        window = torch.tensor(
            [0, 2 * 2, 2, 1], device=gt_coords.device, dtype=torch.int32
        )
        
        gt_level = gt_coords.clone()
        gt_level[:, 1:] //= factor
        
        gidx = ((gt_level // ws) * offset).sum(dim=-1, dtype=torch.int32)
        hidx = (coords * offset).sum(dim=-1, dtype=torch.long)
        hidx = torch.searchsorted(hidx, gidx)
        lidx = ((gt_level % ws) * window).sum(dim=-1, dtype=torch.int32)
        
        occ_mask = torch.zeros(
            len(coords) * 8, device=gt_coords.device, dtype=torch.bool
        )
        occ_mask.index_fill_(dim=0, index=hidx * 8 + lidx, value=True)
        
        return occ_mask
    
    
    @torch.no_grad()
    @torch.compile()
    def _get_up_coords(self, coords, occ_mask, res):
        cube_shifts = torch.nonzero(torch.ones([2, 2, 2], dtype=torch.int32)).int()
        cube_shifts = torch.cat(
            [torch.zeros_like(cube_shifts[:, :1]), cube_shifts], dim=-1
        )
        up_coords = coords.clone().int()
        up_coords[:, 1:] *= 2
        up_coords = (up_coords.unsqueeze(1) + cube_shifts.type_as(up_coords)).view(
            len(up_coords) * 8, 4
        )
        
        up_hidx = occ_mask.nonzero().view(-1)
        up_coords = up_coords[up_hidx]  # take only predicted coords
        
        offset = torch.tensor(
            [res * res * res, res * res, res, 1],
            device=up_coords.device,
            dtype=torch.int32,
        )
        code = (up_coords * offset).sum(dim=-1, dtype=torch.int32)
        perm = torch.argsort(code)
        
        return up_coords[perm], up_hidx[perm]
    
    def chunked_upsample(
            self,
            h: sp.SparseTensor,
            blocks: List[SparseResBlock3d|SparseTransformerBlock],
            layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: # used for eval only because backward with such problems takes too much memory
        
        up_chunks = self.up_chunks
        up_chunks //= 1 << (len(self.up_blocks) - layer_idx - 1)
        up_chunks = max(1, up_chunks)
        
        curr_res = self.latres << layer_idx
        chunk_size = curr_res // up_chunks
        padding = 2 * len(blocks)
        
        hx, hy, hz = h.coords[:, 1:].unbind(dim=-1)
        
        chunk_coords, chunk_feats = [], []
        for i, j, k in product(range(up_chunks), range(up_chunks), range(up_chunks)):
            
            chunk_mask  = (chunk_size * i - padding <= hx) & (hx < chunk_size * (i+1) + padding)
            chunk_mask &= (chunk_size * j - padding <= hy) & (hy < chunk_size * (j+1) + padding)
            chunk_mask &= (chunk_size * k - padding <= hz) & (hz < chunk_size * (k+1) + padding)
            
            _coords = h.coords[chunk_mask]
            if len(_coords) == 0:
                continue
            
            xx, yy, zz = _coords[:, 1:].unbind(dim=-1)
            out_mask  = (chunk_size * i <= xx) & (xx < chunk_size * (i+1))
            out_mask &= (chunk_size * j <= yy) & (yy < chunk_size * (j+1))
            out_mask &= (chunk_size * k <= zz) & (zz < chunk_size * (k+1))
            if out_mask.sum().item() == 0:
                continue
            
            _h = sp.SparseTensor(h.feats[chunk_mask], _coords, h.shape)
            _pos = (_coords.float() + 0.5) / (self.latres << layer_idx) * 2 - 1
            for block_idx, block in enumerate(blocks):
                if self.up_type == "swin":
                    _h = block(_h, _pos)
                else:
                    _h = block(_h)
            chunk_coords.append(_h.coords[out_mask]), chunk_feats.append(_h.feats[out_mask])
        
        chunk_feats = torch.cat(chunk_feats)
        
        chunk_coords = torch.cat(chunk_coords)
        chunk_offset = torch.tensor([
            curr_res * curr_res * curr_res, curr_res * curr_res, curr_res, 1
        ], device=h.device, dtype=torch.int32)
        chunk_code = (chunk_coords * chunk_offset).sum(dim=-1, dtype=torch.int32)
        chunk_order = torch.argsort(chunk_code)
        
        return chunk_feats, chunk_order
    
    @torch.no_grad()
    @torch.compile()
    def partial_upsample(
            self,
            coords: torch.Tensor,
            offsets: List[tuple[int, int]],
            gt_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        res_layer = len(self.up_blocks) - 1
        res_padding = 0
        if self.up_padding:
            while res_layer >= 0:
                res_padding = (res_padding + 1) // 2
                res_padding = res_padding + len(self.up_blocks[res_layer]) * 2
                res_padding = res_padding + self.up_dilation * (res_layer > 0)
                res_layer -= 1
        
        def query_sum(s, bidx, x0, y0, z0, x1, y1, z1):
            x0m, y0m, z0m = (x0 > 0).int()[:, None], (y0 > 0).int()[:, None], (z0 > 0).int()[:, None]
            return (
                    s[bidx, :, x1, y1, z1]
                    - s[bidx, :, x0 - 1, y1, z1] * x0m
                    - s[bidx, :, x1, y0 - 1, z1] * y0m
                    - s[bidx, :, x1, y1, z0 - 1] * z0m
                    + s[bidx, :, x0 - 1, y0 - 1, z1] * x0m * y0m
                    + s[bidx, :, x0 - 1, y1, z0 - 1] * x0m * z0m
                    + s[bidx, :, x1, y0 - 1, z0 - 1] * y0m * z0m
                    - s[bidx, :, x0 - 1, y0 - 1, z0 - 1] * x0m * y0m * z0m
            )
        
        # calculate count of active voxels for each layer
        vox_cnts, vox_limits = [], []
        vox_offs = torch.tensor([self.latres * self.latres * self.latres, self.latres * self.latres, self.latres, 1], device=coords.device, dtype=torch.int32)
        dcoords = dilate_coords(coords, batch_size=len(offsets), dilate=self.up_dilation, grid_res=self.latres)
        vox_code = (dcoords * vox_offs).sum(dim=-1, dtype=torch.int32)
        for layer_idx in range(len(self.up_blocks)):
            fac, fac2 = 1 << (len(self.up_blocks) - layer_idx - 1), 1 << layer_idx
            curr_coords = (gt_coords // torch.tensor([1, fac, fac, fac], device=gt_coords.device, dtype=gt_coords.dtype)).unique(dim=0)
            curr_coords = dilate_coords(
                curr_coords,
                batch_size=len(offsets),
                dilate=self.up_dilation,
                grid_res=self.latres << layer_idx
            ).int()
            curr_code = (curr_coords // torch.tensor([1, fac2, fac2, fac2], device=gt_coords.device, dtype=gt_coords.dtype) * vox_offs).sum(dim=-1, dtype=torch.int32)
            curr_idx = torch.searchsorted(vox_code, curr_code)
            vox_cnt = torch.zeros_like(vox_code).scatter_reduce_(
                dim=0,
                index=curr_idx,
                src=torch.ones_like(curr_code),
                reduce="sum",
                include_self=True,
            )
            vox = torch.zeros(len(offsets), self.latres, self.latres, self.latres, device=coords.device, dtype=torch.int32)
            vox[dcoords.unbind(dim=-1)] = vox_cnt
            vox.cumsum_(1, dtype=torch.int32).cumsum_(2, dtype=torch.int32).cumsum_(3, dtype=torch.int32)
            vox_cnts.append(vox)
            # vox_limits.append(int((2 ** 31) / (16 / 8) / self.up_dims[layer_idx] / len(offsets) * self.up_safe_rate))
            vox_limits.append(int((2 ** 31) / self.up_dims[layer_idx] / len(offsets) * self.up_safe_rate))
        vox_limits[0] = min(vox_limits[0], self.max_latlen)
        vox_cnt, vox_limits = torch.stack(vox_cnts, dim=1), torch.tensor(vox_limits, dtype=torch.int32, device=coords.device)
        
        
        bidx, low_coords = coords[:, 0], coords[:, 1:]
        low = torch.full((len(bidx),), 0, device=bidx.device, dtype=torch.int32)
        hgh = torch.full((len(bidx),), self.latres, device=bidx.device, dtype=torch.int32)
        
        for _ in range(10): # max grid_res = 1024
            mid = (low + hgh + 1) >> 1
            x0, y0, z0 = (low_coords - res_padding).clamp(min=0).unbind(dim=-1)
            x1, y1, z1 = (low_coords + mid[..., None] + res_padding).clamp(max=self.latres - 1).unbind(dim=-1)
            cnt = query_sum(vox_cnt, bidx, x0, y0, z0, x1, y1, z1)
            low = torch.where((cnt <= vox_limits).all(dim=-1), mid, low)
            hgh = torch.where((cnt <= vox_limits).all(dim=-1), hgh, mid - 1)
        up_coords = (low_coords + low[..., None]).clamp(max=self.latres - 1)
        lowx, lowy, lowz = low_coords.unbind(dim=-1)
        upx, upy, upz = up_coords.unbind(dim=-1)
        probs = query_sum(vox_cnt, bidx, lowx, lowy, lowz, upx, upy, upz) * (low[:, None] > 0).float()
        
        sub_mask, _gt_coords, _gt_coords_valid = [], [], []
        for bidx, (start, stop) in enumerate(offsets):
            probs_base = probs[start:stop]
            probs_base = (probs_base / probs_base.sum(dim=0, keepdim=True).clamp(min=1e-6)).prod(dim=-1)
            _low_coords = low_coords[start:stop]
            min_coord, max_coord = _low_coords.min(dim=0, keepdim=True).values, _low_coords.max(dim=0, keepdim=True).values
            
            n_tries = 32
            for try_idx in range(n_tries):
                
                if random.random() < 0.5:
                    _probs = probs_base * ((_low_coords == min_coord) | (_low_coords == max_coord)).any(dim=-1)
                else:
                    _probs = probs_base
                
                cdf = (_probs - _probs.quantile(0.5)).clamp(min=0).cumsum_(0)
                
                sub_idx = torch.searchsorted(cdf, torch.rand(1, device=cdf.device, dtype=cdf.dtype) * cdf[-1])
                coord = coords[start:stop]
                low_coord, up_coord = low_coords[start:stop][sub_idx], up_coords[start:stop][sub_idx]
                low_mask = (low_coord - res_padding <= coord[:, 1:]).all(dim=-1) & (coord[:, 1:] <= up_coord + res_padding).all(dim=-1)
                
                gt_coord = gt_coords[gt_coords[:, 0] == bidx]
                gt_mask_pad  = (((low_coord - res_padding) << (len(self.up_blocks) - 1)) <= gt_coord[:, 1:]).all(dim=-1)
                gt_mask_pad &= (gt_coord[:, 1:] < ((up_coord + res_padding + 1) << (len(self.up_blocks) - 1))).all(dim=-1)
                
                gt_mask  = ((low_coord << (len(self.up_blocks) - 1)) <= gt_coord[:, 1:]).all(dim=-1)
                gt_mask &= (gt_coord[:, 1:] < ((up_coord + 1) << (len(self.up_blocks) - 1))).all(dim=-1)
                
                gt_coords_valid = gt_coord[gt_mask]
                if len(gt_coords_valid) > 0:
                    sub_mask.append(low_mask), _gt_coords.append(gt_coord[gt_mask_pad]), _gt_coords_valid.append(gt_coords_valid)
                    break
                
                if try_idx == n_tries - 1:
                    sub_mask.append(low_mask), _gt_coords.append(gt_coord[gt_mask_pad]), _gt_coords_valid.append(gt_coords[gt_mask_pad])
                    break
        
        return (
            torch.cat(sub_mask),
            torch.cat(_gt_coords),
            torch.cat(_gt_coords_valid)
        )
    
    def forward(
            self, x: sp.SparseTensor, gt_coords=None, train: bool = False,
    ):
        
        h = x.replace(self.input_map(x.feats))
        
        feats = h.feats
        pos = h.coords[:, 1:].float()
        if self.pos_emb:
            pos = (pos + 0.5) / self.latres * 128
            h = h.replace(
                feats
                + get_positional_encoding(
                    pos,
                    feats.shape[-1],
                    scale=10000 ** (-1.0),
                    include_pi=False,
                    logspace=True,
                ).type_as(feats)
            )
        elif self.rope:
            pos = (pos + 0.5) / self.latres * 2 - 1
        
        sp.put_attn_orders(h, self.orders, **self.order_kwargs)
        for block_idx, block in enumerate(self.blocks):
            h = block(h, pos)
        
        # train# and any([lay.stop - lay.start > self.max_latlen for lay in h.layout])
        if self.up_partial and train:
            sub_mask, gt_coords, gt_coords_valid = self.partial_upsample(
                h.coords, [(lay.start, lay.stop) for lay in h.layout], gt_coords
            )
            h = sp.SparseTensor(h.feats[sub_mask], h.coords[sub_mask], h.shape)
        h._spatial_cache.clear()  # remove downsample keys
        up_regs = []
        
        for layer_idx, (blocks, up, prune) in enumerate(
                zip(self.up_blocks, self.ups, self.prunes)
        ):
            
            h = up(h)
            if self.up_dilation > 0 and layer_idx > 0:
                dcoords, orig_idx, orig_msk = self._extend_coords(h.coords, h.shape[0], self.up_dilation, self.latres << layer_idx)
                dfeats = scatter_add(
                    h.feats, orig_idx, dim=0,
                    out=torch.zeros(len(dcoords), h.feats.shape[-1], device=h.feats.device, dtype=h.feats.dtype)
                )
                dh = sp.SparseTensor(dfeats, dcoords, h.shape)
                dh._scale = h._scale
                h = dh
            
            if self.up_type != "res":
                sp.put_attn_orders(h, self.orders, **self.order_kwargs)
            
            pos = (h.coords[:, 1:].float() + 0.5) / (self.latres << layer_idx) * 2 - 1
            
            if self.up_chunks > 1 and not train:
                chunk_feats, chunk_order = self.chunked_upsample(h, blocks, layer_idx)
                torch.cuda.empty_cache()
                h = h.replace(chunk_feats[chunk_order])
                del chunk_feats, chunk_order
                torch.cuda.empty_cache()
            else:
                for block_idx, block in enumerate(blocks):
                    if self.up_type == "res":
                        h = block(h)
                    elif self.up_type == "swin":
                        h = block(
                            h, pos)
            
            if layer_idx < len(self.up_blocks) - 1:
                if self.up_dilation > 0 and layer_idx > 0 and not train:
                    h_orig = sp.SparseTensor(h.feats[orig_msk], h.coords[orig_msk], h.shape)
                    h_orig._scale = h._scale
                    h = h_orig
                
                logits = prune(h).feats.view(len(h.coords) * 8)
                
                if gt_coords is not None:
                    
                    occ_mask = self._get_gt_occ_mask(
                        h.coords,
                        gt_coords,
                        factor=1
                               << (len(self.up_blocks) - (layer_idx + 1) - 1),  # for next res
                        res=self.latres << (layer_idx + 0),  # for curr res
                    )
                    
                    with torch.amp.autocast(
                            enabled=True, dtype=torch.float32, device_type="cuda"
                    ):
                        up_regs.append(
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                logits.view(-1).float(), occ_mask.float()
                            )
                        )
                    
                    if train and self.up_noise > 0.0:
                        with torch.no_grad():
                            # add extra voxels based on model prediction
                            aug_mask = []
                            for lay in h.layout:
                                ratio = self.up_noise / (len(self.up_blocks) - 1) * random.random()
                                lay = slice(lay.start * 8, lay.stop * 8)
                                _logits, _occ_mask = logits[lay], occ_mask[lay]
                                if int(len(_occ_mask) * ratio) > 0:
                                    cdf = (torch.sigmoid(_logits.float()) - _occ_mask.float()).clamp(min=0.0, max=0.5).cumsum_(0)
                                    cdf_idx = torch.searchsorted(cdf, torch.rand(int(len(_occ_mask) * ratio), device=cdf.device, dtype=cdf.dtype) * cdf[-1], out_int32=True)
                                    cdf_idx = cdf_idx.clamp(min=0, max=len(_occ_mask) - 1)
                                    _occ_mask[cdf_idx] = True
                                    # add random noise voxels
                                    _occ_mask = _occ_mask | (torch.rand(*_occ_mask.shape, device=_occ_mask.device, dtype=torch.float16) < ratio)
                                
                                aug_mask.append(_occ_mask)
                            occ_mask = occ_mask | torch.cat(aug_mask)
                else:
                    occ_mask = logits > 0
                    active_voxels = occ_mask.sum(dtype=torch.int64).item()
                    if active_voxels > self.voxel_limits[layer_idx]:
                        return None
                    # print(f"{layer_idx}: {torch.distributed.get_rank()} predicted {active_voxels} active voxels")
                    
                    # if self.up_dilation > 0 and layer_idx > 0 and train:
                    #     occ_mask = occ_mask & orig_msk.repeat_interleave(8, dim=0) # do not up dilated coords
                
                up_coords, up_hidx = self._get_up_coords(
                    h.coords, occ_mask, self.latres << (layer_idx + 1)
                )
                h.register_spatial_cache("up_coords", up_coords)
                h.register_spatial_cache("up_widx", up_hidx)
        
        if self.up_partial and train:
            res = self.latres << (len(self.up_blocks) - 1)
            res_offset = torch.tensor([res * res * res, res * res, res, 1], device=gt_coords_valid.device, dtype=torch.int32)
            if self.up_dilation > 0:
                gt_coords_valid = dilate_coords(gt_coords_valid, batch_size=h.shape[0], dilate=self.up_dilation, grid_res=res)
            valid_idx = torch.searchsorted(
                h.coords.mul(res_offset).sum(dim=-1, dtype=torch.int32),
                gt_coords_valid.mul(res_offset).sum(dim=-1, dtype=torch.int32),
                out_int32=True
            ).clamp(min=0, max=len(h.coords) - 1)
            valid_idx = valid_idx[(h.coords[valid_idx] == gt_coords_valid).all(dim=-1)]
            h = sp.SparseTensor(h.feats[valid_idx], h.coords[valid_idx], h.shape)
        
        h = h.replace(torch.cat([self.out_map(self.out_norm(feat)) for feat in h.feats.split(4194304)]))
        
        attrs, attr_start = [], 0
        for attr_size in ((1 + int(self.sdf_separate)) * 8, 3 * 8, 21):
            attrs.append(h.replace(h.feats[:,attr_start: attr_start + attr_size]))
            attr_start += attr_size
        sdf, deform, weights = attrs
        
        return {
            "sdf": sdf,
            "def": deform,
            "weights": weights,
            "up_regs": up_regs,
        }

class fc_vae(torch.nn.Module):
    """Training wrapper; inference logic lives in LatVaeCore."""

    def __init__(self, config, tracer):
        super().__init__()
        self.config = config 
        from kandi3d_image23d.configs.lat_vae_config import LatVaeConfig
        from kandi3d_image23d.modules.lat_vae.core import LatVaeCore

        self._vae_config = LatVaeConfig.from_geo_config(config)
        self.core = LatVaeCore(self._vae_config, full_config=config, build_encoder=True)
        self.model_config = self.core.model_config
        self.rep_config = self.core.rep_config
        self.latres = self.core.latres
        self.res = self.core.res

    def init(self, device):
        self.device = device
        self.core.to(device)
        self.encoder = getattr(self.core, "encoder", None)
        self.decoder = self.core.decoder
        self.fc = getattr(self.core, "fc", None)
        self.extractor_type = self.core.extractor_type
        self.render_type = self.core.render_type
        self.render_res = self.core.render_res
        self.n_renders = self.core.n_renders
        self.ctx = self.core.ctx
        return self

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        from base_models.optimization_utils import fix_state_dict

        state_dict = fix_state_dict(state_dict, self.core.state_dict())
        return self.core.load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)

    def to(self, device, force=False):
        self.device = device
        self.core.to(device)
        if hasattr(self, "encoder"):
            self.encoder = self.core.encoder
        self.decoder = self.core.decoder
        self.fc = getattr(self.core, "fc", None)
        self.ctx = self.core.ctx
        return self
    
    def encode(self, data_dict, crit_dict, total_iteration):
        
        attrs = {}
        if self.model_config.get("encode_attributes", False):
            cube_idxs = sp.sparse_coords(data_dict["cube_idx"])
            fine_idxs = sp.sparse_coords(data_dict["fine_idxs"])
            cube_offset = torch.tensor(
                [
                    self.res * self.res * self.res,
                    self.res * self.res,
                    self.res, 1
                ],
                device=fine_idxs.device,
                dtype=torch.int32,
            )
            valid_idx = torch.searchsorted(
                cube_idxs.mul(cube_offset).sum(dim=-1, dtype=torch.int32),
                fine_idxs.mul(cube_offset).sum(dim=-1, dtype=torch.int32),
                out_int32=True
            ).clamp(min=0, max=len(cube_idxs) - 1)
            valid_mask = (cube_idxs[valid_idx] == fine_idxs).all(dim=-1)
            valid_idx = valid_idx[valid_mask]
            
            attrs["cube_idxs"] = fine_idxs[valid_mask]
            
            attrs["cube_feat"] = torch.cat([
                torch.cat([
                    v_sdf[cube2vidx].view(-1, 8),
                    v_def[cube2vidx].view(-1, 8 * 3)
                ], dim=-1)
                for v_sdf, v_def, cube2vidx in zip(data_dict["v_sdf"], data_dict["v_def"], data_dict["cube2vidx"])
            ])[valid_idx]
        elif self.model_config.get("use_vox_attn", False):
            
            cube_feat, cube_idxs = [], []
            for fine_idx, v, f in zip(data_dict["fine_idxs"], data_dict["vertices"], data_dict["faces"]):
                
                v_idx = (
                        fine_idx.unsqueeze(1).to(torch.int16) + torch.nonzero(
                    torch.ones([2, 2, 2], device=fine_idx.device, dtype=fine_idx.dtype)
                ).to(torch.int16)).reshape(len(fine_idx) * 8, 3)
                v_idx, cube2vidx = v_idx.unique(dim=0, return_inverse=True)
                v_pos = v_idx.float() / self.res * 2 - 1
                
                dist, tri_idx, tri_uv = get_dist(v_pos, v, f, return_attr=True)
                v0, v1, v2 = v[f[:, 0], :], v[f[:, 1], :], v[f[:, 2], :]
                tri_normal = torch.nn.functional.normalize(
                    torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
                )
                
                tri_pnt = v0[tri_idx] * (1 - tri_uv.sum(dim=-1))[:, None] + v1[tri_idx] * tri_uv[:, 0:1] + v2[tri_idx] * tri_uv[:, 1:2]
                v_feat = torch.cat([
                    dist[:, None] * self.res / 2,
                    (tri_pnt - v_pos) * self.res / 2,
                    tri_normal[tri_idx],
                    ], dim=-1)
                cube_feat.append(v_feat[cube2vidx.view(-1)].view(len(fine_idx), -1))
                cube_idxs.append(fine_idx)
            
            attrs["cube_idxs"] = sp.sparse_coords(cube_idxs)
            attrs["cube_feat"] = torch.cat(cube_feat) if len(cube_feat) > 1 else cube_feat[0]
        
        else:
            attrs["cube_idxs"] = sp.sparse_coords(data_dict["fine_idxs"])
            attrs["pcd"] = data_dict["pcd"].clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            attrs["pcd_normals"] = torch.nn.functional.normalize(data_dict["pcd_normals"].float(), dim=-1)
        
        
        vox_idxs = sp.sparse_coords(data_dict["vox_idxs"])
        
        with torch.amp.autocast(
                enabled=self.config["training"]["fp16"],
                dtype=getattr(torch, self.config["training"]["fp16_type"]),
                device_type="cuda",
        ):
            preds = self.encoder(vox_idxs, **attrs)
        
        layout = preds["mean"].layout
        mean, logvar = preds["mean"].feats.float(), preds["logvar"].feats.float()
        data_dict["latent_mean"], data_dict["latent_logvar"] = mean, logvar
        
        det = (
                      (not data_dict.get("force_vae_sample", False))
                      and data_dict.get("eval_mode", False)
              ) or self.model_config.get("det_vae", False)
        if det:
            z = mean
            data_dict["vox_feats"] = [z[lay] for lay in layout]
        else:
            z = mean + torch.randn_like(logvar) * torch.exp(logvar / 2)
            data_dict["vox_feats"] = [z[lay] for lay in layout]
            
            kl_multiplier = self.model_config.get("kl_multiplier", 1e-6)
            kl = 0.5 * torch.mean(
                mean.pow(2) + torch.exp(logvar) - 1.0 - logvar,
                dim=list(range(1, mean.ndim)),
                )
            
            crit_dict["kl_loss"] = kl.mean() * kl_multiplier
        
        if "gt_vox_idxs" in data_dict and self.model_config.get("vox_zero_weight", 0) > 0 and not data_dict.get("eval_mode", False):
            lat_offset = torch.tensor(
                [
                    self.latres * self.latres * self.latres,
                    self.latres * self.latres,
                    self.latres, 1
                ],
                device=vox_idxs.device,
                dtype=torch.long,
            )
            vox_code = (sp.sparse_coords(data_dict["vox_idxs"]).long() * lat_offset).sum(dim=-1, dtype=torch.int64)
            gt_code = (sp.sparse_coords(data_dict["gt_vox_idxs"]).long() * lat_offset).sum(dim=-1, dtype=torch.int64)
            gt_idx = torch.searchsorted(vox_code, gt_code).clamp(max=len(vox_code) - 1)
            noise_mask = torch.ones(len(vox_code), device=vox_idxs.device, dtype=torch.bool)
            noise_mask[gt_idx] = False
            crit_dict["vox_zero"] = (mean * noise_mask[:, None].type_as(mean)).square().sum() / noise_mask.sum().clamp(min=1) * self.model_config.get("vox_zero_weight", 0)
        
        return data_dict, crit_dict
    
    def decode(self, data_dict, crit_dict, total_iteration, target="all"):
        
        sparse_coords = sp.sparse_coords(data_dict["vox_idxs"])
        sparse_feats = torch.cat([feat for feat in data_dict["vox_feats"]])
        latent = sp.SparseTensor(sparse_feats.float(), sparse_coords)
        
        with torch.amp.autocast(
                enabled=self.config["training"]["fp16"],
                dtype=getattr(torch, self.config["training"]["fp16_type"]),
                device_type="cuda",
        ):
            preds = self.decoder(
                latent,
                sp.sparse_coords(data_dict.get("gt_idxs")) if "gt_idxs" in data_dict else None,
                train=(target in ("train",))
            )
        del sparse_coords, sparse_feats, latent
        
        if preds is None: # predicted invalid mesh
            data_dict["mesh_vertices"] = [torch.zeros(1, 3, device=self.device, dtype=torch.float32)]
            data_dict["mesh_triangles"] = [torch.zeros(1, 3, device=self.device, dtype=torch.int32)]
            return data_dict, crit_dict
        
        sdf, deform, weights = preds["sdf"], preds["def"].feats, preds["weights"].feats
        del preds
        
        if not self.rep_config.get("flexible", True):
            deform, weights = deform * 0, weights * 0
        torch.cuda.empty_cache()
        
        crits, artifacts = defaultdict(list), defaultdict(list)
        for bidx, lay in enumerate(sdf.layout):
            
            with torch.no_grad():
                cube_idx = sdf.coords[lay, 1:]
                cube_shifts = torch.nonzero(
                    torch.ones([2, 2, 2], device=cube_idx.device, dtype=cube_idx.dtype)
                )
                v_idx = (
                        cube_idx.unsqueeze(1).to(torch.int16) + cube_shifts.to(torch.int16)
                ).reshape(len(cube_idx) * 8, 3)
                v_idx, cube2vidx = v_idx.unique(dim=0, return_inverse=True)
                v_idx = v_idx.int()
            
            sdf_separate = self.rep_config.get("sdf_separate", False)
            n_sdf = 1 + int(sdf_separate)
            if len(sdf.layout) > 1:
                cube_sdf, cube_def, cube_weights = sdf.feats[lay], deform[lay], weights[lay]
            else:
                cube_sdf, cube_def, cube_weights = sdf.feats, deform, weights
            
            cube_vattr = torch.cat(
                [cube_sdf.reshape(-1, n_sdf), cube_def.reshape(-1, 3)], dim=-1
            )
            v_attr = scatter_mean(
                cube_vattr,
                cube2vidx,
                dim=0,
                out=torch.zeros(
                    len(v_idx),
                    cube_vattr.shape[-1],
                    device=cube_vattr.device,
                    dtype=cube_vattr.dtype,
                ),
            )
            
            v_sdf_pred, v_def_pred = v_attr[:, :n_sdf].float(), v_attr[:, n_sdf:n_sdf + 3].float()
            
            v_pos = v_idx.float() / self.res * 2 - 1
            v_pos_def = v_pos + torch.tanh(v_def_pred) / self.res * (1 - 1e-8)
            
            # only first component of two taked for sdf
            sdf_first = self.rep_config.get("sdf_first", False)
            if sdf_separate and not sdf_first:
                cube_logit_pred, cube_mag_pred = cube_sdf.reshape(-1, 2).unbind(dim=-1)
                cube_sdf_pred = torch.sign(cube_logit_pred) * cube_mag_pred
                v_logit_pred, v_mag_pred = v_sdf_pred.unbind(dim=-1)
                v_sdf_pred = torch.sign(v_logit_pred) * v_mag_pred
            else:
                cube_sdf_pred = cube_sdf.reshape(-1, n_sdf)[:, 0]
                cube_logit_pred, cube_mag_pred = cube_sdf_pred, cube_sdf_pred.abs()
                v_sdf_pred = v_sdf_pred.view(-1, n_sdf)[:, 0]
                v_logit_pred, v_mag_pred = v_sdf_pred, v_sdf_pred.abs()
            
            if target == "gt":
                v_sdf_pred = get_sdf(v_pos_def, data_dict["vertices"][bidx].float(), data_dict["faces"][bidx].int(), return_attr=False).view(-1) / 2 * self.res
            del cube_sdf, cube_def, cube_vattr, v_attr, v_idx, v_pos, v_def_pred
            torch.cuda.empty_cache() #, gc.collect()
            
            if self.extractor_type == "fc":
                verts, faces, _, colors = self.fc(
                    voxelgrid_vertices=v_pos_def,
                    scalar_field=v_sdf_pred,
                    cube_idx=cube2vidx.view(-1, 8),
                    resolution=self.res,
                    beta=cube_weights[:, :12].float(),
                    alpha=cube_weights[:, 12:20].float(),
                    gamma_f=cube_weights[:, 20].float(),
                    cube_index_map=cube_idx,
                    training=False,
                )
            else:
                verts, faces = fmc(v_pos_def, v_sdf_pred, cube2vidx.view(-1, 8))
            
            artifacts["mesh_vertices"].append(verts.float()), artifacts["mesh_triangles"].append(faces.int())
        
        for name, vals in artifacts.items():
            data_dict[name] = vals
        
        for k, vals in crits.items():
            crit_dict[k] = torch.stack(vals).mean()
        
        return data_dict, crit_dict
    
    def forward(self, data_dict, crit_dict, total_iteration, target="all"):
        return self.core.forward(data_dict, crit_dict, total_iteration, target)
  