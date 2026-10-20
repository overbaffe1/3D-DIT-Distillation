import os
import numpy as np
import torch

try:
    from . import custom_ops  # usual app run
except:
    import custom_ops  # run in current directory

_libsparse = None


def _init(verbose: bool = True):
    global _libsparse
    if _libsparse is None:
        try:
            import libsparse as _libsparse
        except ImportError:
            _libsparse = custom_ops.load_ext(
                "libsparse",
                headers=["../libmesh/cutil_math.h", "../libmesh/bvh.h", "pcd_attn.h"],
                sources=["conv.cu", "vox.cu", "../libmesh/bvh.cu", "pcd_attn.cu", "libsparse.cpp"],
                source_root=os.path.join(os.path.dirname(__file__), "libsparse"),
                verbose=verbose,
                optimize=False,
            )
    return True


class FastSparseConv_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat, weight, bias, indices):
        
        feat = feat.contiguous()
        weight = weight.float().contiguous()
        bias = bias.float().contiguous()
        
        ctx.save_for_backward(feat, weight, bias)
        ctx.indices = indices
        
        out = _libsparse.conv_fwd(feat, weight, bias, indices)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        
        grad_out = grad_out.contiguous()
        feat, weight, bias = ctx.saved_tensors
        indices = ctx.indices
        
        grad_dx = _libsparse.conv_dx(grad_out, weight, indices)
        grad_dw, grad_db = _libsparse.conv_dw(feat.type_as(grad_out), grad_out, indices)
        
        if not bias.requires_grad:
            grad_db = None
        
        return grad_dx, grad_dw, grad_db, None

def get_conv_indices(pos: torch.Tensor) -> torch.Tensor:
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    
    return _libsparse.conv_indices(pos.int().contiguous())

def conv(
        feat: torch.Tensor,
        pos: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        indices: torch.Tensor = None,
):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    if indices is None:
        indices = _libsparse.conv_indices(pos.int().contiguous())
    return FastSparseConv_func.apply(feat, weight, bias, indices)


class FastDenseSparseConv_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat, weight, bias, indices):
        feat = feat.contiguous()
        weight = weight.float().contiguous()
        bias = bias.float().contiguous()
        
        ctx.save_for_backward(feat, weight)
        ctx.indices = indices
        
        out = _libsparse.dense_conv_fwd(feat, weight, bias, indices)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        feat, weight = ctx.saved_tensors
        indices = ctx.indices
        
        grad_dx = _libsparse.dense_conv_dx(grad_out, weight, indices)
        grad_dw, grad_db = _libsparse.dense_conv_dw(feat.type_as(grad_out), grad_out, indices)
        
        return grad_dx, grad_dw, grad_db, None


def dense_conv(
        feat: torch.Tensor,
        pos: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        indices: torch.Tensor = None,
):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    if indices is None:
        indices = _libsparse.conv_indices(pos.int().contiguous())
    return FastDenseSparseConv_func.apply(feat, weight, bias, indices)


def vox_connected_components(pos: torch.Tensor):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    return _libsparse.vox_connected_components(pos.int().contiguous())


def vox_hierarchical_patches(pos: torch.Tensor, code: torch.Tensor, max_patch_size: int):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    pos_c  = pos.int().contiguous()
    code_c = code.long().contiguous()
    return _libsparse.vox_hierarchical_patches(pos_c, code_c, int(max_patch_size))


class PCDAttn_func(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            keypoints,
            key_feat,
            points,
            point_kfeat,
            point_vfeat,
            weights,
            bias,
            head_dim: int,
            K: int = 256,
            radius: float = 2 * np.sqrt(3),
            point_extras=None,
            coord_scale: float = 64.0,
            rel_clamp=None,
            split_q_bwd: int = 1,
            split_k_bwd: int = 1,
            proj_share_group: int = 1,
            nn_idx=None,
            nn_count=None,
    ):
        rel_clamp = float(radius) if rel_clamp is None and radius > 0.0 else (0.0 if rel_clamp is None else float(rel_clamp))
        radius = radius / 2
        points = (points.float() / 2 + 0.5).clamp(0, 1).contiguous()
        keypoints = (keypoints.float() / 2 + 0.5).clamp(0, 1).contiguous()
        
        ctx.req_key_feat = (key_feat is not None and key_feat.requires_grad)
        ctx.req_point_kfeat = (point_kfeat is not None and point_kfeat.requires_grad)
        ctx.req_point_vfeat = (point_vfeat is not None and point_vfeat.requires_grad)
        ctx.req_weights = (weights is not None and weights.requires_grad)
        ctx.req_bias = (bias is not None and bias.requires_grad)
        
        key_feat = key_feat.float().contiguous() if key_feat is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        point_kfeat = point_kfeat.float().contiguous() if point_kfeat is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        point_vfeat = point_vfeat.float().contiguous() if point_vfeat is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        point_extras = point_extras.float().contiguous() if point_extras is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        
        weights_orig = weights
        bias_orig = bias
        weights = weights.float().contiguous() if weights is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        bias = bias.float().contiguous() if bias is not None else torch.empty(0, device=keypoints.device, dtype=torch.float32)
        nn_idx = nn_idx.int().contiguous() if nn_idx is not None else torch.empty(0, device=keypoints.device, dtype=torch.int32)
        nn_count = nn_count.int().contiguous() if nn_count is not None else torch.empty(0, device=keypoints.device, dtype=torch.int32)
        
        out, nn_idx, nn_count, head_max, head_denom = _libsparse.pcd_attn(
            keypoints, key_feat, points, point_kfeat, point_vfeat, point_extras,
            weights, bias, head_dim, K, radius, coord_scale, rel_clamp, proj_share_group,
            nn_idx, nn_count
        )
        ctx.save_for_backward(key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom,
                              weights_orig if weights_orig is not None else weights,
                              bias_orig if bias_orig is not None else bias,
                              out)
        ctx.head_dim = head_dim
        ctx.coord_scale = coord_scale
        ctx.rel_clamp = rel_clamp
        ctx.split_q_bwd = split_q_bwd
        ctx.split_k_bwd = split_k_bwd
        ctx.proj_share_group = proj_share_group
        return out, nn_idx, nn_count, head_max, head_denom
    
    @staticmethod
    def backward(ctx, grad_out, grad_nn_idx=None, grad_nn_count=None, grad_head_max=None, grad_head_denom=None):
        key_feat, keypoints, points, point_kfeat, point_vfeat, point_extras, nn_idx, nn_count, head_max, head_denom, weights, bias, out = ctx.saved_tensors
        
        weights_bwd = weights.float() if weights.numel() > 0 else weights
        bias_bwd = bias.float() if bias.numel() > 0 else bias
        grad_out_bwd = grad_out.float().contiguous()
        
        grad_kf, grad_pkf, grad_pvf, grad_w, grad_b = _libsparse.pcd_attn_bwd(
            grad_out_bwd, key_feat, keypoints, points,
            point_kfeat, point_vfeat, point_extras,
            nn_idx, nn_count, head_max, head_denom, weights_bwd, bias_bwd, out,
            ctx.head_dim, ctx.coord_scale, ctx.rel_clamp, ctx.split_q_bwd, ctx.split_k_bwd, ctx.proj_share_group
        )
        
        if weights.numel() == 0 or not ctx.req_weights:
            grad_w = None
        if bias.numel() == 0 or not ctx.req_bias:
            grad_b = None
        if key_feat.numel() == 0 or not ctx.req_key_feat:
            grad_kf = None
        if point_kfeat.numel() == 0 or not ctx.req_point_kfeat:
            grad_pkf = None
        if point_vfeat.numel() == 0 or not ctx.req_point_vfeat:
            grad_pvf = None
        
        return None, grad_kf, None, grad_pkf, grad_pvf, grad_w, grad_b, None, None, None, None, None, None, None, None, None, None, None


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
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    
    _ = use_wmma
    _ = scale
    _ = key_extras
    if key_feat is None and point_kfeat is None and point_vfeat is None:
        key_feat = torch.empty((*keypoints.shape[:-1], 0), dtype=torch.float32, device=keypoints.device)
    
    return PCDAttn_func.apply(
        keypoints,
        key_feat,
        points,
        point_kfeat,
        point_vfeat,
        weights,
        bias,
        head_dim,
        K,
        radius,
        point_extras,
        coord_scale,
        rel_clamp,
        split_q_bwd,
        split_k_bwd,
        proj_share_group,
        nn_idx,
        nn_count
    )


def pcd_knn(keypoints, points, K: int, radius: float = 0.0):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    
    radius = radius / 2
    points = (points.float() / 2 + 0.5).clamp(0, 1).contiguous()
    keypoints = (keypoints.float() / 2 + 0.5).clamp(0, 1).contiguous()
    return _libsparse.pcd_knn(keypoints, points, K, radius)


def pcd_vox_knn(keypoints, points, grid_res: int, dilation: int):
    if not _init():
        raise RuntimeError(
            f"Cannot load libsparse in {__file__} set VERBOSE=True for more information!"
        )
    
    points = (points.float() / 2 + 0.5).clamp(0, 1).contiguous()
    keypoints = (keypoints.float() / 2 + 0.5).clamp(0, 1).contiguous()
    return _libsparse.pcd_vox_knn(keypoints, points, int(grid_res), int(dilation))


for _name in (
    "get_conv_indices",
    "conv",
    "dense_conv",
    "vox_connected_components",
    "vox_hierarchical_patches",
    "pcd_attn",
    "pcd_knn",
    "pcd_vox_knn",
):
    globals()[_name] = torch._dynamo.disable(globals()[_name])
 
