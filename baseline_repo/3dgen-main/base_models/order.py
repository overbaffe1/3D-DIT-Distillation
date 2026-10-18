from typing import Literal, List, Optional, Union
import torch
import triton
import triton.language as tl

@triton.jit
def _calculate_hilbert_distance(fx, fy, fz):
    
    x = fx.to(tl.int64)
    y = fy.to(tl.int64)
    z = fz.to(tl.int64)
    
    # calculate hilbert distance
    for i in tl.static_range(15, 0, -1):
        q = 1 << i
        p = q - 1
        
        # dim = 0
        x ^= tl.where((x & q) > 0, p, 0)
        
        # dim = 1
        cond = (y & q) > 0
        t = (x ^ y) & p
        x ^= tl.where(cond, p, t)
        y ^= tl.where(cond, 0, t)
        
        # dim = 2
        cond = (z & q) > 0
        t = (x ^ z) & p
        x ^= tl.where(cond, p, t)
        z ^= tl.where(cond, 0, t)
    
    y ^= x
    z ^= y
    
    t = 0
    for i in tl.static_range(15, 0, -1):
        q = 1 << i
        t ^= tl.where((z & q) > 0, q - 1, 0)
    
    x ^= t
    y ^= t
    z ^= t
    
    # write results
    ret = 0
    for i in tl.static_range(0, 16):
        q = 1 << i
        ret |= (x & q) << (2 * i + 2)
        ret |= (y & q) << (2 * i + 1)
        ret |= (z & q) << (2 * i + 0)
    
    return ret


@triton.jit
def _encode_hilbert_kernel(
        xyz_ptr,
        code_ptr,
        N,
        x_offset,
        y_offset,
        z_offset,
        str_xyz_N,
        str_xyz_C,
        BLK: tl.constexpr,
        ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    mask_n = offs_n < N
    
    xyz_ptrs = xyz_ptr + offs_n * str_xyz_N
    bx = tl.load(xyz_ptrs, mask=mask_n)
    fx = tl.load(xyz_ptrs + (1 + x_offset) * str_xyz_C, mask=mask_n)
    fy = tl.load(xyz_ptrs + (1 + y_offset) * str_xyz_C, mask=mask_n)
    fz = tl.load(xyz_ptrs + (1 + z_offset) * str_xyz_C, mask=mask_n)
    ret = _calculate_hilbert_distance(fx, fy, fz)
    
    if ASSIGN_BATCH_INDEX:
        ret |= bx.to(tl.int64) << 48
    
    tl.store(code_ptr + offs_n, ret, mask=mask_n)


def encode_hilbert(
        xyz: torch.Tensor,
        convention="xyz",
        assign_batch_index: bool = True,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): n 4, int coords
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    Returns:
        distance (Tensor): n, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 4, xyz.shape
    assert xyz.dtype == torch.int32, xyz.zdtype
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    N = len(xyz)
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")
    
    code = xyz.new_empty(N, dtype=torch.int64)
    grid = lambda meta: (triton.cdiv(N, meta["BLK"]),)
    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    _encode_hilbert_kernel[grid](
        xyz, code, N, x_offset, y_offset, z_offset, *xyz.stride(), BLK, assign_batch_index
    )
    return code

@triton.jit
def _calculate_zorder(fx, fy, fz):
    x = fx.to(tl.int64)
    y = fy.to(tl.int64)
    z = fz.to(tl.int64)
    
    # calculate z-order
    ret = 0
    for i in tl.static_range(0, 16):
        q = 1 << i
        ret |= (x & q) << (2 * i + 2)
        ret |= (y & q) << (2 * i + 1)
        ret |= (z & q) << (2 * i + 0)
    
    return ret


@triton.jit
def _encode_z_kernel(
        xyz_ptr,
        code_ptr,
        N,
        x_offset,
        y_offset,
        z_offset,
        str_xyz_N,
        str_xyz_C,
        BLK: tl.constexpr,
        ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    mask_n = offs_n < N
    
    xyz_ptrs = xyz_ptr + offs_n * str_xyz_N
    bx = tl.load(xyz_ptrs, mask=mask_n)
    fx = tl.load(xyz_ptrs + (1 + x_offset) * str_xyz_C, mask=mask_n)
    fy = tl.load(xyz_ptrs + (1 + y_offset) * str_xyz_C, mask=mask_n)
    fz = tl.load(xyz_ptrs + (1 + z_offset) * str_xyz_C, mask=mask_n)
    ret = _calculate_zorder(fx, fy, fz)
    
    if ASSIGN_BATCH_INDEX:
        ret |= bx.to(tl.int64) << 48
    
    tl.store(code_ptr + offs_n, ret, mask=mask_n)


def encode_z(
        xyz: torch.Tensor,
        convention="xyz",
        assign_batch_index: bool = True,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): n 4, int coords
        convention (str): xyz offset. Must be one of ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    Returns:
        code (Tensor): n, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 4, xyz.shape
    assert xyz.dtype == torch.int32, xyz.dtype
    convention = convention.lower()
    assert convention in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    N = len(xyz)
    x_offset, y_offset, z_offset = convention.find("x"), convention.find("y"), convention.find("z")
    
    code = xyz.new_empty(N, dtype=torch.int64)
    grid = lambda meta: (triton.cdiv(N, meta["BLK"]),)
    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    _encode_z_kernel[grid](
        xyz, code, N, x_offset, y_offset, z_offset, *xyz.stride(), BLK, assign_batch_index
    )
    return code

@triton.jit
def _calculate_xor(fx, fy, fz):
    x = fx.to(tl.int64)
    y = fy.to(tl.int64)
    z = fz.to(tl.int64)
    
    # calculate coord xor
    ret = x ^ y ^ z
    
    return ret


@triton.jit
def _encode_xor_kernel(
        xyz_ptr,
        code_ptr,
        N,
        str_xyz_N,
        str_xyz_C,
        BLK: tl.constexpr,
        ASSIGN_BATCH_INDEX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    mask_n = offs_n < N
    
    xyz_ptrs = xyz_ptr + offs_n * str_xyz_N
    bx = tl.load(xyz_ptrs, mask=mask_n)
    fx = tl.load(xyz_ptrs + 1 * str_xyz_C, mask=mask_n)
    fy = tl.load(xyz_ptrs + 2 * str_xyz_C, mask=mask_n)
    fz = tl.load(xyz_ptrs + 3 * str_xyz_C, mask=mask_n)
    ret = _calculate_xor(fx, fy, fz)
    
    if ASSIGN_BATCH_INDEX:
        ret |= bx.to(tl.int64) << 48
    
    tl.store(code_ptr + offs_n, ret, mask=mask_n)


def encode_xor(
        xyz: torch.Tensor,
        assign_batch_index: bool = True,
):
    """Returns z-order code from given normalized point cloud
    Args:
        xyz (Tensor): n 4, int coords
    Returns:
        code (Tensor): n, int64
    """
    assert xyz.ndim == 2 and xyz.size(-1) == 4, xyz.shape
    assert xyz.dtype == torch.int32, xyz.dtype
    N = len(xyz)
    
    code = xyz.new_empty(N, dtype=torch.int64)
    grid = lambda meta: (triton.cdiv(N, meta["BLK"]),)
    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    _encode_xor_kernel[grid](
        xyz, code, N, *xyz.stride(), BLK, assign_batch_index
    )
    return code

@triton.jit
def _batch_index_kernel(
        code_ptr,
        bidx_ptr,
        bcode_ptr,
        N,
        BLK: tl.constexpr
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLK + tl.arange(0, BLK)
    mask_n = offs_n < N
    
    code = tl.load(code_ptr + offs_n, mask=mask_n)
    bx = tl.load(bidx_ptr + offs_n, mask=mask_n)
    
    code |= bx.to(tl.int64) << 48
    
    tl.store(bcode_ptr + offs_n, code, mask=mask_n)


def assign_batch_index(code: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    assert code.ndim == 1 and code.dtype == torch.int64, f"{code.shape} {code.dtype}"
    assert batch_idx.ndim == 1 and batch_idx.dtype == torch.int32, f"{batch_idx.shape} {batch_idx.dtype}"
    assert len(code) == len(batch_idx), f"{code.shape} {batch_idx.shape}"
    N = len(code)
    bcode = code.new_empty(N, dtype=torch.int64)
    grid = lambda meta: (triton.cdiv(N, meta["BLK"]),)
    BLK = max(32, min(2048, triton.next_power_of_2(N)))
    _batch_index_kernel[grid](code, batch_idx, bcode, N, BLK)
    return bcode

def encode(
        xyz: torch.Tensor,
        mode: Literal["z", "zt", "h", "ht"] = "z"
) -> torch.Tensor:
    
    if mode == "z":
        return encode_z(xyz, "xyz")
    
    if mode == "zt":
        return encode_z(xyz, "yxz")
    
    if mode == "z1":
        return encode_z(xyz + 1, "xyz")
    
    if mode == "zt1":
        return encode_z(xyz + 1, "yxz")
    
    if mode == "h":
        return encode_hilbert(xyz, "xyz")
    
    if mode == "h1":
        return encode_hilbert(xyz + 1, "xyz")
    
    if mode == "ht":
        return encode_hilbert(xyz, "yxz")
    
    if mode == "ht1":
        return encode_hilbert(xyz + 1, "yxz")
    
    if mode == "x":
        return encode_xor(xyz)
    
    raise NotImplementedError(f"mode {mode} not implemented")


def calc_order_partition(
        coords: torch.Tensor,
        sizes: List[int],
        mode: Literal["z", "zt", "h", "ht", "x"] = "z",
        patch_size: int = 1024,
        adj_first: bool = False,
        sliding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    
    code = encode(coords, mode)
    
    if adj_first:
        
        from ops.sparse import vox_hierarchical_patches
        comp_idx = vox_hierarchical_patches(coords, code, patch_size)
        
        fwd_indices = torch.argsort(comp_idx)
        bwd_indices = torch.zeros_like(fwd_indices).scatter_(
            dim=0,
            index=fwd_indices,
            src=torch.arange(len(fwd_indices), dtype=fwd_indices.dtype, device=fwd_indices.device)
        )
        
        sizes = comp_idx.unique(return_counts=True)[1].int()
        max_len = sizes.max().item()
        cu_seqlens = torch.cat([torch.zeros_like(sizes[:1]), sizes]).cumsum(dim=0, dtype=torch.int32)
    
    else:
        
        fwd_indices = torch.argsort(code)
        bwd_indices = torch.zeros_like(fwd_indices).scatter_(
            dim=0,
            index=fwd_indices,
            src=torch.arange(len(fwd_indices), dtype=fwd_indices.dtype, device=fwd_indices.device)
        )
        
        if sliding:
            seqlens = torch.tensor(sizes, device=fwd_indices.device, dtype=torch.int32)
        else:
            sizes = torch.tensor(sizes, device=fwd_indices.device, dtype=torch.int32)
            seqlens = torch.tensor([patch_size] * len(sizes), device=fwd_indices.device, dtype=torch.int32)
            npatches = (sizes + patch_size - 1) // patch_size
            seqlens = seqlens.repeat_interleave(npatches)
            seqlens[npatches.cumsum(dim=0) - 1] = sizes - (npatches - 1) * patch_size
        
        max_len = seqlens.max().item()
        cu_seqlens = torch.cat([torch.zeros_like(seqlens[:1]), seqlens]).cumsum(dim=0, dtype=torch.int32)
    
    return fwd_indices, bwd_indices, cu_seqlens, max_len


def calc_mixed_partition(
        coords: torch.Tensor,
        sizes: List[int],
        mode: Literal["xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"],
        bucket_size: int = 1 << 9,
        global_size: int = 1 << 9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    
    if mode[0] == "x":
        code = encode_xor(coords, assign_batch_index=False)
    if mode[0] == "z":
        code = encode_z(coords, "xyz", assign_batch_index=False)
    if mode[0] == "h":
        code = encode_hilbert(coords, "xyz", assign_batch_index=False)
    
    if mode.endswith("div"):
        bucket_idx = code // bucket_size
    if mode.endswith("mod"):
        bucket_idx = code % global_size
    
    bucket_idx = assign_batch_index(bucket_idx, coords[:, 0].int().contiguous())
    fwd_indices = torch.argsort(bucket_idx)
    bwd_indices = torch.zeros_like(fwd_indices).scatter_(
        dim=0,
        index=fwd_indices,
        src=torch.arange(len(fwd_indices), dtype=fwd_indices.dtype, device=fwd_indices.device)
    )
    
    _, seqlens = torch.unique_consecutive(bucket_idx[fwd_indices], return_counts=True)
    
    max_len = seqlens.max().item()
    cu_seqlens = torch.cat([torch.zeros_like(seqlens[:1]), seqlens]).cumsum(dim=0, dtype=torch.int32)
    
    return fwd_indices, bwd_indices, cu_seqlens, max_len