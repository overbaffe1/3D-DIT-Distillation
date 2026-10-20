from typing import *
import random
import os
import numpy as np
import torch
from base_models.order import calc_order_partition, calc_mixed_partition

BACKEND = "spconv"
# BACKEND = "torchsparse"

SparseTensorData = None
import importlib

if BACKEND == "spconv":
    SparseTensorData = importlib.import_module(
        "spconv.pytorch"
    ).SparseConvTensor
else:
    SparseTensorData = importlib.import_module("torchsparse").SparseTensor



class SparseTensor:
    """

    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    
    @overload
    def __init__(
            self,
            feats: torch.Tensor,
            coords: torch.Tensor,
            shape: Optional[torch.Size] = None,
            layout: Optional[List[slice]] = None,
            **kwargs,
    ): ...
    
    @overload
    def __init__(
            self,
            data,
            shape: Optional[torch.Size] = None,
            layout: Optional[List[slice]] = None,
            **kwargs,
    ): ...
    
    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend
        # global SparseTensorData
        # if SparseTensorData is None:
        #     import importlib
        #
        #     if BACKEND == "spconv":
        #         SparseTensorData = importlib.import_module(
        #             "spconv.pytorch"
        #         ).SparseConvTensor
        #     else:
        #         SparseTensorData = importlib.import_module("torchsparse").SparseTensor
        
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if "data" in kwargs else 0
        
        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if "feats" in kwargs:
                feats = kwargs["feats"]
                del kwargs["feats"]
            if "coords" in kwargs:
                coords = kwargs["coords"]
                del kwargs["coords"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]
            
            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])
            
            spatial_shape = list(coords.max(0)[0] + 1)[1:]
            
            if BACKEND == 'spconv':
                self.data = SparseTensorData(
                    feats.reshape(feats.shape[0], -1),
                    coords,
                    spatial_shape,
                    shape[0],
                    **kwargs,
                )
                self.data._features = feats
            else:
                self.data = SparseTensorData(feats, coords, **kwargs)
        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            if "data" in kwargs:
                data = kwargs["data"]
                del kwargs["data"]
            if "shape" in kwargs:
                shape = kwargs["shape"]
                del kwargs["shape"]
            if "layout" in kwargs:
                layout = kwargs["layout"]
                del kwargs["layout"]
            
            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])
        
        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get("scale", (1024, 1024, 1024))
        self._spatial_cache = kwargs.get("spatial_cache", {})
    
    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [
            slice((offset[i] - seq_len[i]).item(), offset[i].item())
            for i in range(batch_size)
        ]
        return layout
    
    @property
    def shape(self) -> torch.Size:
        return self._shape
    
    def dim(self) -> int:
        return len(self.shape)
    
    @property
    def layout(self) -> List[slice]:
        return self._layout
    
    @property
    def feats(self) -> torch.Tensor:
        if BACKEND == 'spconv':
            return self.data.features
        return self.data.F
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        if BACKEND == 'spconv':
            self.data.features = value
        else:
            self.data.F = value
    
    @property
    def coords(self) -> torch.Tensor:
        if BACKEND == 'spconv':
            return self.data.indices
        return self.data.C
    
    @coords.setter
    def coords(self, value: torch.Tensor):
        if BACKEND == "spconv":
            self.data.indices = value
        else:
            self.data.C = value
    
    @property
    def dtype(self):
        return self.feats.dtype
    
    @property
    def device(self):
        return self.feats.device
    
    @overload
    def to(self, dtype: torch.dtype) -> "SparseTensor": ...
    
    @overload
    def to(
            self,
            device: Optional[Union[str, torch.device]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> "SparseTensor": ...
    
    def to(self, *args, **kwargs) -> "SparseTensor":
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if "dtype" in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs["dtype"]
        if "device" in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs["device"]
        
        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)
    
    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)
    
    def cpu(self) -> "SparseTensor":
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> "SparseTensor":
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)
    
    def half(self) -> "SparseTensor":
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> "SparseTensor":
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> "SparseTensor":
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)
    
    def dense(self) -> torch.Tensor:
        return self.data.dense()
    
    def reshape(self, *shape) -> "SparseTensor":
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List["SparseTensor"]:
        return sparse_unbind(self, dim)
    
    def replace(
            self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None
    ) -> "SparseTensor":
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        if BACKEND == 'spconv':
            new_data = SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict,
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        else:
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        new_tensor = SparseTensor(
            new_data,
            shape=torch.Size(new_shape),
            layout=self.layout,
            scale=self._scale,
            spatial_cache=self._spatial_cache,
        )
        return new_tensor
    
    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> "SparseTensor":
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(
            -1, 3
        )
        coords = torch.cat(
            [
                torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
                coords.repeat(N, 1),
            ],
            dim=1,
        ).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)
    
    def __merge_sparse_cache(self, other: "SparseTensor") -> dict:
        new_cache = {}
        for k in set(
                list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())
        ):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache
    
    def __neg__(self) -> "SparseTensor":
        return self.replace(-self.feats)
    
    def __elemwise__(
            self, other: Union[torch.Tensor, "SparseTensor"], op: callable
    ) -> "SparseTensor":
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor
    
    def __add__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)
    
    def __radd__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.add)
    
    def __sub__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.sub)
    
    def __rsub__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))
    
    def __mul__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)
    
    def __rmul__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.mul)
    
    def __truediv__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, torch.div)
    
    def __rtruediv__(
            self, other: Union[torch.Tensor, "SparseTensor", float]
    ) -> "SparseTensor":
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (
                    self.shape[0],
                ), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        return SparseTensor(feats=feats, coords=coords)
    
    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value
    
    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(
        input: SparseTensor, other: torch.Tensor, op: callable = torch.add
) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.

    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))


def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors.

    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate.
    """
    if dim == 0:
        start = 0
        coords = []
        for input in inputs:
            coords.append(input.coords.clone())
            coords[-1][:, 0] += start
            start += input.shape[0]
        coords = torch.cat(coords, dim=0)
        feats = torch.cat([input.feats for input in inputs], dim=0)
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        output = inputs[0].replace(feats)
    
    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.

    Args:
        input (SparseTensor): Sparse tensor to unbind.
        dim (int): Dimension to unbind.
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]


def sparse_coords(idxs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            torch.cat(
                [
                    torch.full(
                        (len(vox_idx), 1),
                        item_idx,
                        device=vox_idx.device,
                        dtype=torch.int32,
                    ),
                    vox_idx.int(),
                ],
                dim=-1,
            )
            for item_idx, vox_idx in enumerate(idxs)
        ]
    )


@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
        q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
        q: torch.Tensor, kv: SparseTensor
) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
        q: SparseTensor, k: SparseTensor, v: SparseTensor
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
        q: SparseTensor, k: torch.Tensor, v: torch.Tensor
) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...


@overload
def sparse_scaled_dot_product_attention(
        q: torch.Tensor, k: SparseTensor, v: SparseTensor
) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...


def sparse_scaled_dot_product_attention(*args, **kwargs):
    
    base = None
    
    num_all_args = len(args) + len(kwargs)
    
    if num_all_args == 1:  # qkv
        qkv = args[0] if len(args) > 0 else kwargs["qkv"]
        base = qkv
        
        q_seqlen = [
            qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])
        ]
        kv_seqlen = q_seqlen
        qkv = qkv.feats  # [T, 3, H, C]
        device = qkv.device
    
    elif num_all_args == 2:  # q kv
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        if isinstance(q, SparseTensor):
            base = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, C]
        else:
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)  # [T_Q, H, C]
        device = q.device
        
        if isinstance(kv, SparseTensor):
            kv_seqlen = [
                kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])
            ]
            kv = kv.feats  # [T_KV, 2, H, C]
        else:
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)  # [T_KV, 2, H, C]
    
    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        
        if isinstance(q, SparseTensor):
            base = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats  # [T_Q, H, Ci]
        else:
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]
        device = q.device
        
        if isinstance(k, SparseTensor):
            kv_seqlen = [
                k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])
            ]
            k, v = k.feats, v.feats  # [T_KV, H, Ci]
        else:
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k, v = k.reshape(N * L, H, CI), v.reshape(N * L, H, CO)  # [T_KV, H, Ci]
    
    # import xformers.ops
    # if num_all_args == 1:
    #     q, k, v = qkv.unbind(dim=1)
    # elif num_all_args == 2:
    #     k, v = kv.unbind(dim=1)
    # mask = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    # out = xformers.ops.memory_efficient_attention(q[None,], k[None,], v[None,], mask)[0]
    
    import flash_attn
    
    cu_seqlens_q = (
        torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)])
        .int()
        .to(device)
    )
    if num_all_args in [2, 3]:
        cu_seqlens_kv = (
            torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)])
            .int()
            .to(device)
        )
    if num_all_args == 1:
        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens_q, max(q_seqlen)
        )
    elif num_all_args == 2:
        out = flash_attn.flash_attn_varlen_kvpacked_func(
            q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
        )
    elif num_all_args == 3:
        out = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
        )
    
    if base is not None:
        return base.replace(out)
    return out.reshape(N, L, H, C)



@torch.no_grad()
@torch.compile()
def calc_window_partition(
        coords: torch.Tensor,
        window_size: Union[int, Tuple[int, ...]],
        shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """
    DIM = coords.shape[1] - 1
    shift_window = (
        (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    )
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(
        shift_window, device=coords.device, dtype=torch.int32
    ).unsqueeze(0)
    
    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [(mc + 1 + ws - 1) // ws for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]
    
    shifted_coords[:, 1:] //= torch.tensor(
        window_size, device=coords.device, dtype=torch.int32
    ).unsqueeze(0)
    shifted_indices = (
            shifted_coords
            * torch.tensor(OFFSET, device=coords.device, dtype=torch.int32).unsqueeze(0)
    ).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=coords.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = (
            torch.arange(seq_lens.shape[0], device=coords.device, dtype=torch.int32)
            // OFFSET[0]
    )
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]
    
    max_len = seq_lens.max().item()
    cu_seqlens = torch.cumsum(
        torch.cat([torch.zeros_like(seq_lens[:1]), seq_lens]), dim=0, dtype=torch.int32,
    )
    
    return fwd_indices, bwd_indices, cu_seqlens, max_len


def sparse_windowed_scaled_dot_product_self_attention(
        qkv: SparseTensor,
        window_size: int = 8,
        shift_window: Union[int, Tuple[int, int, int]] = (0, 0, 0),
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """
    serialization_spatial_cache_name = f"window_partition_{window_size}_{shift_window}"
    serialization_spatial_cache = qkv.get_spatial_cache(
        serialization_spatial_cache_name
    )
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, cu_seqlens, max_len = calc_window_partition(
            qkv.coords, window_size, shift_window
        )
        qkv.register_spatial_cache(
            serialization_spatial_cache_name,
            (fwd_indices, bwd_indices, cu_seqlens, max_len),
        )
    else:
        fwd_indices, bwd_indices, cu_seqlens, max_len = serialization_spatial_cache
    
    qkv_feats = qkv.feats[fwd_indices]  # [M, 3, H, C]
    
    # import xformers.ops
    # q, k, v = qkv_feats.unbind(dim=1)                       # [M, H, C]
    # mask = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
    # out = xformers.ops.memory_efficient_attention(q[None,], k[None,], v[None,], mask)[0] # [M, H, C]
    
    import flash_attn
    out = flash_attn.flash_attn_varlen_qkvpacked_func(
        qkv_feats, cu_seqlens, max_len
    )  # [M, H, C]
    
    out = out[bwd_indices]  # [T, H, C]
    
    return qkv.replace(out)

def sparse_windowed_sdp(
        qkv: torch.Tensor,
        fwd_indices: torch.Tensor,
        bwd_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softcap: float = 0.0,
        window: int = -1,
) -> torch.Tensor:
    qkv_feats = qkv[fwd_indices]  # [M, 3, H, C]
    
    flash3 = False
    try:
        import flash_attn_interface
        flash3 = True
    except ImportError:
        import flash_attn
    
    if flash3:
        q, k, v = qkv_feats.unbind(dim=-3)
        out = flash_attn_interface.flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softcap=softcap, window_size=(window, window),
        )
    else:
        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_feats, cu_seqlens, max_seqlen,
            softcap=softcap, window_size=(window, window),
        )
    
    out = out[bwd_indices]  # [T, H, C]
    return out


def sparse_cross_sdp(
        q: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        kv: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_kv: int,
        softcap: float = 0.0,
) -> torch.Tensor:
    
    flash3 = False
    try:
        import flash_attn_interface
        flash3 = True
    except ImportError:
        import flash_attn
    
    
    if flash3:
        k, v = kv.unbind(dim=-3)
        out = flash_attn_interface.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, softcap=softcap,
        )
    else:
        out = flash_attn.flash_attn_varlen_kvpacked_func(
            q, kv, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, softcap=softcap,
        )
    
    return out


@torch.amp.autocast(enabled=False, device_type='cuda')
def sparse_linear_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: List[int]
) -> torch.Tensor:
    
    orig_dtype = q.dtype
    if q.dtype == torch.float16 or k.dtype == torch.float16 or v.dtype == torch.float16: # bf16 is ok
        q, k, v = q.float(), k.float(), v.float()
    
    outs = []
    q, k, v = torch.relu(q), torch.relu(k), v
    for qq, kk, vv in zip(
            q.split(lens), k.split(lens), v.split(lens)
    ):
        qq, kk, vv = qq.movedim(0, -1), kk.movedim(0, -1), vv.movedim(0, -1) # (h, d, n)
        out = torch.matmul(torch.matmul(vv, kk.transpose(-1, -2)), qq)
        denom = (kk.sum(dim=-1, dtype=torch.float32, keepdim=True) * qq).sum(
            dim=-2, keepdim=True, dtype=torch.float32
        )
        out = out / (denom + 1e-6).type_as(out)
        outs.append(out.movedim(-1, 0))
    
    return torch.cat(outs, dim=0).clamp(min=-65536, max=65536).to(orig_dtype)


def put_attn_orders(
        x: SparseTensor,
        orders: List[Literal["full", "swin", "swin_shift", "z", "zt", "h", "ht", "xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"]] = ("swin", "swin_shift"),
        **order_kwargs
):
    
    if order_kwargs.get("shuffle", False):
        random.shuffle(orders)
    
    for order_idx, order in enumerate(orders):
        
        if order == "full":
            fwd_indices = bwd_indices = torch.arange(
                len(x.coords), device=x.device, dtype=torch.long
            )  # already batched
            seq_lens = [lay.stop - lay.start for lay in x.layout]
            max_len = max(seq_lens)
            cu_seqlens = torch.cumsum(torch.tensor([0] + seq_lens, device=x.device, dtype=torch.int32), dim=0, dtype=torch.int32)
        elif order in ("swin", "swin_shift"):
            window_size = order_kwargs.get("window_size", 8)
            fwd_indices, bwd_indices, cu_seqlens, max_len = calc_window_partition(x.coords, window_size, window_size // 2 * int(order == "swin_shift"))
        elif order in ("z", "zt", "h", "ht", "z1", "zt1", "h1", "ht1"):
            patch_size = order_kwargs.get("patch_size", 1024)
            fwd_indices, bwd_indices, cu_seqlens, max_len = calc_order_partition(
                x.coords, [lay.stop - lay.start for lay in x.layout],
                mode=order,
                patch_size=patch_size,
                adj_first=order_kwargs.get("adj_first", False),
                sliding=order_kwargs.get("sliding", False),
            )
        elif order in ("zmod", "zdiv", "xmod", "xdiv", "hmod", "hdiv"):
            bucket_size = order_kwargs.get("bucket_size", 1 << 9)
            global_size = order_kwargs.get("global_size", 1 << 9)
            fwd_indices, bwd_indices, cu_seqlens, max_len = calc_mixed_partition(
                x.coords, [lay.stop - lay.start for lay in x.layout],
                mode=order,
                bucket_size=bucket_size, global_size=global_size
            )
        
        x.register_spatial_cache(f"attn_order{order_idx}", (fwd_indices, bwd_indices, cu_seqlens, max_len))


class SparseSiLU(torch.nn.SiLU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class SparseLinear(torch.nn.Linear):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class SparseGroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        feats = x.feats.float()
        nfeats = torch.zeros_like(feats)
        for k in range(x.shape[0]):
            bfeats = feats[x.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, x.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(x.shape[1], -1).permute(1, 0)
            nfeats[x.layout[k]] = bfeats
        return x.replace(nfeats.type_as(x.feats))


class SparseIdentity(torch.nn.Identity):
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x


class SparseDownsample(torch.nn.Module):
    def __init__(self, in_dim, dim, patch_size=2):
        super(SparseDownsample, self).__init__()
        self.patch_size = patch_size
        self.weight = torch.nn.Parameter(torch.randn(dim, in_dim * patch_size * patch_size * patch_size, dtype=torch.float32) / np.sqrt(in_dim * patch_size ** 2))
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        
        # local patch code
        lx, ly, lz = (x.coords[:, 1:].long() % self.patch_size).unbind(dim=-1)
        lidx = (lx * self.patch_size + ly) * self.patch_size + lz
        
        # window code
        code = x.coords[:, 1:].long() // self.patch_size
        offs = torch.cumprod((code.max(dim=0).values + 1).flip(dims=(-1,)), 0).flip(dims=(-1,)).long()
        offs = torch.nn.functional.pad(offs, (0, 1), value=1)
        code = x.coords[:, 0] * offs[0] + (code * offs[1:]).sum(dim=-1)
        windows, idxs = code.unique(return_inverse=True)
        
        # scatter feats
        feats = x.feats
        widx = idxs * self.patch_size * self.patch_size * self.patch_size + lidx
        h = (
            torch.zeros(len(windows) * self.patch_size * self.patch_size * self.patch_size, feats.shape[-1], device=feats.device, dtype=feats.dtype)
            .scatter_(0, widx.unsqueeze(-1).expand(*feats.shape), feats)
        )
        h = h.view(len(windows), self.patch_size * self.patch_size * self.patch_size * h.shape[-1])
        h = torch.nn.functional.linear(h, self.weight.type_as(h))
        h_coords = torch.cat([windows[:, None] // offs[0], (windows[:, None] % offs[:-1]) // offs[1:]], dim=-1)
        
        # create output tensor with saving spatial cache
        out = SparseTensor(h, h_coords.int())
        out._scale = tuple([s // self.patch_size for s in x._scale])
        out._spatial_cache = x._spatial_cache
        
        out.register_spatial_cache("up_coords", x.coords)
        out.register_spatial_cache("up_widx", widx)
        
        return out


class SparseUpsample(torch.nn.Module):
    def __init__(self, dim, out_dim, patch_size=2, use_skip=False):
        super(SparseUpsample, self).__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.weight = torch.nn.Parameter(torch.randn(patch_size * patch_size * patch_size * out_dim, dim, dtype=torch.float32) / np.sqrt(dim))
        self.use_skip = use_skip
        if use_skip:
            self.skip = torch.nn.Linear(dim, out_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.skip.weight)
    
    def forward(self, x: SparseTensor):
        
        h = torch.nn.functional.linear(x.feats, self.weight.type_as(x.feats))
        h = h.view(h.shape[0] * self.patch_size * self.patch_size * self.patch_size, self.out_dim)
        
        spatial_cache = x.get_spatial_cache()
        if "up_widx" in spatial_cache and "up_coords" in spatial_cache:
            feats = h[spatial_cache["up_widx"]]
            coords = spatial_cache["up_coords"]
            if self.use_skip:
                feats = feats + self.skip(x.feats[spatial_cache["up_widx"] // (self.patch_size * self.patch_size * self.patch_size)])
        else:
            coords = x.coords.clone()
            cube_shifts = torch.nonzero(torch.ones([self.patch_size, self.patch_size, self.patch_size], device=coords.device, dtype=coords.dtype))
            cube_shifts = torch.cat([torch.zeros_like(cube_shifts[:, :1]), cube_shifts], dim=-1)
            coords[:, 1:] *= self.patch_size
            coords = coords.unsqueeze(1) + cube_shifts
            coords = coords.flatten(0, 1)
            
            # restore sorted order
            coords, idx = coords.unique(dim=0, return_inverse=True)
            inv_idx = torch.zeros(len(idx), device=idx.device, dtype=idx.dtype).scatter_(
                0, idx, torch.arange(len(idx), device=idx.device, dtype=idx.dtype)
            )
            feats = h[inv_idx]
        
        out = SparseTensor(feats, coords.int())
        out._scale = tuple([s * self.patch_size for s in x._scale])
        out._spatial_cache = x._spatial_cache
        
        return out

def downsample(x: SparseTensor) -> SparseTensor:
    DIM = x.coords.shape[-1] - 1
    coord = [c // 2 if idx != 0 else c for idx, c in enumerate(x.coords.unbind(dim=-1))]
    
    MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code = sum([c * o for c, o in zip(coord, OFFSET)])
    code, idx = code.unique(return_inverse=True)
    
    new_feats = torch.scatter_reduce(
        torch.zeros(
            code.shape[0], x.feats.shape[1], device=x.feats.device, dtype=x.feats.dtype
        ),
        dim=0,
        index=idx.unsqueeze(1).expand(-1, x.feats.shape[1]),
        src=x.feats,
        reduce="mean",
        include_self=False,
    )
    new_coords = torch.stack(
        [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
        dim=-1,
        )
    out = SparseTensor(
        new_feats,
        new_coords,
        x.shape,
    )
    out._scale = tuple([s // 2 for s in x._scale])
    out._spatial_cache = x._spatial_cache
    
    out.register_spatial_cache(f"upsample_2_coords", x.coords)
    out.register_spatial_cache(f"upsample_2_layout", x.layout)
    out.register_spatial_cache(f"upsample_2_idx", idx)
    
    return out


def upsample(x: SparseTensor) -> SparseTensor:
    
    new_coords = x.get_spatial_cache(f"upsample_2_coords")
    new_layout = x.get_spatial_cache(f"upsample_2_layout")
    idx = x.get_spatial_cache(f"upsample_2_idx")
    if any([x is None for x in [new_coords, new_layout, idx]]):
        raise ValueError(
            "Upsample cache not found. SparseUpsample must be paired with SparseDownsample."
        )
    new_feats = x.feats[idx]
    out = SparseTensor(new_feats, new_coords, x.shape, new_layout)
    out._scale = tuple([s * 2 for s in x._scale])
    out._spatial_cache = x._spatial_cache
    return out


class SparseSubdivide(torch.nn.Module):
    def __init__(self):
        super(SparseSubdivide, self).__init__()
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=x.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)
        factor = n_coords.shape[0]
        
        # double all coords and add to them all {0,1}^3 combinations
        new_coords = x.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(
            new_coords.dtype
        )
        
        # just duplicate feats by factor
        new_feats = x.feats.unsqueeze(1).expand(
            x.feats.shape[0], factor, *x.feats.shape[1:]
        )
        
        # pass new coords to coord dimension
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), x.shape)
        out._scale = x._scale * 2
        out._spatial_cache = x._spatial_cache
        return out
    
    
    
    def forward(self, voxels) -> torch.Tensor:
        
        from warpconvnet.nn.functional.sparse_conv import spatially_sparse_conv
        
        out_voxels = spatially_sparse_conv(
            voxels,
            weight=self.weight,
            bias=self.bias,
            kernel_size=self.kernel_size,
        )
        
        return out_voxels

class WarpConv3d(torch.nn.Module):
    
    def __init__(
            self, *args, **kwargs
    ):
        super(WarpConv3d, self).__init__()
        # compatability with sparse conv 3d
        class _WarpConv3d(torch.nn.Module):
            def __init__(
                    self,
                    dim: int,
                    in_dim: int = None,
                    kernel_size: int = 3,
            ):
                super(_WarpConv3d, self).__init__()
                if in_dim is None:
                    in_dim = dim
                self.kernel_size = kernel_size
                self.weight = torch.nn.Parameter(torch.randn(
                    dim, kernel_size, kernel_size, kernel_size, in_dim, dtype=torch.float32
                ) / np.sqrt((kernel_size ** 3) / 2 * in_dim), requires_grad=True)
                self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)
        
        self.conv = _WarpConv3d(*args, **kwargs)
    
    def forward(self, voxels) -> torch.Tensor:
        
        from warpconvnet.nn.functional.sparse_conv import spatially_sparse_conv
        
        out_voxels = spatially_sparse_conv(
            voxels,
            weight=self.conv.weight.movedim(0, -1).flatten(0, 2),
            bias=self.conv.bias,
            kernel_size=self.conv.kernel_size,
        )
        
        return out_voxels

class FastSparseConv3d(torch.nn.Module):
    
    def __init__(
            self,
            dim: int,
            mix_channels: bool = False,
    ):
        super(FastSparseConv3d, self).__init__()
        # static kernel_size = 3, groups = dim
        kernel_size = 3
        self.weight = torch.nn.Parameter(torch.randn(kernel_size, kernel_size, kernel_size, dim, dtype=torch.float32) / np.sqrt((kernel_size ** 3) / 2), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)
        
        self.mix_channels = mix_channels
        if self.mix_channels:
            self.mixer = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 4),
                torch.nn.SiLU(),
                torch.nn.Linear(dim * 4, dim),
            )
            scale = np.sqrt(3 / (dim * (1 + 4)))
            torch.nn.init.uniform_(self.mixer[-3].weight, -scale, scale)
            torch.nn.init.zeros_(self.mixer[-3].bias)
            torch.nn.init.uniform_(self.mixer[-1].weight, -scale, scale)
            torch.nn.init.zeros_(self.mixer[-1].bias)
    
    def forward(self, feat: torch.Tensor, coord: torch.Tensor, indices: torch.Tensor = None) -> torch.Tensor:
        
        from ops.sparse import conv
        
        out = conv(feat, coord, self.weight, self.bias, indices=indices)
        if self.mix_channels:
            out = self.mixer(out)
        
        return out


class FlexConv3d(torch.nn.Module):
    
    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
    ):
        super(FlexConv3d, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, kernel_size, kernel_size, kernel_size, dim, dtype=torch.float32) / np.sqrt((dim * kernel_size ** 3) / 2), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)
    
    def forward(self,
                feats: torch.Tensor,
                coords: torch.Tensor,
                shape: tuple[int,int,int,int,int],
                neighbours_cache = None,
                ):
        
        input_dtype = feats.dtype
        
        
        LOCAL_CACHE = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH_LOCAL")
        if LOCAL_CACHE is None:
            RANK = os.environ.get("RANK", "0")
            LOCAL_CACHE = os.environ.get(
                'FLEX_GEMM_AUTOTUNE_CACHE_PATH',
                os.path.expanduser('~/.flex_gemm/autotune_cache.json')
            ).replace(".json", f"_{RANK}.json")
            os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH'] = LOCAL_CACHE
            os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH_LOCAL'] = LOCAL_CACHE
        
        import flex_gemm
        flex_gemm.ops.spconv.set_algorithm(
            flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK
        )
        
        out, neighbours_cache = flex_gemm.ops.spconv.sparse_submanifold_conv3d(
            feats.to(torch.float16), coords.to(torch.int32), shape,
            self.weight.to(torch.float16), self.bias.to(torch.float16), neighbor_cache=neighbours_cache
        )
        
        return out.to(input_dtype), neighbours_cache


if BACKEND == 'spconv':
    class SparseConv3d(torch.nn.Module):
        def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                padding=None,
                bias=True,
                indice_key=None,
        ):
            super(SparseConv3d, self).__init__()
            if "spconv" not in globals():
                import spconv.pytorch as spconv
            algo = spconv.ConvAlgo.MaskImplicitGemm
            if stride == 1 and (padding is None):
                self.conv = spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    indice_key=indice_key,
                    algo=algo,
                    groups=groups,
                )
            else:
                self.conv = spconv.SparseConv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    indice_key=indice_key,
                    algo=algo,
                    groups=groups,
                )
            self.stride = (
                tuple(stride)
                if isinstance(stride, (list, tuple))
                else (stride, stride, stride)
            )
            self.padding = padding
        
        def forward(self, x: SparseTensor) -> SparseTensor:
            spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
            
            dtype_ = x.feats.dtype
            x = x.replace(x.feats.type(torch.float32))
            new_data = self.conv(x.data)
            new_shape = [x.shape[0], self.conv.out_channels]
            new_layout = None if spatial_changed else x.layout
            
            if spatial_changed and (x.shape[0] != 1):
                # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
                fwd = new_data.indices[:, 0].argsort()
                bwd = torch.zeros_like(fwd).scatter_(
                    0, fwd, torch.arange(fwd.shape[0], device=fwd.device)
                )
                sorted_feats = new_data.features[fwd]
                sorted_coords = new_data.indices[fwd]
                unsorted_data = new_data
                new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore
            
            out = SparseTensor(
                new_data,
                shape=torch.Size(new_shape),
                layout=new_layout,
                scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
                spatial_cache=x._spatial_cache,
            )
            out = out.replace(out.feats.type(dtype_))
            
            if spatial_changed and (x.shape[0] != 1):
                out.register_spatial_cache(
                    f"conv_{self.stride}_unsorted_data", unsorted_data
                )
                out.register_spatial_cache(f"conv_{self.stride}_sort_bwd", bwd)
            
            return out

if BACKEND == 'torchsparse':
    class SparseConv3d(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True, indice_key=None):
            super(SparseConv3d, self).__init__()
            if 'torchsparse' not in globals():
                import torchsparse
            self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        
        def forward(self, x: SparseTensor) -> SparseTensor:
            dtype = x.feats.dtype
            if dtype not in (torch.float32, torch.float16):
                x = x.replace(x.feats.float())
            out = self.conv(x.data)
            
            spatial_range = out.spatial_range
            
            new_shape = [x.shape[0], self.conv.out_channels]
            out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
            out._spatial_cache = x._spatial_cache
            out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])
            
            out.data.spatial_range = spatial_range
            out = out.replace(out.feats.to(dtype))
            
            return out
    
    
    class SparseInverseConv3d(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
            super(SparseInverseConv3d, self).__init__()
            if 'torchsparse' not in globals():
                import torchsparse
            self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True)
        
        def forward(self, x: SparseTensor) -> SparseTensor:
            out = self.conv(x.data)
            
            new_shape = [x.shape[0], self.conv.out_channels]
            out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
            out._spatial_cache = x._spatial_cache
            out._scale = tuple([s // stride for s, stride in zip(x._scale, self.conv.stride)])
            
            return out