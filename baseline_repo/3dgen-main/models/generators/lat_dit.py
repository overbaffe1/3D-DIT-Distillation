import numpy as np
import torch.nn.functional
from einops import rearrange

from typing import List, Literal
from ops.rope import rope_nd
from base_models.fourier_utils import get_positional_encoding, get_time_encoding
from base_models.norm_utils import LayerNorm
import base_models.sparse as sp


from flash_attn.ops.activations import swiglu

def maybe_dummy_weight(x, weight):
    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    return weight

def layer_norm(x, weight=None, bias=None, eps=1e-6):
    # from flash_attn.ops.layer_norm import layer_norm
    # return layer_norm(x, maybe_dummy_weight(x, weight), bias, eps)
    from flash_attn.ops.triton.layer_norm import layer_norm_fn as layer_norm
    return layer_norm(x, maybe_dummy_weight(x, weight), bias, eps=eps)

def rms_norm(x, weight=None, eps=1e-6):
    # from flash_attn.ops.rms_norm import rms_norm
    # return rms_norm(x, maybe_dummy_weight(x, weight), eps)
    from flash_attn.ops.triton.layer_norm import rms_norm_fn as rms_norm
    return rms_norm(x, maybe_dummy_weight(x, weight), bias=None, eps=eps)


class ScaleNorm(torch.nn.Module):
    
    def __init__(self, hidden_dim,):
        super(ScaleNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
    
    def forward(self, x):
        return rms_norm(x, 1 + self.weight)


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


def zero_module(layer):
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer


def modulate(h: torch.Tensor, sizes: List[int], scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if len(sizes) > 1:
        return torch.cat([layer_norm(hh, 1 + scale[idx], shift[idx]) for idx, hh in enumerate(h.split(sizes))])
    return layer_norm(h, 1 + scale.view(-1), shift.view(-1))


def gate(h: torch.Tensor, sizes: List[int], scale: torch.Tensor) -> torch.Tensor:
    if len(sizes) > 1:
        return torch.cat([hh * scale[idx].type_as(hh) for idx, hh in enumerate(h.split(sizes))])
    return h * scale.view(-1).type_as(h)


class Attention(torch.nn.Module):
    
    def __init__(self, head_dim=64, n_heads=16, rms_norm=False, context_dim=None):
        super(Attention, self).__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        
        hidden_dim = head_dim * n_heads
        if context_dim is None:
            self.to_qkv = torch.nn.Linear(hidden_dim, hidden_dim * 3)
            init_linear(self.to_qkv, n_out=3)
        else:
            self.to_q = torch.nn.Linear(hidden_dim, hidden_dim)
            self.to_kv = torch.nn.Linear(context_dim, hidden_dim * 2)
            init_linear(self.to_q), init_linear(self.to_kv, n_out=2)
        
        self.rms_norm = rms_norm
        if self.rms_norm:
            self.qk_norm = QKNorm(head_dim)
        
        self.to_out = torch.nn.Linear(hidden_dim, hidden_dim)
        init_linear(self.to_out)
    
    def forward(self, x, context=None, pos=None):
        
        if context is None:
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", d=self.head_dim), [q, k, v]
        )
        
        if self.rms_norm:
            q, k = self.qk_norm(q, k)
        
        if pos is not None:
            q, k = rope_nd(q, pos), rope_nd(k, pos)
        
        import flash_attn
        attn_out = flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        # attn_out = xformers.ops.memory_efficient_attention(q, k, v)
        attn_out = rearrange(attn_out, "b n h d -> b n (h d)")
        
        h = self.to_out(attn_out)
        
        return h


class SimpleTransformerBlock(torch.nn.Module):
    
    def __init__(self, head_dim=64, num_heads=16, context_dim=None):
        super(SimpleTransformerBlock, self).__init__()
        
        hidden_dim = head_dim * num_heads
        self.attn_norm = LayerNorm(hidden_dim)
        if context_dim is not None:
            self.context_norm = LayerNorm(context_dim)
        self.attn = Attention(
            context_dim=context_dim,
            head_dim=head_dim,
            n_heads=num_heads,
            rms_norm=False,
        )
        self.ffn_norm = LayerNorm(hidden_dim)
        self.ffn1 = torch.nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn2 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        init_linear(self.ffn1), init_linear(self.ffn2)
    
    def forward(self, x, context=None):
        x = x + self.attn(
            self.attn_norm(x),
            context=self.context_norm(context) if context is not None else context,
        )
        x = x + self.ffn2(
            torch.nn.functional.gelu(self.ffn1(self.ffn_norm(x)), approximate="tanh")
        )
        return x


class SparseResBlock3d(torch.nn.Module):
    def __init__(
            self,
            layer_idx: int,
            hidden_dim: int,
            out_dim: int = None,
            in_dim: int = None,
            clf_dim: int = 768,
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
        
        self.norm = ScaleNorm(in_dim)
        self.conv1 = sp.SparseConv3d(in_dim, hidden_dim, 3, indice_key=f"{layer_idx}")
        
        self.gate = torch.nn.Linear(clf_dim, hidden_dim * 2)
        self.conv2 = sp.SparseConv3d(hidden_dim, out_dim, 3, indice_key=f"{layer_idx}")
        
        
        self.skip = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim != in_dim
            else torch.nn.Identity()
        )
        if in_dim != out_dim:
            init_linear(self.skip)
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.norm(x))
    
    @torch.compile()
    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.gc and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._prepare_input, x, use_reentrant=False
            )
        return self._prepare_input(x)
    
    def _prepare_mid(self, x: torch.Tensor, sizes: List[int], scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(modulate(x, sizes, scale, shift))
    
    @torch.compile()
    def prepare_mid(self, x: torch.Tensor, sizes: List[int], scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        if self.gc and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._prepare_mid, x, sizes, scale, shift, use_reentrant=False
            )
        return self._prepare_mid(x, sizes, scale, shift)
    
    def _forward(self, x: sp.SparseTensor, clf: torch.Tensor) -> sp.SparseTensor:
        
        # input conv
        prepare_input = self._prepare_input if self.save_memory else self.prepare_input
        h = self.conv1(x.replace(prepare_input(x.feats)))
        
        # cond conv
        sizes = [lay.stop - lay.start for lay in x.layout]
        scale, shift = self.gate(clf).chunk(2, dim=-1)
        prepare_mid = self._prepare_mid if self.save_memory else self.prepare_mid
        h = self.conv2(h.replace(prepare_mid(h.feats, sizes, scale, shift)))
        
        # out skip connection
        x = x.replace(self.skip(x.feats) + h.feats)
        
        return x
    
    def forward(self, x: sp.SparseTensor, clf: torch.Tensor) -> sp.SparseTensor:
        if self.save_memory and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, clf, use_reentrant=False
            )
        return self._forward(x, clf)



class QKNorm(torch.nn.Module):
    def __init__(self, num_heads, dim):
        super(QKNorm, self).__init__()
        self.q_scale = torch.nn.Parameter(torch.zeros(num_heads, dim, dtype=torch.float32))
        self.k_scale = torch.nn.Parameter(torch.zeros(num_heads, dim, dtype=torch.float32))
    
    def forward(self, q, k):
        q = rms_norm(q) * (self.q_scale).type_as(q) # TODO
        k = rms_norm(k) * (self.k_scale).type_as(k) # TODO
        return q, k


def swiglu_op(x, ffn0, ffn1):
    import xformers.ops
    w, w_gate = ffn0.weight.chunk(2, dim=0)
    b, b_gate = ffn0.bias.chunk(2, dim=0)
    y = xformers.ops.swiglu_op.swiglu_packed(x, torch.stack([w_gate, w]).type_as(x), torch.stack([b_gate, b]).type_as(x), ffn1.weight.type_as(x), ffn1.bias.type_as(x), op=xformers.ops.SwiGLUPackedFusedOp)
    return y

class SparseTransformerBlock(torch.nn.Module):
    def __init__(
            self,
            head_dim: int = 64,
            num_heads: int = 16,
            rms: bool = True,
            rope: bool = False,
            pos_conv: bool = False,
            ffn_conv: bool = False,
            order_idx: int = 0,
            ffn_rate: float = 4.0,
            cond: bool = False,
            cond_dim: int = None,
            clf_dim: int = None,
            clf_shared: bool = False,
            use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.order_idx = order_idx
        
        self.head_dim = head_dim
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        
        self.use_pos_conv = pos_conv
        if self.use_pos_conv:
            self.pos_norm = ScaleNorm(hidden_dim)
            # self.pos0 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            # self.pos_conv = sp.SparseConv3d(hidden_dim, hidden_dim, 3, indice_key=f"trans")
            self.pos_conv = sp.FastSparseConv3d(hidden_dim, mix_channels=True)
            # self.pos1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            # init_linear(self.pos0, normal=True), init_linear(self.pos1, normal=True)
        
        if clf_dim is None:
            clf_dim = hidden_dim
        self.clf_shared = clf_shared
        # gate cond
        if self.clf_shared:
            self.gates = torch.nn.Parameter(torch.randn(1, hidden_dim * 9, dtype=torch.float32) / np.sqrt(hidden_dim))
        else:
            self.gates = torch.nn.Linear(clf_dim, hidden_dim * 9)
            init_linear(self.gates, n_out=9)
        
        # self attn
        self.to_qkv = torch.nn.Linear(hidden_dim, hidden_dim * 3)
        self.to_out = torch.nn.Linear(hidden_dim, hidden_dim)
        init_linear(self.to_qkv, n_out=3), init_linear(self.to_out)
        
        # cross attn
        
        self.cond = cond
        if cond_dim is None:
            cond_dim = hidden_dim
        if self.cond: # do cond processing
            # cond gates
            if self.clf_shared:
                self.cond_gates = torch.nn.Parameter(torch.randn(1, cond_dim * 9, dtype=torch.float32) / np.sqrt(cond_dim))
            else:
                self.cond_gates = torch.nn.Linear(clf_dim, cond_dim * 9)
                init_linear(self.cond_gates, n_out=9)
            
            # cond self attn
            self.cond_to_qkv = torch.nn.Linear(cond_dim, hidden_dim * 3)
            self.cond_to_out = torch.nn.Linear(hidden_dim, cond_dim)
            init_linear(self.cond_to_qkv, n_out=3), init_linear(self.cond_to_out)
            
            # cond cross attn
            self.to_qkv_cross = torch.nn.Linear(hidden_dim, hidden_dim * 3)
            self.to_out_cross = torch.nn.Linear(hidden_dim, hidden_dim)
            init_linear(self.to_qkv_cross, n_out=3), init_linear(self.to_out_cross)
            
            self.cond_to_qkv_cross = torch.nn.Linear(cond_dim, hidden_dim * 3)
            self.cond_to_out_cross = torch.nn.Linear(hidden_dim, cond_dim)
            init_linear(self.cond_to_qkv_cross, n_out=3), init_linear(self.cond_to_out_cross)
        else:
            self.to_q_cross = torch.nn.Linear(hidden_dim, hidden_dim)
            if self.clf_shared:
                self.cond_gates = torch.nn.Parameter(torch.randn(1, cond_dim * 2, dtype=torch.float32) / np.sqrt(cond_dim))
            else:
                self.cond_gates = torch.nn.Linear(clf_dim, cond_dim * 2)
                init_linear(self.cond_gates, n_out=2)
            self.cond_to_kv_cross = torch.nn.Linear(cond_dim, hidden_dim * 2)
            self.to_out_cross = torch.nn.Linear(hidden_dim, hidden_dim)
            init_linear(self.to_q_cross), init_linear(self.cond_to_kv_cross, n_out=2), init_linear(self.to_out_cross)
        
        
        self.rms = rms
        if rms:
            self.qk_norm = QKNorm(num_heads, head_dim)
            if self.cond:
                self.cond_qk_norm = QKNorm(num_heads, head_dim)
        self.rope = rope
        
        # ffn
        ffn_dim = int(hidden_dim * ffn_rate)
        self.ffn0 = torch.nn.Linear(hidden_dim, ffn_dim * 2)
        self.ffn1 = torch.nn.Linear(ffn_dim, hidden_dim)
        init_linear(self.ffn0, n_out=2), init_linear(self.ffn1)
        self.use_ffn_conv = ffn_conv
        if self.use_ffn_conv:
            self.ffn_conv = sp.FastSparseConv3d(ffn_dim * 2)
        
        if self.cond:
            # cond ffn
            self.cond_ffn0 = torch.nn.Linear(hidden_dim, ffn_dim * 2)
            self.cond_ffn1 = torch.nn.Linear(ffn_dim, hidden_dim)
            init_linear(self.cond_ffn0, n_out=2), init_linear(self.cond_ffn1)
    
    def __forward(
            self,
            x: torch.Tensor,
            coord: torch.Tensor,
            pos: torch.Tensor,
            sizes: List[int],
            fwd_indices: torch.Tensor,
            bwd_indices: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_len: int,
            clf: torch.Tensor,
            clf_shared: torch.Tensor,
            cond: torch.Tensor,
            cond_pos: torch.Tensor,
            q_seqlens: torch.Tensor,
            q_maxlen: int,
            kv_seqlens: torch.Tensor,
            kv_maxlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # window self attention main
        gate_base = self.gates.type_as(clf_shared) + clf_shared if self.clf_shared else self.gates(clf)
        scale_msa, shift_msa, gate_msa, scale_cross, shift_cross, gate_cross, scale_mlp, shift_mlp, gate_mlp = gate_base.chunk(9, dim=-1)
        h = modulate(x, sizes, scale_msa, shift_msa)
        qkv = self.to_qkv(h).view(-1, 3, self.num_heads, self.head_dim)
        if self.rms or self.rope:
            q, k, v = qkv.unbind(dim=-3)
            if self.rms:
                q, k = self.qk_norm(q, k)
            if self.rope:
                q, k = rope_nd(q[None,], pos[None,], freq=128)[0], rope_nd(k[None,], pos[None,], freq=128)[0]
            qkv = torch.stack([q, k, v], dim=-3)
        
        h = sp.sparse_windowed_sdp(qkv, fwd_indices, bwd_indices, cu_seqlens, max_len)
        h = self.to_out(h.view(-1, self.num_heads * self.head_dim))
        x = x + gate(h, sizes, gate_msa)
        
        # full self attention cond
        if self.cond:
            cond_gate_base = self.cond_gates.type_as(clf_shared) + clf_shared if self.clf_shared else self.cond_gates(clf)
            cond_scale_msa, cond_shift_msa, cond_gate_msa, cond_scale_cross, cond_shift_cross, cond_gate_cross, cond_scale_mlp, cond_shift_mlp, cond_gate_mlp = cond_gate_base.chunk(9, dim=-1)
            h_cond = layer_norm(cond) * (1 + cond_scale_msa[:, None]) + cond_shift_msa[:, None]
            qkv_cond = self.cond_to_qkv(h_cond).view(*cond.shape[:2], 3, self.num_heads, self.head_dim)
            if self.rms or self.rope:
                q_cond, k_cond, v_cond = qkv_cond.unbind(dim=-3)
                if self.rms:
                    q_cond, k_cond = self.cond_qk_norm(q_cond, k_cond)
                if self.rope:
                    q_cond = torch.cat([q_cond[:, :5], rope_nd(q_cond[:, 5:], cond_pos, freq=64)], dim=1)
                    k_cond = torch.cat([k_cond[:, :5], rope_nd(k_cond[:, 5:], cond_pos, freq=64)], dim=1)
                qkv_cond = torch.stack([q_cond, k_cond, v_cond], dim=-3)
            
            flash3 = False
            try:
                import flash_attn_interface; flash3 = True
            except ImportError:
                import flash_attn
            
            if flash3:
                h_cond = flash_attn_interface.flash_attn_qkvpacked_func(qkv_cond)
            else:
                h_cond = flash_attn.flash_attn_qkvpacked_func(qkv_cond)
            
            h_cond = self.cond_to_out(h_cond.view(*cond.shape))
            cond = cond + h_cond * cond_gate_msa[:, None]
        
        # cross attention to cond
        if self.cond:
            h = modulate(x, sizes, scale_cross, shift_cross)
            q, k, v = self.to_qkv_cross(h).view(-1, 3, self.num_heads, self.head_dim).unbind(dim=-3)
            
            h_cond = layer_norm(cond) * (1 + cond_scale_cross[:, None]) + cond_shift_cross[:, None]
            q_cond, k_cond, v_cond = self.cond_to_qkv_cross(h_cond).view(cond.shape[0] * cond.shape[1], 3, self.num_heads, self.head_dim).unbind(dim=-3)
            
            h = sp.sparse_cross_sdp(
                q, q_seqlens, q_maxlen,
                torch.stack([k_cond, v_cond], dim=-3), kv_seqlens, kv_maxlen
            )
            h_cond = sp.sparse_cross_sdp(
                q_cond, kv_seqlens, kv_maxlen,
                torch.stack([k, v], dim=-3), q_seqlens, q_maxlen
            )
            
            h = self.to_out_cross(h.view(-1, self.num_heads * self.head_dim))
            x = x + gate(h, sizes, gate_cross)
            
            h_cond = self.cond_to_out_cross(h_cond.view(*cond.shape))
            cond = cond + h_cond * cond_gate_cross[:, None]
        
        else:
            h = modulate(x, sizes, scale_cross, shift_cross)
            q = self.to_q_cross(h).view(-1, self.num_heads, self.head_dim)
            
            cond_gates_base = self.cond_gates.type_as(clf) + clf_shared[:, 3 * self.head_dim * self.num_heads:5 * self.head_dim * self.num_heads] if self.clf_shared else self.cond_gates(clf)
            cond_scale_cross, cond_shift_cross = cond_gates_base.chunk(2, dim=-1)
            h_cond = layer_norm(cond) * (1 + cond_scale_cross[:, None]) + cond_shift_cross[:, None]
            kv = self.cond_to_kv_cross(h_cond).view(cond.shape[0] * cond.shape[1], 2, self.num_heads, self.head_dim)
            
            h = sp.sparse_cross_sdp(
                q, q_seqlens, q_maxlen,
                kv, kv_seqlens, kv_maxlen
            )
            h = self.to_out_cross(h.view(-1, self.num_heads * self.head_dim))
            x = x + gate(h, sizes, gate_cross)
        
        
        # ffn main
        h = modulate(x, sizes, scale_mlp, shift_mlp)
        h = self.ffn0(h)
        if self.use_ffn_conv:
            h = self.ffn_conv(h, coord)
        h, h_gate = h.chunk(2, dim=-1)
        h = self.ffn1(swiglu(h_gate, h))
        # h = swiglu_op(h, self.ffn0, self.ffn1)
        x = x + gate(h, sizes, gate_mlp)
        
        # ffn cond
        if self.cond:
            h_cond = layer_norm(cond) * (1 + cond_scale_mlp[:, None]) + cond_shift_mlp[:, None]
            h_cond, h_cond_gate = self.cond_ffn0(h_cond).chunk(2, dim=-1)
            h_cond = self.cond_ffn1(swiglu(h_cond_gate, h_cond))
            # h_cond = swiglu_op(h_cond, self.cond_ffn0, self.cond_ffn1)
            cond = cond + h_cond * cond_gate_mlp[:, None]
        
        return x, cond
    
    # @torch.compile()
    def _forward(
            self,
            x: torch.Tensor,
            coord: torch.Tensor,
            pos: torch.Tensor,
            sizes: List[int],
            fwd_indices: torch.Tensor,
            bwd_indices: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_len: int,
            clf: torch.Tensor,
            clf_shared: torch.Tensor,
            cond: torch.Tensor,
            cond_pos: torch.Tensor,
            q_seqlens: torch.Tensor,
            q_maxlen: int,
            kv_seqlens: torch.Tensor,
            kv_maxlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.__forward, x, coord, pos, sizes,
                fwd_indices, bwd_indices, cu_seqlens, max_len,
                clf, clf_shared, cond, cond_pos,
                q_seqlens, q_maxlen, kv_seqlens, kv_maxlen, use_reentrant=False
            )
        else:
            return self.__forward(
                x, coord, pos, sizes,
                fwd_indices, bwd_indices, cu_seqlens, max_len,
                clf, clf_shared, cond, cond_pos,
                q_seqlens, q_maxlen, kv_seqlens, kv_maxlen
            )
    
    def _pos_forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        
        h = x.replace(torch.nn.functional.silu(self.pos_norm(x.feats)))
        h = self.pos_conv(h.feats, h.coords)
        x = x.replace(x.feats + h)
        
        return x
    
    def pos_forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._pos_forward, x, use_reentrant=False
            )
        return self._pos_forward(x)
    
    def forward(self, x: sp.SparseTensor, pos: torch.Tensor, clf: torch.Tensor, clf_shared: torch.Tensor, cond: torch.Tensor) -> tuple[sp.SparseTensor, torch.Tensor]:
        
        if self.use_pos_conv:
            x = self.pos_forward(x)
        
        feats, coords = x.feats, x.coords
        sizes = [lay.stop - lay.start for lay in x.layout]
        fwd_indices, bwd_indices, cu_seqlens, max_len = x.get_spatial_cache(f"attn_order{self.order_idx}")
        
        attn_cross = x.get_spatial_cache("attn_cross")
        if attn_cross is None:
            cond_size = int(np.round(np.sqrt(cond.shape[1] - 5)))
            assert cond_size * cond_size == cond.shape[1] - 5
            cond_pos = torch.stack(torch.meshgrid(
                torch.arange(cond_size, device=cond.device, dtype=torch.long),
                torch.arange(cond_size, device=cond.device, dtype=torch.long),
                indexing="ij"
            ), dim=-1).view(-1, 2)
            cond_pos = (cond_pos.float() + 0.5) / cond_size * 2 - 1
            # cond_pos = torch.cat([torch.ones_like(cond_pos[:5]), cond_pos])
            cond_pos = cond_pos[None,].repeat_interleave(len(cond), dim=0).reshape(cond.shape[0], -1, 2)
            # cond_pos = torch.arange(cond.shape[1], device=cond.device, dtype=torch.float32)
            # cond_pos = cond_pos[None,].repeat_interleave(len(cond), dim=0).reshape(*cond.shape[:2], 1)
            q_seqlens = torch.tensor([0] + sizes).cumsum(0).int().to(feats.device)
            q_maxlen = max(sizes)
            kv_seqlens = torch.tensor([0] + [cond.shape[1]] * cond.shape[0]).cumsum(0).int().to(feats.device)
            kv_maxlen = cond.shape[1]
            attn_cross = cond_pos, q_seqlens, q_maxlen, kv_seqlens, kv_maxlen
            x.register_spatial_cache("attn_cross", attn_cross)
        cond_pos, q_seqlens, q_maxlen, kv_seqlens, kv_maxlen = attn_cross
        
        feats, cond = self._forward(
            feats, coords,
            pos, sizes,
            fwd_indices, bwd_indices, cu_seqlens, max_len,
            clf, clf_shared, cond, cond_pos,
            q_seqlens, q_maxlen, kv_seqlens, kv_maxlen
        )
        
        return x.replace(feats), cond


class FlowModel(torch.nn.Module):
    def __init__(
            self,
            res: int = 128,
            latent_dim: int = 16,
            dim: int = 1024,
            cond_depth: int = 8,
            depth: int = 24,
            head_dim: int = 64,
            time_pos_features: int = 256,
            io_conv: bool = False,
            io_dims: tuple[int] = (192, 384, 768),
            io_depths: tuple[int] = (2, 2, 0),
            io_attn: bool = False,
            io_orders: List[Literal["full", "swin", "swin_shift", "z", "zt", "h", "ht", "xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"]] = ("swin", "swin_shift"),
            io_attn_depths: tuple[int] = (4, 4, 0),
            rms: bool = True,
            rope: bool = False,
            pos_emb: bool = True,
            pos_conv: bool = False,
            ffn_conv: bool = False,
            ffn_rate: float = 4.0,
            patch: int = 1,
            orders: List[Literal["full", "swin", "swin_shift", "z", "zt", "h", "ht", "xdiv", "xmod", "zdiv", "zmod", "hdiv", "hmod"]] = ("swin", "swin_shift"),
            shuffle_orders: bool = False,
            window_size: int = 8,
            patch_size: int = 1024,
            bucket_size: int = 4096,
            global_size: int = 128,
            clf_shared: bool = False,
            gc: bool = False,
            save_memory: bool = False,
            **kwargs,
    ):
        super().__init__()
        
        # positional encoding
        self.res = res
        self.pos_emb = pos_emb
        self.rope = rope
        
        # time encoding
        self.time_pos_features = time_pos_features
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.time_pos_features, dim * 4),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(dim * 4, dim),
        )
        init_linear(self.time_embed[0], normal=True)
        init_linear(self.time_embed[-1], normal=True)
        
        self.clf_shared = clf_shared
        if self.clf_shared:
            self.clf_table_map = torch.nn.Linear(dim, dim * 9)
            init_linear(self.clf_table_map, normal=True, n_out=9)
        
        # input down sampling for unet-style arch
        self.io_conv = io_conv
        self.io_attn = io_attn
        if self.io_conv:
            self.res = self.res >> (len(io_dims) - 1)
            self.patchify = torch.nn.Linear(latent_dim, io_dims[0])
            self.down_blocks, self.downs = torch.nn.ModuleList([]), torch.nn.ModuleList([])
            if self.io_attn:
                self.down_attns = torch.nn.ModuleList([])
            for layer_idx in range(len(io_dims)):
                self.down_blocks.append(torch.nn.ModuleList([
                    SparseResBlock3d(
                        layer_idx=layer_idx, hidden_dim=io_dims[layer_idx], clf_dim=dim,
                        gc=gc, save_memory=save_memory,
                    ) for _ in range(io_depths[layer_idx])
                ]))
                if layer_idx < len(io_dims) - 1:
                    self.downs.append(sp.SparseDownsample(io_dims[layer_idx], io_dims[layer_idx + 1]))
                else:
                    self.downs.append(sp.SparseLinear(io_dims[layer_idx], dim, bias=False))
                    init_linear(self.downs[-1], normal=True)
                if self.io_attn:
                    self.down_attns.append(torch.nn.ModuleList([
                        SparseTransformerBlock(
                            head_dim=head_dim, num_heads=io_dims[layer_idx] // head_dim, order_idx=block_idx % len(io_orders),
                            rms=rms, rope=rope, clf_shared=False, clf_dim=dim, cond_dim=dim, cond=False, use_checkpoint=gc,
                        ) for block_idx in range(io_attn_depths[layer_idx])
                    ]))
        else:
            self.patchify = sp.SparseDownsample(latent_dim, dim, patch_size=patch)
        
        self.io_orders = io_orders
        self.orders = orders
        self.order_kwargs = {
            "window_size": window_size,
            "patch_size": patch_size,
            "bucket_size": bucket_size,
            "global_size": global_size,
            "shuffle": shuffle_orders,
        }
        self.blocks = torch.nn.ModuleList([
            SparseTransformerBlock(
                head_dim=head_dim, num_heads=dim // head_dim, order_idx=block_idx % len(orders),
                rms=rms, rope=rope, pos_conv=pos_conv, ffn_conv=ffn_conv, clf_shared=clf_shared, cond=block_idx < cond_depth,
                ffn_rate=ffn_rate, use_checkpoint=gc
            )
            for block_idx in range(depth)
        ])
        
        if self.io_conv:
            self.up_blocks, self.ups = torch.nn.ModuleList([]), torch.nn.ModuleList([])
            if self.io_attn:
                self.up_attns = torch.nn.ModuleList([])
            for layer_idx in reversed(range(len(io_dims))):
                if layer_idx == len(io_dims) - 1:
                    self.ups.append(sp.SparseLinear(dim, io_dims[layer_idx], bias=False))
                    init_linear(self.ups[-1], normal=True)
                else:
                    self.ups.append(sp.SparseUpsample(io_dims[layer_idx + 1], io_dims[layer_idx]))
                self.up_blocks.append(torch.nn.ModuleList([
                    SparseResBlock3d(
                        layer_idx=layer_idx, hidden_dim=io_dims[layer_idx], clf_dim=dim,
                        gc=gc, save_memory=save_memory, in_dim=io_dims[layer_idx] if bidx > 0 or layer_idx == len(io_dims) - 1 else io_dims[layer_idx] * 2
                    ) for bidx in range(io_depths[layer_idx])
                ]))
                if self.io_attn:
                    self.up_attns.append(torch.nn.ModuleList([
                        SparseTransformerBlock(
                            head_dim=head_dim, num_heads=io_dims[layer_idx] // head_dim, order_idx=block_idx % len(io_orders),
                            rms=rms, rope=rope, clf_shared=False, clf_dim=dim, cond_dim=dim, cond=False, use_checkpoint=gc,
                        ) for block_idx in range(io_attn_depths[layer_idx])
                    ]))
            self.out_gates = torch.nn.Linear(dim, 2 * io_dims[0])
            self.unpatchify = sp.SparseLinear(io_dims[0], latent_dim)
        else:
            self.out_gates = torch.nn.Linear(dim, 2 * dim)
            self.unpatchify = sp.SparseUpsample(dim, latent_dim, patch_size=patch)
        
        if kwargs.get("zero_gates", False):
            self.init_weights()
    
    def init_weights(self,):
        if self.io_conv:
            for blocks in self.down_blocks + self.up_blocks:
                for b in blocks:
                    zero_module(b.gate)
            if self.io_attn:
                for blocks in self.down_attns + self.up_attns:
                    for b in blocks:
                        zero_module(b.gates)
                        if b.cond:
                            zero_module(b.cond_gates)
        for b in self.blocks:
            if hasattr(b.gates, 'weight'):
                zero_module(b.gates)
            if b.cond and hasattr(b.cond_gates, 'weight'):
                zero_module(b.cond_gates)
        zero_module(self.out_gates)
    
    def forward(self, x: sp.SparseTensor, t: torch.Tensor, clf: torch.Tensor, cond: torch.Tensor, logger=lambda *args, **kwargs: args) -> sp.SparseTensor:
        
        
        # time embed
        temb = self.time_embed(get_time_encoding(t, self.time_pos_features))
        clf = torch.nn.functional.silu(temb + clf)
        clf_cond = clf
        clf_shared = self.clf_table_map(clf) if self.clf_shared else clf
        logger(clf_cond, "clf")
        
        if self.io_conv:
            h = x.replace(self.patchify(x.feats))
            skips = []
            for layer_idx, (down, blocks) in enumerate(zip(self.downs, self.down_blocks)):
                for block_idx, block in enumerate(blocks[:len(blocks) // 2]):
                    h = block(h, clf.type_as(h.feats))
                    logger(h, f"res_down{layer_idx}_block{block_idx}")
                if self.io_attn and len(self.down_attns[layer_idx]) > 0:
                    sp.put_attn_orders(h, self.io_orders, **self.order_kwargs)
                    for attn_idx, attn in enumerate(self.down_attns[layer_idx]):
                        h, cond = attn(h, clf.type_as(h.feats), cond)
                        logger(h, f"res_down{layer_idx}_attn{attn_idx}")
                for block_idx, block in enumerate(blocks[len(blocks) // 2:]):
                    h = block(h, clf.type_as(h.feats))
                    logger(h, f"res_down{layer_idx}_block{block_idx + len(blocks) // 2}")
                skips.append(h.feats)
                h = down(h)
            skips.pop() # remove last because of hidden dims
        else:
            h = self.patchify(x)
        logger(h, "input")
        
        # pos emb info
        pos = h.coords[:, 1:].detach().float()
        if self.pos_emb:
            feats = h.feats
            pos = (pos + 0.5) / self.res * 128
            pos_emb = get_positional_encoding(
                pos,
                feats.shape[-1],
                scale=10000 ** (-1.0),
                include_pi=False,
                logspace=True,
            ).type_as(feats)
            h = h.replace(feats + pos_emb)
            logger(h, "input_pos")
        elif self.rope:
            pos = (pos + 0.5) / self.res * 2 - 1
        
        # context embed
        cond = cond.type_as(h.feats)
        logger(cond, "cond")
        
        sp.put_attn_orders(h, self.orders, **self.order_kwargs)
        for block_idx, block in enumerate(self.blocks):
            h, cond = block(h, pos, clf_cond, clf_shared, cond)
            logger(h, f"block_{block_idx}")
        
        if self.io_conv:
            for layer_idx, (up, blocks) in enumerate(zip(self.ups, self.up_blocks)):
                h = up(h)
                if layer_idx > 0:
                    h = h.replace(torch.cat([h.feats, skips.pop()], dim=-1))
                for block_idx, block in enumerate(blocks[:len(blocks) // 2]):
                    h = block(h, clf.type_as(h.feats))
                    logger(h, f"res_up{len(self.up_blocks) - 1 - layer_idx}_block{block_idx}")
                if self.io_attn:
                    for attn_idx, attn in enumerate(self.up_attns[layer_idx]):
                        h, cond = attn(h, clf.type_as(h.feats), cond)
                        logger(h, f"res_up{layer_idx}_attn{attn_idx}")
                for block_idx, block in enumerate(blocks[len(blocks) // 2:]):
                    h = block(h, clf.type_as(h.feats))
                    logger(h, f"res_up{len(self.up_blocks) - 1 - layer_idx}_block{block_idx + len(blocks) // 2}")
        
        out_scale, out_shift = self.out_gates(clf).chunk(2, dim=-1)
        h = h.replace(modulate(h.feats, [lay.stop - lay.start for lay in h.layout], out_scale, out_shift))
        logger(h, "out_norm")
        
        x = self.unpatchify(h)
        logger(x, "out")
        
        return x


class lat_dit(torch.nn.Module):
    """Training wrapper; inference logic lives in LatDitCore."""

    def __init__(self, config, tracer):
        super().__init__()
        self.config = config
        from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
        from kandi3d_image23d.modules.lat_dit.core import LatDitCore

        self._dit_config = LatDitConfig.from_geo_config(config)
        self.core = LatDitCore(self._dit_config, full_config=config)
        self.model_config = self._dit_config.model_config
        self.gc = self.model_config.get("gc", False)
        self.log = self.model_config.get("log", False)
        self.use_ema = self.model_config.get("use_ema", False)
        self.cond = self._dit_config.cond
        self.dtype = self.model_config.get("dtype", config["training"].get("fp16_type", "float16"))
        self.use_token = self.model_config.get("use_token", False)
        self.train_stats = self.core.train_stats
        self.latent_scale = self.core.latent_scale

    def init(self, device):
        self.device = device
        self.core.to(device)
        self.flow = self.core.flow
        self.mean = self.core.mean
        self.std = self.core.std
        return self

    def init_stats(self):
        self.core._init_stats()

    @torch.no_grad()
    def normalize(self, x: sp.SparseTensor, train=False, alpha=0.9999) -> sp.SparseTensor:
        return self.core.normalize(x, train=train, alpha=alpha)

    @torch.no_grad()
    def denormalize(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.core.denormalize(x)

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        from base_models.optimization_utils import fix_state_dict

        state_dict = fix_state_dict(state_dict, self.core.state_dict())
        return self.core.load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)
    
    def forward(self, data_dict, crit_dict, total_iteration, mode="sample"):
        return self.core.forward(data_dict, crit_dict, total_iteration, mode)
 