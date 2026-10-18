import os
import sys
import time
import json
import zlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from remotezip import RemoteZip

WEIGHTS_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/kandi3d_weights_aijc_kit.zip"
SAFETENSORS_ARCHIVE_PATH = "kandi3d_weights_aijc_kit/lat_dit/diffusion_pytorch_model.safetensors"


def stream_safetensors_from_remote_zip(url: str, member_path: str):
    """Stream a large .safetensors member from a remote ZIP over a single HTTP Range request."""
    t0 = time.time()
    with RemoteZip(url) as zf:
        info = zf.getinfo(member_path)
        header_offset = info.header_offset
        compress_size = info.compress_size
        compress_type = info.compress_type
        file_size = info.file_size

    print(
        f"Streaming {member_path}: compress_type={compress_type}, "
        f"compress_size={compress_size / (1024**3):.2f} GB, "
        f"uncompressed={file_size / (1024**3):.2f} GB"
    )

    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={header_offset}-{header_offset + compress_size + 1024}"},
    )
    resp = urllib.request.urlopen(req, timeout=120)

    lhdr = resp.read(30)
    magic, ver, flags, ctype, mtime, mdate, crc, csize, usize, fname_len, extra_len = struct.unpack(
        "<4s5H3I2H", lhdr
    )
    assert magic == b"PK\x03\x04", f"Bad local header magic: {magic}"
    resp.read(fname_len + extra_len)

    decompressor = zlib.decompressobj(-15) if compress_type == 8 else None
    buf = bytearray()

    def read_exact(n: int) -> bytes:
        nonlocal buf
        while len(buf) < n:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            if decompressor is not None:
                buf.extend(decompressor.decompress(chunk))
            else:
                buf.extend(chunk)
        res = bytes(memoryview(buf)[:n])
        del buf[:n]
        return res

    header_len = struct.unpack("<Q", read_exact(8))[0]
    header = json.loads(read_exact(header_len).decode("utf-8"))
    items = [(k, v) for k, v in header.items() if k != "__metadata__"]
    items.sort(key=lambda kv: kv[1]["data_offsets"][0])

    state_dict = {}
    stats = {}
    cur_offset = 0

    for idx, (k, meta) in enumerate(items):
        start, end = meta["data_offsets"]
        if start > cur_offset:
            read_exact(start - cur_offset)
        nbytes = end - start
        raw = read_exact(nbytes)
        cur_offset = end

        shape = tuple(meta["shape"])
        tensor_f32 = torch.frombuffer(bytearray(raw), dtype=torch.float32).view(shape)

        if (
            tensor_f32.numel() <= 20000
            or "gates.bias" in k
            or "q_scale" in k
            or "k_scale" in k
            or "ffn_conv.bias" in k
            or k in ("mean", "std")
        ):
            stats[k] = {
                "shape": list(shape),
                "mean": float(tensor_f32.mean().item()),
                "std": float(tensor_f32.std().item()) if tensor_f32.numel() > 1 else 0.0,
                "min": float(tensor_f32.min().item()),
                "max": float(tensor_f32.max().item()),
                "abs_mean": float(tensor_f32.abs().mean().item()),
            }
            if k in ("mean", "std"):
                stats[k]["values"] = tensor_f32.tolist()

        if len(shape) == 2 and tensor_f32.numel() >= 1536 * 1536:
            state_dict[k] = tensor_f32.to(torch.bfloat16)
        else:
            state_dict[k] = tensor_f32

        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  [{idx + 1}/{len(items)}] {time.time() - t0:.1f}s")

    resp.close()
    return state_dict, stats


def precompute_rope_cos_sin(pos: torch.Tensor, base: float, head_dim: int = 96):
    B, N, DIM = pos.shape
    Q = head_dim // (2 * DIM)
    inv_freq = float(base) ** (-torch.arange(Q, dtype=torch.float64, device=pos.device) / Q)
    freq = pos.to(torch.float64).unsqueeze(2).unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, Q)
    cos = torch.cos(freq).to(torch.float32)
    sin = torch.sin(freq).to(torch.float32)
    return cos, sin


def apply_rope_cached(tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, N, H, D = tokens.shape
    DIM, Q = cos.shape[-2], cos.shape[-1]
    tok = tokens.float().view(B, N, H, DIM, 2, Q)
    u = tok[..., 0, :]
    v = tok[..., 1, :]
    out_u = u * cos - v * sin
    out_v = v * cos + u * sin
    return torch.stack([out_u, out_v], dim=-2).reshape(B, N, H, D).to(tokens.dtype)


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = (xf * rrms).to(x.dtype)
    if weight is not None:
        out = out * weight.to(x.dtype)
    return out


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    out = (xf - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def swiglu_ref(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate.float()) * x.float()).to(x.dtype)


def encode_hilbert_cpu(xyz: torch.Tensor, convention: str = "xyz") -> torch.Tensor:
    c = convention.lower()
    xo, yo, zo = c.find("x"), c.find("y"), c.find("z")
    bx = xyz[:, 0].to(torch.int64)
    x = xyz[:, 1 + xo].to(torch.int64).clone()
    y = xyz[:, 1 + yo].to(torch.int64).clone()
    z = xyz[:, 1 + zo].to(torch.int64).clone()

    for i in range(15, 0, -1):
        q = 1 << i
        p = q - 1
        x ^= torch.where((x & q) > 0, p, 0)
        cond = (y & q) > 0
        t = (x ^ y) & p
        x ^= torch.where(cond, p, t)
        y ^= torch.where(cond, 0, t)
        cond = (z & q) > 0
        t = (x ^ z) & p
        x ^= torch.where(cond, p, t)
        z ^= torch.where(cond, 0, t)

    y ^= x
    z ^= y
    t = torch.zeros_like(z)
    for i in range(15, 0, -1):
        q = 1 << i
        t ^= torch.where((z & q) > 0, q - 1, 0)
    x ^= t
    y ^= t
    z ^= t

    ret = torch.zeros_like(x)
    for i in range(16):
        q = 1 << i
        ret |= (x & q) << (2 * i + 2)
        ret |= (y & q) << (2 * i + 1)
        ret |= (z & q) << (2 * i + 0)
    ret |= bx << 48
    return ret


def build_hilbert_orders_cpu(coords: torch.Tensor, orders=("h", "ht1", "h1", "ht"), patch_size=4096):
    N = coords.shape[0]
    order_caches = []
    for mode in orders:
        if mode == "h":
            code = encode_hilbert_cpu(coords, "xyz")
        elif mode == "h1":
            code = encode_hilbert_cpu(coords + 1, "xyz")
        elif mode == "ht":
            code = encode_hilbert_cpu(coords, "yxz")
        elif mode == "ht1":
            code = encode_hilbert_cpu(coords + 1, "yxz")
        else:
            raise ValueError(mode)
        fwd = torch.argsort(code)
        bwd = torch.empty_like(fwd)
        bwd[fwd] = torch.arange(N, dtype=fwd.dtype)
        chunks = []
        for s in range(0, N, patch_size):
            chunks.append(slice(s, min(s + patch_size, N)))
        order_caches.append((fwd, bwd, chunks))
    return order_caches


def build_conv_indices_cpu(coords: torch.Tensor) -> torch.Tensor:
    N = coords.shape[0]
    b = coords[:, 0].to(torch.int64)
    x = coords[:, 1].to(torch.int64)
    y = coords[:, 2].to(torch.int64)
    z = coords[:, 3].to(torch.int64)
    keys = (b << 36) | (x << 24) | (y << 12) | z
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    out = torch.full((N, 27), -1, dtype=torch.int32)
    d = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                xx = x + dx
                yy = y + dy
                zz = z + dz
                valid = (xx >= 0) & (yy >= 0) & (zz >= 0)
                qkeys = (b << 36) | (xx.clamp(min=0) << 24) | (yy.clamp(min=0) << 12) | zz.clamp(min=0)
                idx = torch.searchsorted(sorted_keys, qkeys)
                idx_clamped = idx.clamp(max=N - 1)
                found = valid & (idx < N) & (sorted_keys[idx_clamped] == qkeys)
                out[found, d] = order[idx_clamped[found]].to(torch.int32)
                d += 1
    return out


def sparse_conv3d_ref(
    feat: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_indices: torch.Tensor,
) -> torch.Tensor:
    N, C = feat.shape
    acc = bias.float().unsqueeze(0).expand(N, C).clone()
    w_flat = weight.float().view(27, C)
    feat_f = feat.float()
    for d in range(27):
        idx_d = conv_indices[:, d]
        mask_d = idx_d >= 0
        if mask_d.any():
            acc[mask_d] += feat_f[idx_d[mask_d]] * w_flat[d]
    return acc.to(feat.dtype)


def get_time_encoding_ref(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    omega = torch.exp(
        -np.log(max_period) * torch.arange(0, dim, 2, dtype=torch.float64, device=t.device) / dim
    )
    freqs = t.to(torch.float64)[..., None] * omega
    out = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return out.float()


class FastTeacherRunner:
    """Runs the real 24-layer LatDitModel on CPU using bfloat16 & static geometry precomputation."""

    def __init__(self, state_dict: dict):
        self.sd = state_dict
        self.D = 1536
        self.H = 16
        self.HD = 96
        self.mean = state_dict["mean"].float().view(1, 16)
        self.std = state_dict["std"].float().view(1, 16)
        self.latent_scale = 1.5

    def lin(self, inp: torch.Tensor, prefix: str) -> torch.Tensor:
        w = self.sd[f"{prefix}.weight"].to(inp.dtype)
        b = self.sd.get(f"{prefix}.bias")
        b = b.to(inp.dtype) if b is not None else None
        return F.linear(inp, w, b)

    def prepare_sample(self, vox_idxi: torch.Tensor, dino_feats: torch.Tensor):
        coords_in = torch.cat([torch.zeros(len(vox_idxi), 1, dtype=torch.int32), vox_idxi.int()], dim=-1)
        code = coords_in[:, 1:].long()
        offs = torch.cumprod((code.max(dim=0).values + 1).flip(dims=(-1,)), 0).flip(dims=(-1,)).long()
        offs = F.pad(offs, (0, 1), value=1)
        full_code = coords_in[:, 0].long() * offs[0] + (code * offs[1:]).sum(dim=-1)
        windows, up_widx = full_code.unique(return_inverse=True)
        perm = torch.empty_like(up_widx)
        perm[up_widx] = torch.arange(len(up_widx), dtype=up_widx.dtype)
        h_coords = coords_in[perm].int()

        pos = (h_coords[:, 1:].float() + 0.5) / 64.0 * 2.0 - 1.0
        cos_vox, sin_vox = precompute_rope_cos_sin(pos.unsqueeze(0), base=128.0, head_dim=96)
        order_caches = build_hilbert_orders_cpu(h_coords)
        conv_indices = build_conv_indices_cpu(h_coords)

        cond_size = int(round(np.sqrt(dino_feats.shape[1] - 5)))
        cond_pos = torch.stack(
            torch.meshgrid(
                torch.arange(cond_size, dtype=torch.long),
                torch.arange(cond_size, dtype=torch.long),
                indexing="ij",
            ),
            dim=-1,
        ).view(1, -1, 2)
        cond_pos = (cond_pos.float() + 0.5) / cond_size * 2.0 - 1.0
        cos_cond, sin_cond = precompute_rope_cos_sin(cond_pos, base=64.0, head_dim=96)

        context = self.lin(dino_feats.to(torch.bfloat16), "image_map")

        return {
            "N": len(vox_idxi),
            "perm": perm,
            "up_widx": up_widx,
            "h_coords": h_coords,
            "cos_vox": cos_vox,
            "sin_vox": sin_vox,
            "order_caches": order_caches,
            "conv_indices": conv_indices,
            "cos_cond": cos_cond,
            "sin_cond": sin_cond,
            "context": context,
        }

    def compute_clf(self, t_scalar: float) -> torch.Tensor:
        t_tensor = torch.tensor([t_scalar * 1000.0], dtype=torch.float32)
        t_enc = get_time_encoding_ref(t_tensor, 256).to(torch.bfloat16)
        h = F.silu(self.lin(t_enc, "flow.time_embed.0"))
        temb = self.lin(h, "flow.time_embed.2")
        return F.silu(temb)

    @torch.no_grad()
    def forward_flow(
        self,
        xt_sorted: torch.Tensor,
        t_scalar: float,
        sample_cache: dict,
        uncond: bool = False,
        uncond_full_tokens: bool = False,
        cached_cond_block7: torch.Tensor = None,
        return_cond_block7: bool = False,
    ):
        clf = self.compute_clf(t_scalar)
        N = sample_cache["N"]
        D, H, HD = self.D, self.H, self.HD
        cos_vox, sin_vox = sample_cache["cos_vox"], sample_cache["sin_vox"]
        order_caches = sample_cache["order_caches"]
        conv_indices = sample_cache["conv_indices"]

        w_patch = self.sd["flow.patchify.weight"].to(torch.bfloat16)
        x = F.linear(xt_sorted.to(torch.bfloat16), w_patch)

        if uncond and not uncond_full_tokens:
            cond = torch.zeros(1, 1, D, dtype=torch.bfloat16)
        elif uncond and uncond_full_tokens:
            cond = torch.zeros_like(sample_cache["context"])
            cos_cond, sin_cond = sample_cache["cos_cond"], sample_cache["sin_cond"]
        else:
            cond = sample_cache["context"].clone()
            cos_cond, sin_cond = sample_cache["cos_cond"], sample_cache["sin_cond"]

        ln_cond_cached = None

        for b in range(24):
            p = f"flow.blocks.{b}"
            gate_base = self.lin(clf, f"{p}.gates")
            s_msa, sh_msa, g_msa, s_cr, sh_cr, g_cr, s_mlp, sh_mlp, g_mlp = gate_base.chunk(9, dim=-1)

            # 1. Windowed Self-Attention on x
            h = layer_norm_ref(x, 1 + s_msa.view(-1), sh_msa.view(-1))
            qkv = self.lin(h, f"{p}.to_qkv").view(1, N, 3, H, HD)
            q, k, v = qkv.unbind(dim=-3)
            q = rms_norm_ref(q) * self.sd[f"{p}.qk_norm.q_scale"].to(q.dtype)
            k = rms_norm_ref(k) * self.sd[f"{p}.qk_norm.k_scale"].to(k.dtype)
            q = apply_rope_cached(q, cos_vox, sin_vox).squeeze(0)
            k = apply_rope_cached(k, cos_vox, sin_vox).squeeze(0)
            v = v.squeeze(0)

            fwd, bwd, chunks = order_caches[b % 4]
            q_s, k_s, v_s = q[fwd], k[fwd], v[fwd]
            out_chunks = []
            for sl in chunks:
                out_c = F.scaled_dot_product_attention(
                    q_s[sl].unsqueeze(0).transpose(1, 2),
                    k_s[sl].unsqueeze(0).transpose(1, 2),
                    v_s[sl].unsqueeze(0).transpose(1, 2),
                ).transpose(1, 2).squeeze(0)
                out_chunks.append(out_c)
            attn_out = torch.cat(out_chunks, dim=0)[bwd].reshape(N, D)
            x = x + self.lin(attn_out, f"{p}.to_out") * g_msa.view(1, -1)

            # 2. Conditioning Self-Attention + Cross-Attention
            if b < 8:
                cond_gate_base = self.lin(clf, f"{p}.cond_gates")
                cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cond_gate_base.chunk(9, dim=-1)

                if uncond and not uncond_full_tokens:
                    h_cond = layer_norm_ref(cond, 1 + cs_msa.view(-1), csh_msa.view(-1))
                    v_c = self.lin(h_cond, f"{p}.cond_to_qkv").view(1, 1, 3, H, HD)[:, :, 2]
                    cond = cond + self.lin(v_c.reshape(1, 1, D), f"{p}.cond_to_out") * cg_msa[:, None]

                    h_x = layer_norm_ref(x, 1 + s_cr.view(-1), sh_cr.view(-1))
                    _, k_x, v_x = self.lin(h_x, f"{p}.to_qkv_cross").view(N, 3, H, HD).unbind(dim=-3)

                    h_cond_cr = layer_norm_ref(cond, 1 + cs_cr.view(-1), csh_cr.view(-1))
                    q_cc, _, v_cc = self.lin(h_cond_cr, f"{p}.cond_to_qkv_cross").view(1, 3, H, HD).unbind(dim=-3)

                    x = x + self.lin(v_cc.reshape(1, D), f"{p}.to_out_cross") * g_cr.view(1, -1)

                    h_c_out = F.scaled_dot_product_attention(
                        q_cc.unsqueeze(0).transpose(1, 2),
                        k_x.unsqueeze(0).transpose(1, 2),
                        v_x.unsqueeze(0).transpose(1, 2),
                    ).transpose(1, 2).reshape(1, 1, D)
                    cond = cond + self.lin(h_c_out, f"{p}.cond_to_out_cross") * cg_cr[:, None]

                    h_cond_mlp = layer_norm_ref(cond, 1 + cs_mlp.view(-1), csh_mlp.view(-1))
                    hc1, hc_gate = self.lin(h_cond_mlp, f"{p}.cond_ffn0").chunk(2, dim=-1)
                    cond = cond + self.lin(swiglu_ref(hc_gate, hc1), f"{p}.cond_ffn1") * cg_mlp[:, None]
                else:
                    L = cond.shape[1]
                    h_cond = layer_norm_ref(cond, 1 + cs_msa.view(-1), csh_msa.view(-1))
                    q_c, k_c, v_c = self.lin(h_cond, f"{p}.cond_to_qkv").view(1, L, 3, H, HD).unbind(dim=-3)
                    q_c = rms_norm_ref(q_c) * self.sd[f"{p}.cond_qk_norm.q_scale"].to(q_c.dtype)
                    k_c = rms_norm_ref(k_c) * self.sd[f"{p}.cond_qk_norm.k_scale"].to(k_c.dtype)
                    q_c = torch.cat([q_c[:, :5], apply_rope_cached(q_c[:, 5:], cos_cond, sin_cond)], dim=1)
                    k_c = torch.cat([k_c[:, :5], apply_rope_cached(k_c[:, 5:], cos_cond, sin_cond)], dim=1)
                    attn_c = F.scaled_dot_product_attention(
                        q_c.transpose(1, 2), k_c.transpose(1, 2), v_c.transpose(1, 2)
                    ).transpose(1, 2).reshape(1, L, D)
                    cond = cond + self.lin(attn_c, f"{p}.cond_to_out") * cg_msa[:, None]

                    h_x = layer_norm_ref(x, 1 + s_cr.view(-1), sh_cr.view(-1))
                    q_x, k_x, v_x = self.lin(h_x, f"{p}.to_qkv_cross").view(N, 3, H, HD).unbind(dim=-3)

                    h_cond_cr = layer_norm_ref(cond, 1 + cs_cr.view(-1), csh_cr.view(-1))
                    q_cc, k_cc, v_cc = self.lin(h_cond_cr, f"{p}.cond_to_qkv_cross").view(L, 3, H, HD).unbind(dim=-3)

                    h_x_out = F.scaled_dot_product_attention(
                        q_x.unsqueeze(0).transpose(1, 2),
                        k_cc.unsqueeze(0).transpose(1, 2),
                        v_cc.unsqueeze(0).transpose(1, 2),
                    ).transpose(1, 2).reshape(N, D)
                    x = x + self.lin(h_x_out, f"{p}.to_out_cross") * g_cr.view(1, -1)

                    h_c_out = F.scaled_dot_product_attention(
                        q_cc.unsqueeze(0).transpose(1, 2),
                        k_x.unsqueeze(0).transpose(1, 2),
                        v_x.unsqueeze(0).transpose(1, 2),
                    ).transpose(1, 2).reshape(1, L, D)
                    cond = cond + self.lin(h_c_out, f"{p}.cond_to_out_cross") * cg_cr[:, None]

                    h_cond_mlp = layer_norm_ref(cond, 1 + cs_mlp.view(-1), csh_mlp.view(-1))
                    hc1, hc_gate = self.lin(h_cond_mlp, f"{p}.cond_ffn0").chunk(2, dim=-1)
                    cond = cond + self.lin(swiglu_ref(hc_gate, hc1), f"{p}.cond_ffn1") * cg_mlp[:, None]

                if b == 7:
                    if cached_cond_block7 is not None:
                        cond = cached_cond_block7
                    ln_cond_cached = layer_norm_ref(cond)
            else:
                cs_cr, csh_cr = self.lin(clf, f"{p}.cond_gates").chunk(2, dim=-1)
                h_cond_cr = ln_cond_cached * (1 + cs_cr[:, None]) + csh_cr[:, None]
                if uncond and not uncond_full_tokens:
                    _, v_cc = self.lin(h_cond_cr, f"{p}.cond_to_kv_cross").view(1, 2, H, HD).unbind(dim=-3)
                    x = x + self.lin(v_cc.reshape(1, D), f"{p}.to_out_cross") * g_cr.view(1, -1)
                else:
                    L = cond.shape[1]
                    h_x = layer_norm_ref(x, 1 + s_cr.view(-1), sh_cr.view(-1))
                    q_x = self.lin(h_x, f"{p}.to_q_cross").view(N, H, HD)
                    k_cc, v_cc = self.lin(h_cond_cr, f"{p}.cond_to_kv_cross").view(L, 2, H, HD).unbind(dim=-3)
                    h_x_out = F.scaled_dot_product_attention(
                        q_x.unsqueeze(0).transpose(1, 2),
                        k_cc.unsqueeze(0).transpose(1, 2),
                        v_cc.unsqueeze(0).transpose(1, 2),
                    ).transpose(1, 2).reshape(N, D)
                    x = x + self.lin(h_x_out, f"{p}.to_out_cross") * g_cr.view(1, -1)

            # 3. Main FFN + 3D Sparse Conv + SwiGLU
            h_mlp = layer_norm_ref(x, 1 + s_mlp.view(-1), sh_mlp.view(-1))
            h_mlp = self.lin(h_mlp, f"{p}.ffn0")
            h_mlp = sparse_conv3d_ref(
                h_mlp,
                self.sd[f"{p}.ffn_conv.weight"],
                self.sd[f"{p}.ffn_conv.bias"],
                conv_indices,
            )
            h1, h_gate = h_mlp.chunk(2, dim=-1)
            h_mlp = self.lin(swiglu_ref(h_gate, h1), f"{p}.ffn1")
            x = x + h_mlp * g_mlp.view(1, -1)

        out_s, out_sh = self.lin(clf, "flow.out_gates").chunk(2, dim=-1)
        x = layer_norm_ref(x, 1 + out_s.view(-1), out_sh.view(-1))
        w_unpatch = self.sd["flow.unpatchify.weight"].to(torch.bfloat16)
        out = F.linear(x, w_unpatch).float()
        if return_cond_block7:
            return out, cond
        return out

    def denormalize_and_unsort(self, xt_sorted: torch.Tensor, sample_cache: dict) -> torch.Tensor:
        xt = xt_sorted[sample_cache["up_widx"]].float()
        return (xt * self.std + self.mean) * self.latent_scale


def make_t_seq(num_steps: int, rescale_t: float = 3.0) -> torch.Tensor:
    t_seq = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    return rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)


def make_piecewise_t_seq(n_guided: int, n_unguided: int, rescale_t: float = 3.0, p_guided: float = 1.0, p_unguided: float = 1.0) -> torch.Tensor:
    s1 = torch.linspace(0.0, 1.0, n_guided + 1, dtype=torch.float32) ** p_guided
    u_guided = 1.0 - 0.75 * s1  # 1.0 -> 0.25 (t: 1.0 -> 0.5)
    s2 = torch.linspace(0.0, 1.0, n_unguided + 1, dtype=torch.float32) ** p_unguided
    u_unguided = 0.25 * (1.0 - s2)  # 0.25 -> 0.0 (t: 0.5 -> 0.0)
    u_seq = torch.cat([u_guided[:-1], u_unguided])
    return rescale_t * u_seq / (1.0 + (rescale_t - 1.0) * u_seq)


def evaluate_prediction(pred: torch.Tensor, target: torch.Tensor, name: str, nfe_pos: int, nfe_neg: int):
    diff = pred.float() - target.float()
    mse = diff.pow(2).mean().item()
    rmse = np.sqrt(mse)
    mae = diff.abs().mean().item()
    rel_l2 = (diff.norm() / target.float().norm()).item()
    cos = F.cosine_similarity(pred.float().flatten(), target.float().flatten(), dim=0).item()
    per_vox_cos = F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().item()
    eff_nfe = nfe_pos + 0.42 * nfe_neg
    speedup_est = 88.0 / eff_nfe
    print(
        f"  {name:42s} | pos={nfe_pos:2d} neg={nfe_neg:2d} (eff={eff_nfe:5.1f}, ~{speedup_est:4.1f}x) | "
        f"RMSE={rmse:.4f} MAE={mae:.4f} RelL2={rel_l2:.4f} Cos={cos:.5f} VoxCos={per_vox_cos:.5f}"
    )
    return {
        "name": name,
        "nfe_pos": nfe_pos,
        "nfe_neg": nfe_neg,
        "eff_nfe": eff_nfe,
        "speedup_est": speedup_est,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "cos": cos,
        "per_vox_cos": per_vox_cos,
    }


def run_solver_experiments(runner: FastTeacherRunner):
    print("\n--- Running Trajectory & Solver Experiments on Real Sample ---")
    sample_dir = Path("fetched_info/sample_test_data/public_test/22a9a0954ea14062b20dc3f826081811/cache")
    vox_all = torch.tensor(np.load(sample_dir / "vox_idxi.npz")["vox_idxi"], dtype=torch.int32)
    dino_data = np.load(sample_dir / "4view_000_dinov3.npz")
    dino_feats_full = torch.tensor(dino_data["feats"], dtype=torch.float32)

    # Use 128 real voxels and 8x8 pooled DINO spatial tokens (69 tokens) so 1 forward pass takes ~0.18s on CPU
    vox_sub = vox_all[:128]
    cls_tokens = dino_feats_full[:, :5, :]
    spatial_tokens = dino_feats_full[:, 5:, :].view(1, 64, 64, 1280).permute(0, 3, 1, 2)
    spatial_pooled = F.avg_pool2d(spatial_tokens, kernel_size=8, stride=8).permute(0, 2, 3, 1).reshape(1, 64, 1280)
    dino_sub = torch.cat([cls_tokens, spatial_pooled], dim=1)

    sample_cache = runner.prepare_sample(vox_sub, dino_sub)

    torch.manual_seed(42)
    noise_orig = torch.randn(len(vox_sub), 16, dtype=torch.float32)
    x_init = noise_orig[sample_cache["perm"]].clone()

    # 0. Verify 24-block exact equivalence of 1-token uncond vs full-token uncond!
    t0 = time.time()
    out_uncond_full = runner.forward_flow(x_init, 0.85, sample_cache, uncond=True, uncond_full_tokens=True)
    dt_neg_full = time.time() - t0
    t0 = time.time()
    out_uncond_1tok = runner.forward_flow(x_init, 0.85, sample_cache, uncond=True, uncond_full_tokens=False)
    dt_neg_1tok = time.time() - t0
    uncond_24b_diff = (out_uncond_full - out_uncond_1tok).abs().max().item()
    uncond_24b_cos = F.cosine_similarity(out_uncond_full.flatten(), out_uncond_1tok.flatten(), dim=0).item()
    print(
        f"24-Block Full Uncond vs 1-Token Condensed Uncond: "
        f"max_abs_diff={uncond_24b_diff:.8f}, cos={uncond_24b_cos:.8f} "
        f"({dt_neg_full:.2f}s -> {dt_neg_1tok:.2f}s)"
    )

    t0 = time.time()
    _ = runner.forward_flow(x_init, 1.0, sample_cache, uncond=False)
    dt_pos = time.time() - t0
    print(f"Single 24-block forward pass time on CPU: pos={dt_pos:.2f}s, 1-token neg={dt_neg_1tok:.2f}s")

    guidance_scale = 7.0
    cfg_range = (0.5, 1.0)

    # 1. Baseline 50-step Euler (Teacher Reference)
    print("Computing 50-step Teacher Reference (88 NFEs)...")
    t_seq_50 = make_t_seq(50, rescale_t=3.0)
    xt = x_init.clone()
    teacher_v_pos = []
    teacher_v_neg = []
    for i in range(50):
        t_val = float(t_seq_50[i].item())
        t_next = float(t_seq_50[i + 1].item())
        do_cfg = cfg_range[0] <= t_val <= cfg_range[1]
        v_pos = runner.forward_flow(xt, t_val, sample_cache, uncond=False)
        if do_cfg:
            v_neg = runner.forward_flow(xt, t_val, sample_cache, uncond=True)
            v = v_pos + guidance_scale * (v_pos - v_neg)
            teacher_v_neg.append(v_neg)
        else:
            v = v_pos
        teacher_v_pos.append(v_pos)
        xt = xt - (t_val - t_next) * v

    z_teacher = runner.denormalize_and_unsort(xt, sample_cache)

    cos_v_pos = [
        F.cosine_similarity(teacher_v_pos[i].flatten(), teacher_v_pos[i + 1].flatten(), dim=0).item()
        for i in range(len(teacher_v_pos) - 1)
    ]
    cos_v_neg = [
        F.cosine_similarity(teacher_v_neg[i].flatten(), teacher_v_neg[i + 1].flatten(), dim=0).item()
        for i in range(len(teacher_v_neg) - 1)
    ]
    deltas = [teacher_v_pos[i] - teacher_v_neg[i] for i in range(len(teacher_v_neg))]
    cos_delta = [
        F.cosine_similarity(deltas[i].flatten(), deltas[i + 1].flatten(), dim=0).item()
        for i in range(len(deltas) - 1)
    ]
    print(f"Teacher trajectory step-to-step cosine similarity:")
    print(f"  v_pos mean cos:   {np.mean(cos_v_pos):.5f} (min={np.min(cos_v_pos):.5f})")
    print(f"  v_neg mean cos:   {np.mean(cos_v_neg):.5f} (min={np.min(cos_v_neg):.5f})")
    print(f"  cfg_delta cos:    {np.mean(cos_delta):.5f} (min={np.min(cos_delta):.5f})")

    results = []

    def run_euler(num_steps: int):
        t_seq = make_t_seq(num_steps, rescale_t=3.0)
        xt = x_init.clone()
        n_pos, n_neg = 0, 0
        for i in range(num_steps):
            t_val = float(t_seq[i].item())
            t_next = float(t_seq[i + 1].item())
            do_cfg = cfg_range[0] <= t_val <= cfg_range[1]
            v_pos = runner.forward_flow(xt, t_val, sample_cache, uncond=False)
            n_pos += 1
            if do_cfg:
                v_neg = runner.forward_flow(xt, t_val, sample_cache, uncond=True)
                n_neg += 1
                v = v_pos + guidance_scale * (v_pos - v_neg)
            else:
                v = v_pos
            xt = xt - (t_val - t_next) * v
        z = runner.denormalize_and_unsort(xt, sample_cache)
        return evaluate_prediction(z, z_teacher, f"Euler N={num_steps}", n_pos, n_neg)

    for n_steps in [25, 20, 16, 12, 10, 8, 6]:
        results.append(run_euler(n_steps))

    def run_piecewise_solver(
        n_guided: int,
        n_unguided: int,
        use_ab2: bool = False,
        cfg_stride: int = 1,
        extrapolate_delta: bool = False,
        reuse_neg: bool = False,
        label: str = "",
    ):
        t_seq = make_piecewise_t_seq(n_guided, n_unguided, rescale_t=3.0)
        xt = x_init.clone()
        n_pos, n_neg = 0, 0
        prev_v = None
        prev_dt = None
        last_delta = None
        prev_delta = None
        last_neg = None

        total_steps = n_guided + n_unguided
        for i in range(total_steps):
            t_val = float(t_seq[i].item())
            t_next = float(t_seq[i + 1].item())
            dt = t_val - t_next
            in_guided = i < n_guided

            if i == n_guided:
                prev_v = None
                prev_dt = None

            v_pos = runner.forward_flow(xt, t_val, sample_cache, uncond=False)
            n_pos += 1

            if in_guided:
                if (i % cfg_stride == 0) or (last_delta is None) or (i == 0) or (i == n_guided - 1):
                    v_neg = runner.forward_flow(xt, t_val, sample_cache, uncond=True)
                    n_neg += 1
                    cur_delta = v_pos - v_neg
                    if last_delta is not None:
                        prev_delta = last_delta
                    last_delta = cur_delta
                    last_neg = v_neg
                else:
                    if reuse_neg and last_neg is not None:
                        cur_delta = v_pos - last_neg
                    elif extrapolate_delta and prev_delta is not None:
                        cur_delta = last_delta + 0.5 * (last_delta - prev_delta)
                    else:
                        cur_delta = last_delta
                v = v_pos + guidance_scale * cur_delta
            else:
                v = v_pos

            if use_ab2 and (prev_v is not None) and (prev_dt is not None):
                ratio = min(0.5 * (dt / max(prev_dt, 1e-6)), 0.5)
                v_step = (1.0 + ratio) * v - ratio * prev_v
            else:
                v_step = v

            prev_v = v
            prev_dt = dt
            xt = xt - dt * v_step

        z = runner.denormalize_and_unsort(xt, sample_cache)
        return evaluate_prediction(z, z_teacher, label, n_pos, n_neg)

    for (ng, nu) in [(14, 4), (12, 4), (10, 3), (9, 3), (8, 2), (6, 2)]:
        results.append(
            run_piecewise_solver(ng, nu, use_ab2=False, cfg_stride=1, label=f"PW-Euler ({ng}+{nu})")
        )
        results.append(
            run_piecewise_solver(ng, nu, use_ab2=True, cfg_stride=1, label=f"PW-AB2 ({ng}+{nu})")
        )
        results.append(
            run_piecewise_solver(ng, nu, use_ab2=False, cfg_stride=2, label=f"PW-Euler CFG-delta-s2 ({ng}+{nu})")
        )
        results.append(
            run_piecewise_solver(ng, nu, use_ab2=False, cfg_stride=2, reuse_neg=True, label=f"PW-Euler CFG-neg-s2 ({ng}+{nu})")
        )
        results.append(
            run_piecewise_solver(ng, nu, use_ab2=True, cfg_stride=2, label=f"PW-AB2 CFG-delta-s2 ({ng}+{nu})")
        )

    out_dir = Path("fetched_info")
    with open(out_dir / "solver_experiments.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"PyTorch {torch.__version__}, CPU count: {os.cpu_count()}")

    state_dict, stats = stream_safetensors_from_remote_zip(WEIGHTS_URL, SAFETENSORS_ARCHIVE_PATH)
    out_dir = Path("fetched_info")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "weights_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved weights_stats.json!")
    print("  mean:", stats.get("mean"))
    print("  std:", stats.get("std"))

    runner = FastTeacherRunner(state_dict)
    run_solver_experiments(runner)


if __name__ == "__main__":
    main()
