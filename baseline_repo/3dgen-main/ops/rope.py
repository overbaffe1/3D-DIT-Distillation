import os

import torch

from . import custom_ops

_curope = None


def _init():
    global _curope
    if _curope is None:
        try:
            import librope as _curope
        except ImportError:
            _curope = custom_ops.load_ext(
                "curope",
                headers=[],
                sources=["curope.cpp", "rope.cu"],
                source_root=os.path.join(os.path.dirname(__file__), "curope"),
                verbose=True,
                optimize=False,
            )
    return True


class cuRoPEND_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        tokens = tokens.contiguous()
        positions = positions.float().contiguous()
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        out = torch.empty_like(tokens)
        _curope.rope_nd(tokens, positions, base, F0, out)
        return out

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        out = torch.empty_like(grad_res)
        _curope.rope_nd(grad_res.contiguous(), positions, base, -F0, out)
        return out, None, None, None


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def rope_nd(x, pos, freq=100.0):
    if not _init():
        raise RuntimeError(
            f"Cannot load curope in {__file__} set VERBOSE=True for more information!"
        )
    return cuRoPEND_func.apply(x, pos, freq).type_as(x)


rope_nd = torch._dynamo.disable(rope_nd)
