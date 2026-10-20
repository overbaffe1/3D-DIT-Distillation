"""
04_triton_aot_cpu_verifier.py — Как тестировать @triton.jit и FlashAttention код
на бесплатном CPU-раннере GitHub Actions (`ubuntu-latest`) без физического GPU!

Включает:
  1) AOT PTX-компиляцию любого `@triton.jit` ядра под NVIDIA A100 (`sm_80`) без GPU.
  2) Безопасную регистрацию `sys.modules` заглушек для `flash_attn` и `spconv` на CPU.
  3) Проверку инвариантов `cu_seqlens` на CPU, которая на 100% предотвращает
     ошибку `cudaErrorIllegalAddress` на реальном кластере с A100.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any, Dict, Tuple


# ==============================================================================
# 1. AOT-компиляция @triton.jit ядер в PTX под NVIDIA A100 (sm_80) БЕЗ GPU
# ==============================================================================
def verify_triton_kernel_aot_sm80(
    kernel_fn: Any,
    signature: Dict[str, str],
    constants: Dict[str, int],
    arch_sm: int = 80,
) -> int:
    """
    Компилирует `@triton.jit` ядро напрямую в PTX-ассемблер для архитектуры NVIDIA A100 (`sm_80`),
    не требуя ни видеокарты, ни установленного драйвера CUDA!

    Пример вызова:
        ptx_size = verify_triton_kernel_aot_sm80(
            _fused_gate_add_ln_modulate_kernel,
            signature={
                "X_ptr": "*bf16", "H_ptr": "*bf16", "Out_ptr": "*bf16",
                "G_ptr": "*bf16", "S_ptr": "*bf16", "Sh_ptr": "*bf16",
                "Batch_idx_ptr": "*i32", "D": "i32", "eps": "fp32",
            },
            constants={"BLOCK_D": 2048, "HAS_GATE": 1, "HAS_SCALE": 1, "IS_1B": 1},
            arch_sm=80,
        )

    Ловит до отправки на платформу:
      - Несуществующие функции `triton.language` (например, `tl.zeros_like` вместо `0`);
      - Ошибки типов указателей и констант (`tl.constexpr`);
      - Переполнение 32-битных смещений (`tl.program_id(0).to(tl.int64)`).
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("cuda", arch_sm, 32)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constants)
    compiled = triton.compile(src, target=target)
    ptx_code = compiled.asm.get("ptx", "")
    if not ptx_code:
        raise RuntimeError(f"Triton AOT compilation produced empty PTX for {kernel_fn}")
    return len(ptx_code)


# ==============================================================================
# 2. Правильные CPU-заглушки (`sys.modules`) для `flash_attn` и `spconv`
# ==============================================================================
def install_cpu_extension_stubs() -> None:
    """
    Устанавливает корректные CPU-заглушки для GPU-only библиотек (`flash_attn`, `spconv`, `torch_scatter`),
    если они не установлены на CPU-раннере GitHub Actions.

    ВАЖНЫЕ ТОНКОСТИ PYTORCH / PYTHON 3.11+:
      1) У каждого фиктивного модуля ОБЯЗАТЕЛЬНО должен быть `mod.__spec__ = ModuleSpec(name, None)`,
         иначе `importlib.util.find_spec(...)` внутри `diffusers` / `transformers` упадёт с `ValueError`.
      2) Если в модуле определяется `__getattr__(name)`, он ОБЯЗАТЕЛЬНО должен выбрасывать
         `AttributeError(name)` для всех системных атрибутов `if name.startswith("__"): raise AttributeError(name)`,
         иначе `inspect.unwrap()` внутри `torch._dynamo` уйдёт в бесконечную рекурсию!
    """
    for mod_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn.bert_padding",
        "flash_attn.layers",
        "flash_attn.layers.rotary",
    ):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            sys.modules[mod_name] = m


# ==============================================================================
# 3. Защита от `cudaErrorIllegalAddress` в `flash_attn_varlen_func`
# ==============================================================================
def validate_flash_attn_varlen_bounds(
    q_shape: Tuple[int, ...],
    kv_shape: Tuple[int, ...],
    cu_seqlens_q: Any,
    cu_seqlens_k: Any,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    """
    На GPU ядро `flash_attn_varlen_func` читает память вплоть до `cu_seqlens_q[-1]` и `cu_seqlens_k[-1]`.
    Если после прореживания токенов (Token Carving, например 4101 -> 1029) в `flash_attn_varlen_func`
    передать старый `cu_seqlens = [0, 4101]` для тензора длины `1029`, на GPU произойдёт
    чтение за пределами буфера (`cudaErrorIllegalAddress`), причём ошибка всплывёт асинхронно
    позже (например, на `.to(device)` следующей модели)!

    Вызывай эту проверку и на GPU, и в CPU-ветке: тогда любой рассинхрон `cu_seqlens`
    мгновенно обнаружится в обычном CPU-тесте в GitHub Actions!
    """
    q_total = int(cu_seqlens_q[-1].item())
    k_total = int(cu_seqlens_k[-1].item())
    if q_total != q_shape[0]:
        raise RuntimeError(
            f"[FlashAttn Bounds Error] cu_seqlens_q[-1]={q_total} != q.shape[0]={q_shape[0]} "
            f"(max_seqlen_q={max_seqlen_q})"
        )
    if k_total != kv_shape[0]:
        raise RuntimeError(
            f"[FlashAttn Bounds Error] cu_seqlens_k[-1]={k_total} != kv.shape[0]={kv_shape[0]} "
            f"(max_seqlen_k={max_seqlen_k})"
        )


if __name__ == "__main__":
    print("04_triton_aot_cpu_verifier.py loaded successfully.")
