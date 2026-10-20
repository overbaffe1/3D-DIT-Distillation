"""
02_remotezip_and_chunking.py — Инструментарий для работы с гигабайтными архивами,
потоковым чтением .safetensors по HTTP Range (RemoteZip) и нарезкой файлов на чанки (< 45 MB)
для сохранения в Git без нарушения лимита GitHub (100 MB).

Запускается внутри GitHub Actions (или локально для сборки/нарезки чанков).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ==============================================================================
# 1. Инспекция удалённого ZIP-архива (10+ ГБ) за 1 секунду через HTTP Range
# ==============================================================================
def inspect_remote_zip(url: str) -> List[Dict[str, Any]]:
    """
    Читает только Central Directory удалённого ZIP-архива через HTTP Range-запросы
    (требуется пакет `remotezip`: `pip install remotezip`).
    Возвращает список всех файлов внутри архива с их размерами без скачивания архива!
    """
    from remotezip import RemoteZip

    entries: List[Dict[str, Any]] = []
    with RemoteZip(url) as zf:
        for info in zf.infolist():
            entries.append(
                {
                    "filename": info.filename,
                    "file_size_bytes": info.file_size,
                    "file_size_mb": round(info.file_size / (1024 * 1024), 2),
                    "compress_size_mb": round(info.compress_size / (1024 * 1024), 2),
                }
            )
    return entries


# ==============================================================================
# 2. Потоковое чтение заголовка и нужных слоёв из .safetensors внутри RemoteZip
# ==============================================================================
def inspect_safetensors_header_in_remote_zip(
    zip_url: str, safetensors_member_path: str
) -> Dict[str, Any]:
    """
    Считывает только JSON-заголовок файла `.safetensors` (первые несколько десятков КБ)
    внутри гигабайтного удалённого ZIP-архива.
    Возвращает словарь `{tensor_name: {"dtype": ..., "shape": [...], "data_offsets": [start, end]}}`.
    """
    from remotezip import RemoteZip

    with RemoteZip(zip_url) as zf:
        with zf.open(safetensors_member_path) as fp:
            header_len = struct.unpack("<Q", fp.read(8))[0]
            header_json = fp.read(header_len).decode("utf-8")
            header = json.loads(header_json)
    return header


def stream_safetensors_subset_from_remote_zip(
    zip_url: str,
    safetensors_member_path: str,
    prefix_filter: str = "flow.blocks.0.",
    max_byte_offset: int = 80_000_000,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    ВАЖНОЕ ПРАВИЛО РАБОТЫ С REMOTEZIP:
    Сжатый deflate-поток внутри ZIP нельзя перематывать назад (`seek` назад заставляет
    качать файл с самого начала). Поэтому мы:
      1) Читаем 8-байтную длину заголовка и сам JSON-заголовок `.safetensors`.
      2) Отбираем нужные тензоры и СОРТИРУЕМ их строго по возрастанию `data_offsets[0]`.
      3) Идём строго вперёд (`cur_offset`), пропуская ненужные промежутки через `fp.read(skip)`.
    """
    import numpy as np
    import torch
    from remotezip import RemoteZip

    t0 = time.perf_counter()
    tensors: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, Any]] = {}

    with RemoteZip(zip_url) as zf:
        with zf.open(safetensors_member_path) as fp:
            header_len = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_len).decode("utf-8"))

            selected = [
                (k, v)
                for k, v in header.items()
                if k != "__metadata__"
                and k.startswith(prefix_filter)
                and v["data_offsets"][1] <= max_byte_offset
            ]
            # Сортируем строго по начальному байтовому смещению!
            selected.sort(key=lambda item: item[1]["data_offsets"][0])

            cur_offset = 0
            for name, meta in selected:
                start, end = meta["data_offsets"]
                if start > cur_offset:
                    fp.read(start - cur_offset)
                nbytes = end - start
                raw = fp.read(nbytes)
                cur_offset = end

                dtype_str = meta.get("dtype", "F32")
                np_dtype = np.float16 if dtype_str == "F16" else np.float32
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(meta["shape"]).copy()
                tensors[name] = torch.from_numpy(arr)
                stats[name] = {
                    "shape": meta["shape"],
                    "dtype": dtype_str,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }

    elapsed = time.perf_counter() - t0
    print(f"[RemoteZip] Streamed {len(tensors)} tensors in {elapsed:.1f}s")
    return tensors, stats


# ==============================================================================
# 3. Безопасная распаковка архива с кодом бейзлайна (без вложенного .git!)
# ==============================================================================
def extract_code_archive_safely(
    zip_url: str,
    extract_dir: str | Path = "baseline_repo",
    max_single_file_mb: float = 10.0,
) -> Path:
    """
    Скачивает архив с кодом бейзлайна, распаковывает в `extract_dir` и:
      1) Удаляет любые вложенные папки `.git` (иначе `git add` добавит пустой gitlink 160000!).
      2) Удаляет слишком большие бинарные файлы (> max_single_file_mb), чтобы не засорять Git.
    """
    extract_path = Path(extract_dir)
    tmp_zip = Path("/tmp/_downloaded_archive.zip")
    print(f"Downloading {zip_url} -> {tmp_zip} ...")
    urllib.request.urlretrieve(zip_url, tmp_zip)

    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(extract_path)

    tmp_zip.unlink(missing_ok=True)

    # КРИТИЧЕСКИ ВАЖНО: удаляем все вложенные директории .git!
    for git_dir in list(extract_path.rglob(".git")):
        if git_dir.is_dir():
            print(f"Removing nested .git directory: {git_dir}")
            shutil.rmtree(git_dir)

    max_bytes = int(max_single_file_mb * 1024 * 1024)
    for p in list(extract_path.rglob("*")):
        if p.is_file() and p.stat().st_size > max_bytes:
            print(f"Removing oversized file ({p.stat().st_size / 1e6:.1f} MB): {p}")
            p.unlink()

    return extract_path


# ==============================================================================
# 4. Нарезка больших файлов на части (Chunking < 45 MB) и обратная склейка
# ==============================================================================
def split_file_into_chunks(
    src_file: str | Path,
    out_dir: str | Path,
    chunk_size_mb: int = 45,
    prefix: str = "chunk",
) -> Path:
    """
    Разрезает любой большой файл (например, веса 120 MB..500 MB) на части по `chunk_size_mb` (45 MB),
    чтобы их можно было спокойно закоммитить и запушить в GitHub (лимит GitHub = 100 MB на файл,
    рекомендуемый порог без предупреждений = 45 MB).
    Создаёт `manifest.json` с контрольными суммами SHA256.
    """
    src_path = Path(src_file)
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    chunk_bytes = chunk_size_mb * 1024 * 1024
    chunks_meta: List[Dict[str, Any]] = []
    total_hasher = hashlib.sha256()
    total_size = 0

    with open(src_path, "rb") as fp:
        idx = 0
        while True:
            data = fp.read(chunk_bytes)
            if not data:
                break
            chunk_name = f"{prefix}_{idx:03d}.part"
            chunk_path = dst_dir / chunk_name
            with open(chunk_path, "wb") as out_fp:
                out_fp.write(data)

            chunk_sha = hashlib.sha256(data).hexdigest()
            total_hasher.update(data)
            total_size += len(data)
            chunks_meta.append(
                {
                    "index": idx,
                    "filename": chunk_name,
                    "size_bytes": len(data),
                    "sha256": chunk_sha,
                }
            )
            idx += 1

    manifest = {
        "original_filename": src_path.name,
        "total_size_bytes": total_size,
        "total_sha256": total_hasher.hexdigest(),
        "chunk_size_mb": chunk_size_mb,
        "num_chunks": len(chunks_meta),
        "chunks": chunks_meta,
    }
    manifest_path = dst_dir / f"{prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(
        f"[Chunker] Split {src_path.name} ({total_size / 1e6:.2f} MB) into "
        f"{len(chunks_meta)} chunks in {dst_dir}"
    )
    return manifest_path


def reassemble_chunks_to_file(
    manifest_path: str | Path,
    output_file: Optional[str | Path] = None,
    verify_sha256: bool = True,
) -> Path:
    """
    Собирает разрезанный по частям файл обратно из `*_manifest.json` и проверяет SHA256.
    """
    mpath = Path(manifest_path)
    chunks_dir = mpath.parent
    with open(mpath, encoding="utf-8") as mf:
        manifest = json.load(mf)

    out_path = Path(output_file) if output_file else (chunks_dir / manifest["original_filename"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_hasher = hashlib.sha256()

    with open(out_path, "wb") as out_fp:
        for ch in sorted(manifest["chunks"], key=lambda x: x["index"]):
            cpath = chunks_dir / ch["filename"]
            data = cpath.read_bytes()
            if verify_sha256:
                sha = hashlib.sha256(data).hexdigest()
                if sha != ch["sha256"]:
                    raise ValueError(f"SHA256 mismatch in chunk {cpath}: {sha} != {ch['sha256']}")
            out_fp.write(data)
            total_hasher.update(data)

    if verify_sha256 and total_hasher.hexdigest() != manifest["total_sha256"]:
        raise ValueError("Total reassembled file SHA256 mismatch!")

    print(f"[Chunker] Reassembled {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) OK")
    return out_path


if __name__ == "__main__":
    # Быстрый самотест нарезки и обратной сборки в памяти /tmp
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample_bin = tmp_path / "test_weights.bin"
        sample_bin.write_bytes(os.urandom(250_000))
        manifest_file = split_file_into_chunks(sample_bin, tmp_path / "parts", chunk_size_mb=1)
        restored_file = reassemble_chunks_to_file(manifest_file, tmp_path / "restored.bin")
        assert sample_bin.read_bytes() == restored_file.read_bytes()
        print("Self-test passed: split_file_into_chunks & reassemble_chunks_to_file OK!")
