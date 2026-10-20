import os
import hashlib
import re
import uuid
import shutil
import torch
import torch.utils.cpp_extension

__cached_ops = {}

def load_ext(name, headers, sources, source_root, verbose=False, optimize=True, cflags=(), cuda_flags=()):
    
    if name not in __cached_ops:
    
        files = [os.path.join(source_root, fname) for fname in headers + sources]
    
        hash_md5 = hashlib.md5()
        for file in files:
            with open(file, "rb") as f:
                hash_md5.update(f.read())
    
        build_top_dir = torch.utils.cpp_extension._get_build_directory(name, verbose=True)
        cached_build_dir = os.path.join(
            build_top_dir,
            f"{hash_md5.hexdigest()}-{re.sub('[^0-9a-zA-Z]+', '-', torch.cuda.get_device_name().lower())}",
        )
        def _ensure_cached_src_dir():
            tmpdir = f"{build_top_dir}/srctmp-{uuid.uuid4().hex}"
            os.makedirs(tmpdir, exist_ok=True)
            for file in files:
                shutil.copyfile(file, os.path.join(tmpdir, os.path.basename(file)))
            try:
                os.replace(tmpdir, cached_build_dir)
            except OSError:
                shutil.rmtree(tmpdir)
                if not os.path.isdir(cached_build_dir):
                    raise

        if not os.path.isdir(cached_build_dir):
            _ensure_cached_src_dir()

        extra_includes = [cached_build_dir]

        extra_cuda_flags = list(cuda_flags)

        __cached_ops[name] = torch.utils.cpp_extension.load(
            name=name,
            build_directory=cached_build_dir,
            extra_include_paths=extra_includes,
            sources=[os.path.join(cached_build_dir, os.path.basename(src)) for src in sources],
            verbose=verbose,
            extra_cflags=(["-O3"] if optimize else []) + list(cflags),
            extra_cuda_cflags=(["-O3"] if optimize else []) + extra_cuda_flags,
        )

    return __cached_ops[name]
