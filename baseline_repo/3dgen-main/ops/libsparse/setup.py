import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import include_paths, library_paths

SOURCE_ROOT = os.path.dirname(__file__)

sources = {
    "cpp": ["libsparse.cpp"],
    "cuda": ["conv.cu", "vox.cu", "pcd_attn.cu", "../libmesh/bvh.cu"],
}

cuda_extension = CUDAExtension(
    name="libsparse",
    sources=[os.path.join(SOURCE_ROOT, src) for src in sources["cuda"] + sources["cpp"]],
    extra_compile_args={
        "nvcc": [
            # "-O3",
            # "-gencode=arch=compute_90,code=sm_90",
            # "-gencode=arch=compute_120,code=sm_120",
            # "--use_fast_math",
            # "--extra-device-vectorization",
        ],
    },
    include_dirs=include_paths() + [SOURCE_ROOT, os.path.join(SOURCE_ROOT, "..", "libmesh")],
    library_dirs=library_paths(),
)

setup(
    name="libsparse",
    version="1.0",
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": BuildExtension},
)
