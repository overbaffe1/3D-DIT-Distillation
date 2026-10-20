import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

from torch.utils.cpp_extension import include_paths, library_paths

SOURCE_ROOT = os.path.dirname(__file__)
sources = {
    "cpp": ["curope.cpp"],           # Все обычные C++ файлы
    "cuda": ["rope.cu"]  # Все CUDA файлы
}

cuda_extension = CUDAExtension(
    name="librope",
    sources=[os.path.join(SOURCE_ROOT, src) for src in sources["cuda"] + sources["cpp"]],
    extra_compile_args={
        "nvcc": ["-O3"],
    },
    include_dirs=include_paths() + [SOURCE_ROOT],
    library_dirs=library_paths(),
)

setup(
    name="librope",
    version="1.0",
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": BuildExtension}
)