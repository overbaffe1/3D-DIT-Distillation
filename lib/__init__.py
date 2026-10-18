"""Accelerated inference and distillation library for Kandinsky3D LatDiT."""

from lib.fast_kernels import preload_extensions

preload_extensions(verbose=False)

from lib.fast_flow import OptimizedLatDitRunner
from lib.fast_sampler import SolverConfig

__all__ = ["preload_extensions", "OptimizedLatDitRunner", "SolverConfig"]
