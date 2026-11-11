#!/usr/bin/env python3
"""Build script for the causa Python extension.

Uses only the standard Python C API - no external dependencies.

Build:
    python setup.py build_ext --inplace

Test:
    python test_causa.py
    python test_causa_scm.py
"""

import os
import sys
from setuptools import setup, Extension

HERE = os.path.abspath(os.path.dirname(__file__))

# The C sources that make up the core library
CORE_SOURCES = [
    os.path.join("src", "causa_arena.c"),
    os.path.join("src", "causa_tensor.c"),
    os.path.join("src", "causa_graph.c"),
    os.path.join("src", "causa_backend_cpu.c"),
    os.path.join("src", "causa_backend_cuda.c"),
    os.path.join("src", "causa_memory.c"),
    os.path.join("src", "causa_scm.c"),
    os.path.join("src", "causa_io.c"),
]

PY_SOURCES = CORE_SOURCES + [os.path.join("src", "causa_py.c")]

causa_ext = Extension(
    "causa",
    sources=PY_SOURCES,
    include_dirs=[os.path.join(HERE, "include")],
    depends=[os.path.join(HERE, "include", h)
             for h in os.listdir(os.path.join(HERE, "include"))
             if h.endswith(".h")],
    extra_compile_args=[],
    define_macros=[],
)

if sys.platform == "win32":
    causa_ext.extra_compile_args = ["/O2", "/std:c11", "/D_CRT_SECURE_NO_WARNINGS"]
else:
    causa_ext.extra_compile_args = ["-O2", "-std=c11"]

setup(
    name="causa",
    version="0.1.0",
    description="Tensor computation with differentiable memory and causal reasoning",
    ext_modules=[causa_ext],
    python_requires=">=3.8",
)
