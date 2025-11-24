# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

ext_modules = [
    Extension(
        "pfun_cma_engine",
        sources=["pfun_cma_engine.c"],   # adjust path if you put file in subdir
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="pfun-cma-engine",
    version="0.0.1",
    description="C engine for pfun CMA model",
    ext_modules=ext_modules,
)

