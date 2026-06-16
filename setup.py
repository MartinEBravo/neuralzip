import os

from setuptools import Extension, setup


compile_args = ["/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]


setup(
    ext_modules=[
        Extension(
            "neuralzip._huffman_cpp",
            sources=["neuralzip/_huffman_cpp.cpp"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]
)
