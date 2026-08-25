import os

from setuptools import Extension, setup


compile_args = ["/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
# This extension contains both C++ bindings and upstream Zstd C sources.
# Clang rejects a C++ language-standard flag while compiling the latter.
huff0_compile_args = compile_args if os.name == "nt" else ["-O3", "-pthread"]
huff0_link_args = [] if os.name == "nt" else ["-pthread"]


setup(
    ext_modules=[
        Extension(
            "neuralzip._huff0_cpp",
            sources=[
                "neuralzip/_huff0_cpp.cpp",
                "third_party/zstd/lib/common/debug.c",
                "third_party/zstd/lib/common/entropy_common.c",
                "third_party/zstd/lib/common/error_private.c",
                "third_party/zstd/lib/common/fse_decompress.c",
                "third_party/zstd/lib/compress/fse_compress.c",
                "third_party/zstd/lib/compress/hist.c",
                "third_party/zstd/lib/compress/huf_compress.c",
                "third_party/zstd/lib/decompress/huf_decompress.c",
            ],
            include_dirs=[
                "third_party/zstd/lib/common",
                "third_party/zstd/lib/compress",
                "third_party/zstd/lib/decompress",
            ],
            define_macros=[
                ("ZSTD_DISABLE_ASM", "1"),
                ("HUF_DISABLE_FAST_DECODE", "0"),
            ],
            language="c++",
            extra_compile_args=huff0_compile_args,
            extra_link_args=huff0_link_args,
        ),
    ]
)
