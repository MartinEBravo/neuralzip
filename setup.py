import hashlib
import os
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

if sys.platform == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import SetupError

# Official Zstd revision pinned for reproducible, byte-compatible builds. The
# archive is not trusted until its SHA-256 matches this pin.
ZSTD_REVISION = "5233c58e6ca0b1c4c6b353ad79649191ed195bdc"
ZSTD_ROOT = f"zstd-{ZSTD_REVISION}"
ZSTD_URL = f"https://codeload.github.com/facebook/zstd/tar.gz/{ZSTD_REVISION}"
ZSTD_SHA256 = "4c8e7f2ab048339cf523e0ad62c119d4e4d49f4abf94028dffd5897706b1b68c"
ZSTD_ARCHIVE_ENV = "NEURALZIP_ZSTD_ARCHIVE"
ZSTD_FILES = (
    "lib/common/entropy_common.c",
    "lib/common/error_private.c",
    "lib/common/fse_decompress.c",
    "lib/compress/fse_compress.c",
    "lib/compress/hist.c",
    "lib/compress/huf_compress.c",
    "lib/decompress/huf_decompress.c",
    "lib/zstd.h",
    "lib/zstd_errors.h",
    "lib/common/bitstream.h",
    "lib/common/bits.h",
    "lib/common/compiler.h",
    "lib/common/cpu.h",
    "lib/common/debug.h",
    "lib/common/error_private.h",
    "lib/common/fse.h",
    "lib/common/huf.h",
    "lib/common/mem.h",
    "lib/common/portability_macros.h",
    "lib/common/xxhash.h",
    "lib/common/zstd_deps.h",
    "lib/common/zstd_internal.h",
    "lib/common/zstd_trace.h",
    "lib/compress/hist.h",
)
ZSTD_SOURCES = tuple(path for path in ZSTD_FILES if path.endswith(".c"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_zstd_archive(dependency_dir):
    override = os.environ.get(ZSTD_ARCHIVE_ENV)
    archive = (
        Path(override).resolve() if override else dependency_dir / f"{ZSTD_ROOT}.tar.gz"
    )
    if override and not archive.is_file():
        raise SetupError(f"{ZSTD_ARCHIVE_ENV} does not name a file: {archive}")

    if not archive.is_file():
        dependency_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (
                urlopen(ZSTD_URL, timeout=60) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        except OSError as error:
            archive.unlink(missing_ok=True)
            raise SetupError(
                f"Could not download pinned Zstd sources: {error}. Set "
                f"{ZSTD_ARCHIVE_ENV} to the archive path for an offline build."
            ) from error

    actual = sha256(archive)
    if actual != ZSTD_SHA256:
        if not override:
            archive.unlink(missing_ok=True)
        raise SetupError(
            f"Zstd archive SHA-256 mismatch: expected {ZSTD_SHA256}, got {actual}."
        )
    return archive


def extract_zstd(archive, dependency_dir):
    source_root = dependency_dir / ZSTD_ROOT
    complete = source_root / ".complete"
    if complete.is_file() and all(
        (source_root / relative).is_file() for relative in ZSTD_FILES
    ):
        return source_root
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True)

    try:
        with tarfile.open(archive, "r:gz") as tar:
            for relative in ZSTD_FILES:
                member = tar.getmember(f"{ZSTD_ROOT}/{relative}")
                if not member.isfile():
                    raise SetupError(f"Refusing non-file Zstd member: {relative}")
                source = tar.extractfile(member)
                if source is None:
                    raise SetupError(f"Could not read Zstd member: {relative}")
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        complete.touch()
    except (KeyError, OSError, tarfile.TarError) as error:
        raise SetupError(f"Could not extract pinned Zstd sources: {error}") from error
    return source_root


class ZstdBuildExt(build_ext):
    def build_extensions(self):
        dependency_dir = Path(self.build_temp).resolve() / "_deps"
        source_root = extract_zstd(get_zstd_archive(dependency_dir), dependency_dir)
        include_dirs = [
            source_root / "lib",
            source_root / "lib" / "common",
            source_root / "lib" / "compress",
            source_root / "lib" / "decompress",
        ]
        for extension in self.extensions:
            extension.sources.extend(
                str(source_root / relative) for relative in ZSTD_SOURCES
            )
            extension.include_dirs.extend(str(path) for path in include_dirs)
            extension.define_macros.append(("ZSTD_DISABLE_ASM", "1"))
        super().build_extensions()


# C and C++ sources share these flags, so Unix builds use the compiler's C++
# default instead of passing a C++-only -std flag to Zstd's C files.
compile_args = ["/std:c++17"] if os.name == "nt" else ["-O3", "-pthread"]
link_args = [] if os.name == "nt" else ["-pthread"]
setup(
    ext_modules=[
        Extension(
            "neuralzip._huff0_cpp",
            sources=["neuralzip/_huff0_cpp.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": ZstdBuildExt},
)
