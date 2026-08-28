import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "13.0")

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

compile_args = (
    ["/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17", "-pthread"]
)
huff0_link_args = [] if os.name == "nt" else ["-pthread"]
if sys.platform == "linux":
    huff0_link_args.extend(["-ldl", "-lm"])


class GoBuildExt(build_ext):
    """Build klauspost/huff0 once and link it into the Python extension."""

    def build_extensions(self):
        go = shutil.which("go")
        if go is None:
            raise RuntimeError(
                "Building NeuralZip from source requires Go 1.24 or newer."
            )

        root = Path(__file__).resolve().parent
        source = root / "native" / "gohuff0"
        destination = Path(self.build_temp) / "gohuff0"
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / "libneuralzip_huff0.a"
        environment = os.environ.copy()
        environment["CGO_ENABLED"] = "1"
        if sys.platform == "darwin":
            target = environment["MACOSX_DEPLOYMENT_TARGET"]
            flag = f"-mmacosx-version-min={target}"
            for name in ("CGO_CFLAGS", "CGO_LDFLAGS"):
                environment[name] = f"{environment.get(name, '')} {flag}".strip()
        subprocess.run(
            [
                go,
                "build",
                "-trimpath",
                "-mod=readonly",
                "-buildmode=c-archive",
                "-o",
                str(archive.resolve()),
                ".",
            ],
            cwd=source,
            env=environment,
            check=True,
        )
        go_inputs = [
            Path("native/gohuff0/go.mod"),
            Path("native/gohuff0/go.sum"),
            *Path("native/gohuff0").rglob("*.go"),
        ]
        for extension in self.extensions:
            extension.extra_objects = [
                *getattr(extension, "extra_objects", ()),
                str(archive),
            ]
            extension.depends = [
                *getattr(extension, "depends", ()),
                *(str(path) for path in go_inputs),
                str(archive),
            ]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "neuralzip._huff0_cpp",
            sources=["neuralzip/_huff0_cpp.cpp"],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=huff0_link_args,
        ),
    ],
    cmdclass={"build_ext": GoBuildExt},
)
