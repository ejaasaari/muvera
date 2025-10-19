import os
import sys
import subprocess
import platform
from pathlib import Path

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


ext_modules = [
    Extension(
        "fde",
        ["python_bindings.cc", "fixed_dimensional_encoding.cc"],
        include_dirs=["."],
        language="c++",
    )
]


def has_flag(compiler, flagname):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False

    return True


def clang_info(compiler):
    try:
        cmd = getattr(compiler, "compiler", None) or getattr(compiler, "compiler_so", None) or []
        if not cmd:
            return False
        out = subprocess.check_output(cmd + ["--version"], stderr=subprocess.STDOUT)
        txt = out.decode("utf-8", errors="ignore").lower()
        is_clang = "clang" in txt and "gcc" not in txt
        is_apple_clang = "apple clang" in txt and "gcc" not in txt
        return is_clang, is_apple_clang
    except Exception:
        return False, False


def get_architecture():
    machine = platform.machine().lower()

    if machine in ["x86_64", "amd64", "i386", "i686"]:
        return "x86-64" if "64" in machine else "x86"
    elif machine in ["arm64", "aarch64"]:
        return "arm64"
    elif "arm" in machine:
        return "arm"
    else:
        return machine


def native_flags(compiler):
    c = compiler.lower()
    a = get_architecture()

    mn = ["-march=native", "-mtune=native"]
    mc = ["-mcpu=native"]

    flags = {
        "gcc": {"x86-64": mn, "arm64": mc},
        "clang": {"x86-64": mn, "arm64": mc},
    }
    defaults = {
        "gcc": mn,
        "clang": mc,
    }

    if c not in flags:
        raise ValueError(f"Unknown compiler '{compiler}'")

    return flags[c].get(a, defaults[c])


def find_eigen():
    """Find Eigen installation."""
    bundled_eigen = Path(__file__).parent.absolute()
    if (bundled_eigen / "Eigen").exists():
        return str(bundled_eigen)

    eigen_paths = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]

    for path in eigen_paths:
        if os.path.exists(path):
            return path

    try:
        result = subprocess.run(
            ["pkg-config", "--cflags-only-I", "eigen3"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().replace("-I", "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Eigen not found. Please install Eigen:\n"
            "  macOS: brew install eigen\n"
            "  Ubuntu/Debian: sudo apt-get install libeigen3-dev\n"
            "  Fedora/RHEL: sudo dnf install eigen3-devel"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options.

    Assume that C++17 is available.
    """

    c_opts = {
        "unix": [
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-flax-vector-conversions",
            "-DNDEBUG",
            "-Wno-unknown-pragmas",
            "-Wno-unknown-warning-option",
            "-Wno-unused-function",
            "-Wl,--no-undefined",
        ],
        "msvc": ["/std:c++17", "/O2", "/EHsc", "/DNDEBUG", "/wd4244"],
    }
    link_opts = {
        "unix": ["-pthread"],
        "msvc": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.link_opts.get(ct, [])

        if ct == "unix":
            opts.extend(
                [
                    "-fassociative-math",
                    "-fno-signaling-nans",
                    "-fno-trapping-math",
                    "-fno-signed-zeros",
                    "-freciprocal-math",
                    "-fno-math-errno",
                ]
            )

            is_clang, is_apple_clang = clang_info(self.compiler)
            native = native_flags("clang" if is_clang else "gcc")

            for flag in native + ["-fvisibility=hidden"]:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

            if sys.platform == "darwin":
                opts.append("-mmacosx-version-min=11.0")
                link_opts.append("-mmacosx-version-min=11.0")

                if is_clang and has_flag(self.compiler, "-stdlib=libc++"):
                    opts.append("-stdlib=libc++")
                    link_opts.append("-stdlib=libc++")

        eigen_include = find_eigen()
        print(f"Using Eigen from: {eigen_include}")

        import pybind11

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)
            ext.include_dirs.extend(
                [
                    eigen_include,
                    pybind11.get_include(),
                ]
            )

        build_ext.build_extensions(self)


setup(
    name="fde",
    version="0.1.0",
    author="FDE Library",
    description="Fixed Dimensional Encoding library for point clouds",
    long_description="Fixed Dimensional Encoding (FDE) implementation for point clouds",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="point cloud, fixed dimensional encoding, chamfer similarity",
    zip_safe=False,
    ext_modules=ext_modules,
    install_requires=["numpy", "pybind11>=2.6.0"],
    cmdclass={"build_ext": BuildExt},
)
