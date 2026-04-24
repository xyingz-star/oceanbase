"""Build libq_kmeanspp_cuda.so into package directory (no PyTorch required; needs nvcc)."""

import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "csrc", "kmeans_pp_l2.cu")
SO_NAME = "libq_kmeanspp_cuda.so"


def _run_nvcc(out_so: str) -> None:
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        CU,
        "-o",
        out_so,
        "-lcudart",
    ]
    extra = os.environ.get("CUDA_ARCH", "").strip()
    if extra:
        cmd.extend(extra.split())
    print("q_kmeanspp: " + " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)


class build_py(build_py_orig):
    def run(self) -> None:
        pkg_dir = os.path.join(HERE, "q_kmeanspp")
        os.makedirs(pkg_dir, exist_ok=True)
        out_so = os.path.join(pkg_dir, SO_NAME)
        _run_nvcc(out_so)
        super().run()


setup(
    name="q-kmeanspp",
    version="0.1.0",
    description="CUDA k-means++ (fp32 / int8) for numpy",
    packages=find_packages(),
    package_data={"q_kmeanspp": [SO_NAME]},
    include_package_data=True,
    cmdclass={"build_py": build_py},
    python_requires=">=3.6",
    zip_safe=False,
)
