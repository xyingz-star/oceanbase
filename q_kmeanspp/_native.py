"""CUDA k-means++ (fp32 / int8): load libq_kmeanspp_cuda.so and expose numpy API."""

import ctypes
import os
from typing import Optional

import numpy as np

__version__ = "0.1.0"

_LIB: Optional[ctypes.CDLL] = None


def _find_so_path() -> str:
    """Resolve shared library: Q_KMEANSPP_CUDA_SO / OB_EXTERNAL_KMEANS_PP_CUDA_SO, then package lib."""
    for key in ("Q_KMEANSPP_CUDA_SO", "OB_EXTERNAL_KMEANS_PP_CUDA_SO"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(pkg_dir, "libq_kmeanspp_cuda.so")
    if os.path.isfile(p):
        return p
    raise ImportError(
        "q_kmeanspp: libq_kmeanspp_cuda.so not found (pip install -e in test/q_kmeanspp, or set Q_KMEANSPP_CUDA_SO)."
    )


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_find_so_path())
    return _LIB


def _call_native(
    fn_name: str,
    x: np.ndarray,
    k: int,
    seed: int,
    cosine: bool,
) -> np.ndarray:
    fn = getattr(_lib(), fn_name)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int

    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = int(x.shape[0]), int(x.shape[1])
    if k <= 0 or k > n:
        raise ValueError("need 0 < k <= n")
    out = np.empty((k, d), dtype=np.float32)
    err_sz = 512
    err = ctypes.create_string_buffer(err_sz)
    ret = fn(
        x.ctypes.data_as(ctypes.c_void_p),
        n,
        d,
        k,
        ctypes.c_uint64(seed & ((1 << 64) - 1)),
        1 if cosine else 0,
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.cast(err, ctypes.c_void_p),
        ctypes.c_size_t(err_sz),
        None,
    )
    if ret != 0:
        msg = err.value.decode("utf-8", errors="replace").strip("\x00")
        raise RuntimeError("k-means++ %s failed ret=%d: %s" % (fn_name, ret, msg))
    return out


def kmeans_pp_l2(
    x: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    cosine: bool = False,
    int8: bool = True,
) -> np.ndarray:
    """GPU k-means++ initialization. x: (n, d) float32; returns (k, d) float32 center rows (from original X).

    int8=True (default): symmetric global int8 quantization for distances only (see csrc/kmeans_pp_l2.cu).
    int8=False: full fp32 squared-L2 distances (kmeans_pp_l2_cuda).
    """
    fn = "kmeans_pp_l2_int8_cuda" if int8 else "kmeans_pp_l2_cuda"
    return _call_native(fn, x, k, seed, cosine)


def kmeans_pp_l2_int8(x: np.ndarray, k: int, *, seed: int = 42, cosine: bool = False) -> np.ndarray:
    """Same as kmeans_pp_l2(..., int8=True)."""
    return kmeans_pp_l2(x, k, seed=seed, cosine=cosine, int8=True)
