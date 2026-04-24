"""q_kmeanspp — CUDA k-means++ (fp32 / int8), numpy API similar in spirit to flash_kmeans (import + one call).

Default: int8-quantized distances in kmeans_pp_l2; pass int8=False for full fp32.

Install (from repo):

  cd test/q_kmeanspp && pip install -e .

Or build .so only:

  make -C test/q_kmeanspp

Override library path:

  export Q_KMEANSPP_CUDA_SO=/path/to/libq_kmeanspp_cuda.so
"""

from ._native import __version__, kmeans_pp_l2, kmeans_pp_l2_int8

__all__ = ["kmeans_pp_l2", "kmeans_pp_l2_int8", "__version__"]
