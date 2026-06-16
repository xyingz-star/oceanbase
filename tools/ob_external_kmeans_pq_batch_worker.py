#!/usr/bin/env python3
"""Batch PQ codebook k-means for ObMultiKmeansExecutor (Phase 1 / A2 micro-batch).

Input bin: ObExtKmeansPqBatchInHeader + N * full_dim float32 samples (one transfer).
Output bin: ObExtKmeansPqBatchOutHeader + m * k * sub_dim float32 centroids.

Must stay in sync with ob_vector_kmeans_ctx.cpp (magic OBKB, struct layout).
"""
from __future__ import annotations

import os
import struct
import sys
import time

# Reuse single-job helpers (k-means++, flash path, profiling).
from ob_external_kmeans_worker import (  # noqa: E402
    FLAG_WORKER_GPU_KMEANSPP,
    VIDA_L2,
    _gpu_kmeans_plus_plus_native,
    _gpu_kmeans_plus_plus_torch,
    _prof,
    _prof_reset,
    _read_exact,
    _write_err_and_log,
)

MAGIC_PQ_BATCH = 0x424B424F  # 'OBKB' little-endian
VERSION_PQ_BATCH = 1

# struct ObExtKmeansPqBatchInHeader (packed, LE)
IN_HDR_FMT = "<IIqqqqqiiI"
IN_HDR_SIZE = struct.calcsize(IN_HDR_FMT)

OUT_HDR_FMT = "<IIqqq"
OUT_HDR_SIZE = struct.calcsize(OUT_HDR_FMT)


def _env_micro_batch(default: int = 16) -> int:
    v = os.environ.get("OB_EXTERNAL_KMEANS_PQ_MICRO_BATCH", "").strip()
    if v.isdigit():
        n = int(v)
        if 1 <= n <= 256:
            return n
    return default


def _env_gpu_kpp() -> bool:
    v = os.environ.get("OB_EXTERNAL_GPU_KMEANSPP", "").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    return True


def _run_pq_micro_batches(
    x_full,
    m: int,
    sub_dim: int,
    k: int,
    max_iters: int,
    micro_batch: int,
    gpu_kpp: bool,
):
    import numpy as np
    import torch
    from flash_kmeans import batch_kmeans_Euclid

    n, full_dim = x_full.shape
    if full_dim != m * sub_dim:
        raise ValueError("full_dim %d != m*sub_dim %d" % (full_dim, m * sub_dim))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _prof("pq_batch: import torch + flash_kmeans done device=%s" % dev)
    x_t = torch.as_tensor(x_full, dtype=torch.float32, device=dev)
    _prof("pq_batch: H2D full samples (%d,%d)" % (n, full_dim))

    out = np.zeros((m, k, sub_dim), dtype=np.float32)
    use_native_kpp = os.environ.get("OB_USE_CUDA_KMEANS_PP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    rounds = (m + micro_batch - 1) // micro_batch
    for r, s0 in enumerate(range(0, m, micro_batch)):
        s1 = min(s0 + micro_batch, m)
        b = s1 - s0
        t0 = time.perf_counter()
        chunks = [x_t[:, (s0 + i) * sub_dim : (s0 + i + 1) * sub_dim] for i in range(b)]
        x_batch = torch.stack(chunks, dim=0)
        ic = None
        if gpu_kpp:
            ic_rows = []
            for i in range(b):
                xi = x_batch[i].detach().cpu().numpy()
                init = None
                if use_native_kpp and str(dev).startswith("cuda"):
                    init = _gpu_kmeans_plus_plus_native(xi, k, VIDA_L2)
                if init is None:
                    init = _gpu_kmeans_plus_plus_torch(xi, k, VIDA_L2, dev)
                ic_rows.append(init)
            ic = torch.as_tensor(np.stack(ic_rows, axis=0), dtype=torch.float32, device=dev)
        _, centroids, n_iters = batch_kmeans_Euclid(
            x_batch,
            k,
            max_iters=max_iters,
            tol=1e-8,
            init_centroids=ic,
            verbose=False,
        )
        centroids_np = centroids.detach().float().cpu().numpy()
        for i in range(b):
            out[s0 + i] = centroids_np[i]
        _prof(
            "pq_batch: round %d/%d subspaces [%d,%d) B=%d iters=%d wall=%.1fms"
            % (r + 1, rounds, s0, s1, b, int(n_iters), (time.perf_counter() - t0) * 1000.0)
        )
    return out


def main():
    import numpy as np

    if len(sys.argv) != 3:
        print(
            "usage: ob_external_kmeans_pq_batch_worker.py <input.bin> <output.bin>",
            file=sys.stderr,
        )
        return 2
    in_path, out_path = sys.argv[1], sys.argv[2]
    _prof_reset()
    _prof("pq_batch worker start in=%s out=%s" % (in_path, out_path))

    with open(in_path, "rb") as fin:
        hdr_bytes = _read_exact(fin, IN_HDR_SIZE)
        (
            magic,
            ver,
            n_samples,
            full_dim,
            m,
            sub_dim,
            k,
            max_iters,
            micro_batch,
            flags,
        ) = struct.unpack(IN_HDR_FMT, hdr_bytes)
        _prof(
            "pq_batch header n=%d full_dim=%d m=%d sub_dim=%d k=%d max_iters=%d micro_batch=%d flags=%d"
            % (n_samples, full_dim, m, sub_dim, k, max_iters, micro_batch, flags)
        )
        if magic != MAGIC_PQ_BATCH or ver != VERSION_PQ_BATCH:
            print("bad pq batch magic/version", magic, ver, file=sys.stderr)
            return 1
        if n_samples <= 0 or full_dim <= 0 or m <= 0 or sub_dim <= 0 or k <= 0:
            print("invalid pq batch dimensions", file=sys.stderr)
            return 1
        if full_dim != m * sub_dim:
            print("full_dim != m*sub_dim", full_dim, m, sub_dim, file=sys.stderr)
            return 1
        if micro_batch <= 0:
            micro_batch = _env_micro_batch(16)
        sample_bytes = int(n_samples * full_dim * 4)
        raw = _read_exact(fin, sample_bytes)
        x = np.frombuffer(raw, dtype=np.float32).reshape(int(n_samples), int(full_dim))
        _prof("pq_batch read samples bytes=%d" % sample_bytes)

    gpu_kpp = bool(flags & FLAG_WORKER_GPU_KMEANSPP) or _env_gpu_kpp()
    use_flash = os.environ.get("USE_FLASH_KMEANS", "").strip().lower() in ("1", "true", "yes")
    if not use_flash:
        _write_err_and_log("pq_batch requires USE_FLASH_KMEANS=1")
        return 1

    try:
        centroids = _run_pq_micro_batches(
            x,
            int(m),
            int(sub_dim),
            int(k),
            int(max_iters),
            int(micro_batch),
            gpu_kpp,
        )
    except Exception as e:
        import traceback

        _write_err_and_log("pq_batch flash-kmeans failed: %r\n%s" % (e, traceback.format_exc()))
        return 1

    with open(out_path, "wb") as fout:
        fout.write(struct.pack(OUT_HDR_FMT, MAGIC_PQ_BATCH, VERSION_PQ_BATCH, m, k, sub_dim))
        fout.write(np.asarray(centroids, dtype=np.float32).tobytes())
    _prof("pq_batch write output ok m=%d k=%d sub_dim=%d" % (m, k, sub_dim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
