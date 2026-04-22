#!/usr/bin/env python3
# Reference worker for ObExternalGpuKmeansAlgo (ob_vector_kmeans_ctx.cpp).
#
# Optional native GPU k-means++ (recommended): pip install -e test/q_kmeanspp → import q_kmeanspp.
#   OB_USE_CUDA_KMEANS_PP=1 (default) — try q_kmeanspp / legacy .so when CUDA is available, else PyTorch.
#   OB_USE_CUDA_KMEANS_PP=0 — always use PyTorch _gpu_kmeans_plus_plus_torch.
#   OB_EXTERNAL_KMEANS_PP_CUDA_SO / Q_KMEANSPP_CUDA_SO — override shared library path.
#   OB_CUDA_KMEANS_PP_INT8 — int8 quantize distance path (default ON). Set 0/false/no/off for fp32 kmeans_pp_l2_cuda.
# Legacy: make -C tools/cuda_kmeans_pp still produces libkmeans_pp_l2.so from test/q_kmeanspp/csrc/*.cu.
#
# dist_algo MUST match oceanbase ObVectorIndexDistAlgorithm in ob_vector_index_util.h:
#   VIDA_L2 = 0  -> Euclidean (squared L2) k-means, flash: batch_kmeans_Euclid
#   VIDA_IP = 1  -> dot-product similarity k-means, flash: batch_kmeans_Dot
#   VIDA_COS = 2 -> cosine similarity k-means, flash: batch_kmeans_Cosine (unit vectors)

import os
import struct
import sys
import time
import traceback

# Native CUDA k-means++ (libkmeans_pp_l2.so); loaded lazily via ctypes
_KMEANS_PP_CUDA_SO = None  # None=unresolved, False=load failed, CDLL=loaded

MAGIC = 0x4D4B424F

# Profiling: monotonic ms. stderr often ends up only in observer logs (not in kmeans_*.log).
# Same lines are appended to $HOME/log/ob_external_kmeans_worker.log (override: OB_EXTERNAL_KMEANS_WORKER_LOG).
_PROF_T0 = 0.0
_PROF_LAST = 0.0
_PROF_FILE = None


def _ensure_prof_file():
    global _PROF_FILE
    if _PROF_FILE is not None:
        return
    path = os.environ.get("OB_EXTERNAL_KMEANS_WORKER_LOG")
    if not path:
        home = os.environ.get("HOME")
        if not home:
            return
        log_dir = os.path.join(home, "log")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            return
        path = os.path.join(log_dir, "ob_external_kmeans_worker.log")
    try:
        _PROF_FILE = open(path, "a", buffering=1)
        _PROF_FILE.write(
            "\n======== ob_external_kmeans_worker pid=%d time=%s ========\n"
            % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        _PROF_FILE.flush()
    except OSError:
        _PROF_FILE = None


def _prof_reset():
    global _PROF_T0, _PROF_LAST
    _ensure_prof_file()
    _PROF_T0 = time.perf_counter()
    _PROF_LAST = _PROF_T0


def _prof(msg):
    global _PROF_LAST
    now = time.perf_counter()
    dt_prev = (now - _PROF_LAST) * 1000.0
    dt_tot = (now - _PROF_T0) * 1000.0
    line = "[ob_external_kmeans] %s | +%.1fms (prev) %.1fms (total)\n" % (msg, dt_prev, dt_tot)
    sys.stderr.write(line)
    sys.stderr.flush()
    if _PROF_FILE is not None:
        try:
            _PROF_FILE.write("[%d] %s" % (int(time.time() * 1000), line))
            _PROF_FILE.flush()
        except OSError:
            pass
    _PROF_LAST = now


def _write_err_and_log(text):
    """Same destination as _prof lines: stderr + ob_external_kmeans_worker.log (if available)."""
    _ensure_prof_file()
    if not text.endswith("\n"):
        text = text + "\n"
    sys.stderr.write(text)
    sys.stderr.flush()
    if _PROF_FILE is not None:
        try:
            _PROF_FILE.write("[%d] %s" % (int(time.time() * 1000), text))
            _PROF_FILE.flush()
        except OSError:
            pass


VERSION = 1
FLAG_HAS_INIT = 1 << 0
# Must match OB_EXT_KMEANS_FLAG_WORKER_GPU_KMEANSPP in ob_vector_kmeans_ctx.cpp (OB defaults to worker k-means++; OB_EXTERNAL_GPU_KMEANSPP=0 uses OB CPU init)
FLAG_WORKER_GPU_KMEANSPP = 1 << 1

# ObVectorIndexDistAlgorithm (keep in sync with ob_vector_index_util.h)
VIDA_L2 = 0
VIDA_IP = 1
VIDA_COS = 2


def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("unexpected EOF")
        buf.extend(chunk)
    return bytes(buf)


def _lloyd_numpy_l2(x, k, max_iters, init):
    """Euclidean Lloyd; same objective as flash batch_kmeans_Euclid."""
    import numpy as np

    n, d = x.shape
    if n < k:
        raise ValueError("n_samples (%d) < k (%d)" % (n, k))
    if init is not None:
        centroids = init.astype(np.float32, copy=True)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=k, replace=False)
        centroids = x[idx].copy()
    for _ in range(max_iters):
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_c = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            c = int(mask.sum())
            if c > 0:
                new_c[j] = x[mask].mean(axis=0)
            else:
                new_c[j] = centroids[j]
        shift = float(np.linalg.norm(new_c - centroids, axis=1).max())
        centroids = new_c
        if shift < 1e-8:
            break
    return centroids


def _lloyd_numpy_dot(x, k, max_iters, init):
    """Maximize within-cluster dot similarity; aligns with flash batch_kmeans_Dot."""
    import numpy as np

    n, d = x.shape
    if n < k:
        raise ValueError("n_samples (%d) < k (%d)" % (n, k))
    if init is not None:
        centroids = init.astype(np.float32, copy=True)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=k, replace=False)
        centroids = x[idx].copy()
    for _ in range(max_iters):
        sims = np.dot(x, centroids.T)
        labels = sims.argmax(axis=1)
        new_c = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if int(mask.sum()) > 0:
                new_c[j] = x[mask].mean(axis=0)
            else:
                new_c[j] = centroids[j]
        shift = float(np.linalg.norm(new_c - centroids, axis=1).max())
        centroids = new_c
        if shift < 1e-8:
            break
    return centroids


def _lloyd_numpy_cosine(x, k, max_iters, init):
    """Spherical / cosine k-means on L2-normalized rows; aligns with flash batch_kmeans_Cosine."""
    import numpy as np

    eps = 1e-12
    n, d = x.shape
    if n < k:
        raise ValueError("n_samples (%d) < k (%d)" % (n, k))
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    if init is not None:
        cn = init.astype(np.float32, copy=True)
        cn = cn / np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), eps)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=k, replace=False)
        cn = xn[idx].copy()
    for _ in range(max_iters):
        sims = np.dot(xn, cn.T)
        labels = sims.argmax(axis=1)
        new_cn = np.zeros_like(cn)
        for j in range(k):
            mask = labels == j
            if int(mask.sum()) > 0:
                mu = xn[mask].mean(axis=0)
                nm = np.linalg.norm(mu)
                new_cn[j] = mu / max(nm, eps)
            else:
                new_cn[j] = cn[j]
        shift = float(np.linalg.norm(new_cn - cn, axis=1).max())
        cn = new_cn
        if shift < 1e-8:
            break
    return cn


def _numpy_kmeans(x, k, max_iters, init, dist_algo):
    if dist_algo == VIDA_L2:
        return _lloyd_numpy_l2(x, k, max_iters, init)
    if dist_algo == VIDA_IP:
        return _lloyd_numpy_dot(x, k, max_iters, init)
    if dist_algo == VIDA_COS:
        return _lloyd_numpy_cosine(x, k, max_iters, init)
    raise ValueError("unsupported dist_algo=%r (expected 0=L2, 1=IP, 2=COS)" % (dist_algo,))


def _kmeans_plus_plus_numpy(x, k, dist_algo):
    """Arthur–Vassilvitskii k-means++ on rows of x (L2 / squared distance); COS uses row-normalized x."""
    import numpy as np

    n, _d = x.shape
    xw = np.asarray(x, dtype=np.float32, copy=False)
    if dist_algo == VIDA_COS:
        eps = 1e-12
        xw = xw / np.maximum(np.linalg.norm(xw, axis=1, keepdims=True), eps)
    seed_s = os.environ.get("OB_KMEANSPP_SEED", "").strip()
    seed = int(seed_s) if seed_s.isdigit() else None
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centers = [xw[first]]
    min_d2 = np.sum((xw - xw[first]) ** 2, axis=1)
    for _ in range(k - 1):
        s = float(min_d2.sum())
        if s <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            probs = min_d2 / s
            idx = int(rng.choice(n, p=probs))
        centers.append(xw[idx])
        dist_new = np.sum((xw - xw[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, dist_new)
    return np.stack(centers, axis=0)


def _gpu_kmeans_plus_plus_torch(x, k, dist_algo, device):
    """k-means++ init on device; x is float32 numpy (n,d), possibly padded. Returns (k,d) float32 CPU.

    Uses incremental min squared distance (O(k*n*d) memory O(n*d)), not a full (n,m,d) diff tensor, to avoid OOM when k is large.

    Important: avoid ``.item()`` / implicit sync inside the (k-1) iteration loop — each sync stalls the GPU
    and makes wall time comparable to CPU k-means++ even though compute is on-device. Indices stay as CUDA tensors
    until the final ``.cpu().numpy()``.
    """
    import torch

    seed_s = os.environ.get("OB_KMEANSPP_SEED", "").strip()
    if seed_s.isdigit():
        s = int(seed_s)
        torch.manual_seed(s)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(s)
    n, _d = x.shape
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    if dist_algo == VIDA_COS:
        eps = 1e-12
        X = X / torch.clamp(torch.norm(X, dim=1, keepdim=True), min=eps)
    idxs = torch.empty(k, dtype=torch.long, device=device)
    idxs[0] = torch.randint(0, n, (1,), device=device).squeeze()
    min_d2 = (X - X[idxs[0]]).pow(2).sum(dim=1)
    uni = torch.full((n,), 1.0 / float(n), device=device, dtype=X.dtype)
    for i in range(1, k):
        s = min_d2.sum()
        probs = torch.where(s > 0, min_d2 / (s + 1e-30), uni)
        ni = torch.multinomial(probs, 1).squeeze(0)
        idxs[i] = ni
        dist_new = (X - X[ni]).pow(2).sum(dim=1)
        min_d2 = torch.minimum(min_d2, dist_new)
    return X[idxs].detach().float().cpu().numpy()


def _resolve_kmeans_pp_cuda_so_path():
    p = os.environ.get("OB_EXTERNAL_KMEANS_PP_CUDA_SO", "").strip()
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "cuda_kmeans_pp", "libkmeans_pp_l2.so")


def _kpp_seed_u64():
    s = os.environ.get("OB_KMEANSPP_SEED", "").strip()
    if s.isdigit():
        return int(s) & ((1 << 64) - 1)
    import random

    return random.getrandbits(64)


def _env_cuda_kmeans_pp_int8():
    """Use int8 distance path by default; set OB_CUDA_KMEANS_PP_INT8=0|false|no|off for fp32."""
    v = os.environ.get("OB_CUDA_KMEANS_PP_INT8", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def _gpu_kmeans_plus_plus_native_legacy(x, k, dist_algo):
    """ctypes: libkmeans_pp_l2.so under tools/cuda_kmeans_pp (same symbols as q_kmeanspp)."""
    import ctypes
    import numpy as np

    global _KMEANS_PP_CUDA_SO
    if _KMEANS_PP_CUDA_SO is False:
        return None
    if _KMEANS_PP_CUDA_SO is None:
        so_path = _resolve_kmeans_pp_cuda_so_path()
        try:
            _KMEANS_PP_CUDA_SO = ctypes.CDLL(so_path)
        except OSError as e:
            _prof("flash: kmeans++ CUDA .so not loaded (%s): %s" % (so_path, e))
            _KMEANS_PP_CUDA_SO = False
            return None
    want_int8 = _env_cuda_kmeans_pp_int8()
    sym = "kmeans_pp_l2_int8_cuda" if want_int8 else "kmeans_pp_l2_cuda"
    lib = _KMEANS_PP_CUDA_SO
    try:
        fn = getattr(lib, sym)
    except AttributeError:
        if want_int8:
            _prof("flash: kmeans++ %s not in .so, using kmeans_pp_l2_cuda" % sym)
            fn = lib.kmeans_pp_l2_cuda
            sym = "kmeans_pp_l2_cuda"
        else:
            raise
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
    out = np.empty((k, d), dtype=np.float32)
    err_sz = 512
    err = ctypes.create_string_buffer(err_sz)
    seed = _kpp_seed_u64()
    cosine = 1 if dist_algo == VIDA_COS else 0
    ret = fn(
        x.ctypes.data_as(ctypes.c_void_p),
        n,
        d,
        k,
        ctypes.c_uint64(seed),
        cosine,
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.cast(err, ctypes.c_void_p),
        ctypes.c_size_t(err_sz),
        None,
    )
    if ret != 0:
        msg = err.value.decode("utf-8", errors="replace").strip("\x00")
        _prof("flash: native kmeans++ failed ret=%d %s" % (ret, msg))
        return None
    _prof("flash: native k-means++ legacy symbol=%s" % sym)
    return out


def _gpu_kmeans_plus_plus_native(x, k, dist_algo):
    """Prefer package q_kmeanspp (pip install -e test/q_kmeanspp); else ctypes legacy .so.

    Returns float32 (k, d) numpy on success, or None to signal fallback to PyTorch.
    """
    try:
        from q_kmeanspp import kmeans_pp_l2
    except ImportError:
        return _gpu_kmeans_plus_plus_native_legacy(x, k, dist_algo)
    try:
        out = kmeans_pp_l2(
            x,
            k,
            seed=_kpp_seed_u64(),
            cosine=(dist_algo == VIDA_COS),
            int8=_env_cuda_kmeans_pp_int8(),
        )
    except ImportError as e:
        _prof("flash: q_kmeanspp .so missing (%s), trying legacy ctypes" % (e,))
        return _gpu_kmeans_plus_plus_native_legacy(x, k, dist_algo)
    except RuntimeError as e:
        _prof("flash: q_kmeanspp failed: %s" % (e,))
        return None
    _prof(
        "flash: k-means++ path=q_kmeanspp (int8=%s)"
        % ("yes" if _env_cuda_kmeans_pp_int8() else "no",)
    )
    return out


def _ceil_pow2(n):
    """Smallest power of 2 >= n (n >= 1). Triton tl.arange(0, D) in flash-kmeans needs Po2 D."""
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _try_flash_kmeans(x, k, max_iters, init, dist_algo, worker_kpp=False):
    import numpy as np

    _, d_orig = x.shape
    d_pad = _ceil_pow2(d_orig)
    if d_pad > d_orig:
        _prof("flash: pad dim %d -> %d (Triton requires power-of-2 D)" % (d_orig, d_pad))
        x = np.pad(x, ((0, 0), (0, d_pad - d_orig)), mode="constant", constant_values=0)
        if init is not None:
            init = np.pad(init, ((0, 0), (0, d_pad - d_orig)), mode="constant", constant_values=0)

    _prof("flash: import torch + flash_kmeans (cold start may be slow)")
    import torch
    from flash_kmeans import batch_kmeans_Cosine, batch_kmeans_Dot, batch_kmeans_Euclid

    _prof("flash: imports done")
    cuda_ok = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda_ok else "cpu")
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _prof(
        "flash: CUDA_VISIBLE_DEVICES=%r device=%s cuda_available=%s"
        % (cv if cv else "(unset)", dev, cuda_ok)
    )
    if cuda_ok:
        try:
            free_b, total_b = torch.cuda.mem_get_info(dev)
            name = torch.cuda.get_device_name(dev)
            _prof(
                "flash: gpu mem free=%.2fGiB total=%.2fGiB name=%s"
                % (free_b / (1024.0**3), total_b / (1024.0**3), name)
            )
        except Exception as e:
            _prof("flash: gpu mem probe failed: %s" % (e,))
        try:
            torch.cuda.synchronize(dev)
        except Exception:
            pass
    if worker_kpp and init is None:
        _prof("flash: worker GPU k-means++ init (k=%d dist_algo=%d)" % (k, dist_algo))
        t0 = time.perf_counter()
        init = None
        use_native = os.environ.get("OB_USE_CUDA_KMEANS_PP", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if use_native and cuda_ok:
            init = _gpu_kmeans_plus_plus_native(x, k, dist_algo)
            if init is not None:
                _prof("flash: k-means++ native ok (q_kmeanspp or legacy .so; see path= lines above)")
        if init is None:
            init = _gpu_kmeans_plus_plus_torch(x, k, dist_algo, dev)
            _prof("flash: k-means++ path=pytorch")
        _prof("flash: k-means++ init done in %.1fms" % ((time.perf_counter() - t0) * 1000.0,))
    x_t = torch.as_tensor(x, dtype=torch.float16, device=dev).unsqueeze(0)
    ic = None
    if init is not None:
        ic = torch.as_tensor(init, dtype=torch.float16, device=dev).unsqueeze(0)
    _prof("flash: tensors on device (H2D done if CUDA)")
    if dist_algo == VIDA_L2:
        fn = batch_kmeans_Euclid
    elif dist_algo == VIDA_IP:
        fn = batch_kmeans_Dot
    elif dist_algo == VIDA_COS:
        fn = batch_kmeans_Cosine
    else:
        raise ValueError("unsupported dist_algo=%r" % (dist_algo,))
    _prof("flash: batch_kmeans start k=%d max_iters=%d" % (k, max_iters))
    _, centroids, _ = fn(
        x_t, k, max_iters=max_iters, tol=1e-8, init_centroids=ic, verbose=False
    )
    _prof("flash: batch_kmeans done")
    out = centroids[0].float().detach().cpu().numpy()
    if d_pad > d_orig:
        out = np.ascontiguousarray(out[:, :d_orig])
    _prof("flash: centroids D2H + numpy done (trimmed to dim=%d)" % d_orig)
    return out


def main():
    import numpy as np

    if len(sys.argv) != 3:
        print("usage: ob_external_kmeans_worker.py <input.bin> <output.bin>", file=sys.stderr)
        return 2
    in_path, out_path = sys.argv[1], sys.argv[2]
    _prof_reset()
    _prof("worker start in=%s out=%s wall=%s" % (in_path, out_path, time.strftime("%Y-%m-%d %H:%M:%S")))

    with open(in_path, "rb") as fin:
        fmt = "<IIqqqiiI"
        hdr_bytes = _read_exact(fin, struct.calcsize(fmt))
        magic, ver, n_samples, dim, k, max_iters, dist_algo, flags = struct.unpack(fmt, hdr_bytes)
        _prof("read header ok n=%d dim=%d k=%d max_iters=%d dist_algo=%d flags=%d" % (n_samples, dim, k, max_iters, dist_algo, flags))
        if magic != MAGIC or ver != VERSION:
            print("bad input magic/version", magic, ver, file=sys.stderr)
            return 1
        if dist_algo not in (VIDA_L2, VIDA_IP, VIDA_COS):
            print("unsupported dist_algo=%d (need 0=L2, 1=IP, 2=COS)" % dist_algo, file=sys.stderr)
            return 1
        worker_kpp = bool(flags & FLAG_WORKER_GPU_KMEANSPP)
        sample_bytes = int(n_samples * dim * 4)
        raw = _read_exact(fin, sample_bytes)
        _prof("read sample bytes=%d" % sample_bytes)
        x = np.frombuffer(raw, dtype=np.float32).reshape(int(n_samples), int(dim))
        _prof("numpy reshape x (%d,%d)" % (int(n_samples), int(dim)))
        init = None
        if worker_kpp:
            _prof("flags: WORKER_GPU_KMEANSPP (init from k-means++ in worker)")
        elif flags & FLAG_HAS_INIT:
            raw_i = _read_exact(fin, int(k * dim * 4))
            init = np.frombuffer(raw_i, dtype=np.float32).reshape(int(k), int(dim))
            _prof("read init centers (%d,%d)" % (int(k), int(dim)))

    _prof("input file closed")
    use_flash = os.environ.get("USE_FLASH_KMEANS", "").strip().lower() in ("1", "true", "yes")
    k, max_iters = int(k), int(max_iters)
    _prof("USE_FLASH_KMEANS=%r -> use_flash=%s" % (os.environ.get("USE_FLASH_KMEANS", ""), use_flash))
    if use_flash:
        try:
            centroids = _try_flash_kmeans(x, k, max_iters, init, dist_algo, worker_kpp=worker_kpp)
        except Exception as e:
            tb = traceback.format_exc()
            _write_err_and_log("flash-kmeans failed, fallback numpy: %r\n%s" % (e, tb))
            _prof("numpy fallback after flash error")
            if worker_kpp and init is None:
                init = _kmeans_plus_plus_numpy(x, k, dist_algo)
            centroids = _numpy_kmeans(x, k, max_iters, init, dist_algo)
            _prof("numpy kmeans done")
    else:
        _prof("numpy kmeans start (no flash)")
        if worker_kpp and init is None:
            init = _kmeans_plus_plus_numpy(x, k, dist_algo)
        centroids = _numpy_kmeans(x, k, max_iters, init, dist_algo)
        _prof("numpy kmeans done")

    with open(out_path, "wb") as fout:
        fout.write(struct.pack("<IIqq", MAGIC, VERSION, k, int(dim)))
        fout.write(np.asarray(centroids, dtype=np.float32).tobytes())
    _prof("write output ok path=%s" % out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        _write_err_and_log("ob_external_kmeans_worker failed: %r\n%s" % (exc, traceback.format_exc()))
        sys.exit(1)
