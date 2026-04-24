from typing import Optional
import ctypes
import os
import torch
import triton
import triton.language as tl

# ===============================================================
# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# Inputs:
#   x           : (B, N, D)  float16 / float32
#   centroids   : (B, K, D)  same dtype as x
#   x_sq        : (B, N)     float32 – pre-computed ||x||^2 per point
# Output:
#   cluster_ids : (B, N)     int32   – nearest centroid index per point
# ===============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# Shared-memory budget (per-block opt-in limit) + clamp heuristic configs
# -----------------------------------------------------------------------------

_cudart_lib = None
_smem_optin_per_device: dict[int, int] = {}


def _get_cudart():
    """Resolve libcudart next to PyTorch; None if unavailable."""
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib if _cudart_lib is not False else None
    candidates = []
    try:
        torch_dir = os.path.dirname(torch.__file__)
        candidates.extend(
            [
                os.path.join(torch_dir, "lib", "libcudart.so.12"),
                os.path.join(torch_dir, "lib", "libcudart.so.11.0"),
            ]
        )
    except Exception:
        pass
    for p in candidates:
        if os.path.isfile(p):
            try:
                _cudart_lib = ctypes.CDLL(p)
                return _cudart_lib
            except OSError:
                pass
    for name in ("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"):
        try:
            _cudart_lib = ctypes.CDLL(name)
            return _cudart_lib
        except OSError:
            continue
    _cudart_lib = False
    return None


def _cuda_max_shared_memory_per_block_optin(device: torch.device) -> int:
    """Bytes; matches Triton 'Hardware limit' when opt-in extended smem applies."""
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    if idx in _smem_optin_per_device:
        return _smem_optin_per_device[idx]

    lib = _get_cudart()
    if lib is None:
        _smem_optin_per_device[idx] = 49152
        return 49152

    fn = lib.cudaDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int

    CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8

    val = ctypes.c_int()
    err = fn(ctypes.byref(val), CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, idx)
    if err != 0 or val.value <= 0:
        err2 = fn(ctypes.byref(val), CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, idx)
        if err2 != 0 or val.value <= 0:
            _smem_optin_per_device[idx] = 49152
            return 49152

    optin = int(val.value)
    _smem_optin_per_device[idx] = optin
    return optin


def _floor_pow2_at_least(n: int, least: int = 8) -> int:
    """Largest power of 2 <= n, but not below ``least`` (Triton tl.arange tile sizes)."""
    n = int(n)
    if n <= least:
        return least
    return max(least, 1 << (n.bit_length() - 1))


def _euclid_assign_smem_upper_bound(
    block_n: int, block_k: int, d: int, num_stages: int, num_warps: int
) -> int:
    """Conservative bytes upper bound vs Triton-reported metadata.shared for this kernel.

    Calibrated point: (BN=64,BK=64,D=1024,warp=8,stage=1) required 196608 == 3*BN*D bytes.
    """
    bn = max(1, block_n)
    bk = max(1, block_k)
    dd = max(1, d)
    core = max(3 * bn * dd, 2 * bk * dd)
    return int(core * max(1, num_stages) + 2048 * num_warps + 12288)


def _clamp_euclid_config_to_shared_budget(
    cfg: dict, d: int, device: torch.device
) -> dict:
    """Shrink BLOCK_N / BLOCK_K / stages / warps so the assign kernel fits per-block smem."""
    bn = max(8, int(cfg["BLOCK_N"]))
    bk = max(8, int(cfg["BLOCK_K"]))
    nw = int(cfg["num_warps"])
    ns = int(cfg["num_stages"])
    nw = 4 if nw <= 4 else 8
    ns = max(1, ns)

    limit = _cuda_max_shared_memory_per_block_optin(device)
    budget = max(16384, int(0.88 * limit))
    overhead = 2048 * nw + 12288
    head = budget - overhead
    if head > 0 and d > 0:
        cap_bn = max(8, head // (3 * d))
        cap_bk = max(8, head // (2 * d))
        bn = min(bn, cap_bn)
        bk = min(bk, cap_bk)

    guard = 0
    while _euclid_assign_smem_upper_bound(bn, bk, d, ns, nw) > budget and guard < 48:
        guard += 1
        if bk >= bn and bk > 8:
            bk = max(8, bk // 2)
        elif bn > 8:
            bn = max(8, bn // 2)
        elif bk > 8:
            bk = max(8, bk // 2)
        elif nw > 4:
            nw = 4
        elif ns > 1:
            ns = 1
        else:
            bn = max(8, bn // 2)
            bk = max(8, bk // 2)

    bn = _floor_pow2_at_least(bn)
    bk = _floor_pow2_at_least(bk)
    guard2 = 0
    while _euclid_assign_smem_upper_bound(bn, bk, d, ns, nw) > budget and guard2 < 16:
        guard2 += 1
        if bn > bk:
            bn = max(8, bn // 2)
        else:
            bk = max(8, bk // 2)
        bn = _floor_pow2_at_least(bn)
        bk = _floor_pow2_at_least(bk)

    return {"BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "num_stages": ns}


# -----------------------------------------------------------------------------
# Auto-tuning setup – explore various tile sizes / warp counts
# -----------------------------------------------------------------------------

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True

_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))

def _heuristic_euclid_config(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
):
    """Architecture-aware heuristic config selection without autotune.

    Keep one unified heuristic entry and diverge inside by GPU family:
    - H200: existing hand-tuned heuristic
    - A100: heuristic derived from A100 grid tuning results
    - others: conservative fallback to reduce OOR risk

    The returned config is then clamped to the current GPU's per-block shared-memory
    budget (cudaDevAttrMaxSharedMemoryPerBlockOptin) using a conservative smem model.
    """
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()

    if "H200" in gpu_name:
        # Keep the original H200 heuristic as-is.
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            num_warps = 4
            num_stages = 2
        else:
            # D <= 128
            if K >= 4096:
                block_k = 128
                if D >= 128:
                    num_warps = 8
                    num_stages = 2
                else:
                    num_warps = 4
                    num_stages = 4
            else:
                block_k = 64
                num_warps = 4
                num_stages = 1

        # D=64 with large K tends to prefer smaller BLOCK_N and deeper pipeline.
        if D <= 64 and K >= 4096:
            block_n = 64
            block_k = 128
            num_warps = 4
            num_stages = 4

        # Smaller N favors smaller BLOCK_N to reduce wasted work.
        if N < 65536:
            block_n = 64

        return _clamp_euclid_config_to_shared_budget(
            {
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            D,
            device,
        )

    if "H100" in gpu_name:
        # H100 tuned heuristic (more conservative on D=64 mid-K vs H200).
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            if K <= 1024:
                num_warps = 8
                num_stages = 1
            elif K <= 16384:
                num_warps = 4
                num_stages = 1
            else:
                num_warps = 8
                num_stages = 1
        else:
            # D <= 128
            if D <= 64:
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 16384:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 65536:
                    block_k = 128
                    num_warps = 4
                    num_stages = 4
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4
            else:
                # D == 128
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 1
                elif K <= 65536:
                    block_k = 128
                    num_warps = 8
                    num_stages = 2
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4

        if N < 65536:
            block_n = 64

        return _clamp_euclid_config_to_shared_budget(
            {
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            D,
            device,
        )

    if "A100" in gpu_name:
        # Robust default on A100 across tuned grid.
        block_n = 128
        block_k = 32
        num_warps = 4
        num_stages = 2

        if D == 128:
            # Small-N cases tend to prefer a larger K tile.
            if N <= 65536:
                block_k = 64
        elif D == 256:
            # D=256 benefits from deeper pipeline at larger K.
            if K >= 65536:
                block_k = 32
                num_stages = 4
            elif K >= 1024 and N <= 262144:
                block_k = 64
                num_stages = 4

        return _clamp_euclid_config_to_shared_budget(
            {
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            D,
            device,
        )

    # Conservative fallback for unknown architectures (prioritize avoiding OOR).
    return _clamp_euclid_config_to_shared_budget(
        {
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_warps": 4,
            "num_stages": 1,
        },
        D,
        device,
    )


@triton.jit
def _euclid_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)          # tile index along N dimension
    pid_b = tl.program_id(1)          # batch index
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D).to(tl.int64)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Pre-load x_sq for the tile  (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # load c_sq for the tile  (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # # Compute centroid squared norms (BLOCK_K,)
        # cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

_euclid_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K", "D"])(_euclid_assign_kernel)

@triton.jit
def _cosine_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)          # tile index along N dimension
    pid_b = tl.program_id(1)          # batch index
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D).to(tl.int64)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)  # less is worse 
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # Compute cosine distance (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], cross, 0.0)

        curr_max = tl.max(dist, axis=1)
        curr_idx = tl.argmax(dist, axis=1)

        update = curr_max > best_dist
        best_dist = tl.where(update, curr_max, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

# Cosine assign uses :func:`_direct_optimal_assign_config` + ``_cosine_assign_kernel`` only
# (no Triton autotuner — avoids long multi-config trial at runtime).

# ---------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------

def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    c_sq: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 – ||x||^2 per point (on CUDA)
        out       : (B, N)    int32   – (option) pre-allocated output tensor (on CUDA)
        c_sq      : (B, K)    float32 – (option) ||centroids||^2 per centroid (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    Extra:
        config        : {"BLOCK_N","BLOCK_K","num_warps","num_stages"} to force a config
        use_heuristic : use a fixed heuristic config instead of autotune
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(N, K, D, device=x.device)

    if selected_config is not None:
        _euclid_assign_kernel[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
            BLOCK_N=selected_config["BLOCK_N"],
            BLOCK_K=selected_config["BLOCK_K"],
            num_warps=selected_config["num_warps"],
            num_stages=selected_config["num_stages"],
        )
    else:
        _euclid_assign_kernel_autotuned[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
        )
    return out


# Minimum tile side for tl.dot / tensor-core path in ``_cosine_assign_kernel`` (Triton 3.x).
_COSINE_TRITON_DOT_MIN_TILE = 16


def _direct_optimal_assign_config(d: int, device: torch.device) -> Optional[dict]:
    """Pick a single Triton assign config without autotune trials.

    Enumerates power-of-2 ``BLOCK_N``/``BLOCK_K`` in [16, 128] and small
    ``num_stages``/``num_warps`` grids, keeps configs that satisfy
    ``_euclid_assign_smem_upper_bound <= per-block opt-in limit``, and chooses
    the one with the largest ``BLOCK_N * BLOCK_K`` (then prefers smaller
    ``num_stages``, ``num_warps``, then lower estimated smem). Deterministic for
    a given ``(d, device)``.

    Returns:
        dict with BLOCK_N, BLOCK_K, num_stages, num_warps, or None if no Triton
        kernel can run (caller should use PyTorch assign).
    """
    limit = _cuda_max_shared_memory_per_block_optin(device)
    slack = 2048
    budget = max(16384, limit - slack)
    mt = _COSINE_TRITON_DOT_MIN_TILE
    tiles: list[int] = []
    t = 128
    while t >= mt:
        tiles.append(t)
        t //= 2

    best: Optional[dict] = None
    best_key: Optional[tuple] = None

    for ns in (1, 2, 4):
        for nw in (4, 8):
            for bn in tiles:
                for bk in tiles:
                    need = _euclid_assign_smem_upper_bound(bn, bk, d, ns, nw)
                    if need > budget:
                        continue
                    key = (bn * bk, -ns, -nw, -need)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "num_stages": ns,
                            "num_warps": nw,
                        }
    return best


def _cosine_assign_pytorch_gpu(
    x: torch.Tensor, centroids: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Cosine/dot nearest-centroid ids via batched GEMM; avoids Triton smem / dot tile limits."""
    sim = torch.einsum("bnd,bkd->bnk", x.float(), centroids.float())
    out.copy_(sim.argmax(dim=-1).to(torch.int32))
    return out


def _cosine_triton_launch_infeasible(cfg: dict, d: int, device: torch.device) -> bool:
    """True if Triton assign kernel cannot run (smem budget or tl.dot tile minimum)."""
    limit = _cuda_max_shared_memory_per_block_optin(device)
    slack = 2048
    ceiling = max(0, limit - slack)
    mt = _COSINE_TRITON_DOT_MIN_TILE
    if _euclid_assign_smem_upper_bound(mt, mt, d, 1, 4) > ceiling:
        return True
    bn = int(cfg["BLOCK_N"])
    bk = int(cfg["BLOCK_K"])
    if bn < mt or bk < mt:
        return True
    need = _euclid_assign_smem_upper_bound(
        bn, bk, d, int(cfg["num_stages"]), int(cfg["num_warps"])
    )
    return need > ceiling


def cosine_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    out: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Return nearest(cosine similarity)-centroid indices using Triton kernel.

    By default (no explicit ``config`` / warp overrides), uses
    :func:`_direct_optimal_assign_config`: a **deterministic** search over a small
    grid of tile sizes and warp/stage counts that fit the GPU's per-block shared
    memory — **no Triton autotuner trial loop**. If no Triton config fits, falls
    back to batched PyTorch on GPU.

    ``use_heuristic`` is kept for API compatibility; it no longer switches to
    autotune (removed from the default path to avoid long compile/tune stalls).

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        config    : Optional {"BLOCK_N","BLOCK_K","num_warps","num_stages"} override.

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    _ = use_heuristic  # API compat; assign path is always direct-optimal or override
    assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config: Optional[dict] = None
    if config is not None:
        selected_config = _clamp_euclid_config_to_shared_budget(dict(config), D, x.device)
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = _clamp_euclid_config_to_shared_budget(
            {
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
            },
            D,
            x.device,
        )
    else:
        selected_config = _direct_optimal_assign_config(D, x.device)

    if selected_config is None:
        return _cosine_assign_pytorch_gpu(x, centroids, out)

    if _cosine_triton_launch_infeasible(selected_config, D, x.device):
        return _cosine_assign_pytorch_gpu(x, centroids, out)

    _cosine_assign_kernel[grid](
        x,
        centroids,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_out_b,
        stride_out_n,
        BLOCK_N=selected_config["BLOCK_N"],
        BLOCK_K=selected_config["BLOCK_K"],
        num_warps=selected_config["num_warps"],
        num_stages=selected_config["num_stages"],
    )
    return out

# ---------------------------------------------------------------
# Quick correctness & performance check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128
    K = 1000
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    cent = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)

    # Reference
    dist = (
        x_sq.unsqueeze(-1) + (cent.to(torch.float32) ** 2).sum(-1).unsqueeze(1) - 2.0 * torch.einsum("bnd,bkd->bnk", x, cent).to(torch.float32)
    ).clamp_min_(0.0)
    ref_ids = dist.argmin(dim=-1)

    tri_ids = euclid_assign_triton(x, cent, x_sq, out)

    print("Correct:", torch.equal(ref_ids.cpu(), tri_ids.cpu()))


    dist_cos = torch.einsum("bnd,bkd->bnk", x.to(torch.float32), cent.to(torch.float32))
    ref_ids_cos = dist_cos.argmax(dim=-1)
    tri_ids_cos = cosine_assign_triton(x, cent, out)

    print("Cosine Correct:", torch.equal(ref_ids_cos.cpu(), tri_ids_cos.cpu()))

    # Simple timing
    repeats = 20
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        euclid_assign_triton(x, cent, x_sq, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids[10, 69344]=}, {tri_ids[10, 69344]=}, {dist[10, 69344, ref_ids[10, 69344]]=}, {dist[10, 69344, tri_ids[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids, tri_ids.to(ref_ids.dtype))
    except Exception as e:
        print("Assertion failed:", e)

    start.record()
    for _ in range(repeats):
        cosine_assign_triton(x, cent, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton Cosine: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids_cos[10, 69344]=}, {tri_ids_cos[10, 69344]=}, {dist_cos[10, 69344, ref_ids_cos[10, 69344]]=}, {dist_cos[10, 69344, tri_ids_cos[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids_cos, tri_ids_cos.to(ref_ids_cos.dtype))
    except Exception as e:
        print("Assertion failed:", e)