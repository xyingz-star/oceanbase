/*
 * k-means++ (L2 / squared Euclidean) on GPU — reference implementation.
 *
 * - Outer (k-1) rounds are sequential (algorithm requirement).
 * - Each round: O(n·d) distance + O(n) scan + O(log n) search on device.
 * - Weighted sampling uses thrust inclusive_scan + lower_bound (no full min_d2 H2D).
 * - Two kernels per inner round: kernel_sq_dist_to_center -> d_dist, kernel_elementwise_min -> d_min (center in shared).
 *
 * Build shared lib:
 *   nvcc -O3 -std=c++17 -Xcompiler -fPIC -shared kmeans_pp_l2.cu -o libkmeans_pp_l2.so -lcudart
 * int8 path: kmeans_pp_l2_int8_cuda(...) — same signature; symmetric global quantize to [-127,127],
 *   approximate sq L2 = (max_abs/127)^2 * sum (Δq)^2; centers still copied from float X rows.
 * Build test (prints H2D / normalize / quantize(int8) / k-means++ GPU / D2H center timings when non-NULL):
 *   make -C ... test && ./kmeans_pp_test [n] [d] [k] [cosine] [mode]
 *   mode: fp32 (default) | int8 | compare
 *   nvcc -O3 -std=c++17 kmeans_pp_l2.cu -o kmeans_pp_test -D KMEANS_PP_STANDALONE_MAIN -lcudart
 */

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/* Optional: pass NULL from lib callers (no cudaEvent overhead). Standalone test fills this. */
typedef struct kmeans_pp_l2_timings {
  float h2d_ms;                 /* Host -> device copy of X */
  float normalize_ms;           /* cosine row normalize kernel (0 if cosine == 0) */
  float kmeanspp_compute_ms;    /* GPU k-means++: first sq_dist + (k-1) rounds */
  float d2h_centers_ms;         /* Device -> host: k center rows */
  /*
   * Breakdown of the for (round=1..k-1) loop (sums over those rounds; 0 if k<=1).
   * Wall-clock ms via std::chrono + cudaDeviceSynchronize after GPU sections (only when this struct is passed).
   *   scan:        inclusive_scan + device sync
   *   total_d2h:   cudaMemcpy scalar total (sync)
   *   pick:        RNG and/or lower_bound + device sync
   *   sq_dist:     kernel_sq_dist_to_center only (inner rounds)
   *   min:         kernel_elementwise_min only (inner rounds)
   */
  float kpp_round_sum_scan_ms;
  float kpp_round_sum_total_d2h_ms;
  float kpp_round_sum_pick_ms;
  float kpp_round_sum_sq_dist_ms;
  float kpp_round_sum_min_ms;
  float kpp_first_center_sq_dist_ms; /* launch_sq_dist for chosen[0] only; 0 if k<=0 */
  float kpp_quantize_ms;             /* f32->int8 quantize + sync; int8 path only, else 0 */
  /*
   * Optional per-round arrays of length k-1; set kpp_round_detail_cap >= k-1 and non-NULL pointers to fill.
   */
  int kpp_round_detail_cap;
  float *kpp_round_scan_ms;
  float *kpp_round_total_d2h_ms;
  float *kpp_round_pick_ms;
  float *kpp_round_sq_dist_ms;
  float *kpp_round_min_ms;
} kmeans_pp_l2_timings_t;

#ifdef __cplusplus
}
#endif

namespace {

struct FabsFunctor {
  __host__ __device__ float operator()(float x) const { return fabsf(x); }
};

struct CudaTimingEvents {
  bool active = false;
  cudaEvent_t e_h2d0{}, e_h2d1{}, e_n0{}, e_n1{}, e_c0{}, e_c1{}, e_d0{}, e_d1{};

  explicit CudaTimingEvents(bool on) : active(on) {
    if (!active) {
      return;
    }
    cudaEventCreate(&e_h2d0);
    cudaEventCreate(&e_h2d1);
    cudaEventCreate(&e_n0);
    cudaEventCreate(&e_n1);
    cudaEventCreate(&e_c0);
    cudaEventCreate(&e_c1);
    cudaEventCreate(&e_d0);
    cudaEventCreate(&e_d1);
  }

  ~CudaTimingEvents() {
    if (!active) {
      return;
    }
    cudaEventDestroy(e_h2d0);
    cudaEventDestroy(e_h2d1);
    cudaEventDestroy(e_n0);
    cudaEventDestroy(e_n1);
    cudaEventDestroy(e_c0);
    cudaEventDestroy(e_c1);
    cudaEventDestroy(e_d0);
    cudaEventDestroy(e_d1);
  }
};

}  // namespace

__global__ void kernel_sq_dist_to_center(
    const float *__restrict__ X, int n, int d, int center_idx, float *__restrict__ out) {
  extern __shared__ float s_center[];
  const float *xc_g = X + static_cast<size_t>(center_idx) * d;
  for (int t = threadIdx.x; t < d; t += blockDim.x) {
    s_center[t] = xc_g[t];
  }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float *xi = X + static_cast<size_t>(i) * d;
  float s = 0.f;
  for (int t = 0; t < d; ++t) {
    float diff = xi[t] - s_center[t];
    s += diff * diff;
  }
  out[i] = s;
}

__global__ void kernel_elementwise_min(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = fminf(a[i], b[i]);
}

static cudaError_t launch_sq_dist(
    const float *d_X, int n, int d, int center_idx, float *d_out, cudaStream_t stream = 0) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const size_t shmem = sizeof(float) * static_cast<size_t>(d);
  kernel_sq_dist_to_center<<<blocks, threads, shmem, stream>>>(d_X, n, d, center_idx, d_out);
  return cudaGetLastError();
}

static cudaError_t launch_min(
    const float *a, const float *b, float *out, int n, cudaStream_t stream = 0) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  kernel_elementwise_min<<<blocks, threads, 0, stream>>>(a, b, out, n);
  return cudaGetLastError();
}

/* Global symmetric quantization: q = round(x * (127/max_abs)), clamp to [-127,127]. scale_sq = (max_abs/127)^2. */
__global__ void kernel_quantize_f32_to_i8(const float *__restrict__ X, int8_t *__restrict__ Q, int64_t nd, float inv_scale) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nd) {
    return;
  }
  const float v = X[idx] * inv_scale;
  const int qi = __float2int_rn(fminf(127.f, fmaxf(-127.f, v)));
  Q[idx] = static_cast<int8_t>(qi);
}

__global__ void kernel_sq_dist_int8_to_center(
    const int8_t *__restrict__ Q, int n, int d, int center_idx, float scale_sq, float *__restrict__ out) {
  extern __shared__ int8_t s_qc[];
  const int8_t *xc_g = Q + static_cast<size_t>(center_idx) * d;
  for (int t = threadIdx.x; t < d; t += blockDim.x) {
    s_qc[t] = xc_g[t];
  }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int8_t *qi = Q + static_cast<size_t>(i) * d;
  int32_t acc = 0;
  for (int t = 0; t < d; ++t) {
    const int diff = static_cast<int>(qi[t]) - static_cast<int>(s_qc[t]);
    acc += diff * diff;
  }
  out[i] = scale_sq * static_cast<float>(acc);
}

static cudaError_t launch_quantize_f32_i8(
    const float *d_X, int8_t *d_Q, int64_t nd, float inv_scale, cudaStream_t stream = 0) {
  const int threads = 256;
  const int blocks = static_cast<int>((nd + threads - 1) / threads);
  kernel_quantize_f32_to_i8<<<blocks, threads, 0, stream>>>(d_X, d_Q, nd, inv_scale);
  return cudaGetLastError();
}

static cudaError_t launch_sq_dist_int8(
    const int8_t *d_Q, int n, int d, int center_idx, float scale_sq, float *d_out, cudaStream_t stream = 0) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const size_t shmem = sizeof(int8_t) * static_cast<size_t>(d);
  kernel_sq_dist_int8_to_center<<<blocks, threads, shmem, stream>>>(d_Q, n, d, center_idx, scale_sq, d_out);
  return cudaGetLastError();
}

__global__ void kernel_row_l2_normalize(float *__restrict__ X, int n, int d, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float *row = X + static_cast<size_t>(i) * d;
  float norm2 = 0.f;
  for (int t = 0; t < d; ++t) {
    norm2 += row[t] * row[t];
  }
  float inv = rsqrtf(fmaxf(norm2, eps * eps));
  for (int t = 0; t < d; ++t) {
    row[t] *= inv;
  }
}

static cudaError_t launch_normalize(float *d_X, int n, int d, cudaStream_t stream = 0) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  kernel_row_l2_normalize<<<blocks, threads, 0, stream>>>(d_X, n, d, 1e-12f);
  return cudaGetLastError();
}

/*
 * Returns 0 on success, non-zero on failure. err_buf optional.
 * h_centers_out: k * d row-major (k center vectors).
 * timings: optional; if non-NULL, fills cudaEvent-based ms (H2D / normalize / k-means++ GPU / D2H centers).
 */
extern "C" int kmeans_pp_l2_cuda(const float *h_X,
    int n,
    int d,
    int k,
    uint64_t seed,
    int cosine, /* 1 = L2 on row-normalized X (same spirit as COS in worker), 0 = raw L2 */
    float *h_centers_out,
    char *err_buf,
    size_t err_buf_sz,
    kmeans_pp_l2_timings_t *timings) {
  if (err_buf && err_buf_sz) {
    err_buf[0] = '\0';
  }
  if (n <= 0 || d <= 0 || k <= 0 || k > n || !h_X || !h_centers_out) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "invalid args");
    }
    return -1;
  }

  if (timings) {
    timings->h2d_ms = timings->normalize_ms = timings->kmeanspp_compute_ms = timings->d2h_centers_ms = 0.f;
    timings->kpp_round_sum_scan_ms = timings->kpp_round_sum_total_d2h_ms = timings->kpp_round_sum_pick_ms =
        timings->kpp_round_sum_sq_dist_ms = timings->kpp_round_sum_min_ms = timings->kpp_first_center_sq_dist_ms =
        timings->kpp_quantize_ms = 0.f;
  }
  CudaTimingEvents ev(timings != nullptr);

  thrust::device_vector<float> d_X(static_cast<size_t>(n) * d);
  float *d_X_raw = thrust::raw_pointer_cast(d_X.data());

  if (ev.active) {
    cudaEventRecord(ev.e_h2d0, 0);
  }
  cudaError_t st = cudaMemcpy(d_X_raw, h_X, sizeof(float) * static_cast<size_t>(n) * d, cudaMemcpyHostToDevice);
  if (ev.active) {
    cudaEventRecord(ev.e_h2d1, 0);
    cudaEventSynchronize(ev.e_h2d1);
    cudaEventElapsedTime(&timings->h2d_ms, ev.e_h2d0, ev.e_h2d1);
  }
  if (st != cudaSuccess) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "cudaMemcpy H2D: %s", cudaGetErrorString(st));
    }
    return -2;
  }

  if (cosine) {
    if (ev.active) {
      cudaEventRecord(ev.e_n0, 0);
    }
    st = launch_normalize(d_X_raw, n, d);
    if (ev.active) {
      cudaEventRecord(ev.e_n1, 0);
      cudaEventSynchronize(ev.e_n1);
      cudaEventElapsedTime(&timings->normalize_ms, ev.e_n0, ev.e_n1);
    }
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "normalize: %s", cudaGetErrorString(st));
      }
      return -3;
    }
  }

  thrust::device_vector<float> min_d2(n), dist(n), cumsum(n);
  float *d_Xp = thrust::raw_pointer_cast(d_X.data());
  float *d_min = thrust::raw_pointer_cast(min_d2.data());
  float *d_dist = thrust::raw_pointer_cast(dist.data());
  float *d_cum = thrust::raw_pointer_cast(cumsum.data());

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> uni_idx(0, n - 1);
  std::uniform_real_distribution<float> uni01(0.f, 1.f);

  std::vector<int> chosen(static_cast<size_t>(k));
  chosen[0] = uni_idx(gen);

  if (ev.active) {
    cudaEventRecord(ev.e_c0, 0);
  }

  if (timings) {
    const auto t_fc0 = std::chrono::steady_clock::now();
    st = launch_sq_dist(d_Xp, n, d, chosen[0], d_min);
    cudaDeviceSynchronize();
    const auto t_fc1 = std::chrono::steady_clock::now();
    timings->kpp_first_center_sq_dist_ms =
        std::chrono::duration<float, std::milli>(t_fc1 - t_fc0).count();
  } else {
    st = launch_sq_dist(d_Xp, n, d, chosen[0], d_min);
  }
  if (st != cudaSuccess) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "sq_dist0: %s", cudaGetErrorString(st));
    }
    return -4;
  }

  for (int round = 1; round < k; ++round) {
    const int ridx = round - 1;

    if (timings) {
      const auto t_scan0 = std::chrono::steady_clock::now();
      thrust::inclusive_scan(thrust::cuda::par.on(0), min_d2.begin(), min_d2.end(), cumsum.begin());
      cudaDeviceSynchronize();
      const auto t_scan1 = std::chrono::steady_clock::now();
      const float ms_scan =
          std::chrono::duration<float, std::milli>(t_scan1 - t_scan0).count();
      timings->kpp_round_sum_scan_ms += ms_scan;
      if (timings->kpp_round_scan_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_scan_ms[ridx] = ms_scan;
      }
    } else {
      thrust::inclusive_scan(thrust::cuda::par.on(0), min_d2.begin(), min_d2.end(), cumsum.begin());
    }

    float total = 0.f;
    if (timings) {
      const auto t_d2h0 = std::chrono::steady_clock::now();
      st = cudaMemcpy(&total, d_cum + (n - 1), sizeof(float), cudaMemcpyDeviceToHost);
      const auto t_d2h1 = std::chrono::steady_clock::now();
      const float ms_d2h = std::chrono::duration<float, std::milli>(t_d2h1 - t_d2h0).count();
      timings->kpp_round_sum_total_d2h_ms += ms_d2h;
      if (timings->kpp_round_total_d2h_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_total_d2h_ms[ridx] = ms_d2h;
      }
    } else {
      st = cudaMemcpy(&total, d_cum + (n - 1), sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "memcpy total: %s", cudaGetErrorString(st));
      }
      return -5;
    }

    if (timings) {
      const auto t_pick0 = std::chrono::steady_clock::now();
      int idx = 0;
      if (!(total > 1e-30f)) {
        idx = uni_idx(gen);
      } else {
        float u = uni01(gen) * total;
        auto it = thrust::lower_bound(thrust::cuda::par.on(0), cumsum.begin(), cumsum.end(), u);
        idx = static_cast<int>(it - cumsum.begin());
        if (idx < 0) {
          idx = 0;
        }
        if (idx >= n) {
          idx = n - 1;
        }
      }
      cudaDeviceSynchronize();
      const auto t_pick1 = std::chrono::steady_clock::now();
      const float ms_pick = std::chrono::duration<float, std::milli>(t_pick1 - t_pick0).count();
      timings->kpp_round_sum_pick_ms += ms_pick;
      if (timings->kpp_round_pick_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_pick_ms[ridx] = ms_pick;
      }
      chosen[static_cast<size_t>(round)] = idx;
    } else {
      int idx = 0;
      if (!(total > 1e-30f)) {
        idx = uni_idx(gen);
      } else {
        float u = uni01(gen) * total;
        auto it = thrust::lower_bound(thrust::cuda::par.on(0), cumsum.begin(), cumsum.end(), u);
        idx = static_cast<int>(it - cumsum.begin());
        if (idx < 0) {
          idx = 0;
        }
        if (idx >= n) {
          idx = n - 1;
        }
      }
      chosen[static_cast<size_t>(round)] = idx;
    }

    const int idx = chosen[static_cast<size_t>(round)];

    if (timings) {
      const auto t_sq0 = std::chrono::steady_clock::now();
      st = launch_sq_dist(d_Xp, n, d, idx, d_dist);
      cudaDeviceSynchronize();
      const auto t_sq1 = std::chrono::steady_clock::now();
      const float ms_sq = std::chrono::duration<float, std::milli>(t_sq1 - t_sq0).count();
      timings->kpp_round_sum_sq_dist_ms += ms_sq;
      if (timings->kpp_round_sq_dist_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_sq_dist_ms[ridx] = ms_sq;
      }
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "sq_dist: %s", cudaGetErrorString(st));
        }
        return -6;
      }
      const auto t_mn0 = std::chrono::steady_clock::now();
      st = launch_min(d_min, d_dist, d_min, n);
      cudaDeviceSynchronize();
      const auto t_mn1 = std::chrono::steady_clock::now();
      const float ms_mn = std::chrono::duration<float, std::milli>(t_mn1 - t_mn0).count();
      timings->kpp_round_sum_min_ms += ms_mn;
      if (timings->kpp_round_min_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_min_ms[ridx] = ms_mn;
      }
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "min: %s", cudaGetErrorString(st));
        }
        return -7;
      }
    } else {
      st = launch_sq_dist(d_Xp, n, d, idx, d_dist);
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "sq_dist: %s", cudaGetErrorString(st));
        }
        return -6;
      }
      st = launch_min(d_min, d_dist, d_min, n);
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "min: %s", cudaGetErrorString(st));
        }
        return -7;
      }
    }
  }

  if (ev.active) {
    cudaEventRecord(ev.e_c1, 0);
    cudaEventSynchronize(ev.e_c1);
    cudaEventElapsedTime(&timings->kmeanspp_compute_ms, ev.e_c0, ev.e_c1);
    cudaEventRecord(ev.e_d0, 0);
  }

  for (int c = 0; c < k; ++c) {
    const int row = chosen[static_cast<size_t>(c)];
    st = cudaMemcpy(
        h_centers_out + static_cast<size_t>(c) * d, d_Xp + static_cast<size_t>(row) * d, sizeof(float) * static_cast<size_t>(d), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "memcpy center %d: %s", c, cudaGetErrorString(st));
      }
      return -8;
    }
  }

  if (ev.active) {
    cudaEventRecord(ev.e_d1, 0);
    cudaEventSynchronize(ev.e_d1);
    cudaEventElapsedTime(&timings->d2h_centers_ms, ev.e_d0, ev.e_d1);
  }

  return 0;
}

/*
 * Same as kmeans_pp_l2_cuda but distances use symmetric int8 quantization of X after H2D (+ optional cosine):
 * global max_abs = max |X_ij|, q_ij = round(127 * x_ij / max_abs) clamped to [-127,127],
 * approximate squared L2 = (max_abs/127)^2 * sum_t (q_it - q_jt)^2.
 * Center rows copied from float d_X (exact rows), not dequantized Q.
 */
extern "C" int kmeans_pp_l2_int8_cuda(const float *h_X,
    int n,
    int d,
    int k,
    uint64_t seed,
    int cosine,
    float *h_centers_out,
    char *err_buf,
    size_t err_buf_sz,
    kmeans_pp_l2_timings_t *timings) {
  if (err_buf && err_buf_sz) {
    err_buf[0] = '\0';
  }
  if (n <= 0 || d <= 0 || k <= 0 || k > n || !h_X || !h_centers_out) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "invalid args");
    }
    return -1;
  }

  if (timings) {
    timings->h2d_ms = timings->normalize_ms = timings->kmeanspp_compute_ms = timings->d2h_centers_ms = 0.f;
    timings->kpp_round_sum_scan_ms = timings->kpp_round_sum_total_d2h_ms = timings->kpp_round_sum_pick_ms =
        timings->kpp_round_sum_sq_dist_ms = timings->kpp_round_sum_min_ms = timings->kpp_first_center_sq_dist_ms =
        timings->kpp_quantize_ms = 0.f;
  }
  CudaTimingEvents ev(timings != nullptr);

  thrust::device_vector<float> d_X(static_cast<size_t>(n) * d);
  float *d_X_raw = thrust::raw_pointer_cast(d_X.data());

  if (ev.active) {
    cudaEventRecord(ev.e_h2d0, 0);
  }
  cudaError_t st = cudaMemcpy(d_X_raw, h_X, sizeof(float) * static_cast<size_t>(n) * d, cudaMemcpyHostToDevice);
  if (ev.active) {
    cudaEventRecord(ev.e_h2d1, 0);
    cudaEventSynchronize(ev.e_h2d1);
    cudaEventElapsedTime(&timings->h2d_ms, ev.e_h2d0, ev.e_h2d1);
  }
  if (st != cudaSuccess) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "cudaMemcpy H2D: %s", cudaGetErrorString(st));
    }
    return -2;
  }

  if (cosine) {
    if (ev.active) {
      cudaEventRecord(ev.e_n0, 0);
    }
    st = launch_normalize(d_X_raw, n, d);
    if (ev.active) {
      cudaEventRecord(ev.e_n1, 0);
      cudaEventSynchronize(ev.e_n1);
      cudaEventElapsedTime(&timings->normalize_ms, ev.e_n0, ev.e_n1);
    }
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "normalize: %s", cudaGetErrorString(st));
      }
      return -3;
    }
  }

  const int64_t nd = static_cast<int64_t>(n) * d;
  float max_abs = thrust::transform_reduce(
      thrust::cuda::par.on(0),
      thrust::device_ptr<float>(d_X_raw),
      thrust::device_ptr<float>(d_X_raw) + static_cast<size_t>(nd),
      FabsFunctor{},
      0.f,
      thrust::maximum<float>());
  if (!(max_abs > 1e-30f)) {
    max_abs = 1.f;
  }
  const float inv_scale = 127.f / max_abs;
  const float scale_sq = (max_abs / 127.f) * (max_abs / 127.f);

  thrust::device_vector<int8_t> d_Q(static_cast<size_t>(nd));
  int8_t *d_Q_raw = thrust::raw_pointer_cast(d_Q.data());

  if (timings) {
    const auto t_q0 = std::chrono::steady_clock::now();
    st = launch_quantize_f32_i8(d_X_raw, d_Q_raw, nd, inv_scale);
    cudaDeviceSynchronize();
    const auto t_q1 = std::chrono::steady_clock::now();
    timings->kpp_quantize_ms = std::chrono::duration<float, std::milli>(t_q1 - t_q0).count();
  } else {
    st = launch_quantize_f32_i8(d_X_raw, d_Q_raw, nd, inv_scale);
  }
  if (st != cudaSuccess) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "quantize: %s", cudaGetErrorString(st));
    }
    return -9;
  }

  thrust::device_vector<float> min_d2(n), dist(n), cumsum(n);
  float *d_Xp = thrust::raw_pointer_cast(d_X.data());
  float *d_min = thrust::raw_pointer_cast(min_d2.data());
  float *d_dist = thrust::raw_pointer_cast(dist.data());
  float *d_cum = thrust::raw_pointer_cast(cumsum.data());

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> uni_idx(0, n - 1);
  std::uniform_real_distribution<float> uni01(0.f, 1.f);

  std::vector<int> chosen(static_cast<size_t>(k));
  chosen[0] = uni_idx(gen);

  if (ev.active) {
    cudaEventRecord(ev.e_c0, 0);
  }

  if (timings) {
    const auto t_fc0 = std::chrono::steady_clock::now();
    st = launch_sq_dist_int8(d_Q_raw, n, d, chosen[0], scale_sq, d_min);
    cudaDeviceSynchronize();
    const auto t_fc1 = std::chrono::steady_clock::now();
    timings->kpp_first_center_sq_dist_ms =
        std::chrono::duration<float, std::milli>(t_fc1 - t_fc0).count();
  } else {
    st = launch_sq_dist_int8(d_Q_raw, n, d, chosen[0], scale_sq, d_min);
  }
  if (st != cudaSuccess) {
    if (err_buf && err_buf_sz) {
      snprintf(err_buf, err_buf_sz, "sq_dist0: %s", cudaGetErrorString(st));
    }
    return -4;
  }

  for (int round = 1; round < k; ++round) {
    const int ridx = round - 1;

    if (timings) {
      const auto t_scan0 = std::chrono::steady_clock::now();
      thrust::inclusive_scan(thrust::cuda::par.on(0), min_d2.begin(), min_d2.end(), cumsum.begin());
      cudaDeviceSynchronize();
      const auto t_scan1 = std::chrono::steady_clock::now();
      const float ms_scan =
          std::chrono::duration<float, std::milli>(t_scan1 - t_scan0).count();
      timings->kpp_round_sum_scan_ms += ms_scan;
      if (timings->kpp_round_scan_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_scan_ms[ridx] = ms_scan;
      }
    } else {
      thrust::inclusive_scan(thrust::cuda::par.on(0), min_d2.begin(), min_d2.end(), cumsum.begin());
    }

    float total = 0.f;
    if (timings) {
      const auto t_d2h0 = std::chrono::steady_clock::now();
      st = cudaMemcpy(&total, d_cum + (n - 1), sizeof(float), cudaMemcpyDeviceToHost);
      const auto t_d2h1 = std::chrono::steady_clock::now();
      const float ms_d2h = std::chrono::duration<float, std::milli>(t_d2h1 - t_d2h0).count();
      timings->kpp_round_sum_total_d2h_ms += ms_d2h;
      if (timings->kpp_round_total_d2h_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_total_d2h_ms[ridx] = ms_d2h;
      }
    } else {
      st = cudaMemcpy(&total, d_cum + (n - 1), sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "memcpy total: %s", cudaGetErrorString(st));
      }
      return -5;
    }

    if (timings) {
      const auto t_pick0 = std::chrono::steady_clock::now();
      int idx = 0;
      if (!(total > 1e-30f)) {
        idx = uni_idx(gen);
      } else {
        float u = uni01(gen) * total;
        auto it = thrust::lower_bound(thrust::cuda::par.on(0), cumsum.begin(), cumsum.end(), u);
        idx = static_cast<int>(it - cumsum.begin());
        if (idx < 0) {
          idx = 0;
        }
        if (idx >= n) {
          idx = n - 1;
        }
      }
      cudaDeviceSynchronize();
      const auto t_pick1 = std::chrono::steady_clock::now();
      const float ms_pick = std::chrono::duration<float, std::milli>(t_pick1 - t_pick0).count();
      timings->kpp_round_sum_pick_ms += ms_pick;
      if (timings->kpp_round_pick_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_pick_ms[ridx] = ms_pick;
      }
      chosen[static_cast<size_t>(round)] = idx;
    } else {
      int idx = 0;
      if (!(total > 1e-30f)) {
        idx = uni_idx(gen);
      } else {
        float u = uni01(gen) * total;
        auto it = thrust::lower_bound(thrust::cuda::par.on(0), cumsum.begin(), cumsum.end(), u);
        idx = static_cast<int>(it - cumsum.begin());
        if (idx < 0) {
          idx = 0;
        }
        if (idx >= n) {
          idx = n - 1;
        }
      }
      chosen[static_cast<size_t>(round)] = idx;
    }

    const int idx = chosen[static_cast<size_t>(round)];

    if (timings) {
      const auto t_sq0 = std::chrono::steady_clock::now();
      st = launch_sq_dist_int8(d_Q_raw, n, d, idx, scale_sq, d_dist);
      cudaDeviceSynchronize();
      const auto t_sq1 = std::chrono::steady_clock::now();
      const float ms_sq = std::chrono::duration<float, std::milli>(t_sq1 - t_sq0).count();
      timings->kpp_round_sum_sq_dist_ms += ms_sq;
      if (timings->kpp_round_sq_dist_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_sq_dist_ms[ridx] = ms_sq;
      }
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "sq_dist: %s", cudaGetErrorString(st));
        }
        return -6;
      }
      const auto t_mn0 = std::chrono::steady_clock::now();
      st = launch_min(d_min, d_dist, d_min, n);
      cudaDeviceSynchronize();
      const auto t_mn1 = std::chrono::steady_clock::now();
      const float ms_mn = std::chrono::duration<float, std::milli>(t_mn1 - t_mn0).count();
      timings->kpp_round_sum_min_ms += ms_mn;
      if (timings->kpp_round_min_ms && ridx < timings->kpp_round_detail_cap) {
        timings->kpp_round_min_ms[ridx] = ms_mn;
      }
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "min: %s", cudaGetErrorString(st));
        }
        return -7;
      }
    } else {
      st = launch_sq_dist_int8(d_Q_raw, n, d, idx, scale_sq, d_dist);
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "sq_dist: %s", cudaGetErrorString(st));
        }
        return -6;
      }
      st = launch_min(d_min, d_dist, d_min, n);
      if (st != cudaSuccess) {
        if (err_buf && err_buf_sz) {
          snprintf(err_buf, err_buf_sz, "min: %s", cudaGetErrorString(st));
        }
        return -7;
      }
    }
  }

  if (ev.active) {
    cudaEventRecord(ev.e_c1, 0);
    cudaEventSynchronize(ev.e_c1);
    cudaEventElapsedTime(&timings->kmeanspp_compute_ms, ev.e_c0, ev.e_c1);
    cudaEventRecord(ev.e_d0, 0);
  }

  for (int c = 0; c < k; ++c) {
    const int row = chosen[static_cast<size_t>(c)];
    st = cudaMemcpy(
        h_centers_out + static_cast<size_t>(c) * d, d_Xp + static_cast<size_t>(row) * d, sizeof(float) * static_cast<size_t>(d), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
      if (err_buf && err_buf_sz) {
        snprintf(err_buf, err_buf_sz, "memcpy center %d: %s", c, cudaGetErrorString(st));
      }
      return -8;
    }
  }

  if (ev.active) {
    cudaEventRecord(ev.e_d1, 0);
    cudaEventSynchronize(ev.e_d1);
    cudaEventElapsedTime(&timings->d2h_centers_ms, ev.e_d0, ev.e_d1);
  }

  return 0;
}

#ifdef KMEANS_PP_STANDALONE_MAIN
#include <cstdlib>
#include <cstring>

static void print_timings(const char *label, const kmeans_pp_l2_timings_t &tm, double wall_ms, int k, int n, int d, int cosine,
    const std::vector<float> &rd_scan, const std::vector<float> &rd_d2h, const std::vector<float> &rd_pick,
    const std::vector<float> &rd_sq, const std::vector<float> &rd_min) {
  const float ev_sum = tm.h2d_ms + tm.normalize_ms + tm.kmeanspp_compute_ms + tm.d2h_centers_ms;
  printf("%s timings (cudaEvent, default stream):\n", label);
  printf("  h2d_ms (X Host->Device)     = %.3f\n", tm.h2d_ms);
  printf("  normalize_ms (cosine=%d)    = %.3f\n", cosine, tm.normalize_ms);
  printf("  kpp_quantize_ms (int8 only) = %.3f\n", tm.kpp_quantize_ms);
  printf("  kmeanspp_compute_ms (GPU)  = %.3f\n", tm.kmeanspp_compute_ms);
  printf("  d2h_centers_ms (k rows)     = %.3f\n", tm.d2h_centers_ms);
  printf("  cuda_events_sum_ms          = %.3f\n", ev_sum);
  printf("  wall_clock_total_ms         = %.3f  (includes host CPU between phases; use cuda_* for GPU breakdown)\n", wall_ms);
  printf("k-means++ first center (kernel_sq_dist -> d_min):\n");
  printf("  first_center_sq_dist_ms          = %.3f\n", tm.kpp_first_center_sq_dist_ms);
  if (k > 1) {
    const float rs = tm.kpp_round_sum_scan_ms + tm.kpp_round_sum_total_d2h_ms + tm.kpp_round_sum_pick_ms +
        tm.kpp_round_sum_sq_dist_ms + tm.kpp_round_sum_min_ms;
    printf("k-means++ inner loop (sum over rounds 1..k-1; wall ms per phase, sync after each kernel):\n");
    printf("  sum_scan_ms (inclusive_scan)     = %.3f\n", tm.kpp_round_sum_scan_ms);
    printf("  sum_total_d2h_ms (scalar total)   = %.3f\n", tm.kpp_round_sum_total_d2h_ms);
    printf("  sum_pick_ms (RNG + lower_bound)  = %.3f\n", tm.kpp_round_sum_pick_ms);
    printf("  sum_sq_dist_ms (kernel_sq_dist)  = %.3f\n", tm.kpp_round_sum_sq_dist_ms);
    printf("  sum_min_ms (kernel_elementwise)  = %.3f\n", tm.kpp_round_sum_min_ms);
    printf("  inner_loop_segments_sum_ms       = %.3f  (scan+d2h+pick+sq+min)\n", rs);
    const int max_print = 40;
    if (k - 1 <= max_print) {
      printf("  per-round (round r = 1..k-1):\n");
      printf("    r   scan  total_d2h    pick  sq_dist  min\n");
      for (int r = 0; r < k - 1; ++r) {
        printf(
            "    %d %7.3f %10.3f %7.3f %8.3f %6.3f\n", r + 1, rd_scan[static_cast<size_t>(r)], rd_d2h[static_cast<size_t>(r)],
            rd_pick[static_cast<size_t>(r)], rd_sq[static_cast<size_t>(r)], rd_min[static_cast<size_t>(r)]);
      }
    } else {
      printf("  (per-round table omitted: k-1=%d > %d; use smaller k to print)\n", k - 1, max_print);
    }
  }
}

int main(int argc, char **argv) {
  int n = 1000;
  int d = 32;
  int k = 10;
  int cosine = 0;
  const char *mode = "fp32";
  if (argc >= 2) {
    n = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    d = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    k = std::atoi(argv[3]);
  }
  if (argc >= 5) {
    cosine = std::atoi(argv[4]);
  }
  if (argc >= 6) {
    mode = argv[5];
  }
  if (n <= 0 || d <= 0 || k <= 0 || k > n) {
    fprintf(
        stderr,
        "usage: %s [n] [d] [k] [cosine] [mode]\n"
        "  mode: fp32 | int8 | compare  (default fp32)\n",
        argv[0]);
    return 2;
  }

  std::vector<float> X(static_cast<size_t>(n) * d);
  std::mt19937_64 g(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto &v : X) {
    v = nd(g);
  }
  std::vector<float> centers(static_cast<size_t>(k) * d);
  char ebuf[256];
  /* Warm up CUDA context so wall_clock_total_ms is comparable to cuda events (not 100x inflate on first launch). */
  cudaFree(0);

  const bool mode_int8 =
      (std::strcmp(mode, "int8") == 0 || std::strcmp(mode, "1") == 0 || std::strcmp(mode, "i8") == 0);
  const bool mode_compare =
      (std::strcmp(mode, "compare") == 0 || std::strcmp(mode, "both") == 0 || std::strcmp(mode, "cmp") == 0);

  auto fill_round_detail = [&](kmeans_pp_l2_timings_t *ptm, std::vector<float> &a, std::vector<float> &b, std::vector<float> &c,
                                std::vector<float> &e, std::vector<float> &f) {
    if (k > 1) {
      const int nr = k - 1;
      a.resize(static_cast<size_t>(nr));
      b.resize(static_cast<size_t>(nr));
      c.resize(static_cast<size_t>(nr));
      e.resize(static_cast<size_t>(nr));
      f.resize(static_cast<size_t>(nr));
      ptm->kpp_round_detail_cap = nr;
      ptm->kpp_round_scan_ms = a.data();
      ptm->kpp_round_total_d2h_ms = b.data();
      ptm->kpp_round_pick_ms = c.data();
      ptm->kpp_round_sq_dist_ms = e.data();
      ptm->kpp_round_min_ms = f.data();
    }
  };

  auto run_fp32 = [&]() -> int {
    kmeans_pp_l2_timings_t tm{};
    std::vector<float> rd_scan, rd_d2h, rd_pick, rd_sq, rd_min;
    fill_round_detail(&tm, rd_scan, rd_d2h, rd_pick, rd_sq, rd_min);
    auto t0 = std::chrono::steady_clock::now();
    int rc = kmeans_pp_l2_cuda(X.data(), n, d, k, 42, cosine, centers.data(), ebuf, sizeof(ebuf), &tm);
    auto t1 = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (rc != 0) {
      fprintf(stderr, "kmeans_pp_l2_cuda failed %d: %s\n", rc, ebuf);
      return rc;
    }
    print_timings("fp32 (kmeans_pp_l2_cuda)", tm, wall_ms, k, n, d, cosine, rd_scan, rd_d2h, rd_pick, rd_sq, rd_min);
    printf("ok fp32 k=%d n=%d d=%d cosine=%d first_center[0]=%f\n", k, n, d, cosine, centers[0]);
    return 0;
  };

  auto run_int8 = [&]() -> int {
    kmeans_pp_l2_timings_t tm{};
    std::vector<float> rd_scan, rd_d2h, rd_pick, rd_sq, rd_min;
    fill_round_detail(&tm, rd_scan, rd_d2h, rd_pick, rd_sq, rd_min);
    auto t0 = std::chrono::steady_clock::now();
    int rc = kmeans_pp_l2_int8_cuda(X.data(), n, d, k, 42, cosine, centers.data(), ebuf, sizeof(ebuf), &tm);
    auto t1 = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (rc != 0) {
      fprintf(stderr, "kmeans_pp_l2_int8_cuda failed %d: %s\n", rc, ebuf);
      return rc;
    }
    print_timings("int8 (kmeans_pp_l2_int8_cuda)", tm, wall_ms, k, n, d, cosine, rd_scan, rd_d2h, rd_pick, rd_sq, rd_min);
    printf("ok int8 k=%d n=%d d=%d cosine=%d first_center[0]=%f\n", k, n, d, cosine, centers[0]);
    return 0;
  };

  if (mode_compare) {
    printf("=== compare fp32 vs int8 (same X, seed=42) ===\n\n");
    if (run_fp32() != 0) {
      return 1;
    }
    printf("\n");
    if (run_int8() != 0) {
      return 1;
    }
    printf(
        "\n--- latency summary (cudaEvent kmeanspp_compute_ms; int8 adds quantize outside that window) ---\n"
        "  See kmeanspp_compute_ms and kpp_quantize_ms above.\n");
    return 0;
  }
  if (mode_int8) {
    return run_int8() != 0 ? 1 : 0;
  }
  return run_fp32() != 0 ? 1 : 0;
}
#endif
