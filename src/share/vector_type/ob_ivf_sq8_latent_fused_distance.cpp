/**
 * Copyright (c) 2026 OceanBase
 * OceanBase CE is licensed under the Mulan PubL v2.
 *
 * Fused latent IVF_SQ8 distance: stream dequant + accumulate (scalar / AVX2 / AVX512),
 * matching ob_ivf_sq8_latent_decode.h dequant rules.
 */

#define USING_LOG_PREFIX SHARE
#include "ob_ivf_sq8_latent_fused_distance.h"
#include "common/object/ob_object.h"
#include "ob_vector_op_common.h"
#include <cmath>
#include <cstring>

namespace oceanbase
{
namespace common
{

namespace {

enum class VecDisType : int
{
  COSINE = 0,
  DOT = 1,
  EUCLIDEAN = 2,
  MANHATTAN = 3,
  EUCLIDEAN_SQUARED = 4,
};

OB_INLINE float latent_decode_one(
    const int64_t i,
    const float *meta_min,
    const float *meta_step,
    const uint8_t code)
{
  const float epsilon = static_cast<float>(1e-10);
  if (fabsf(meta_step[i]) < epsilon) {
    return meta_min[i];
  }
  return meta_min[i] + meta_step[i] * (static_cast<float>(code) + static_cast<float>(0.5));
}

OB_INLINE double cosine_dist_from_similarity(double similarity)
{
  if (similarity > 1.0) {
    similarity = 1.0;
  } else if (similarity < -1.0) {
    similarity = -1.0;
  }
  return 1.0 - similarity;
}

OB_INLINE int fused_finalize_cosine(const double cross, const double sum_q2, const double sum_y2, double &out_distance)
{
  int ret = OB_SUCCESS;
  const double nq = std::sqrt(sum_q2);
  const double ny = std::sqrt(sum_y2);
  if (nq <= 0.0 || ny <= 0.0) {
    ret = OB_ERR_NULL_VALUE;
  } else {
    double sim = cross / (nq * ny);
    if (std::isnan(sim)) {
      ret = OB_NUMERIC_OVERFLOW;
    } else {
      out_distance = cosine_dist_from_similarity(sim);
    }
  }
  return ret;
}

// need_norm_candidate: same as decode path — q·(y/||y||) with q already unit; always divide by ||y|| when non-zero.
OB_INLINE int fused_finalize_dot(const double cross, const double sum_y2, const bool need_norm_candidate, double &out_distance)
{
  int ret = OB_SUCCESS;
  if (need_norm_candidate) {
    const double ny = std::sqrt(sum_y2);
    if (ny <= 0.0) {
      ret = OB_ERR_NULL_VALUE;
    } else {
      out_distance = cross / ny;
      if (std::isnan(out_distance)) {
        ret = OB_NUMERIC_OVERFLOW;
      }
    }
  } else {
    out_distance = cross;
  }
  return ret;
}

OB_INLINE int fused_impl_scalar(
    const float *query,
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const uint8_t *codes,
    const int vec_dis_type_int,
    const bool need_norm_candidate,
    double &out_distance)
{
  int ret = OB_SUCCESS;
  out_distance = 0.0;
  const VecDisType dt = static_cast<VecDisType>(vec_dis_type_int);
  switch (dt) {
    case VecDisType::COSINE: {
      double cross = 0.0;
      double sum_q2 = 0.0;
      double sum_y2 = 0.0;
      for (int64_t i = 0; i < dim; ++i) {
        const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
        const double q = static_cast<double>(query[i]);
        const double y = static_cast<double>(yi);
        cross += q * y;
        sum_q2 += q * q;
        sum_y2 += y * y;
      }
      ret = fused_finalize_cosine(cross, sum_q2, sum_y2, out_distance);
      break;
    }
    case VecDisType::DOT: {
      double cross = 0.0;
      double sum_y2 = 0.0;
      for (int64_t i = 0; i < dim; ++i) {
        const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
        const double q = static_cast<double>(query[i]);
        const double y = static_cast<double>(yi);
        cross += q * y;
        sum_y2 += y * y;
      }
      ret = fused_finalize_dot(cross, sum_y2, need_norm_candidate, out_distance);
      break;
    }
    case VecDisType::EUCLIDEAN:
    case VecDisType::EUCLIDEAN_SQUARED:
    case VecDisType::MANHATTAN: {
      double acc = 0.0;
      for (int64_t i = 0; i < dim; ++i) {
        const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
        const double q = static_cast<double>(query[i]);
        const double y = static_cast<double>(yi);
        if (dt == VecDisType::MANHATTAN) {
          acc += std::fabs(q - y);
        } else {
          const double d = q - y;
          acc += d * d;
        }
      }
      if (dt == VecDisType::EUCLIDEAN) {
        out_distance = std::sqrt(acc);
        if (std::isnan(out_distance)) {
          ret = OB_NUMERIC_OVERFLOW;
        }
      } else {
        out_distance = acc;
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
      break;
  }
  return ret;
}

} // namespace

OB_DECLARE_DEFAULT_CODE(
    int ivf_sq8_latent_fused_distance_vs_query_impl(
        const float *query,
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        const int vec_dis_type_int,
        const bool need_norm_candidate,
        double &out_distance)
    {
      int ret = OB_SUCCESS;
      out_distance = 0.0;
      if (OB_ISNULL(query) || OB_ISNULL(meta_min) || OB_ISNULL(meta_step) || OB_ISNULL(codes) || dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
      } else {
        ret = fused_impl_scalar(query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
      }
      return ret;
    }
)

#if defined(__GNUC__) && defined(__x86_64__)

OB_DECLARE_AVX2_SPECIFIC_CODE(
    OB_INLINE double horizontal_sum_m256_ps(const __m256 v)
    {
      const __m128 hi = _mm256_extractf128_ps(v, 1);
      const __m128 lo = _mm256_castps256_ps128(v);
      __m128 s = _mm_add_ps(lo, hi);
      s = _mm_hadd_ps(s, s);
      s = _mm_hadd_ps(s, s);
      return static_cast<double>(_mm_cvtss_f32(s));
    }

    int ivf_sq8_latent_fused_distance_vs_query_impl(
        const float *query,
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        const int vec_dis_type_int,
        const bool need_norm_candidate,
        double &out_distance)
    {
      int ret = OB_SUCCESS;
      out_distance = 0.0;
      if (OB_ISNULL(query) || OB_ISNULL(meta_min) || OB_ISNULL(meta_step) || OB_ISNULL(codes) || dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
      } else if (common::is_arch_supported(ObTargetArch::AVX2)) {
        const VecDisType dt = static_cast<VecDisType>(vec_dis_type_int);
        const __m256 veps = _mm256_set1_ps(1e-10f);
        const __m256 vhalf = _mm256_set1_ps(0.5f);
        const __m256 vsign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        int64_t i = 0;
        const int64_t dim8 = dim & ~static_cast<int64_t>(7);

        switch (dt) {
          case VecDisType::COSINE:
          case VecDisType::DOT: {
            double cross = 0.0;
            double sum_y2 = 0.0;
            double sum_q2 = 0.0;
            const bool accum_q2 = (dt == VecDisType::COSINE);
            for (; i < dim8; i += 8) {
              const __m256 vmin = _mm256_loadu_ps(meta_min + i);
              const __m256 vstep = _mm256_loadu_ps(meta_step + i);
              const __m256 abs_step = _mm256_and_ps(vstep, vsign_mask);
              int64_t packed64 = 0;
              std::memcpy(&packed64, codes + i, sizeof(packed64));
              const __m128i c8 = _mm_cvtsi64_si128(packed64);
              const __m256i ci = _mm256_cvtepu8_epi32(c8);
              const __m256 cf = _mm256_cvtepi32_ps(ci);
              const __m256 ch = _mm256_add_ps(cf, vhalf);
              const __m256 fmadd = _mm256_fmadd_ps(vstep, ch, vmin);
              const __m256 mask_small = _mm256_cmp_ps(abs_step, veps, _CMP_LT_OQ);
              const __m256 vy = _mm256_blendv_ps(fmadd, vmin, mask_small);
              const __m256 vq = _mm256_loadu_ps(query + i);
              cross += horizontal_sum_m256_ps(_mm256_mul_ps(vq, vy));
              sum_y2 += horizontal_sum_m256_ps(_mm256_mul_ps(vy, vy));
              if (accum_q2) {
                sum_q2 += horizontal_sum_m256_ps(_mm256_mul_ps(vq, vq));
              }
            }
            for (; i < dim; ++i) {
              const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
              const double q = static_cast<double>(query[i]);
              const double y = static_cast<double>(yi);
              cross += q * y;
              sum_y2 += y * y;
              if (accum_q2) {
                sum_q2 += q * q;
              }
            }
            if (dt == VecDisType::COSINE) {
              ret = fused_finalize_cosine(cross, sum_q2, sum_y2, out_distance);
            } else {
              ret = fused_finalize_dot(cross, sum_y2, need_norm_candidate, out_distance);
            }
            break;
          }
          case VecDisType::EUCLIDEAN:
          case VecDisType::EUCLIDEAN_SQUARED:
          case VecDisType::MANHATTAN: {
            double acc = 0.0;
            for (; i < dim8; i += 8) {
              const __m256 vmin = _mm256_loadu_ps(meta_min + i);
              const __m256 vstep = _mm256_loadu_ps(meta_step + i);
              const __m256 abs_step = _mm256_and_ps(vstep, vsign_mask);
              int64_t packed64 = 0;
              std::memcpy(&packed64, codes + i, sizeof(packed64));
              const __m128i c8 = _mm_cvtsi64_si128(packed64);
              const __m256i ci = _mm256_cvtepu8_epi32(c8);
              const __m256 cf = _mm256_cvtepi32_ps(ci);
              const __m256 ch = _mm256_add_ps(cf, vhalf);
              const __m256 fmadd = _mm256_fmadd_ps(vstep, ch, vmin);
              const __m256 mask_small = _mm256_cmp_ps(abs_step, veps, _CMP_LT_OQ);
              const __m256 vy = _mm256_blendv_ps(fmadd, vmin, mask_small);
              const __m256 vq = _mm256_loadu_ps(query + i);
              if (dt == VecDisType::MANHATTAN) {
                const __m256 diff = _mm256_sub_ps(vq, vy);
                const __m256 absd = _mm256_and_ps(diff, vsign_mask);
                acc += horizontal_sum_m256_ps(absd);
              } else {
                const __m256 diff = _mm256_sub_ps(vq, vy);
                acc += horizontal_sum_m256_ps(_mm256_mul_ps(diff, diff));
              }
            }
            for (; i < dim; ++i) {
              const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
              const double q = static_cast<double>(query[i]);
              const double y = static_cast<double>(yi);
              if (dt == VecDisType::MANHATTAN) {
                acc += std::fabs(q - y);
              } else {
                const double d = q - y;
                acc += d * d;
              }
            }
            if (dt == VecDisType::EUCLIDEAN) {
              out_distance = std::sqrt(acc);
              if (std::isnan(out_distance)) {
                ret = OB_NUMERIC_OVERFLOW;
              }
            } else {
              out_distance = acc;
            }
            break;
          }
          default:
            ret = OB_INVALID_ARGUMENT;
            break;
        }
      } else {
        ret = oceanbase::common::specific::normal::ivf_sq8_latent_fused_distance_vs_query_impl(
            query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
      }
      return ret;
    }
)

OB_DECLARE_AVX512_SPECIFIC_CODE(
    OB_INLINE double horizontal_sum_m512_ps(const __m512 v)
    {
#if defined(__AVX512F__)
      return static_cast<double>(_mm512_reduce_add_ps(v));
#else
      const __m256 lo = _mm512_castps512_ps256(v);
      // Upper 8 floats via f64x4 extract (AVX512F); avoid _mm512_extractf32x8_ps (needs AVX512DQ).
      const __m256 hi =
          _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
      return oceanbase::common::specific::avx2::horizontal_sum_m256_ps(lo)
          + oceanbase::common::specific::avx2::horizontal_sum_m256_ps(hi);
#endif
    }

    int ivf_sq8_latent_fused_distance_vs_query_impl(
        const float *query,
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        const int vec_dis_type_int,
        const bool need_norm_candidate,
        double &out_distance)
    {
      int ret = OB_SUCCESS;
      out_distance = 0.0;
      if (OB_ISNULL(query) || OB_ISNULL(meta_min) || OB_ISNULL(meta_step) || OB_ISNULL(codes) || dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
      } else if (common::is_arch_supported(ObTargetArch::AVX512)) {
        const VecDisType dt = static_cast<VecDisType>(vec_dis_type_int);
        const __m512 veps = _mm512_set1_ps(1e-10f);
        const __m512 vhalf = _mm512_set1_ps(0.5f);
        const __m512i vsign_mask_i = _mm512_set1_epi32(0x7fffffff);
        int64_t i = 0;
        const int64_t dim16 = dim & ~static_cast<int64_t>(15);

        switch (dt) {
          case VecDisType::COSINE:
          case VecDisType::DOT: {
            double cross = 0.0;
            double sum_y2 = 0.0;
            double sum_q2 = 0.0;
            const bool accum_q2 = (dt == VecDisType::COSINE);
            for (; i < dim16; i += 16) {
              const __m512 vmin = _mm512_loadu_ps(meta_min + i);
              const __m512 vstep = _mm512_loadu_ps(meta_step + i);
              const __m512 abs_step = _mm512_castsi512_ps(
                  _mm512_and_si512(_mm512_castps_si512(vstep), vsign_mask_i));
              const __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codes + i));
              const __m512i ci = _mm512_cvtepu8_epi32(c16);
              const __m512 cf = _mm512_cvtepi32_ps(ci);
              const __m512 ch = _mm512_add_ps(cf, vhalf);
              const __m512 fmadd = _mm512_fmadd_ps(vstep, ch, vmin);
              const __mmask16 mask_small = _mm512_cmp_ps_mask(abs_step, veps, _CMP_LT_OQ);
              const __m512 vy = _mm512_mask_blend_ps(mask_small, fmadd, vmin);
              const __m512 vq = _mm512_loadu_ps(query + i);
              cross += horizontal_sum_m512_ps(_mm512_mul_ps(vq, vy));
              sum_y2 += horizontal_sum_m512_ps(_mm512_mul_ps(vy, vy));
              if (accum_q2) {
                sum_q2 += horizontal_sum_m512_ps(_mm512_mul_ps(vq, vq));
              }
            }
            for (; i < dim; ++i) {
              const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
              const double q = static_cast<double>(query[i]);
              const double y = static_cast<double>(yi);
              cross += q * y;
              sum_y2 += y * y;
              if (accum_q2) {
                sum_q2 += q * q;
              }
            }
            if (dt == VecDisType::COSINE) {
              ret = fused_finalize_cosine(cross, sum_q2, sum_y2, out_distance);
            } else {
              ret = fused_finalize_dot(cross, sum_y2, need_norm_candidate, out_distance);
            }
            break;
          }
          case VecDisType::EUCLIDEAN:
          case VecDisType::EUCLIDEAN_SQUARED:
          case VecDisType::MANHATTAN: {
            double acc = 0.0;
            for (; i < dim16; i += 16) {
              const __m512 vmin = _mm512_loadu_ps(meta_min + i);
              const __m512 vstep = _mm512_loadu_ps(meta_step + i);
              const __m512 abs_step = _mm512_castsi512_ps(
                  _mm512_and_si512(_mm512_castps_si512(vstep), vsign_mask_i));
              const __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codes + i));
              const __m512i ci = _mm512_cvtepu8_epi32(c16);
              const __m512 cf = _mm512_cvtepi32_ps(ci);
              const __m512 ch = _mm512_add_ps(cf, vhalf);
              const __m512 fmadd = _mm512_fmadd_ps(vstep, ch, vmin);
              const __mmask16 mask_small = _mm512_cmp_ps_mask(abs_step, veps, _CMP_LT_OQ);
              const __m512 vy = _mm512_mask_blend_ps(mask_small, fmadd, vmin);
              const __m512 vq = _mm512_loadu_ps(query + i);
              if (dt == VecDisType::MANHATTAN) {
                const __m512 diff = _mm512_sub_ps(vq, vy);
                const __m512 absd = _mm512_castsi512_ps(
                    _mm512_and_si512(_mm512_castps_si512(diff), vsign_mask_i));
                acc += horizontal_sum_m512_ps(absd);
              } else {
                const __m512 diff = _mm512_sub_ps(vq, vy);
                acc += horizontal_sum_m512_ps(_mm512_mul_ps(diff, diff));
              }
            }
            for (; i < dim; ++i) {
              const float yi = latent_decode_one(i, meta_min, meta_step, codes[i]);
              const double q = static_cast<double>(query[i]);
              const double y = static_cast<double>(yi);
              if (dt == VecDisType::MANHATTAN) {
                acc += std::fabs(q - y);
              } else {
                const double d = q - y;
                acc += d * d;
              }
            }
            if (dt == VecDisType::EUCLIDEAN) {
              out_distance = std::sqrt(acc);
              if (std::isnan(out_distance)) {
                ret = OB_NUMERIC_OVERFLOW;
              }
            } else {
              out_distance = acc;
            }
            break;
          }
          default:
            ret = OB_INVALID_ARGUMENT;
            break;
        }
      } else {
        ret = oceanbase::common::specific::avx2::ivf_sq8_latent_fused_distance_vs_query_impl(
            query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
      }
      return ret;
    }
)

#endif // x86_64

int ivf_sq8_latent_fused_distance_vs_query(
    const float *query,
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const uint8_t *codes,
    int vec_dis_type_int,
    bool need_norm_candidate,
    double &out_distance)
{
#if OB_USE_MULTITARGET_CODE
  if (oceanbase::common::is_arch_supported(oceanbase::common::ObTargetArch::AVX512)) {
    return oceanbase::common::specific::avx512::ivf_sq8_latent_fused_distance_vs_query_impl(
        query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
  }
  if (oceanbase::common::is_arch_supported(oceanbase::common::ObTargetArch::AVX2)) {
    return oceanbase::common::specific::avx2::ivf_sq8_latent_fused_distance_vs_query_impl(
        query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
  }
#endif
  return oceanbase::common::specific::normal::ivf_sq8_latent_fused_distance_vs_query_impl(
      query, dim, meta_min, meta_step, codes, vec_dis_type_int, need_norm_candidate, out_distance);
}

} // namespace common
} // namespace oceanbase
