/**
 * Copyright (c) 2026 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 *
 * IVF_SQ8 latent dequant: min + step * (u8 + 0.5), or min if |step| < eps.
 * Multitarget dispatch mirrors ob_vector_l2_distance.cpp (AVX512 / AVX2 / normal).
 */

#ifndef OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_DECODE_H_
#define OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_DECODE_H_

#include "ob_vector_op_common.h"
#include <cstdint>
#include <cmath>
#include <cstring>

namespace oceanbase
{
namespace common
{

OB_DECLARE_DEFAULT_CODE(
    inline void ivf_sq8_latent_decode(
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        float *decoded)
    {
      const float epsilon = static_cast<float>(1e-10);
      for (int64_t i = 0; i < dim; ++i) {
        if (fabsf(meta_step[i]) < epsilon) {
          decoded[i] = meta_min[i];
        } else {
          decoded[i] = meta_min[i] + meta_step[i] * (static_cast<float>(codes[i]) + static_cast<float>(0.5));
        }
      }
    }
)

OB_DECLARE_AVX2_SPECIFIC_CODE(
    inline void ivf_sq8_latent_decode(
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        float *decoded)
    {
      if (common::is_arch_supported(ObTargetArch::AVX2)) {
        const __m256 veps = _mm256_set1_ps(1e-10f);
        const __m256 vhalf = _mm256_set1_ps(0.5f);
        const __m256 vsign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        int64_t i = 0;
        const int64_t dim8 = dim & ~static_cast<int64_t>(7);
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
          const __m256 out = _mm256_blendv_ps(fmadd, vmin, mask_small);
          _mm256_storeu_ps(decoded + i, out);
        }
        common::specific::normal::ivf_sq8_latent_decode(
            dim - i, meta_min + i, meta_step + i, codes + i, decoded + i);
      } else {
        common::specific::normal::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
      }
    }
)

OB_DECLARE_AVX512_SPECIFIC_CODE(
    inline void ivf_sq8_latent_decode(
        const int64_t dim,
        const float *meta_min,
        const float *meta_step,
        const uint8_t *codes,
        float *decoded)
    {
      if (common::is_arch_supported(ObTargetArch::AVX512)) {
        const __m512 veps = _mm512_set1_ps(1e-10f);
        const __m512 vhalf = _mm512_set1_ps(0.5f);
        // Use integer AND to avoid requiring extra AVX-512 subsets on some toolchains.
        const __m512i vsign_mask_i = _mm512_set1_epi32(0x7fffffff);
        int64_t i = 0;
        const int64_t dim16 = dim & ~static_cast<int64_t>(15);
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
          const __m512 out = _mm512_mask_blend_ps(mask_small, fmadd, vmin);
          _mm512_storeu_ps(decoded + i, out);
        }
        const int64_t rem = dim - i;
        if (rem >= 8) {
          common::specific::avx2::ivf_sq8_latent_decode(
              rem, meta_min + i, meta_step + i, codes + i, decoded + i);
        } else if (rem > 0) {
          common::specific::normal::ivf_sq8_latent_decode(
              rem, meta_min + i, meta_step + i, codes + i, decoded + i);
        }
      } else {
        common::specific::avx2::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
      }
    }
)

void ivf_sq8_legacy_bin_center_u8_decode(
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const uint8_t *codes,
    float *decoded);

} // namespace common
} // namespace oceanbase

#endif // OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_DECODE_H_
