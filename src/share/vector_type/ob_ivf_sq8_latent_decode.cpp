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
 */

#include "ob_ivf_sq8_latent_decode.h"

namespace oceanbase
{
namespace common
{

void ivf_sq8_legacy_bin_center_u8_decode(
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const uint8_t *codes,
    float *decoded)
{
#if OB_USE_MULTITARGET_CODE
  if (common::is_arch_supported(ObTargetArch::AVX512)) {
    common::specific::avx512::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
  } else if (common::is_arch_supported(ObTargetArch::AVX2)) {
    common::specific::avx2::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
  } else {
    common::specific::normal::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
  }
#elif defined(__aarch64__)
  common::specific::normal::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
#else
  common::specific::normal::ivf_sq8_latent_decode(dim, meta_min, meta_step, codes, decoded);
#endif
}

} // namespace common
} // namespace oceanbase
