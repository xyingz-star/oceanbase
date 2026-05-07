/**
 * Copyright (c) 2026 OceanBase
 * OceanBase CE is licensed under the Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 *
 * Fused latent IVF_SQ8: stream dequant (same rule as ob_ivf_sq8_latent_decode.h) and
 * accumulate distance vs float query without staging float[dim].
 *
 * vec_dis_type_int matches sql::ObExprVectorDistance::ObVecDisType:
 *   0=COSINE, 1=DOT, 2=EUCLIDEAN, 3=MANHATTAN, 4=EUCLIDEAN_SQUARED
 */

#ifndef OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_FUSED_DISTANCE_H_
#define OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_FUSED_DISTANCE_H_

#include <cstdint>

#include "lib/ob_errno.h"

namespace oceanbase
{
namespace common
{

/// `need_norm_candidate`: IVF rowkey path sets true when stored cid vectors must be L2-normalized
/// for cosine/dot (same meaning as `need_norm_` on ObDASIvfScanIter). Implies DOT is computed as
/// q·y/||y|| when q is already unit (post-cosine-query rewrite to inner product).
int ivf_sq8_latent_fused_distance_vs_query(
    const float *query,
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const uint8_t *codes,
    int vec_dis_type_int,
    bool need_norm_candidate,
    double &out_distance);

} // namespace common
} // namespace oceanbase

#endif // OCEANBASE_SHARE_VECTOR_TYPE_OB_IVF_SQ8_LATENT_FUSED_DISTANCE_H_
