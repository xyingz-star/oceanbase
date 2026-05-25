/**
 * Copyright (c) 2021 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You can use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#ifndef OCEANBASE_SQL_DAS_ITER_OB_DAS_IVF_PER_QUERY_STATS_H_
#define OCEANBASE_SQL_DAS_ITER_OB_DAS_IVF_PER_QUERY_STATS_H_

#include "lib/utility/ob_macro_utils.h"

namespace oceanbase
{
namespace sql
{

/// Per-query IVF stats: one pinned owner worker thread records latency breakdown + cache session
/// stats at query granularity (no per-CID progress spam on other threads).
/// Enable: OB_IVF_LATENCY_BREAKDOWN=1 and/or OB_IVF_CID_CLUSTER_CACHE_STATS=1 (default on).
bool ob_ivf_latency_breakdown_enabled();
bool ob_ivf_per_query_stats_any_enabled();

/// Call at the start of each IVF query (process_ivf_scan). Pins owner thread, decides emit for this query.
/// Returns true iff this thread should record latency breakdown timers and cache session logs for this query.
bool ob_ivf_per_query_stats_begin_query(const int64_t dataset_rows, const int64_t dim, const int64_t nlist_centers);

/// Whether the current query on this thread was selected for per-query logging (set by begin_query).
bool ob_ivf_per_query_stats_emit_this_query();

} // namespace sql
} // namespace oceanbase

#endif /* OCEANBASE_SQL_DAS_ITER_OB_DAS_IVF_PER_QUERY_STATS_H_ */
