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

#define USING_LOG_PREFIX SQL_DAS

#include "sql/das/iter/ob_das_ivf_per_query_stats.h"
#include "share/vector_index/ob_ivf_cid_cluster_cache.h"
#include "lib/utility/ob_macro_utils.h"
#include "lib/time/ob_time_utility.h"

namespace oceanbase
{
namespace sql
{

namespace {

constexpr int64_t IVF_PER_QUERY_STATS_IDLE_NEW_ROUND_GAP_US = 60LL * 1000000LL;
static int64_t g_ivf_per_query_stats_last_scan_us = 0;
static int64_t g_ivf_per_query_stats_owner_tid = 0;

thread_local bool tls_ivf_per_query_stats_emit = false;

OB_INLINE void ob_ivf_per_query_stats_on_scan_start()
{
  const int64_t now_us = ObTimeUtility::current_time();
  const int64_t prev_us = ATOMIC_LOAD(&g_ivf_per_query_stats_last_scan_us);
  if (prev_us != 0LL && (now_us - prev_us) > IVF_PER_QUERY_STATS_IDLE_NEW_ROUND_GAP_US) {
    ATOMIC_STORE(&g_ivf_per_query_stats_owner_tid, 0);
  }
  ATOMIC_STORE(&g_ivf_per_query_stats_last_scan_us, now_us);
}

/// Default 0: emit every query on the owner thread. <=0 means every query.
constexpr int64_t IVF_PER_QUERY_STATS_SAMPLE_EVERY_N_DEFAULT = 0;

OB_INLINE int64_t ob_ivf_per_query_stats_sample_every_n_env()
{
  const char *const e = ::getenv("OB_IVF_LATENCY_BREAKDOWN_SAMPLE_EVERY_N");
  if (e == nullptr || e[0] == '\0') {
    return IVF_PER_QUERY_STATS_SAMPLE_EVERY_N_DEFAULT;
  }
  char *endptr = nullptr;
  const long long parsed = std::strtoll(e, &endptr, 10);
  if (endptr == e) {
    return IVF_PER_QUERY_STATS_SAMPLE_EVERY_N_DEFAULT;
  }
  return static_cast<int64_t>(parsed);
}

OB_INLINE bool ob_ivf_per_query_stats_should_sample_for_dataset(
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist_centers,
    const int64_t every_n)
{
  thread_local bool tls_ds_inited = false;
  thread_local int64_t tls_ds_rows = 0;
  thread_local int64_t tls_ds_dim = 0;
  thread_local int64_t tls_ds_nlist = 0;
  thread_local int64_t tls_query_seq = 0;

  if (!tls_ds_inited || dataset_rows != tls_ds_rows || dim != tls_ds_dim || nlist_centers != tls_ds_nlist) {
    tls_ds_inited = true;
    tls_ds_rows = dataset_rows;
    tls_ds_dim = dim;
    tls_ds_nlist = nlist_centers;
    tls_query_seq = 0;
  }
  ++tls_query_seq;
  if (every_n <= 0) {
    return true;
  }
  return ((tls_query_seq - 1) % every_n) == 0;
}

} // namespace

bool ob_ivf_latency_breakdown_enabled()
{
  const char *const e = ::getenv("OB_IVF_LATENCY_BREAKDOWN");
  return e != nullptr && e[0] == '1';
}

bool ob_ivf_per_query_stats_any_enabled()
{
  return ob_ivf_latency_breakdown_enabled() || share::is_ivf_cid_cluster_cache_stats_enabled();
}

bool ob_ivf_per_query_stats_begin_query(
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist_centers)
{
  tls_ivf_per_query_stats_emit = false;
  if (!ob_ivf_per_query_stats_any_enabled()) {
    return false;
  }
  ob_ivf_per_query_stats_on_scan_start();
  const int64_t tid = GETTID();
  int64_t owner_tid = ATOMIC_LOAD(&g_ivf_per_query_stats_owner_tid);
  if (owner_tid == 0) {
    if (ATOMIC_BCAS(&g_ivf_per_query_stats_owner_tid, 0, tid)) {
      owner_tid = tid;
    } else {
      owner_tid = ATOMIC_LOAD(&g_ivf_per_query_stats_owner_tid);
    }
  }
  if (owner_tid != tid) {
    return false;
  }
  const int64_t every_n = ob_ivf_per_query_stats_sample_every_n_env();
  if (!ob_ivf_per_query_stats_should_sample_for_dataset(dataset_rows, dim, nlist_centers, every_n)) {
    return false;
  }
  tls_ivf_per_query_stats_emit = true;
  return true;
}

bool ob_ivf_per_query_stats_emit_this_query()
{
  return tls_ivf_per_query_stats_emit;
}

} // namespace sql
} // namespace oceanbase
