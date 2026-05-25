/**
 * Copyright (c) 2021 OceanBase
 * OceanBase CE is licensed under Mulan PubL v2.
 * You may use this software according to the terms and conditions of the Mulan PubL v2.
 * You may obtain a copy of Mulan PubL v2 at:
 *          http://license.coscl.org.cn/MulanPubL-2.0
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PubL v2 for more details.
 */

#define USING_LOG_PREFIX SHARE

#include "share/vector_index/ob_ivf_cid_cluster_cache.h"
#include "share/vector_index/ob_ivf_cid_cluster_kv_cache.h"
#include "lib/container/ob_array.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/atomic/ob_atomic.h"
#include "lib/file/file_directory_utils.h"
#include "lib/lock/ob_thread_cond.h"
#include "lib/time/ob_time_utility.h"
#include "lib/utility/ob_macro_utils.h"
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace oceanbase
{
namespace share
{

using namespace common;

static const int64_t IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES = 1024;
static const int64_t DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_MB = 128;
static const int64_t DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT = 2;
/// Min probe access_cnt on a cid before a miss may become FILL leader (default 5: early touches STORAGE_ONLY).
static const int64_t DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS = 5;
/// Default cap on how many distinct cids may compete for FILL (aligns with ~128MB / ~5MB per cluster).
static const int64_t DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_TOP_K = 24;
/// Rough bytes per full 50K/56 IVF_FLAT 1536-dim cluster for auto top-K sizing.
static const int64_t IVF_CID_CLUSTER_CACHE_EST_BYTES_PER_CLUSTER = 5 * 1024 * 1024;

int64_t ivf_cid_cluster_cache_fill_min_probe_access()
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS");
  if (env == nullptr || env[0] == '\0') {
    return DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS;
  }
  const int64_t v = static_cast<int64_t>(atoll(env));
  return v > 0 ? v : DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS;
}

int64_t ivf_cid_cluster_cache_fill_top_k(const int64_t max_bytes)
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_FILL_TOP_K");
  if (env != nullptr && env[0] != '\0') {
    const int64_t v = static_cast<int64_t>(atoll(env));
    if (v > 0) {
      return v;
    }
  }
  int64_t k = DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_TOP_K;
  if (max_bytes > 0) {
    const int64_t by_cap = max_bytes / IVF_CID_CLUSTER_CACHE_EST_BYTES_PER_CLUSTER;
    if (by_cap > 0) {
      k = by_cap;
    }
  }
  return OB_MAX(1, k);
}
static const ObMemAttr IVF_CID_CLUSTER_CACHE_LABEL(OB_SERVER_TENANT_ID, "IvfCidClu");

typedef common::hash::ObHashMap<ObIvfCidClusterCacheMgrKey, ObIvfCidClusterCache *> IvfCidClusterCacheMgrMap;
static IvfCidClusterCacheMgrMap g_ivf_cid_cluster_cache_mgr_map;
static lib::ObMutex g_ivf_cid_cluster_cache_mgr_lock(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK);
static bool g_ivf_cid_cluster_cache_mgr_inited = false;

ObIvfCidClusterEntry::~ObIvfCidClusterEntry()
{
  rows_.reset();
  if (OB_NOT_NULL(arena_)) {
    arena_->~ObArenaAllocator();
    ob_free(arena_);
    arena_ = nullptr;
  }
  if (session_owned_ && OB_NOT_NULL(rowkey_objs_)) {
    ob_free(rowkey_objs_);
    rowkey_objs_ = nullptr;
  }
  session_kv_handle_.reset();
  kv_flat_buf_ = nullptr;
}

bool is_ivf_cid_cluster_cache_enabled()
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_ENABLED");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return !(env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F');
}

bool is_ivf_cid_cluster_cache_stats_enabled()
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_STATS");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return !(env[0] == '0' || env[0] == 'n' || env[0] == 'N' || env[0] == 'f' || env[0] == 'F');
}

int64_t ivf_cid_cluster_cache_stats_every_n_cid()
{
  if (!is_ivf_cid_cluster_cache_stats_enabled()) {
    return 0;
  }
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_STATS_EVERY_N_CID");
  if (env != nullptr && env[0] != '\0') {
    const int64_t v = static_cast<int64_t>(atoll(env));
    return v >= 0 ? v : 1;
  }
  const char *live = getenv("OB_IVF_CID_CLUSTER_CACHE_STATS_LIVE");
  if (live != nullptr && (live[0] == '0' || live[0] == 'n' || live[0] == 'N' || live[0] == 'f' || live[0] == 'F')) {
    // Explicit off: progress disabled; session/final only (faster for benchmarks).
    return 0;
  }
  if (live != nullptr && (live[0] == '1' || live[0] == 'y' || live[0] == 'Y' || live[0] == 't' || live[0] == 'T')) {
    return 1;
  }
  // Default with STATS=1: per-query final on owner thread only; no per-CID progress spam.
  return 0;
}

void ObIvfCidClusterCacheStats::reset()
{
  get_hit_cnt_ = 0;
  get_miss_cnt_ = 0;
  get_stale_epoch_cnt_ = 0;
  put_ok_cnt_ = 0;
  put_fail_cnt_ = 0;
  put_skip_rows_limit_cnt_ = 0;
  put_skip_pinned_cnt_ = 0;
  put_skip_cap_cnt_ = 0;
  fill_wait_cnt_ = 0;
  fill_bypass_cnt_ = 0;
  evict_entry_cnt_ = 0;
  evict_bytes_ = 0;
  invalidate_cnt_ = 0;
  phase_learning_cnt_ = 0;
  phase_refreshing_cnt_ = 0;
  phase_serving_cnt_ = 0;
  cur_entry_cnt_ = 0;
  cur_bytes_ = 0;
}

void ObIvfCidClusterCacheSessionStats::reset()
{
  fill_cid_cnt_ = 0;
  replay_cid_cnt_ = 0;
  replay_row_cnt_ = 0;
  fill_row_cnt_ = 0;
  put_cid_ok_cnt_ = 0;
  put_cid_fail_cnt_ = 0;
  put_cid_skip_pinned_cnt_ = 0;
  put_rows_total_ = 0;
  storage_only_cid_cnt_ = 0;
  cid_switch_cnt_ = 0;
  acquire_us_ = 0;
  fill_wait_us_ = 0;
  fill_wait_cnt_ = 0;
  storage_fetch_us_ = 0;
  replay_serve_us_ = 0;
  fill_append_us_ = 0;
  has_cache_snap_begin_ = false;
  cache_snap_begin_.reset();
}

void ObIvfCidClusterCacheLogSnapshot::reset()
{
  valid_ = false;
  cache_active_ = false;
  index_epoch_ = 0;
  algorithm_type_ = 0;
  session_.reset();
  cluster_cache_ = nullptr;
}

namespace {

int64_t ob_ivf_cid_cluster_cache_query_cache_hit(const ObIvfCidClusterCacheSessionStats &session)
{
  return session.replay_cid_cnt_ > 0 ? 1 : 0;
}

double ob_ivf_cid_cluster_cache_cid_hit_rate(const ObIvfCidClusterCacheSessionStats &session)
{
  const int64_t probe = session.cid_switch_cnt_ > 0
      ? session.cid_switch_cnt_
      : (session.fill_cid_cnt_ + session.replay_cid_cnt_ + session.storage_only_cid_cnt_);
  return probe > 0 ? static_cast<double>(session.replay_cid_cnt_) / static_cast<double>(probe) : 0.0;
}

double ob_ivf_cid_cluster_cache_row_replay_rate(const ObIvfCidClusterCacheSessionStats &session)
{
  const int64_t total = session.fill_row_cnt_ + session.replay_row_cnt_;
  return total > 0 ? static_cast<double>(session.replay_row_cnt_) / static_cast<double>(total) : 0.0;
}

static int ob_ivf_cid_cluster_cache_create_log_dir(const char *dir_path)
{
  if (OB_ISNULL(dir_path) || dir_path[0] == '\0') {
    return OB_INVALID_ARGUMENT;
  }
  return common::FileDirectoryUtils::create_full_path(dir_path);
}

static void ob_ivf_cid_cluster_cache_format_stats_line(
    char *line,
    const int line_cap,
    const ObIvfCidClusterCacheLogSnapshot &snap,
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist,
    const ObIvfCidClusterCacheStats *cache_end,
    const ObIvfCidClusterCacheStats *cache_begin,
    const char *phase,
    const uint64_t current_cid,
    const char *current_mode)
{
  if (OB_ISNULL(line) || line_cap <= 0) {
    return;
  }
  const int64_t tid = GETTID();
  const int64_t ts_us = ObTimeUtility::current_time();
  if (!snap.valid_) {
    (void)snprintf(line,
        line_cap,
        "[OB_IVF_CID_CLUSTER_CACHE_STATS] ts_us=%lld tid=%lld cache_active=0\n",
        static_cast<long long>(ts_us),
        static_cast<long long>(tid));
    return;
  }
  const ObIvfCidClusterCacheSessionStats &s = snap.session_;
  const int64_t cid_probe = s.cid_switch_cnt_ > 0
      ? s.cid_switch_cnt_
      : (s.fill_cid_cnt_ + s.replay_cid_cnt_ + s.storage_only_cid_cnt_);
  const double cid_hit_rate = ob_ivf_cid_cluster_cache_cid_hit_rate(s);
  const double row_replay_rate = ob_ivf_cid_cluster_cache_row_replay_rate(s);
  int64_t d_fill_wait = 0;
  int64_t d_fill_bypass = 0;
  int64_t d_get_hit = 0;
  int64_t d_get_miss = 0;
  int64_t d_put_ok = 0;
  int64_t d_evict_entry = 0;
  if (OB_NOT_NULL(cache_end) && OB_NOT_NULL(cache_begin)) {
    d_fill_wait = cache_end->fill_wait_cnt_ - cache_begin->fill_wait_cnt_;
    d_fill_bypass = cache_end->fill_bypass_cnt_ - cache_begin->fill_bypass_cnt_;
    d_get_hit = cache_end->get_hit_cnt_ - cache_begin->get_hit_cnt_;
    d_get_miss = cache_end->get_miss_cnt_ - cache_begin->get_miss_cnt_;
    d_put_ok = cache_end->put_ok_cnt_ - cache_begin->put_ok_cnt_;
    d_evict_entry = cache_end->evict_entry_cnt_ - cache_begin->evict_entry_cnt_;
  }
  const char *phase_str = (phase != nullptr && phase[0] != '\0') ? phase : "final";
  const char *mode_str = (current_mode != nullptr && current_mode[0] != '\0') ? current_mode : "-";
  (void)snprintf(line,
      line_cap,
      "[OB_IVF_CID_CLUSTER_CACHE_STATS] phase=%s current_cid=%llu current_mode=%s "
      "ts_us=%lld tid=%lld n=%lld d=%lld c=%lld "
      "index_epoch=%llu algo=%lld cid_switch_cnt=%lld storage_only_cid_cnt=%lld replay_cid_cnt=%lld "
      "fill_cid_cnt=%lld cid_hit_rate=%f storage_fetch_us=%lld replay_serve_us=%lld fill_append_us=%lld "
      "acquire_us=%lld fill_wait_us=%lld fill_wait_cnt=%lld fill_row_cnt=%lld replay_row_cnt=%lld "
      "row_replay_rate=%f put_cid_ok=%lld put_cid_fail=%lld put_skip_pinned=%lld put_rows=%lld "
      "cache_d_fill_wait=%lld cache_d_fill_bypass=%lld cache_d_get_hit=%lld cache_d_get_miss=%lld "
      "cache_d_put_ok=%lld cache_d_evict_entry=%lld cur_entry_cnt=%lld cur_bytes=%lld max_bytes=%lld\n",
      phase_str,
      static_cast<unsigned long long>(current_cid),
      mode_str,
      static_cast<long long>(ts_us),
      static_cast<long long>(tid),
      static_cast<long long>(dataset_rows),
      static_cast<long long>(dim),
      static_cast<long long>(nlist),
      static_cast<unsigned long long>(snap.index_epoch_),
      static_cast<long long>(snap.algorithm_type_),
      static_cast<long long>(s.cid_switch_cnt_),
      static_cast<long long>(s.storage_only_cid_cnt_),
      static_cast<long long>(s.replay_cid_cnt_),
      static_cast<long long>(s.fill_cid_cnt_),
      cid_hit_rate,
      static_cast<long long>(s.storage_fetch_us_),
      static_cast<long long>(s.replay_serve_us_),
      static_cast<long long>(s.fill_append_us_),
      static_cast<long long>(s.acquire_us_),
      static_cast<long long>(s.fill_wait_us_),
      static_cast<long long>(s.fill_wait_cnt_),
      static_cast<long long>(s.fill_row_cnt_),
      static_cast<long long>(s.replay_row_cnt_),
      row_replay_rate,
      static_cast<long long>(s.put_cid_ok_cnt_),
      static_cast<long long>(s.put_cid_fail_cnt_),
      static_cast<long long>(s.put_cid_skip_pinned_cnt_),
      static_cast<long long>(s.put_rows_total_),
      static_cast<long long>(d_fill_wait),
      static_cast<long long>(d_fill_bypass),
      static_cast<long long>(d_get_hit),
      static_cast<long long>(d_get_miss),
      static_cast<long long>(d_put_ok),
      static_cast<long long>(d_evict_entry),
      OB_NOT_NULL(cache_end) ? static_cast<long long>(cache_end->cur_entry_cnt_) : 0LL,
      OB_NOT_NULL(cache_end) ? static_cast<long long>(cache_end->cur_bytes_) : 0LL,
      OB_NOT_NULL(snap.cluster_cache_) ? static_cast<long long>(snap.cluster_cache_->get_max_bytes()) : 0LL);
}

} // namespace

void ob_ivf_cid_cluster_cache_write_user_log_file(
    const ObIvfCidClusterCacheLogSnapshot &snap,
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist,
    const char *phase,
    const uint64_t current_cid,
    const char *current_mode)
{
  if (!is_ivf_cid_cluster_cache_stats_enabled()) {
    return;
  }
  char path_buf[common::FileDirectoryUtils::MAX_PATH + 1];
  path_buf[0] = '\0';
  const char *env_file = getenv("OB_IVF_CID_CLUSTER_CACHE_LOG_FILE");
  const int64_t tid = GETTID();
  if (env_file != nullptr && env_file[0] != '\0') {
    const int nf = snprintf(path_buf, sizeof(path_buf), "%s", env_file);
    if (nf <= 0 || nf >= static_cast<int>(sizeof(path_buf))) {
      return;
    }
  } else {
    const char *env_dir = getenv("OB_IVF_CID_CLUSTER_CACHE_LOG_DIR");
    const char *home = getenv("HOME");
    const char *base_dir = (env_dir != nullptr && env_dir[0] != '\0')
        ? env_dir
        : ((home != nullptr && home[0] != '\0') ? nullptr : nullptr);
    if (base_dir != nullptr) {
      (void)ob_ivf_cid_cluster_cache_create_log_dir(base_dir);
      const int nf = snprintf(path_buf,
          sizeof(path_buf),
          "%s/ob_ivf_cid_cluster_cache.n%lld_d%lld_c%lld.%lld.log",
          base_dir,
          static_cast<long long>(dataset_rows),
          static_cast<long long>(dim),
          static_cast<long long>(nlist),
          static_cast<long long>(tid));
      if (nf <= 0 || nf >= static_cast<int>(sizeof(path_buf))) {
        return;
      }
    } else if (home != nullptr && home[0] != '\0') {
      char dir_buf[common::FileDirectoryUtils::MAX_PATH + 1];
      const int nd = snprintf(dir_buf, sizeof(dir_buf), "%s/log", home);
      if (nd > 0 && nd < static_cast<int>(sizeof(dir_buf))) {
        (void)ob_ivf_cid_cluster_cache_create_log_dir(dir_buf);
      }
      const int nf = snprintf(path_buf,
          sizeof(path_buf),
          "%s/log/ob_ivf_cid_cluster_cache.n%lld_d%lld_c%lld.%lld.log",
          home,
          static_cast<long long>(dataset_rows),
          static_cast<long long>(dim),
          static_cast<long long>(nlist),
          static_cast<long long>(tid));
      if (nf <= 0 || nf >= static_cast<int>(sizeof(path_buf))) {
        return;
      }
    } else {
      return;
    }
  }
  FILE *fp = fopen(path_buf, "ae");
  if (OB_ISNULL(fp)) {
    fp = fopen(path_buf, "a");
  }
  if (OB_ISNULL(fp)) {
    int ret = OB_IO_ERROR;
    LOG_WARN("failed to open ivf cid cluster cache stats log", K(ret), K(path_buf));
    return;
  }
  ObIvfCidClusterCacheStats cache_end;
  ObIvfCidClusterCacheStats cache_begin;
  cache_end.reset();
  cache_begin.reset();
  const ObIvfCidClusterCacheStats *p_begin = nullptr;
  const ObIvfCidClusterCacheStats *p_end = nullptr;
  if (snap.valid_ && snap.session_.has_cache_snap_begin_) {
    cache_begin = snap.session_.cache_snap_begin_;
    p_begin = &cache_begin;
  }
  if (OB_NOT_NULL(snap.cluster_cache_)) {
    snap.cluster_cache_->get_stats(cache_end);
    p_end = &cache_end;
  }
  char line[4096];
  ob_ivf_cid_cluster_cache_format_stats_line(
      line, static_cast<int>(sizeof(line)), snap, dataset_rows, dim, nlist, p_end, p_begin, phase, current_cid, current_mode);
  const size_t nl = strlen(line);
  if (nl > 0) {
    (void)fwrite(line, 1, nl, fp);
  }
  (void)fflush(fp);
  (void)fclose(fp);
}

void log_ivf_cid_cluster_cache_snapshot_to_observer(const ObIvfCidClusterCacheLogSnapshot &snap)
{
  if (!is_ivf_cid_cluster_cache_stats_enabled()) {
    return;
  }
  if (!snap.valid_) {
    LOG_INFO("[OB_IVF_CID_CLUSTER_CACHE_STATS] session",
             "cache_iter", false,
             "cache_active", false,
             "query_cache_hit", 0);
    return;
  }
  const ObIvfCidClusterCacheSessionStats &s = snap.session_;
  const int64_t query_cache_hit = ob_ivf_cid_cluster_cache_query_cache_hit(s);
  LOG_INFO("[OB_IVF_CID_CLUSTER_CACHE_STATS] session",
           K(snap.index_epoch_),
           K(snap.algorithm_type_),
           K(snap.cache_active_),
           K(query_cache_hit),
           K(s.cid_switch_cnt_),
           K(s.storage_only_cid_cnt_),
           K(s.fill_cid_cnt_),
           K(s.replay_cid_cnt_),
           "cid_hit_rate", ob_ivf_cid_cluster_cache_cid_hit_rate(s),
           K(s.storage_fetch_us_),
           K(s.replay_serve_us_),
           K(s.fill_append_us_),
           K(s.acquire_us_),
           K(s.fill_wait_us_),
           K(s.fill_wait_cnt_),
           K(s.fill_row_cnt_),
           K(s.replay_row_cnt_),
           "row_replay_rate", ob_ivf_cid_cluster_cache_row_replay_rate(s),
           K(s.put_cid_ok_cnt_),
           K(s.put_cid_fail_cnt_),
           K(s.put_cid_skip_pinned_cnt_),
           K(s.put_rows_total_));
  if (OB_NOT_NULL(snap.cluster_cache_)) {
    ObIvfCidClusterCacheStats cs;
    cs.reset();
    snap.cluster_cache_->get_stats(cs);
    const int64_t get_total = cs.get_hit_cnt_ + cs.get_miss_cnt_;
    const double get_hit_rate = get_total > 0 ? static_cast<double>(cs.get_hit_cnt_) / static_cast<double>(get_total) : 0.0;
    LOG_INFO("[OB_IVF_CID_CLUSTER_CACHE_STATS] cache_cumulative",
             "tag", "with_latency",
             K(cs.get_hit_cnt_),
             K(cs.get_miss_cnt_),
             "get_hit_rate", get_hit_rate,
             K(cs.fill_wait_cnt_),
             K(cs.fill_bypass_cnt_),
             K(cs.put_ok_cnt_),
             K(cs.put_fail_cnt_),
             K(cs.put_skip_pinned_cnt_),
             K(cs.put_skip_cap_cnt_),
             K(cs.evict_entry_cnt_),
             K(cs.evict_bytes_),
             K(cs.cur_entry_cnt_),
             K(cs.cur_bytes_));
  }
}

int ob_ivf_cid_cluster_cache_append_latency_log_file(FILE *fp, const ObIvfCidClusterCacheLogSnapshot &snap)
{
  if (OB_ISNULL(fp)) {
    return 0;
  }
  ObIvfCidClusterCacheStats cache_end;
  ObIvfCidClusterCacheStats cache_begin;
  cache_end.reset();
  cache_begin.reset();
  const ObIvfCidClusterCacheStats *p_begin = nullptr;
  const ObIvfCidClusterCacheStats *p_end = nullptr;
  if (snap.valid_ && snap.session_.has_cache_snap_begin_) {
    cache_begin = snap.session_.cache_snap_begin_;
    p_begin = &cache_begin;
  }
  if (OB_NOT_NULL(snap.cluster_cache_)) {
    snap.cluster_cache_->get_stats(cache_end);
    p_end = &cache_end;
  }
  char line[4096];
  ob_ivf_cid_cluster_cache_format_stats_line(
      line, static_cast<int>(sizeof(line)), snap, 0, 0, 0, p_end, p_begin, "final", 0, nullptr);
  const size_t nl = strlen(line);
  if (nl > 0) {
    (void)::fwrite(line, 1, nl, fp);
  }
  return static_cast<int>(nl);
}

void log_ivf_cid_cluster_cache_session_stats(
    const ObIvfCidClusterCacheSessionStats &session,
    const uint64_t index_epoch,
    const int64_t algorithm_type,
    const ObIvfCidClusterCache *cache)
{
  if (!is_ivf_cid_cluster_cache_stats_enabled()) {
    return;
  }
  ObIvfCidClusterCacheLogSnapshot snap;
  snap.valid_ = true;
  snap.cache_active_ = true;
  snap.index_epoch_ = index_epoch;
  snap.algorithm_type_ = algorithm_type;
  snap.session_ = session;
  snap.cluster_cache_ = const_cast<ObIvfCidClusterCache *>(cache);
  log_ivf_cid_cluster_cache_snapshot_to_observer(snap);
  ob_ivf_cid_cluster_cache_write_user_log_file(snap, 0, 0, 0);
}

uint64_t ObIvfCidClusterCacheMgrKey::hash() const
{
  uint64_t h = murmurhash(&tenant_id_, sizeof(tenant_id_), 0);
  h = murmurhash(&index_tablet_id_, sizeof(index_tablet_id_), h);
  h = murmurhash(&cid_vec_table_id_, sizeof(cid_vec_table_id_), h);
  h = murmurhash(&data_table_id_, sizeof(data_table_id_), h);
  h = murmurhash(&algorithm_type_, sizeof(algorithm_type_), h);
  return h;
}

bool ObIvfCidClusterCacheMgrKey::operator==(const ObIvfCidClusterCacheMgrKey &o) const
{
  return tenant_id_ == o.tenant_id_
      && index_tablet_id_ == o.index_tablet_id_
      && cid_vec_table_id_ == o.cid_vec_table_id_
      && data_table_id_ == o.data_table_id_
      && algorithm_type_ == o.algorithm_type_;
}

static int ensure_cache_mgr_map_inited()
{
  int ret = OB_SUCCESS;
  lib::ObMutexGuard guard(g_ivf_cid_cluster_cache_mgr_lock);
  if (!g_ivf_cid_cluster_cache_mgr_inited) {
    if (OB_FAIL(g_ivf_cid_cluster_cache_mgr_map.create(64, "IvfCidCluMap", "IvfCidCluMap", OB_SERVER_TENANT_ID))) {
      LOG_WARN("failed to create ivf cid cluster cache mgr map", K(ret));
    } else {
      g_ivf_cid_cluster_cache_mgr_inited = true;
    }
  }
  return ret;
}

struct ObIvfCidFillGate
{
  explicit ObIvfCidFillGate(uint64_t cid) : cid_(cid), filling_(false), ref_cnt_(1), cond_() {}
  void inc_ref() { ATOMIC_INC(&ref_cnt_); }
  void dec_ref()
  {
    if (0 == ATOMIC_SAF(&ref_cnt_, 1)) {
      cond_.destroy();
      ob_free(this);
    }
  }
  uint64_t cid_;
  bool filling_;
  int64_t ref_cnt_;
  common::ObThreadCond cond_;
};

struct FillGateWakeAndFreeFn
{
  int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidFillGate *> &pair)
  {
    ObIvfCidFillGate *gate = pair.second;
    if (OB_NOT_NULL(gate)) {
      if (OB_SUCCESS == gate->cond_.lock()) {
        gate->filling_ = false;
        (void)gate->cond_.broadcast();
        (void)gate->cond_.unlock();
      }
      gate->dec_ref();
    }
    return OB_SUCCESS;
  }
};

struct IvfPinFreeFn
{
  explicit IvfPinFreeFn(ObIvfCidClusterCache &cache) : cache_(cache) {}
  int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *> &pair)
  {
    cache_.release_pin_slot_(pair.second);
    return OB_SUCCESS;
  }
  ObIvfCidClusterCache &cache_;
};

/// Online invalidate: keep slots with active REPLAY pins; cleanup on unpin.
struct IvfPinInvalidateFn
{
  IvfPinInvalidateFn(common::hash::ObHashMap<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *> &pin_map,
      ObIvfCidClusterCache &cache)
      : pin_map_(pin_map), cache_(cache)
  {}
  int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *> &pair)
  {
    ObIvfCidClusterCache::ObIvfCidClusterPinSlot *slot = pair.second;
    if (OB_NOT_NULL(slot)) {
      (void)pin_map_.erase_refactored(pair.first);
      cache_.release_pin_slot_(slot);
    }
    return OB_SUCCESS;
  }
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *> &pin_map_;
  ObIvfCidClusterCache &cache_;
};

static const int64_t IVF_MAP_OVERWRITE = 1;
typedef lib::ObMutex IvfPinShardMutex;

ObIvfCidClusterCache::ObIvfCidClusterCache()
  : inited_(false),
    mgr_key_(),
    current_index_epoch_(1),
    max_bytes_(0),
    max_rows_per_cid_(0),
    replay_heat_weight_(DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT),
    ledger_bytes_(0),
    ref_cnt_(0),
    cache_phase_(static_cast<int64_t>(ObIvfCidClusterCachePhase::LEARNING)),
    total_access_samples_(0),
    serving_miss_samples_(0),
    refresh_targets_remaining_(0),
    refresh_target_map_(),
    ledger_map_(),
    fill_gates_(),
    ledger_lock_(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK),
    probe_heat_lock_(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK),
    fill_gates_lock_(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK)
{
  for (int64_t i = 0; i < IVF_CID_PIN_MAP_SHARD_CNT; ++i) {
    pin_shard_locks_[i] = nullptr;
  }
  for (int64_t i = 0; i < IVF_CID_PIN_MAP_SHARD_CNT; ++i) {
    pin_shard_locks_[i] = OB_NEW(IvfPinShardMutex, IVF_CID_CLUSTER_CACHE_LABEL, common::ObLatchIds::VECTOR_IVF_CACHE_LOCK);
  }
}

int64_t ObIvfCidClusterCache::pin_shard_idx_(const uint64_t cid) const
{
  return static_cast<int64_t>(cid % static_cast<uint64_t>(IVF_CID_PIN_MAP_SHARD_CNT));
}

common::hash::ObHashMap<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *>
    &ObIvfCidClusterCache::pin_map_shard_(const uint64_t cid)
{
  return pin_map_shards_[pin_shard_idx_(cid)];
}

const common::hash::ObHashMap<uint64_t, ObIvfCidClusterCache::ObIvfCidClusterPinSlot *>
    &ObIvfCidClusterCache::pin_map_shard_(const uint64_t cid) const
{
  return pin_map_shards_[pin_shard_idx_(cid)];
}

lib::ObMutex &ObIvfCidClusterCache::pin_shard_lock_(const uint64_t cid)
{
  return *pin_shard_locks_[pin_shard_idx_(cid)];
}

const lib::ObMutex &ObIvfCidClusterCache::pin_shard_lock_(const uint64_t cid) const
{
  return *pin_shard_locks_[pin_shard_idx_(cid)];
}

uint64_t ObIvfCidClusterCache::load_index_epoch_() const
{
  return ATOMIC_LOAD(&current_index_epoch_);
}

void ObIvfCidClusterCache::release_dormant_pin_shard_locked_(const uint64_t cid, ObIvfCidClusterPinSlot *slot)
{
  if (OB_NOT_NULL(slot)) {
    (void)pin_map_shard_(cid).erase_refactored(cid);
    release_pin_slot_(slot);
  }
}

int ObIvfCidClusterCache::ledger_remove_(const uint64_t cid, const bool count_evict, const bool erase_kv)
{
  {
    lib::ObMutexGuard pin_guard(pin_shard_lock_(cid));
    ObIvfCidClusterPinSlot *pin_slot = nullptr;
    if (OB_SUCCESS == pin_map_shard_(cid).get_refactored(cid, pin_slot) && OB_NOT_NULL(pin_slot)) {
      release_dormant_pin_shard_locked_(cid, pin_slot);
    }
  }
  lib::ObMutexGuard ledger_guard(ledger_lock_);
  return ledger_remove_locked_(cid, count_evict, erase_kv);
}

ObIvfCidClusterCache::~ObIvfCidClusterCache()
{
  destroy();
}

int ObIvfCidClusterCache::init(const ObIvfCidClusterCacheMgrKey &key, int64_t max_bytes, int64_t max_rows_per_cid, int64_t replay_heat_weight)
{
  int ret = OB_SUCCESS;
  ObMemAttr attr(OB_SERVER_TENANT_ID, "IvfCidClu");
  if (inited_) {
    ret = OB_INIT_TWICE;
  } else if (!key.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < IVF_CID_PIN_MAP_SHARD_CNT; ++i) {
      if (OB_FAIL(pin_map_shards_[i].create(128, attr, attr))) {
        LOG_WARN("failed to create pin map shard", K(ret), K(i));
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ledger_map_.create(128, attr, attr))) {
      LOG_WARN("failed to create ledger map", K(ret));
    } else if (OB_FAIL(fill_gates_.create(64, attr, attr))) {
      LOG_WARN("failed to create fill gates map", K(ret));
    } else if (OB_FAIL(probe_heat_map_.create(128, attr, attr))) {
      LOG_WARN("failed to create probe heat map", K(ret));
    } else if (OB_FAIL(refresh_target_map_.create(128, attr, attr))) {
      LOG_WARN("failed to create refresh target map", K(ret));
    } else {
      mgr_key_ = key;
      current_index_epoch_ = 1;
      cache_phase_ = static_cast<int64_t>(ObIvfCidClusterCachePhase::LEARNING);
      total_access_samples_ = 0;
      serving_miss_samples_ = 0;
      refresh_targets_remaining_ = 0;
      max_bytes_ = max_bytes > 0 ? max_bytes : DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_MB * 1024L * 1024L;
      max_rows_per_cid_ = max_rows_per_cid;
      replay_heat_weight_ = replay_heat_weight > 0 ? replay_heat_weight : DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT;
      ledger_bytes_ = 0;
      stats_.reset();
      (void)get_ivf_cid_cluster_kv_cache().init_cache();
      inited_ = true;
    }
  }
  return ret;
}

void ObIvfCidClusterCache::get_stats(ObIvfCidClusterCacheStats &out) const
{
  // Lock-free snapshot for stats/logging: counters are updated with atomics on the hot path.
  out.get_hit_cnt_ = ATOMIC_LOAD(&stats_.get_hit_cnt_);
  out.get_miss_cnt_ = ATOMIC_LOAD(&stats_.get_miss_cnt_);
  out.get_stale_epoch_cnt_ = ATOMIC_LOAD(&stats_.get_stale_epoch_cnt_);
  out.put_ok_cnt_ = ATOMIC_LOAD(&stats_.put_ok_cnt_);
  out.put_fail_cnt_ = ATOMIC_LOAD(&stats_.put_fail_cnt_);
  out.put_skip_rows_limit_cnt_ = ATOMIC_LOAD(&stats_.put_skip_rows_limit_cnt_);
  out.put_skip_pinned_cnt_ = ATOMIC_LOAD(&stats_.put_skip_pinned_cnt_);
  out.put_skip_cap_cnt_ = ATOMIC_LOAD(&stats_.put_skip_cap_cnt_);
  out.fill_wait_cnt_ = ATOMIC_LOAD(&stats_.fill_wait_cnt_);
  out.fill_bypass_cnt_ = ATOMIC_LOAD(&stats_.fill_bypass_cnt_);
  out.evict_entry_cnt_ = ATOMIC_LOAD(&stats_.evict_entry_cnt_);
  out.evict_bytes_ = ATOMIC_LOAD(&stats_.evict_bytes_);
  out.invalidate_cnt_ = ATOMIC_LOAD(&stats_.invalidate_cnt_);
  out.cur_entry_cnt_ = ATOMIC_LOAD(&stats_.cur_entry_cnt_);
  {
    lib::ObMutexGuard guard(ledger_lock_);
    out.cur_bytes_ = ledger_bytes_;
  }
}

void ObIvfCidClusterCache::log_stats(const char *tag) const
{
  if (!is_ivf_cid_cluster_cache_stats_enabled()) {
    return;
  }
  ObIvfCidClusterCacheStats s;
  get_stats(s);
  const int64_t get_total = s.get_hit_cnt_ + s.get_miss_cnt_;
  const double get_hit_rate = get_total > 0 ? static_cast<double>(s.get_hit_cnt_) / static_cast<double>(get_total) : 0.0;
  LOG_INFO("[OB_IVF_CID_CLUSTER_CACHE_STATS] cache",
           "tag", tag == nullptr ? "" : tag,
           K(mgr_key_),
           K(max_bytes_),
           K(s.get_hit_cnt_),
           K(s.get_miss_cnt_),
           "get_hit_rate", get_hit_rate,
           K(s.put_ok_cnt_),
           K(s.put_fail_cnt_),
           K(s.put_skip_rows_limit_cnt_),
           K(s.put_skip_pinned_cnt_),
           K(s.put_skip_cap_cnt_),
           K(s.evict_entry_cnt_),
           K(s.evict_bytes_),
           K(s.cur_entry_cnt_),
           K(s.cur_bytes_),
           K(ref_cnt_));
}

void ObIvfCidClusterCache::destroy()
{
  clear_all_fill_gates_();
  (void)fill_gates_.destroy();
  IvfPinFreeFn pin_free_fn(*this);
  for (int64_t i = 0; i < IVF_CID_PIN_MAP_SHARD_CNT; ++i) {
    if (OB_NOT_NULL(pin_shard_locks_[i])) {
      lib::ObMutexGuard pin_guard(*pin_shard_locks_[i]);
      (void)pin_map_shards_[i].foreach_refactored(pin_free_fn);
      pin_map_shards_[i].destroy();
    }
  }
  for (int64_t i = 0; i < IVF_CID_PIN_MAP_SHARD_CNT; ++i) {
    if (OB_NOT_NULL(pin_shard_locks_[i])) {
      OB_DELETE(IvfPinShardMutex, IVF_CID_CLUSTER_CACHE_LABEL, pin_shard_locks_[i]);
      pin_shard_locks_[i] = nullptr;
    }
  }
  common::ObArray<uint64_t> ledger_cids;
  struct CollectLedgerCidFn
  {
    explicit CollectLedgerCidFn(common::ObArray<uint64_t> &cids) : cids_(cids) {}
    int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterLedgerRecord> &pair)
    {
      return cids_.push_back(pair.first);
    }
    common::ObArray<uint64_t> &cids_;
  } collect_ledger_fn(ledger_cids);
  {
    lib::ObMutexGuard guard(ledger_lock_);
    (void)ledger_map_.foreach_refactored(collect_ledger_fn);
  }
  for (int64_t i = 0; i < ledger_cids.count(); ++i) {
    (void)ledger_remove_(ledger_cids.at(i), false);
  }
  {
    lib::ObMutexGuard guard(ledger_lock_);
    ledger_map_.destroy();
  }
  {
    lib::ObMutexGuard probe_guard(probe_heat_lock_);
    (void)probe_heat_map_.destroy();
  }
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    (void)refresh_target_map_.destroy();
  }
  inited_ = false;
  ledger_bytes_ = 0;
}

void ObIvfCidClusterCache::clear_all_fill_gates_()
{
  if (!inited_) {
    return;
  }
  lib::ObMutexGuard fg_guard(fill_gates_lock_);
  FillGateWakeAndFreeFn free_fn;
  (void)fill_gates_.foreach_refactored(free_fn);
  (void)fill_gates_.clear();
}

void ObIvfCidClusterCache::inc_ref()
{
  ATOMIC_INC(&ref_cnt_);
}

void ObIvfCidClusterCache::dec_ref()
{
  ATOMIC_SAF(&ref_cnt_, 1);
}

double ObIvfCidClusterCache::heat_score_(const ObIvfCidClusterHeat &heat, int64_t byte_denom) const
{
  const int64_t denom = OB_MAX(byte_denom, IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
  return static_cast<double>(heat.access_cnt_ + replay_heat_weight_ * heat.replay_cnt_)
      / static_cast<double>(denom);
}

double ObIvfCidClusterCache::prospective_heat_score_locked_(uint64_t cid) const
{
  ObIvfCidClusterHeat heat;
  if (OB_SUCCESS != probe_heat_map_.get_refactored(cid, heat)) {
    heat.access_cnt_ = 1;
    heat.replay_cnt_ = 0;
  }
  return heat_score_(heat, IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
}

bool ObIvfCidClusterCache::hotter_than_coldest_locked_(uint64_t skip_cid, double incoming_score) const
{
  double min_score = DBL_MAX;
  uint64_t victim_cid = 0;
  return find_coldest_evictable_ledger_locked_(skip_cid, min_score, victim_cid) && incoming_score > min_score;
}

void ObIvfCidClusterCache::release_pin_slot_(ObIvfCidClusterPinSlot *slot)
{
  if (OB_ISNULL(slot)) {
  } else if (OB_NOT_NULL(slot->view_entry_)) {
    slot->view_entry_->rowkey_objs_ = nullptr;
    slot->view_entry_->kv_flat_buf_ = nullptr;
    if (OB_NOT_NULL(slot->rowkey_objs_)) {
      ob_free(slot->rowkey_objs_);
      slot->rowkey_objs_ = nullptr;
    }
    OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, slot->view_entry_);
    slot->view_entry_ = nullptr;
  }
  slot->kv_handle_.reset();
  ob_free(slot);
}

int ObIvfCidClusterCache::install_pin_slot_locked_(uint64_t cid,
    ObIvfCidClusterEntry *view_entry,
    ObObj *rowkey_objs,
    ObKVCacheHandle &kv_handle)
{
  int ret = OB_SUCCESS;
  ObIvfCidClusterPinSlot *existing = nullptr;
  if (OB_SUCCESS == pin_map_shard_(cid).get_refactored(cid, existing) && OB_NOT_NULL(existing)) {
    release_dormant_pin_shard_locked_(cid, existing);
  }
  ObIvfCidClusterPinSlot *slot = static_cast<ObIvfCidClusterPinSlot *>(
      ob_malloc(sizeof(ObIvfCidClusterPinSlot), IVF_CID_CLUSTER_CACHE_LABEL));
  if (OB_ISNULL(slot)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    new (slot) ObIvfCidClusterPinSlot();
    slot->view_entry_ = view_entry;
    slot->rowkey_objs_ = rowkey_objs;
    view_entry->rowkey_objs_ = rowkey_objs;
    if (OB_FAIL(slot->kv_handle_.assign(kv_handle))) {
      LOG_WARN("failed to assign kv handle", K(ret), K(cid));
    } else if (OB_FAIL(pin_map_shard_(cid).set_refactored(cid, slot, IVF_MAP_OVERWRITE, 0, IVF_MAP_OVERWRITE))) {
      LOG_WARN("failed to set pin slot", K(ret), K(cid));
    } else {
      return OB_SUCCESS;
    }
    release_pin_slot_(slot);
  }
  return ret;
}

int ObIvfCidClusterCache::ensure_ledger_room_locked_(
    uint64_t skip_cid,
    int64_t need_bytes,
    double incoming_score)
{
  if (ledger_bytes_ + need_bytes <= max_bytes_) {
    return OB_SUCCESS;
  }
  if (!hotter_than_coldest_locked_(skip_cid, incoming_score)) {
    return OB_BUF_NOT_ENOUGH;
  }
  return ledger_evict_until_locked_(skip_cid, need_bytes);
}

int ObIvfCidClusterCache::ledger_remove_locked_(uint64_t cid, const bool count_evict, const bool erase_kv)
{
  int ret = OB_SUCCESS;
  ObIvfCidClusterLedgerRecord rec;
  if (OB_FAIL(ledger_map_.erase_refactored(cid, &rec))) {
    if (ret != OB_HASH_NOT_EXIST) {
      LOG_WARN("failed to erase ledger", K(ret), K(cid));
    }
  } else {
    ledger_bytes_ -= rec.bytes_;
    ATOMIC_DEC(&stats_.cur_entry_cnt_);
    if (count_evict) {
      ATOMIC_INC(&stats_.evict_entry_cnt_);
      ATOMIC_AAF(&stats_.evict_bytes_, rec.bytes_);
    }
    if (erase_kv) {
      ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
      (void)get_ivf_cid_cluster_kv_cache().erase_key(kv_key);
    }
  }
  return ret;
}

int ObIvfCidClusterCache::ledger_put_locked_(uint64_t cid, int64_t bytes, const ObIvfCidClusterHeat &heat)
{
  int ret = OB_SUCCESS;
  ObIvfCidClusterLedgerRecord rec;
  ObIvfCidClusterLedgerRecord old_rec;
  const bool had_old = (OB_SUCCESS == ledger_map_.get_refactored(cid, old_rec));
  const int overwrite_flag = 1;
  const int overwrite_key = 1;
  if (had_old) {
    ledger_bytes_ -= old_rec.bytes_;
  } else {
    ATOMIC_INC(&stats_.cur_entry_cnt_);
  }
  rec.bytes_ = bytes;
  rec.heat_ = heat;
  ledger_bytes_ += bytes;
  if (OB_FAIL(ledger_map_.set_refactored(cid, rec, overwrite_flag, 0, overwrite_key))) {
    LOG_WARN("failed to set ledger", K(ret), K(cid));
    ledger_bytes_ -= bytes;
    if (had_old) {
      ledger_bytes_ += old_rec.bytes_;
    } else {
      ATOMIC_DEC(&stats_.cur_entry_cnt_);
    }
  }
  return ret;
}

bool ObIvfCidClusterCache::find_coldest_evictable_ledger_locked_(
    uint64_t skip_cid,
    double &min_score,
    uint64_t &victim_cid) const
{
  min_score = DBL_MAX;
  victim_cid = 0;
  int64_t oldest_access = INT64_MAX;
  common::ObArray<uint64_t> cand_cids;
  common::ObArray<double> cand_scores;
  common::ObArray<int64_t> cand_access_us;
  struct CollectFn
  {
    CollectFn(const ObIvfCidClusterCache &cache,
        common::ObArray<uint64_t> &cids,
        common::ObArray<double> &scores,
        common::ObArray<int64_t> &access_us)
        : cache_(cache), cids_(cids), scores_(scores), access_us_(access_us)
    {}
    int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterLedgerRecord> &pair)
    {
      int ret = OB_SUCCESS;
      if (OB_FAIL(cids_.push_back(pair.first))) {
      } else if (OB_FAIL(scores_.push_back(cache_.heat_score_(pair.second.heat_, pair.second.bytes_)))) {
      } else if (OB_FAIL(access_us_.push_back(pair.second.heat_.last_access_us_))) {
      }
      return ret;
    }
    const ObIvfCidClusterCache &cache_;
    common::ObArray<uint64_t> &cids_;
    common::ObArray<double> &scores_;
    common::ObArray<int64_t> &access_us_;
  } collect_fn(*this, cand_cids, cand_scores, cand_access_us);
  {
    lib::ObMutexGuard guard(const_cast<ObIvfCidClusterCache *>(this)->ledger_lock_);
    (void)ledger_map_.foreach_refactored(collect_fn);
  }
  for (int64_t i = 0; i < cand_cids.count(); ++i) {
    const uint64_t cid = cand_cids.at(i);
    if (cid == skip_cid) {
    } else {
      const double score = cand_scores.at(i);
      const int64_t access_us = cand_access_us.at(i);
      if (score < min_score || (score == min_score && access_us < oldest_access)) {
        min_score = score;
        oldest_access = access_us;
        victim_cid = cid;
      }
    }
  }
  return victim_cid > 0;
}

int ObIvfCidClusterCache::ledger_evict_until_locked_(uint64_t skip_cid, int64_t need_bytes)
{
  int ret = OB_SUCCESS;
  while (OB_SUCC(ret) && ledger_bytes_ + need_bytes > max_bytes_) {
    double min_score = DBL_MAX;
    uint64_t victim_cid = 0;
    if (!find_coldest_evictable_ledger_locked_(skip_cid, min_score, victim_cid)) {
      ret = OB_BUF_NOT_ENOUGH;
      break;
    } else {
      {
        lib::ObMutexGuard pin_guard(pin_shard_lock_(victim_cid));
        ObIvfCidClusterPinSlot *pin_slot = nullptr;
        if (OB_SUCCESS == pin_map_shard_(victim_cid).get_refactored(victim_cid, pin_slot) && OB_NOT_NULL(pin_slot)) {
          release_dormant_pin_shard_locked_(victim_cid, pin_slot);
        }
      }
      if (OB_FAIL(ledger_remove_locked_(victim_cid, true))) {
        LOG_WARN("failed to remove victim ledger", K(ret), K(victim_cid));
      }
    }
  }
  return ret;
}

void ObIvfCidClusterCache::bump_probe_heat_locked_(uint64_t cid)
{
  const int64_t now = ObTimeUtility::current_time();
  ObIvfCidClusterHeat heat;
  const int overwrite_flag = 1;
  const int overwrite_key = 1;
  if (OB_SUCCESS == probe_heat_map_.get_refactored(cid, heat)) {
    heat.access_cnt_++;
    heat.last_access_us_ = now;
    (void)probe_heat_map_.set_refactored(cid, heat, overwrite_flag, 0, overwrite_key);
  } else {
    heat.access_cnt_ = 1;
    heat.replay_cnt_ = 0;
    heat.fill_cnt_ = 0;
    heat.last_access_us_ = now;
    (void)probe_heat_map_.set_refactored(cid, heat, overwrite_flag, 0, overwrite_key);
  }
}

void ObIvfCidClusterCache::clear_probe_heat_locked_(uint64_t cid)
{
  lib::ObMutexGuard probe_guard(probe_heat_lock_);
  (void)probe_heat_map_.erase_refactored(cid);
}

void ObIvfCidClusterCache::invalidate_all()
{
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    FillGateWakeAndFreeFn free_fn;
    (void)fill_gates_.foreach_refactored(free_fn);
    (void)fill_gates_.clear();
  }
  for (int64_t s = 0; s < IVF_CID_PIN_MAP_SHARD_CNT; ++s) {
    if (OB_ISNULL(pin_shard_locks_[s])) {
    } else {
      lib::ObMutexGuard pin_guard(*pin_shard_locks_[s]);
      IvfPinInvalidateFn pin_invalidate_fn(pin_map_shards_[s], *this);
      (void)pin_map_shards_[s].foreach_refactored(pin_invalidate_fn);
    }
  }
  ATOMIC_INC(&current_index_epoch_);
  common::ObArray<uint64_t> cids;
  struct CollectLedgerCidFn
  {
    explicit CollectLedgerCidFn(common::ObArray<uint64_t> &cids) : cids_(cids) {}
    int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterLedgerRecord> &pair)
    {
      return cids_.push_back(pair.first);
    }
    common::ObArray<uint64_t> &cids_;
  } collect_ledger_fn(cids);
  {
    lib::ObMutexGuard guard(ledger_lock_);
    (void)ledger_map_.foreach_refactored(collect_ledger_fn);
  }
  for (int64_t i = 0; i < cids.count(); ++i) {
    (void)ledger_remove_(cids.at(i), false);
  }
  {
    lib::ObMutexGuard probe_guard(probe_heat_lock_);
    (void)probe_heat_map_.clear();
  }
  ATOMIC_INC(&stats_.invalidate_cnt_);
  store_phase_(ObIvfCidClusterCachePhase::LEARNING);
  ATOMIC_STORE(&total_access_samples_, 0);
  ATOMIC_STORE(&serving_miss_samples_, 0);
  ATOMIC_STORE(&refresh_targets_remaining_, 0);
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    (void)refresh_target_map_.clear();
  }
  ATOMIC_INC(&stats_.phase_learning_cnt_);
}

ObIvfCidClusterCachePhase ObIvfCidClusterCache::load_phase_() const
{
  return static_cast<ObIvfCidClusterCachePhase>(ATOMIC_LOAD(&cache_phase_));
}

void ObIvfCidClusterCache::store_phase_(ObIvfCidClusterCachePhase phase)
{
  ATOMIC_STORE(&cache_phase_, static_cast<int64_t>(phase));
}

bool ObIvfCidClusterCache::cas_phase_(ObIvfCidClusterCachePhase expected, ObIvfCidClusterCachePhase desired)
{
  const int64_t exp = static_cast<int64_t>(expected);
  const int64_t des = static_cast<int64_t>(desired);
  return ATOMIC_VCAS(&cache_phase_, exp, des) == exp;
}

int64_t ObIvfCidClusterCache::learning_start_threshold_() const
{
  const int64_t min_probe = ivf_cid_cluster_cache_fill_min_probe_access();
  return min_probe > 0 ? min_probe : DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS;
}

void ObIvfCidClusterCache::record_access_(uint64_t cid, ObIvfCidClusterCachePhase phase)
{
  ATOMIC_INC(&total_access_samples_);
  if (phase == ObIvfCidClusterCachePhase::LEARNING) {
    lib::ObMutexGuard probe_guard(probe_heat_lock_);
    bump_probe_heat_locked_(cid);
  }
}

bool ObIvfCidClusterCache::is_refresh_target_cid_(uint64_t cid) const
{
  int8_t state = 1;
  lib::ObMutexGuard fg_guard(const_cast<lib::ObMutex &>(fill_gates_lock_));
  if (OB_SUCCESS != refresh_target_map_.get_refactored(cid, state)) {
    return false;
  }
  return state == 0;
}

void ObIvfCidClusterCache::mark_refresh_target_done_(uint64_t cid)
{
  lib::ObMutexGuard fg_guard(fill_gates_lock_);
  int8_t state = 0;
  if (OB_SUCCESS == refresh_target_map_.get_refactored(cid, state) && state == 0) {
  const int overwrite_flag = 1;
  const int overwrite_key = 1;
    (void)refresh_target_map_.set_refactored(cid, static_cast<int8_t>(1), overwrite_flag, 0, overwrite_key);
    ATOMIC_DEC(&refresh_targets_remaining_);
  }
}

void ObIvfCidClusterCache::prepare_refresh_targets_(bool use_ledger_heat)
{
  common::ObArray<uint64_t> cids;
  common::ObArray<double> scores;
  const int64_t top_k = ivf_cid_cluster_cache_fill_top_k(max_bytes_);
  const int64_t min_probe = ivf_cid_cluster_cache_fill_min_probe_access();
  if (use_ledger_heat) {
    lib::ObMutexGuard ledger_guard(ledger_lock_);
    struct CollectLedgerFn
    {
      CollectLedgerFn(const ObIvfCidClusterCache &cache, common::ObArray<uint64_t> &cids, common::ObArray<double> &scores)
          : cache_(cache), cids_(cids), scores_(scores)
      {}
      int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterLedgerRecord> &pair)
      {
        int ret = OB_SUCCESS;
        const double score = cache_.heat_score_(pair.second.heat_, pair.second.bytes_ > 0 ? pair.second.bytes_ : IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
        if (OB_FAIL(cids_.push_back(pair.first))) {
        } else if (OB_FAIL(scores_.push_back(score))) {
        }
        return ret;
      }
      const ObIvfCidClusterCache &cache_;
      common::ObArray<uint64_t> &cids_;
      common::ObArray<double> &scores_;
    } collect_fn(*this, cids, scores);
    (void)ledger_map_.foreach_refactored(collect_fn);
  } else {
    lib::ObMutexGuard probe_guard(probe_heat_lock_);
    struct CollectProbeFn
    {
      CollectProbeFn(const ObIvfCidClusterCache &cache,
          const int64_t min_probe,
          common::ObArray<uint64_t> &cids,
          common::ObArray<double> &scores)
          : cache_(cache), min_probe_(min_probe), cids_(cids), scores_(scores)
      {}
      int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterHeat> &pair)
      {
        int ret = OB_SUCCESS;
        if (pair.second.access_cnt_ < min_probe_) {
        } else {
          const double score = cache_.heat_score_(pair.second, IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
          if (OB_FAIL(cids_.push_back(pair.first))) {
          } else if (OB_FAIL(scores_.push_back(score))) {
          }
        }
        return ret;
      }
      const ObIvfCidClusterCache &cache_;
      int64_t min_probe_;
      common::ObArray<uint64_t> &cids_;
      common::ObArray<double> &scores_;
    } collect_fn(*this, min_probe, cids, scores);
    (void)probe_heat_map_.foreach_refactored(collect_fn);
  }
  lib::ObMutexGuard fg_guard(fill_gates_lock_);
  (void)refresh_target_map_.clear();
  int64_t picked = 0;
  for (int64_t pick = 0; pick < top_k && pick < cids.count(); ++pick) {
    int64_t best_i = pick;
    for (int64_t j = pick + 1; j < cids.count(); ++j) {
      if (scores.at(j) > scores.at(best_i)) {
        best_i = j;
      }
    }
    if (best_i != pick) {
      const uint64_t tmp_cid = cids.at(pick);
      const double tmp_score = scores.at(pick);
      cids.at(pick) = cids.at(best_i);
      scores.at(pick) = scores.at(best_i);
      cids.at(best_i) = tmp_cid;
      scores.at(best_i) = tmp_score;
    }
    (void)refresh_target_map_.set_refactored(cids.at(pick), static_cast<int8_t>(0));
    ++picked;
  }
  ATOMIC_STORE(&refresh_targets_remaining_, picked);
}

void ObIvfCidClusterCache::try_begin_refresh_from_learning_()
{
  if (load_phase_() != ObIvfCidClusterCachePhase::LEARNING) {
    return;
  }
  const int64_t threshold = learning_start_threshold_();
  int64_t hot_cnt = 0;
  {
    lib::ObMutexGuard probe_guard(probe_heat_lock_);
    struct CountHotFn
    {
      explicit CountHotFn(const int64_t threshold, int64_t &hot_cnt) : threshold_(threshold), hot_cnt_(hot_cnt) {}
      int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterHeat> &pair)
      {
        if (pair.second.access_cnt_ >= threshold_) {
          hot_cnt_++;
        }
        return OB_SUCCESS;
      }
      int64_t threshold_;
      int64_t &hot_cnt_;
    } count_fn(threshold, hot_cnt);
    (void)probe_heat_map_.foreach_refactored(count_fn);
  }
  if (hot_cnt <= 0) {
    return;
  }
  if (!cas_phase_(ObIvfCidClusterCachePhase::LEARNING, ObIvfCidClusterCachePhase::REFRESHING)) {
    return;
  }
  prepare_refresh_targets_(false);
  if (ATOMIC_LOAD(&refresh_targets_remaining_) <= 0) {
    (void)cas_phase_(ObIvfCidClusterCachePhase::REFRESHING, ObIvfCidClusterCachePhase::LEARNING);
    return;
  }
  ATOMIC_INC(&stats_.phase_refreshing_cnt_);
}

void ObIvfCidClusterCache::try_begin_refresh_from_serving_()
{
  if (load_phase_() != ObIvfCidClusterCachePhase::SERVING) {
    return;
  }
  const int64_t access = ATOMIC_LOAD(&total_access_samples_);
  const int64_t miss = ATOMIC_LOAD(&serving_miss_samples_);
  if (access < 1000 || miss * 100 / access < 30) {
    return;
  }
  if (!cas_phase_(ObIvfCidClusterCachePhase::SERVING, ObIvfCidClusterCachePhase::REFRESHING)) {
    return;
  }
  prepare_refresh_targets_(true);
  if (ATOMIC_LOAD(&refresh_targets_remaining_) <= 0) {
    (void)cas_phase_(ObIvfCidClusterCachePhase::REFRESHING, ObIvfCidClusterCachePhase::SERVING);
    return;
  }
  ATOMIC_STORE(&total_access_samples_, 0);
  ATOMIC_STORE(&serving_miss_samples_, 0);
  ATOMIC_INC(&stats_.phase_refreshing_cnt_);
}

int ObIvfCidClusterCache::warm_readonly_pin_slots_()
{
  common::ObArray<uint64_t> cids;
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    struct CollectTargetFn
    {
      explicit CollectTargetFn(common::ObArray<uint64_t> &cids) : cids_(cids) {}
      int operator()(const common::hash::HashMapPair<uint64_t, int8_t> &pair)
      {
        if (pair.second == 1) {
          return cids_.push_back(pair.first);
        }
        return OB_SUCCESS;
      }
      common::ObArray<uint64_t> &cids_;
    } collect_fn(cids);
    (void)refresh_target_map_.foreach_refactored(collect_fn);
  }
  for (int64_t i = 0; i < cids.count(); ++i) {
    (void)warm_pin_slot_(cids.at(i));
  }
  return OB_SUCCESS;
}

void ObIvfCidClusterCache::finalize_refresh_to_serving_()
{
  (void)warm_readonly_pin_slots_();
  if (cas_phase_(ObIvfCidClusterCachePhase::REFRESHING, ObIvfCidClusterCachePhase::SERVING)) {
    ATOMIC_INC(&stats_.phase_serving_cnt_);
    ATOMIC_STORE(&total_access_samples_, 0);
    ATOMIC_STORE(&serving_miss_samples_, 0);
  }
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    (void)refresh_target_map_.clear();
  }
  ATOMIC_STORE(&refresh_targets_remaining_, 0);
}

void ObIvfCidClusterCache::try_finalize_refresh_if_complete_()
{
  if (load_phase_() != ObIvfCidClusterCachePhase::REFRESHING) {
    return;
  }
  if (ATOMIC_LOAD(&refresh_targets_remaining_) > 0) {
    return;
  }
  finalize_refresh_to_serving_();
}

void ObIvfCidClusterCache::notify_refresh_put_done(uint64_t cid)
{
  mark_refresh_target_done_(cid);
  try_finalize_refresh_if_complete_();
}

int ObIvfCidClusterCache::lookup_readonly_serving_(uint64_t cid, ObIvfCidClusterEntry *&entry)
{
  int ret = OB_SUCCESS;
  entry = nullptr;
  const uint64_t epoch = load_index_epoch_();
  {
    lib::ObMutexGuard guard(pin_shard_lock_(cid));
    ObIvfCidClusterPinSlot *slot = nullptr;
    if (OB_SUCCESS == pin_map_shard_(cid).get_refactored(cid, slot) && OB_NOT_NULL(slot) && OB_NOT_NULL(slot->view_entry_)) {
      if (slot->view_entry_->index_epoch_ == epoch) {
        entry = slot->view_entry_;
        entry->session_borrowed_ = true;
        ATOMIC_INC(&stats_.get_hit_cnt_);
        return OB_SUCCESS;
      }
    }
  }
  ObIvfKvPinPrep prep;
  const int prep_ret = load_kv_pin_prep_(cid, epoch, prep);
  if (prep_ret != OB_SUCCESS) {
    ATOMIC_INC(&stats_.get_miss_cnt_);
    ATOMIC_INC(&serving_miss_samples_);
    return prep_ret;
  }
  prep.view_entry_->session_owned_ = true;
  if (OB_FAIL(prep.view_entry_->session_kv_handle_.assign(prep.kv_handle_))) {
    discard_kv_pin_prep_(prep);
    ATOMIC_INC(&stats_.get_miss_cnt_);
    ATOMIC_INC(&serving_miss_samples_);
    return OB_ERR_UNEXPECTED;
  }
  entry = prep.view_entry_;
  prep.view_entry_ = nullptr;
  prep.kv_handle_.reset();
  ATOMIC_INC(&stats_.get_hit_cnt_);
  return OB_SUCCESS;
}

void ObIvfCidClusterCache::release_session_entry(ObIvfCidClusterEntry *entry)
{
  if (OB_ISNULL(entry)) {
    return;
  }
  if (entry->session_owned_) {
    entry->session_kv_handle_.reset();
    entry->session_owned_ = false;
    OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, entry);
    return;
  }
  if (entry->session_borrowed_) {
    entry->session_borrowed_ = false;
  }
}

int ObIvfCidClusterCache::warm_pin_slot_(uint64_t cid)
{
  const uint64_t epoch = load_index_epoch_();
  {
    lib::ObMutexGuard guard(pin_shard_lock_(cid));
    ObIvfCidClusterPinSlot *slot = nullptr;
    if (OB_SUCCESS == pin_map_shard_(cid).get_refactored(cid, slot) && OB_NOT_NULL(slot)
        && OB_NOT_NULL(slot->view_entry_) && slot->view_entry_->index_epoch_ == epoch) {
      return OB_SUCCESS;
    }
  }
  ObIvfKvPinPrep prep;
  int ret = load_kv_pin_prep_(cid, epoch, prep);
  if (OB_HASH_NOT_EXIST == ret) {
    reconcile_kv_pin_miss_locked_(cid);
    (void)ledger_remove_(cid, false);
    return ret;
  } else if (OB_FAIL(ret)) {
    return ret;
  }
  {
    lib::ObMutexGuard guard(pin_shard_lock_(cid));
    ret = install_pin_slot_locked_(cid, prep.view_entry_, prep.rowkey_objs_, prep.kv_handle_);
  }
  if (OB_SUCC(ret)) {
    prep.view_entry_ = nullptr;
    prep.rowkey_objs_ = nullptr;
    prep.kv_handle_.reset();
  } else {
    discard_kv_pin_prep_(prep);
  }
  return ret;
}

void ObIvfCidClusterCache::discard_kv_pin_prep_(ObIvfKvPinPrep &prep)
{
  if (OB_NOT_NULL(prep.view_entry_)) {
    prep.view_entry_->rowkey_objs_ = nullptr;
    prep.view_entry_->kv_flat_buf_ = nullptr;
    if (OB_NOT_NULL(prep.rowkey_objs_)) {
      ob_free(prep.rowkey_objs_);
      prep.rowkey_objs_ = nullptr;
    }
    OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, prep.view_entry_);
    prep.view_entry_ = nullptr;
  }
  prep.kv_handle_.reset();
  prep.flat_buf_ = nullptr;
  prep.flat_len_ = 0;
  prep.rowkey_obj_cnt_ = 0;
}

int ObIvfCidClusterCache::load_kv_pin_prep_(uint64_t cid, uint64_t index_epoch, ObIvfKvPinPrep &prep)
{
  int ret = OB_SUCCESS;
  discard_kv_pin_prep_(prep);
  ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
  const char *flat_buf = nullptr;
  int64_t flat_len = 0;
  ObKVCacheHandle kv_handle;
  if (OB_FAIL(get_ivf_cid_cluster_kv_cache().get_flat(kv_key, flat_buf, flat_len, kv_handle))) {
    if (ret == OB_ENTRY_NOT_EXIST) {
      ATOMIC_INC(&stats_.get_miss_cnt_);
      return OB_HASH_NOT_EXIST;
    }
    LOG_WARN("failed to get flat cluster from kv", K(ret), K(cid));
    return ret;
  }
  if (OB_ISNULL(flat_buf) || flat_len <= 0) {
    return OB_ERR_UNEXPECTED;
  }
  const ObIvfCidFlatHeader *hdr = reinterpret_cast<const ObIvfCidFlatHeader *>(flat_buf);
  if (hdr->index_epoch_ != index_epoch) {
    ATOMIC_INC(&stats_.get_stale_epoch_cnt_);
    ATOMIC_INC(&stats_.get_miss_cnt_);
    return OB_HASH_NOT_EXIST;
  }
  ObIvfCidClusterEntry *view_entry = nullptr;
  ObObj *rk_objs = nullptr;
  int64_t rk_cnt = 0;
  if (OB_FAIL(ivf_cid_flat_attach_entry(flat_buf, flat_len, view_entry, rk_objs, rk_cnt))) {
    LOG_WARN("failed to attach flat entry", K(ret), K(cid));
    return ret;
  }
  prep.flat_buf_ = flat_buf;
  prep.flat_len_ = flat_len;
  prep.view_entry_ = view_entry;
  prep.rowkey_objs_ = rk_objs;
  prep.rowkey_obj_cnt_ = rk_cnt;
  if (OB_FAIL(prep.kv_handle_.assign(kv_handle))) {
    LOG_WARN("failed to assign kv handle for kv pin prep", K(ret), K(cid));
    discard_kv_pin_prep_(prep);
    return ret;
  }
  return OB_SUCCESS;
}

void ObIvfCidClusterCache::reconcile_kv_pin_miss_locked_(uint64_t cid)
{
  ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
  const char *flat_buf = nullptr;
  int64_t flat_len = 0;
  ObKVCacheHandle kv_handle;
  if (OB_SUCCESS == get_ivf_cid_cluster_kv_cache().get_flat(kv_key, flat_buf, flat_len, kv_handle)) {
    const ObIvfCidFlatHeader *hdr = reinterpret_cast<const ObIvfCidFlatHeader *>(flat_buf);
    if (OB_NOT_NULL(hdr) && hdr->index_epoch_ != load_index_epoch_()) {
      (void)get_ivf_cid_cluster_kv_cache().erase_key(kv_key);
    }
  }
}

int ObIvfCidClusterCache::lookup_cid(
    uint64_t cid,
    ObIvfCidClusterEntry *&entry,
    ObIvfCidClusterLookupResult &result,
    ObIvfCidClusterCacheSessionStats *session_stats)
{
  int ret = OB_SUCCESS;
  entry = nullptr;
  result = ObIvfCidClusterLookupResult::MISS;
  if (!inited_) {
    ret = OB_NOT_INIT;
  } else {
    const ObIvfCidClusterCachePhase phase = load_phase_();
    record_access_(cid, phase);
    if (phase == ObIvfCidClusterCachePhase::LEARNING) {
      try_begin_refresh_from_learning_();
    } else if (phase == ObIvfCidClusterCachePhase::REFRESHING) {
      ret = lookup_cid_refreshing_(cid, entry, result, session_stats);
    } else {
      const int serving_ret = lookup_readonly_serving_(cid, entry);
      if (OB_SUCCESS == serving_ret) {
        result = ObIvfCidClusterLookupResult::HIT;
      } else if (OB_HASH_NOT_EXIST == serving_ret) {
        result = ObIvfCidClusterLookupResult::MISS;
        try_begin_refresh_from_serving_();
      } else {
        ret = serving_ret;
      }
    }
  }
  return ret;
}

int ObIvfCidClusterCache::lookup_cid_refreshing_(
    uint64_t cid,
    ObIvfCidClusterEntry *&entry,
    ObIvfCidClusterLookupResult &result,
    ObIvfCidClusterCacheSessionStats *session_stats)
{
  int ret = OB_SUCCESS;
  entry = nullptr;
  result = ObIvfCidClusterLookupResult::MISS;
  if (!is_refresh_target_cid_(cid)) {
    return OB_SUCCESS;
  }
  bool is_fill_leader = false;
  ObIvfCidFillGate *gate = nullptr;
  while (OB_SUCC(ret)) {
      bool wait_fill = false;
      {
        lib::ObMutexGuard fg_guard(fill_gates_lock_);
        if (OB_FAIL(fill_gates_.get_refactored(cid, gate))) {
          if (ret == OB_HASH_NOT_EXIST) {
            ret = OB_SUCCESS;
            void *buf = ob_malloc(sizeof(ObIvfCidFillGate), IVF_CID_CLUSTER_CACHE_LABEL);
            if (OB_ISNULL(buf)) {
              ret = OB_ALLOCATE_MEMORY_FAILED;
            } else {
              gate = new (buf) ObIvfCidFillGate(cid);
              if (OB_FAIL(gate->cond_.init(common::ObWaitEventIds::THREAD_IDLING_COND_WAIT))) {
                LOG_WARN("failed to init fill gate cond", K(ret), K(cid));
                gate->dec_ref();
                gate = nullptr;
              } else if (OB_FAIL(fill_gates_.set_refactored(cid, gate))) {
                if (ret == OB_HASH_EXIST) {
                  ret = OB_SUCCESS;
                  gate->dec_ref();
                  gate = nullptr;
                  (void)fill_gates_.get_refactored(cid, gate);
                } else {
                  LOG_WARN("failed to set fill gate", K(ret), K(cid));
                  gate->dec_ref();
                  gate = nullptr;
                }
              }
            }
            if (OB_SUCC(ret) && OB_NOT_NULL(gate)) {
              if (OB_FAIL(gate->cond_.lock())) {
                LOG_WARN("failed to lock new fill gate", K(ret), K(cid));
                (void)fill_gates_.erase_refactored(cid);
                gate->dec_ref();
                gate = nullptr;
                ret = OB_SUCCESS;
              } else {
                gate->filling_ = true;
                (void)gate->cond_.unlock();
                is_fill_leader = true;
                gate->inc_ref();
                break;
              }
            }
          } else {
            LOG_WARN("failed to get fill gate", K(ret), K(cid));
          }
        } else if (OB_NOT_NULL(gate) && gate->filling_) {
          wait_fill = true;
          gate->inc_ref();
        } else if (OB_NOT_NULL(gate)) {
          (void)fill_gates_.erase_refactored(cid);
          gate->dec_ref();
        }
      }

      if (OB_SUCC(ret) && wait_fill && OB_NOT_NULL(gate)) {
        if (OB_FAIL(gate->cond_.lock())) {
          LOG_WARN("failed to lock fill gate for wait", K(ret), K(cid));
          gate->dec_ref();
        } else {
          const int64_t wait_beg_us = ObTimeUtility::current_time();
          while (OB_SUCC(ret) && gate->filling_) {
            ATOMIC_INC(&stats_.fill_wait_cnt_);
            if (OB_NOT_NULL(session_stats)) {
              session_stats->fill_wait_cnt_++;
            }
            if (OB_FAIL(gate->cond_.wait())) {
              LOG_WARN("failed to wait on fill gate", K(ret), K(cid));
            }
          }
          if (OB_NOT_NULL(session_stats)) {
            session_stats->fill_wait_us_ += ObTimeUtility::current_time() - wait_beg_us;
          }
          (void)gate->cond_.unlock();
          gate->dec_ref();
          continue;
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (is_fill_leader) {
        result = ObIvfCidClusterLookupResult::FILL_LEADER;
      } else {
        result = ObIvfCidClusterLookupResult::MISS;
      }
    }
  return ret;
}

void ObIvfCidClusterCache::finish_cid_fill(uint64_t cid)
{
  ObIvfCidFillGate *gate = nullptr;
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    if (OB_SUCCESS != fill_gates_.get_refactored(cid, gate) || OB_ISNULL(gate)) {
      return;
    }
    (void)fill_gates_.erase_refactored(cid);
  }
  if (OB_SUCCESS == gate->cond_.lock()) {
    gate->filling_ = false;
    (void)gate->cond_.broadcast();
    (void)gate->cond_.unlock();
  }
  // Leader inc_ref in acquire; gate also has its alloc ref — drop both after erase from map.
  gate->dec_ref();
  gate->dec_ref();
}

int ObIvfCidClusterCache::put(ObIvfCidClusterEntry &entry)
{
  int ret = OB_SUCCESS;
  if (load_phase_() != ObIvfCidClusterCachePhase::REFRESHING) {
    return OB_STATE_NOT_MATCH;
  } else if (max_rows_per_cid_ > 0 && entry.row_count_ > max_rows_per_cid_) {
    ret = OB_SIZE_OVERFLOW;
    ATOMIC_INC(&stats_.put_skip_rows_limit_cnt_);
  } else {
    const uint64_t cid = entry.cid_;
    const double incoming_score = prospective_heat_score_locked_(entry.cid_);
    if (ledger_bytes_ + entry.entry_bytes_ > max_bytes_ && !hotter_than_coldest_locked_(cid, incoming_score)) {
      ATOMIC_INC(&stats_.put_skip_cap_cnt_);
      return OB_BUF_NOT_ENOUGH;
    }
    {
      lib::ObMutexGuard pin_guard(pin_shard_lock_(cid));
      ObIvfCidClusterPinSlot *pin_slot = nullptr;
      if (OB_SUCCESS == pin_map_shard_(cid).get_refactored(cid, pin_slot) && OB_NOT_NULL(pin_slot)) {
        release_dormant_pin_shard_locked_(cid, pin_slot);
      }
    }
    {
      lib::ObMutexGuard ledger_guard(ledger_lock_);
      const uint64_t fill_epoch = entry.index_epoch_;
      const uint64_t cur_epoch = load_index_epoch_();
      if (fill_epoch != 0 && fill_epoch != cur_epoch) {
        ATOMIC_INC(&stats_.get_stale_epoch_cnt_);
        return OB_STATE_NOT_MATCH;
      }
      entry.index_epoch_ = cur_epoch;
      const int cap_ret = ensure_ledger_room_locked_(cid, entry.entry_bytes_, incoming_score);
      if (cap_ret == OB_BUF_NOT_ENOUGH) {
        ATOMIC_INC(&stats_.put_skip_cap_cnt_);
        return OB_BUF_NOT_ENOUGH;
      } else if (OB_FAIL(cap_ret)) {
        return cap_ret;
      }
      // Drop ledger only; put_flat(overwrite) replaces KV. erase_key here races with concurrent get_flat.
      (void)ledger_remove_locked_(cid, false, false);
      char *flat_buf = nullptr;
      int64_t flat_len = 0;
      if (OB_FAIL(ivf_cid_flat_encode_entry(entry, flat_buf, flat_len))) {
        LOG_WARN("failed to encode flat entry", K(ret), K(cid));
      } else {
        ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
        if (OB_FAIL(get_ivf_cid_cluster_kv_cache().put_flat(kv_key, flat_buf, flat_len, true))) {
          LOG_WARN("failed to put flat into kv cache", K(ret), K(cid));
        } else {
          if (entry.heat_.access_cnt_ == 0) {
            entry.heat_.access_cnt_ = 1;
          }
          entry.heat_.fill_cnt_++;
          entry.heat_.last_access_us_ = ObTimeUtility::current_time();
          ObIvfCidClusterHeat ledger_heat = entry.heat_;
          ObIvfCidClusterHeat probe_heat;
          {
            lib::ObMutexGuard probe_guard(probe_heat_lock_);
            if (OB_SUCCESS == probe_heat_map_.get_refactored(cid, probe_heat)) {
              ledger_heat.access_cnt_ += probe_heat.access_cnt_;
              ledger_heat.replay_cnt_ += probe_heat.replay_cnt_;
            }
          }
          if (OB_FAIL(ledger_put_locked_(cid, entry.entry_bytes_, ledger_heat))) {
            LOG_WARN("failed to update ledger", K(ret), K(cid));
            (void)get_ivf_cid_cluster_kv_cache().erase_key(kv_key);
          } else {
            ATOMIC_INC(&stats_.put_ok_cnt_);
            clear_probe_heat_locked_(cid);
          }
        }
        ivf_cid_flat_free_buf(flat_buf);
      }
      if (OB_FAIL(ret) && ret != OB_SIZE_OVERFLOW && ret != OB_BUF_NOT_ENOUGH && ret != OB_EAGAIN
          && ret != OB_STATE_NOT_MATCH) {
        ATOMIC_INC(&stats_.put_fail_cnt_);
      }
    }
  }
  return ret;
}

int acquire_ivf_cid_cluster_cache(const ObIvfCidClusterCacheMgrKey &key, ObIvfCidClusterCache *&cache)
{
  int ret = OB_SUCCESS;
  cache = nullptr;
  if (!is_ivf_cid_cluster_cache_enabled()) {
  } else if (OB_FAIL(ensure_cache_mgr_map_inited())) {
    LOG_WARN("failed to init mgr map", K(ret));
  } else {
    lib::ObMutexGuard guard(g_ivf_cid_cluster_cache_mgr_lock);
    if (OB_FAIL(g_ivf_cid_cluster_cache_mgr_map.get_refactored(key, cache))) {
      if (ret == OB_HASH_NOT_EXIST) {
        ret = OB_SUCCESS;
        ObIvfCidClusterCache *created = OB_NEW(ObIvfCidClusterCache, IVF_CID_CLUSTER_CACHE_LABEL);
        if (OB_ISNULL(created)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_FAIL(created->init(key, 0, 0, 0))) {
          LOG_WARN("failed to init cache", K(ret));
          OB_DELETE(ObIvfCidClusterCache, IVF_CID_CLUSTER_CACHE_LABEL, created);
        } else if (OB_FAIL(g_ivf_cid_cluster_cache_mgr_map.set_refactored(key, created))) {
          if (ret == OB_HASH_EXIST) {
            ret = OB_SUCCESS;
            OB_DELETE(ObIvfCidClusterCache, IVF_CID_CLUSTER_CACHE_LABEL, created);
            g_ivf_cid_cluster_cache_mgr_map.get_refactored(key, cache);
          } else {
            LOG_WARN("failed to set cache mgr", K(ret));
            OB_DELETE(ObIvfCidClusterCache, IVF_CID_CLUSTER_CACHE_LABEL, created);
          }
        } else {
          cache = created;
          // Mgr-map residency pin: keeps probe_heat / ledger across DAS iter release.
          cache->inc_ref();
        }
      } else {
        LOG_WARN("failed to get cache", K(ret));
      }
    }
    // Per-holder pin (e.g. ObDASIvfCidVecCacheScanIter); released in release_ivf_cid_cluster_cache.
    if (OB_SUCC(ret) && OB_NOT_NULL(cache)) {
      cache->inc_ref();
    }
  }
  return ret;
}

void invalidate_ivf_cid_cluster_cache_for_index_tablet(
    const uint64_t tenant_id,
    const common::ObTabletID &index_tablet_id)
{
  if (!is_ivf_cid_cluster_cache_enabled() || !index_tablet_id.is_valid()) {
    return;
  }
  if (OB_SUCCESS != ensure_cache_mgr_map_inited()) {
    return;
  }
  lib::ObMutexGuard guard(g_ivf_cid_cluster_cache_mgr_lock);
  struct InvalidateTabletFn
  {
    explicit InvalidateTabletFn(const uint64_t tenant_id, const common::ObTabletID &index_tablet_id)
      : tenant_id_(tenant_id), index_tablet_id_(index_tablet_id)
    {}
    int operator()(const common::hash::HashMapPair<ObIvfCidClusterCacheMgrKey, ObIvfCidClusterCache *> &pair)
    {
      ObIvfCidClusterCache *cache = pair.second;
      if (OB_NOT_NULL(cache)
          && pair.first.tenant_id_ == tenant_id_
          && pair.first.index_tablet_id_ == index_tablet_id_) {
        cache->invalidate_all();
      }
      return OB_SUCCESS;
    }
    uint64_t tenant_id_;
    common::ObTabletID index_tablet_id_;
  } fn(tenant_id, index_tablet_id);
  (void)g_ivf_cid_cluster_cache_mgr_map.foreach_refactored(fn);
}

void release_ivf_cid_cluster_cache(ObIvfCidClusterCache *cache)
{
  if (OB_ISNULL(cache)) {
  } else if (!g_ivf_cid_cluster_cache_mgr_inited) {
    cache->dec_ref();
  } else {
    int ret = OB_SUCCESS;
    bool destroy_cache = false;
    {
      lib::ObMutexGuard guard(g_ivf_cid_cluster_cache_mgr_lock);
      const int64_t ref_after_dec = ATOMIC_SAF(&cache->ref_cnt_, 1);
      // ref_cnt_ == 1 after dec: only mgr-map residency remains — keep cache for cross-query warmup.
      if (ref_after_dec > 1) {
      } else if (ref_after_dec == 1) {
      } else if (ref_after_dec < 0) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("ivf cid cluster cache ref underflow", K(ret), K(ref_after_dec), K(cache->mgr_key_));
        cache->inc_ref();
      } else {
        const ObIvfCidClusterCacheMgrKey &key = cache->mgr_key_;
        ObIvfCidClusterCache *stored = nullptr;
        const int get_ret = g_ivf_cid_cluster_cache_mgr_map.get_refactored(key, stored);
        if (OB_SUCCESS == get_ret && stored == cache) {
          const int erase_ret = g_ivf_cid_cluster_cache_mgr_map.erase_refactored(key);
          if (OB_SUCCESS == erase_ret || OB_HASH_NOT_EXIST == erase_ret) {
            destroy_cache = true;
          } else {
            ret = erase_ret;
            LOG_WARN("failed to erase ivf cid cluster cache from mgr map", K(ret), K(key));
            cache->inc_ref();
          }
        } else {
          destroy_cache = true;
        }
      }
    }
    if (destroy_cache) {
      cache->destroy();
      OB_DELETE(ObIvfCidClusterCache, IVF_CID_CLUSTER_CACHE_LABEL, cache);
    }
  }
}

} // namespace share
} // namespace oceanbase
