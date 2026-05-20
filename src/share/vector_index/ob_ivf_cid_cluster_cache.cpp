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
  // Default with STATS=1: progress every cid + session/final on query end.
  return 1;
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

typedef common::hash::HashMapPair<ObIvfCidClusterCache::EntryMapKey, ObIvfCidClusterEntry *> CidClusterEntryPair;

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

struct EvictVictimPicker
{
  explicit EvictVictimPicker(int64_t replay_heat_weight)
    : replay_heat_weight_(replay_heat_weight),
      min_score_(DBL_MAX),
      oldest_access_(INT64_MAX),
      victim_key_(),
      victim_(nullptr)
  {}
  int operator()(const CidClusterEntryPair &entry)
  {
    int ret = OB_SUCCESS;
    ObIvfCidClusterEntry *e = entry.second;
    if (OB_ISNULL(e) || e->pin_cnt_ > 0) {
    } else {
      const int64_t denom = OB_MAX(e->entry_bytes_, IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
      const double score =
          static_cast<double>(e->heat_.access_cnt_ + replay_heat_weight_ * e->heat_.replay_cnt_)
          / static_cast<double>(denom);
      if (score < min_score_ || (score == min_score_ && e->heat_.last_access_us_ < oldest_access_)) {
        min_score_ = score;
        oldest_access_ = e->heat_.last_access_us_;
        victim_key_ = entry.first;
        victim_ = e;
      }
    }
    return ret;
  }
  int64_t replay_heat_weight_;
  double min_score_;
  int64_t oldest_access_;
  ObIvfCidClusterCache::EntryMapKey victim_key_;
  ObIvfCidClusterEntry *victim_;
};

ObIvfCidClusterCache::ObIvfCidClusterCache()
  : inited_(false),
    mgr_key_(),
    current_index_epoch_(1),
    max_bytes_(0),
    max_rows_per_cid_(0),
    replay_heat_weight_(DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT),
    total_bytes_(0),
    ref_cnt_(0),
    entry_map_(),
    fill_gates_(),
    lock_(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK),
    fill_gates_lock_(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK)
{}

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
  } else if (OB_FAIL(entry_map_.create(128, attr, attr))) {
    LOG_WARN("failed to create entry map", K(ret));
  } else if (OB_FAIL(fill_gates_.create(64, attr, attr))) {
    LOG_WARN("failed to create fill gates map", K(ret));
  } else if (OB_FAIL(probe_heat_map_.create(128, attr, attr))) {
    LOG_WARN("failed to create probe heat map", K(ret));
  } else {
    mgr_key_ = key;
    current_index_epoch_ = 1;
    max_bytes_ = max_bytes > 0 ? max_bytes : DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_MB * 1024L * 1024L;
    max_rows_per_cid_ = max_rows_per_cid;
    replay_heat_weight_ = replay_heat_weight > 0 ? replay_heat_weight : DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT;
    total_bytes_ = 0;
    stats_.reset();
    inited_ = true;
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
  out.cur_bytes_ = ATOMIC_LOAD(&total_bytes_);
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
  {
    lib::ObMutexGuard guard(lock_);
    (void)probe_heat_map_.destroy();
  }
  lib::ObMutexGuard guard(lock_);
  struct EntryFreeFn
  {
    int operator()(const CidClusterEntryPair &entry)
    {
      ObIvfCidClusterEntry *to_del = entry.second;
      if (OB_NOT_NULL(to_del)) {
        OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, to_del);
      }
      return OB_SUCCESS;
    }
  } free_fn;
  entry_map_.foreach_refactored(free_fn);
  entry_map_.destroy();
  inited_ = false;
  total_bytes_ = 0;
}

void ObIvfCidClusterCache::clear_all_fill_gates_()
{
  if (!inited_) {
    return;
  }
  lib::ObMutexGuard fg_guard(fill_gates_lock_);
  FillGateWakeAndFreeFn free_fn;
  (void)fill_gates_.foreach_refactored(free_fn);
}

void ObIvfCidClusterCache::inc_ref()
{
  ATOMIC_INC(&ref_cnt_);
}

void ObIvfCidClusterCache::dec_ref()
{
  ATOMIC_SAF(&ref_cnt_, 1);
}

double ObIvfCidClusterCache::calc_heat_score(const ObIvfCidClusterEntry &e) const
{
  const int64_t denom = OB_MAX(e.entry_bytes_, IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
  return static_cast<double>(e.heat_.access_cnt_ + replay_heat_weight_ * e.heat_.replay_cnt_) / static_cast<double>(denom);
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

double ObIvfCidClusterCache::calc_prospective_heat_score_(uint64_t cid) const
{
  ObIvfCidClusterHeat heat;
  if (OB_SUCCESS != probe_heat_map_.get_refactored(cid, heat)) {
    heat.access_cnt_ = 1;
    heat.replay_cnt_ = 0;
  }
  return calc_probe_heat_score_(heat);
}

double ObIvfCidClusterCache::calc_probe_heat_score_(const ObIvfCidClusterHeat &heat) const
{
  return static_cast<double>(heat.access_cnt_ + replay_heat_weight_ * heat.replay_cnt_)
      / static_cast<double>(IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
}

bool ObIvfCidClusterCache::is_probe_top_k_for_fill_locked_(uint64_t cid, const double target_score) const
{
  const int64_t top_k = ivf_cid_cluster_cache_fill_top_k(max_bytes_);
  const int64_t min_probe_access = ivf_cid_cluster_cache_fill_min_probe_access();
  int64_t strictly_higher = 0;
  struct CountStrictlyHigherFn
  {
    CountStrictlyHigherFn(const uint64_t cid,
        const double target_score,
        const int64_t min_probe_access,
        const int64_t replay_heat_weight,
        int64_t &strictly_higher)
        : cid_(cid),
          target_score_(target_score),
          min_probe_access_(min_probe_access),
          replay_heat_weight_(replay_heat_weight),
          strictly_higher_(strictly_higher)
    {}
    int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidClusterHeat> &pair)
    {
      const ObIvfCidClusterHeat &heat = pair.second;
      if (heat.access_cnt_ < min_probe_access_) {
      } else if (pair.first != cid_) {
        const double score = static_cast<double>(heat.access_cnt_ + replay_heat_weight_ * heat.replay_cnt_)
            / static_cast<double>(IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES);
        if (score > target_score_) {
          strictly_higher_++;
        }
      }
      return OB_SUCCESS;
    }
    uint64_t cid_;
    double target_score_;
    int64_t min_probe_access_;
    int64_t replay_heat_weight_;
    int64_t &strictly_higher_;
  } count_fn(cid, target_score, min_probe_access, replay_heat_weight_, strictly_higher);
  (void)probe_heat_map_.foreach_refactored(count_fn);
  return strictly_higher < top_k;
}

bool ObIvfCidClusterCache::should_admit_fill_locked_(uint64_t cid)
{
  bump_probe_heat_locked_(cid);
  ObIvfCidClusterHeat probe_heat;
  int64_t probe_access = 1;
  if (OB_SUCCESS == probe_heat_map_.get_refactored(cid, probe_heat)) {
    probe_access = probe_heat.access_cnt_;
  }
  const int64_t min_probe_access = ivf_cid_cluster_cache_fill_min_probe_access();
  if (probe_access < min_probe_access) {
    ATOMIC_INC(&stats_.fill_bypass_cnt_);
    return false;
  }
  const double new_score = calc_prospective_heat_score_(cid);
  if (!is_probe_top_k_for_fill_locked_(cid, new_score)) {
    ATOMIC_INC(&stats_.fill_bypass_cnt_);
    return false;
  }
  if (total_bytes_ + IVF_CID_CLUSTER_CACHE_MIN_ENTRY_BYTES <= max_bytes_) {
    return true;
  }
  EvictVictimPicker picker(replay_heat_weight_);
  (void)entry_map_.foreach_refactored(picker);
  if (OB_ISNULL(picker.victim_)) {
    ATOMIC_INC(&stats_.fill_bypass_cnt_);
    return false;
  }
  // Only replace when strictly hotter than the coldest evictable entry (equal → STORAGE_ONLY, avoid thrash).
  if (new_score <= picker.min_score_) {
    ATOMIC_INC(&stats_.fill_bypass_cnt_);
    return false;
  }
  return true;
}

void ObIvfCidClusterCache::clear_probe_heat_locked_(uint64_t cid)
{
  (void)probe_heat_map_.erase_refactored(cid);
}

int ObIvfCidClusterCache::erase_entry(const EntryMapKey &key, const bool count_evict)
{
  int ret = OB_SUCCESS;
  ObIvfCidClusterEntry *entry = nullptr;
  if (OB_FAIL(entry_map_.erase_refactored(key, &entry))) {
    if (ret != OB_HASH_NOT_EXIST) {
      LOG_WARN("failed to erase entry", K(ret), K(key.cid_));
    }
  } else if (OB_NOT_NULL(entry)) {
    if (count_evict) {
      ATOMIC_INC(&stats_.evict_entry_cnt_);
      ATOMIC_AAF(&stats_.evict_bytes_, entry->entry_bytes_);
    }
    total_bytes_ -= entry->entry_bytes_;
    ATOMIC_DEC(&stats_.cur_entry_cnt_);
    OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, entry);
  }
  return ret;
}

int ObIvfCidClusterCache::evict_until(int64_t need_bytes)
{
  int ret = OB_SUCCESS;
  while (total_bytes_ + need_bytes > max_bytes_ && entry_map_.size() > 0) {
    EvictVictimPicker picker(replay_heat_weight_);
    if (OB_FAIL(entry_map_.foreach_refactored(picker))) {
      LOG_WARN("failed to foreach entry map", K(ret));
      break;
    } else if (OB_ISNULL(picker.victim_)) {
      ret = OB_BUF_NOT_ENOUGH;
      break;
    } else if (OB_FAIL(erase_entry(picker.victim_key_, true))) {
      LOG_WARN("failed to erase victim", K(ret));
      break;
    }
  }
  if (OB_SUCC(ret) && total_bytes_ + need_bytes > max_bytes_) {
    ret = OB_BUF_NOT_ENOUGH;
  }
  return ret;
}

void ObIvfCidClusterCache::invalidate_all()
{
  {
    lib::ObMutexGuard fg_guard(fill_gates_lock_);
    FillGateWakeAndFreeFn free_fn;
    (void)fill_gates_.foreach_refactored(free_fn);
  }
  lib::ObMutexGuard guard(lock_);
  current_index_epoch_++;
  common::ObArray<uint64_t> cids_to_erase;
  struct CollectUnpinnedFn
  {
    explicit CollectUnpinnedFn(common::ObArray<uint64_t> &cids) : cids_(cids) {}
    int operator()(const CidClusterEntryPair &entry)
    {
      ObIvfCidClusterEntry *e = entry.second;
      if (OB_NOT_NULL(e) && e->pin_cnt_ == 0) {
        (void)cids_.push_back(entry.first.cid_);
      }
      return OB_SUCCESS;
    }
    common::ObArray<uint64_t> &cids_;
  } collect_fn(cids_to_erase);
  (void)entry_map_.foreach_refactored(collect_fn);
  for (int64_t i = 0; i < cids_to_erase.count(); ++i) {
    (void)erase_entry(EntryMapKey(cids_to_erase.at(i)), false);
  }
  ATOMIC_INC(&stats_.invalidate_cnt_);
}

int ObIvfCidClusterCache::unpin_entry(ObIvfCidClusterEntry *entry)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(entry)) {
  } else {
    lib::ObMutexGuard guard(lock_);
    EntryMapKey key(entry->cid_);
    ObIvfCidClusterEntry *stored = nullptr;
    if (OB_FAIL(entry_map_.get_refactored(key, stored))) {
      if (ret != OB_HASH_NOT_EXIST) {
        LOG_WARN("failed to lookup entry on unpin", K(ret), K(key.cid_));
      }
    } else if (stored == entry && entry->pin_cnt_ > 0) {
      entry->pin_cnt_--;
      if (entry->pin_cnt_ == 0 && entry->index_epoch_ != current_index_epoch_) {
        (void)erase_entry(key, false);
      }
    }
  }
  return ret;
}

int ObIvfCidClusterCache::try_pin_ready_entry_(uint64_t cid, ObIvfCidClusterEntry *&entry)
{
  int ret = OB_SUCCESS;
  entry = nullptr;
  const int64_t now = ObTimeUtility::current_time();
  const EntryMapKey key(cid);
  if (OB_FAIL(entry_map_.get_refactored(key, entry))) {
    if (ret == OB_HASH_NOT_EXIST) {
      ATOMIC_INC(&stats_.get_miss_cnt_);
    } else {
      LOG_WARN("failed to get cluster entry", K(ret), K(cid));
    }
  } else if (OB_ISNULL(entry)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (entry->index_epoch_ != current_index_epoch_) {
    ATOMIC_INC(&stats_.get_stale_epoch_cnt_);
    ATOMIC_INC(&stats_.get_miss_cnt_);
    entry = nullptr;
    ret = OB_HASH_NOT_EXIST;
  } else {
    ATOMIC_INC(&stats_.get_hit_cnt_);
    entry->pin_cnt_++;
    entry->heat_.access_cnt_++;
    entry->heat_.replay_cnt_++;
    entry->heat_.last_access_us_ = now;
    ObIvfCidClusterHeat probe_heat;
    if (OB_SUCCESS == probe_heat_map_.get_refactored(cid, probe_heat)) {
      entry->heat_.access_cnt_ += probe_heat.access_cnt_;
      entry->heat_.replay_cnt_ += probe_heat.replay_cnt_;
      (void)probe_heat_map_.erase_refactored(cid);
    }
  }
  return ret;
}

int ObIvfCidClusterCache::acquire_cid_cluster(
    uint64_t cid,
    ObIvfCidClusterEntry *&entry,
    bool &is_fill_leader,
    bool &storage_only,
    ObIvfCidClusterCacheSessionStats *session_stats)
{
  int ret = OB_SUCCESS;
  is_fill_leader = false;
  storage_only = false;
  entry = nullptr;
  if (!inited_) {
    ret = OB_NOT_INIT;
  }
  while (OB_SUCC(ret)) {
    {
      lib::ObMutexGuard guard(lock_);
      ret = try_pin_ready_entry_(cid, entry);
    }
    if (OB_SUCCESS == ret) {
      break;
    }
    if (OB_HASH_NOT_EXIST != ret) {
      break;
    }
    ret = OB_SUCCESS;

    {
      lib::ObMutexGuard guard(lock_);
      if (!should_admit_fill_locked_(cid)) {
        storage_only = true;
        break;
      }
    }

    ObIvfCidFillGate *gate = nullptr;
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
            } else {
              gate->filling_ = true;
              (void)gate->cond_.unlock();
              {
                lib::ObMutexGuard guard(lock_);
                if (OB_SUCCESS == try_pin_ready_entry_(cid, entry)) {
                  if (OB_SUCCESS == gate->cond_.lock()) {
                    gate->filling_ = false;
                    (void)gate->cond_.broadcast();
                    (void)gate->cond_.unlock();
                  }
                  (void)fill_gates_.erase_refactored(cid);
                  gate->dec_ref();
                  break;
                }
              }
              is_fill_leader = true;
              break;
            }
          }
        } else {
          LOG_WARN("failed to get fill gate", K(ret), K(cid));
        }
      } else if (OB_NOT_NULL(gate) && gate->filling_) {
        wait_fill = true;
      }
    }

    if (OB_SUCC(ret) && wait_fill && OB_NOT_NULL(gate)) {
      gate->inc_ref();
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
      }
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
  gate->dec_ref();
}

int ObIvfCidClusterCache::put(ObIvfCidClusterEntry &entry)
{
  int ret = OB_SUCCESS;
  if (max_rows_per_cid_ > 0 && entry.row_count_ > max_rows_per_cid_) {
    ret = OB_SIZE_OVERFLOW;
    ATOMIC_INC(&stats_.put_skip_rows_limit_cnt_);
  } else {
    lib::ObMutexGuard guard(lock_);
    entry.index_epoch_ = current_index_epoch_;
    const EntryMapKey key(entry.cid_);
    ObIvfCidClusterEntry *stored = nullptr;
    if (OB_FAIL(entry_map_.get_refactored(key, stored))) {
      if (ret == OB_HASH_NOT_EXIST) {
        ret = OB_SUCCESS;
      } else {
        LOG_WARN("failed to lookup entry", K(ret));
      }
    }
    if (OB_SUCC(ret) && OB_NOT_NULL(stored)) {
      if (stored->pin_cnt_ > 0) {
        ATOMIC_INC(&stats_.put_skip_pinned_cnt_);
        return OB_EAGAIN;
      } else if (stored->index_epoch_ == current_index_epoch_) {
        if (OB_FAIL(erase_entry(key, false))) {
          LOG_WARN("failed to erase old cluster entry before put", K(ret), K(key.cid_));
        }
      } else if (OB_FAIL(erase_entry(key, false))) {
        LOG_WARN("failed to erase stale epoch cluster entry before put", K(ret), K(key.cid_));
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(evict_until(entry.entry_bytes_))) {
        if (ret == OB_BUF_NOT_ENOUGH) {
          ATOMIC_INC(&stats_.put_skip_cap_cnt_);
          return OB_BUF_NOT_ENOUGH;
        }
        LOG_WARN("failed to evict", K(ret), K(entry.entry_bytes_));
      } else if (OB_ISNULL(stored = OB_NEW(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        stored->index_epoch_ = entry.index_epoch_;
        stored->cid_ = entry.cid_;
        stored->row_count_ = entry.row_count_;
        stored->entry_bytes_ = entry.entry_bytes_;
        stored->heat_ = entry.heat_;
        stored->arena_ = entry.arena_;
        entry.arena_ = nullptr;
        if (OB_FAIL(stored->rows_.assign(entry.rows_))) {
          LOG_WARN("failed to assign rows", K(ret));
          OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, stored);
        } else if (OB_FAIL(entry_map_.set_refactored(key, stored))) {
          LOG_WARN("failed to set entry", K(ret));
          OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, stored);
        } else {
          if (stored->heat_.access_cnt_ == 0) {
            stored->heat_.access_cnt_ = 1;
          }
          stored->heat_.fill_cnt_++;
          stored->heat_.last_access_us_ = ObTimeUtility::current_time();
          total_bytes_ += stored->entry_bytes_;
          ATOMIC_INC(&stats_.put_ok_cnt_);
          ATOMIC_INC(&stats_.cur_entry_cnt_);
          clear_probe_heat_locked_(entry.cid_);
        }
      }
    }
    if (OB_FAIL(ret) && ret != OB_SIZE_OVERFLOW) {
      ATOMIC_INC(&stats_.put_fail_cnt_);
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
          cache->inc_ref();
        }
      } else {
        LOG_WARN("failed to get cache", K(ret));
      }
    } else if (OB_NOT_NULL(cache)) {
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
      if (ref_after_dec > 0) {
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
