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
/// Rough bytes per full 50K/56 IVF_FLAT 1536-dim cluster for admission when entry size unknown.
static const int64_t IVF_CID_CLUSTER_CACHE_EST_BYTES_PER_CLUSTER = 5 * 1024 * 1024;
static const int64_t DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_CID = 4096;

int64_t ivf_cid_cluster_cache_fill_min_probe_access()
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS");
  if (env == nullptr || env[0] == '\0') {
    return DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS;
  }
  const int64_t v = static_cast<int64_t>(atoll(env));
  return v > 0 ? v : DEFAULT_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS;
}

int64_t ivf_cid_cluster_cache_max_cid()
{
  const char *env = getenv("OB_IVF_CID_CLUSTER_CACHE_MAX_CID");
  if (env == nullptr || env[0] == '\0') {
    return DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_CID;
  }
  const int64_t v = static_cast<int64_t>(atoll(env));
  return v > 0 ? v : DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_CID;
}

static const ObMemAttr IVF_CID_CLUSTER_CACHE_LABEL(OB_SERVER_TENANT_ID, "IvfCidClu");

typedef common::hash::ObHashMap<ObIvfCidClusterCacheMgrKey, ObIvfCidClusterCache *> IvfCidClusterCacheMgrMap;
static IvfCidClusterCacheMgrMap g_ivf_cid_cluster_cache_mgr_map;
static lib::ObMutex g_ivf_cid_cluster_cache_mgr_lock(common::ObLatchIds::VECTOR_IVF_CACHE_LOCK);
static bool g_ivf_cid_cluster_cache_mgr_inited = false;

ObIvfCidClusterEntry::~ObIvfCidClusterEntry()
{
  rows_.reset();
  if (OB_NOT_NULL(flat_fill_)) {
    ivf_cid_flat_fill_destroy(flat_fill_);
    flat_fill_ = nullptr;
  }
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
  int64_t d_get_hit = 0;
  int64_t d_get_miss = 0;
  int64_t d_put_ok = 0;
  int64_t d_put_skip_cap = 0;
  if (OB_NOT_NULL(cache_end) && OB_NOT_NULL(cache_begin)) {
    d_get_hit = cache_end->get_hit_cnt_ - cache_begin->get_hit_cnt_;
    d_get_miss = cache_end->get_miss_cnt_ - cache_begin->get_miss_cnt_;
    d_put_ok = cache_end->put_ok_cnt_ - cache_begin->put_ok_cnt_;
    d_put_skip_cap = cache_end->put_skip_cap_cnt_ - cache_begin->put_skip_cap_cnt_;
  }
  const char *phase_str = (phase != nullptr && phase[0] != '\0') ? phase : "final";
  const char *mode_str = (current_mode != nullptr && current_mode[0] != '\0') ? current_mode : "-";
  (void)snprintf(line,
      line_cap,
      "[OB_IVF_CID_CLUSTER_CACHE_STATS] phase=%s current_cid=%llu current_mode=%s "
      "ts_us=%lld tid=%lld n=%lld d=%lld c=%lld "
      "index_epoch=%llu algo=%lld cid_switch_cnt=%lld storage_only_cid_cnt=%lld replay_cid_cnt=%lld "
      "fill_cid_cnt=%lld cid_hit_rate=%f storage_fetch_us=%lld replay_serve_us=%lld fill_append_us=%lld "
      "acquire_us=%lld fill_row_cnt=%lld replay_row_cnt=%lld "
      "row_replay_rate=%f put_cid_ok=%lld put_cid_fail=%lld put_skip_pinned=%lld put_rows=%lld "
      "cache_d_get_hit=%lld cache_d_get_miss=%lld "
      "cache_d_put_ok=%lld cache_d_put_skip_cap=%lld cur_entry_cnt=%lld cur_bytes=%lld max_bytes=%lld\n",
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
      static_cast<long long>(s.fill_row_cnt_),
      static_cast<long long>(s.replay_row_cnt_),
      row_replay_rate,
      static_cast<long long>(s.put_cid_ok_cnt_),
      static_cast<long long>(s.put_cid_fail_cnt_),
      static_cast<long long>(s.put_cid_skip_pinned_cnt_),
      static_cast<long long>(s.put_rows_total_),
      static_cast<long long>(d_get_hit),
      static_cast<long long>(d_get_miss),
      static_cast<long long>(d_put_ok),
      static_cast<long long>(d_put_skip_cap),
      OB_NOT_NULL(cache_end) ? static_cast<long long>(cache_end->cur_entry_cnt_) : 0LL,
      OB_NOT_NULL(cache_end) ? static_cast<long long>(cache_end->cur_bytes_) : 0LL,
      OB_NOT_NULL(snap.cluster_cache_) ? static_cast<long long>(snap.cluster_cache_->get_max_bytes()) : 0LL);
}

} // namespace

int ob_ivf_cid_cluster_cache_query_hit_flag(const ObIvfCidClusterCacheLogSnapshot &snap)
{
  if (!snap.valid_ || !snap.cache_active_) {
    return 0;
  }
  return snap.session_.replay_cid_cnt_ > 0 ? 1 : 0;
}

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
             K(cs.put_ok_cnt_),
             K(cs.put_fail_cnt_),
             K(cs.put_skip_pinned_cnt_),
             K(cs.put_skip_cap_cnt_),
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
  explicit ObIvfCidFillGate(const uint64_t cid) : cid_(cid), filling_(false) {}
  uint64_t cid_;
  bool filling_;
};

struct FillGateFreeFn
{
  int operator()(const common::hash::HashMapPair<uint64_t, ObIvfCidFillGate *> &pair)
  {
    if (OB_NOT_NULL(pair.second)) {
      ob_free(pair.second);
    }
    return OB_SUCCESS;
  }
};

typedef lib::ObMutex IvfFillGateShardMutex;

ObIvfCidClusterCache::ObIvfCidClusterCache()
  : inited_(false),
    mgr_key_(),
    current_index_epoch_(1),
    max_bytes_(0),
    max_rows_per_cid_(0),
    replay_heat_weight_(DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT),
    max_cid_cnt_(0),
    ledger_bytes_total_(0),
    ref_cnt_(0),
    cid_states_(nullptr)
{
  for (int64_t i = 0; i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
    fill_gate_shard_locks_[i] = nullptr;
  }
  for (int64_t i = 0; i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
    fill_gate_shard_locks_[i] = OB_NEW(IvfFillGateShardMutex, IVF_CID_CLUSTER_CACHE_LABEL, common::ObLatchIds::VECTOR_IVF_CACHE_LOCK);
  }
}

int64_t ObIvfCidClusterCache::fill_gate_shard_idx_(const uint64_t cid) const
{
  return static_cast<int64_t>(cid % static_cast<uint64_t>(IVF_CID_FILL_GATE_SHARD_CNT));
}

common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *>
    &ObIvfCidClusterCache::fill_gate_map_shard_(const uint64_t cid)
{
  return fill_gate_map_shards_[fill_gate_shard_idx_(cid)];
}

const common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *>
    &ObIvfCidClusterCache::fill_gate_map_shard_(const uint64_t cid) const
{
  return fill_gate_map_shards_[fill_gate_shard_idx_(cid)];
}

lib::ObMutex &ObIvfCidClusterCache::fill_gate_shard_lock_(const uint64_t cid)
{
  return *fill_gate_shard_locks_[fill_gate_shard_idx_(cid)];
}

const lib::ObMutex &ObIvfCidClusterCache::fill_gate_shard_lock_(const uint64_t cid) const
{
  return *fill_gate_shard_locks_[fill_gate_shard_idx_(cid)];
}

uint64_t ObIvfCidClusterCache::load_index_epoch_() const
{
  return ATOMIC_LOAD(&current_index_epoch_);
}

int ObIvfCidClusterCache::ledger_remove_(const uint64_t cid, const bool erase_kv)
{
  ledger_drop_cached_(cid, erase_kv);
  return OB_SUCCESS;
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
    for (int64_t i = 0; OB_SUCC(ret) && i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
      if (OB_FAIL(fill_gate_map_shards_[i].create(64, attr, attr))) {
        LOG_WARN("failed to create fill gate map shard", K(ret), K(i));
      }
    }
    if (OB_FAIL(ret)) {
    } else {
      max_cid_cnt_ = ivf_cid_cluster_cache_max_cid();
      const int64_t state_bytes = max_cid_cnt_ * static_cast<int64_t>(sizeof(ObIvfCidPerCidState));
      cid_states_ = static_cast<ObIvfCidPerCidState *>(ob_malloc(state_bytes, IVF_CID_CLUSTER_CACHE_LABEL));
      if (OB_ISNULL(cid_states_)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("failed to alloc per-cid states", K(ret), K(max_cid_cnt_), K(state_bytes));
      } else {
        memset(cid_states_, 0, static_cast<size_t>(state_bytes));
      }
    }
    if (OB_SUCC(ret)) {
      mgr_key_ = key;
      current_index_epoch_ = 1;
      max_bytes_ = max_bytes > 0 ? max_bytes : DEFAULT_IVF_CID_CLUSTER_CACHE_MAX_MB * 1024L * 1024L;
      max_rows_per_cid_ = max_rows_per_cid;
      replay_heat_weight_ = replay_heat_weight > 0 ? replay_heat_weight : DEFAULT_IVF_CID_CLUSTER_REPLAY_HEAT_WEIGHT;
      ledger_bytes_total_ = 0;
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
  out.invalidate_cnt_ = ATOMIC_LOAD(&stats_.invalidate_cnt_);
  out.cur_entry_cnt_ = ATOMIC_LOAD(&stats_.cur_entry_cnt_);
  out.cur_bytes_ = ATOMIC_LOAD(&ledger_bytes_total_);
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
           K(s.cur_entry_cnt_),
           K(s.cur_bytes_),
           K(ref_cnt_));
}

void ObIvfCidClusterCache::destroy()
{
  clear_all_fill_gates_();
  for (int64_t i = 0; i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
    if (OB_NOT_NULL(fill_gate_shard_locks_[i])) {
      fill_gate_map_shards_[i].destroy();
    }
  }
  for (int64_t i = 0; i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
    if (OB_NOT_NULL(fill_gate_shard_locks_[i])) {
      OB_DELETE(IvfFillGateShardMutex, IVF_CID_CLUSTER_CACHE_LABEL, fill_gate_shard_locks_[i]);
      fill_gate_shard_locks_[i] = nullptr;
    }
  }
  if (OB_NOT_NULL(cid_states_) && max_cid_cnt_ > 0) {
    for (uint64_t cid = 0; cid < static_cast<uint64_t>(max_cid_cnt_); ++cid) {
      if (ATOMIC_LOAD(&cid_states_[cid].cached_bytes_) > 0) {
        (void)ledger_remove_(cid, false);
      }
    }
    ob_free(cid_states_);
    cid_states_ = nullptr;
  }
  max_cid_cnt_ = 0;
  inited_ = false;
  ledger_bytes_total_ = 0;
}

void ObIvfCidClusterCache::clear_all_fill_gates_()
{
  if (!inited_) {
    return;
  }
  FillGateFreeFn free_fn;
  for (int64_t i = 0; i < IVF_CID_FILL_GATE_SHARD_CNT; ++i) {
    if (OB_ISNULL(fill_gate_shard_locks_[i])) {
    } else {
      lib::ObMutexGuard fg_guard(*fill_gate_shard_locks_[i]);
      (void)fill_gate_map_shards_[i].foreach_refactored(free_fn);
      (void)fill_gate_map_shards_[i].clear();
    }
  }
}

void ObIvfCidClusterCache::inc_ref()
{
  ATOMIC_INC(&ref_cnt_);
}

void ObIvfCidClusterCache::dec_ref()
{
  ATOMIC_SAF(&ref_cnt_, 1);
}

ObIvfCidPerCidState *ObIvfCidClusterCache::cid_state_(const uint64_t cid)
{
  if (OB_ISNULL(cid_states_) || cid >= static_cast<uint64_t>(max_cid_cnt_)) {
    return nullptr;
  }
  return &cid_states_[cid];
}

const ObIvfCidPerCidState *ObIvfCidClusterCache::cid_state_(const uint64_t cid) const
{
  if (OB_ISNULL(cid_states_) || cid >= static_cast<uint64_t>(max_cid_cnt_)) {
    return nullptr;
  }
  return &cid_states_[cid];
}

bool ObIvfCidClusterCache::probe_heat_meets_fill_threshold_(const uint64_t cid) const
{
  const ObIvfCidPerCidState *st = cid_state_(cid);
  if (OB_ISNULL(st)) {
    return false;
  }
  const int64_t min_probe = ivf_cid_cluster_cache_fill_min_probe_access();
  return ATOMIC_LOAD(&st->probe_access_) >= static_cast<uint64_t>(min_probe);
}

bool ObIvfCidClusterCache::has_cache_space_(const uint64_t cid, int64_t need_bytes) const
{
  int64_t need = need_bytes;
  if (need <= 0) {
    need = IVF_CID_CLUSTER_CACHE_EST_BYTES_PER_CLUSTER;
  }
  const ObIvfCidPerCidState *st = cid_state_(cid);
  const int64_t old_bytes = OB_NOT_NULL(st) ? ATOMIC_LOAD(&st->cached_bytes_) : 0;
  const int64_t total = ATOMIC_LOAD(&ledger_bytes_total_);
  return total - old_bytes + need <= max_bytes_;
}

void ObIvfCidClusterCache::ledger_drop_cached_(const uint64_t cid, const bool erase_kv)
{
  ObIvfCidPerCidState *st = cid_state_(cid);
  if (OB_ISNULL(st)) {
    return;
  }
  const int64_t old_bytes = ATOMIC_LOAD(&st->cached_bytes_);
  if (old_bytes > 0) {
    ATOMIC_SAF(&ledger_bytes_total_, old_bytes);
    ATOMIC_STORE(&st->cached_bytes_, 0);
    ATOMIC_STORE(&st->phase_, static_cast<int64_t>(ObIvfCidClusterCidPhase::LEARNING));
    ATOMIC_DEC(&stats_.cur_entry_cnt_);
  }
  if (erase_kv) {
    ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
    (void)get_ivf_cid_cluster_kv_cache().erase_key(kv_key);
  }
}

void ObIvfCidClusterCache::ledger_commit_(const uint64_t cid, const int64_t bytes)
{
  ObIvfCidPerCidState *st = cid_state_(cid);
  if (OB_ISNULL(st)) {
    return;
  }
  const int64_t old_bytes = ATOMIC_LOAD(&st->cached_bytes_);
  if (old_bytes == 0) {
    ATOMIC_INC(&stats_.cur_entry_cnt_);
  }
  ATOMIC_AAF(&ledger_bytes_total_, bytes - old_bytes);
  ATOMIC_STORE(&st->cached_bytes_, bytes);
  ATOMIC_STORE(&st->phase_, static_cast<int64_t>(ObIvfCidClusterCidPhase::READY));
}

void ObIvfCidClusterCache::reset_all_cid_states_()
{
  if (OB_ISNULL(cid_states_) || max_cid_cnt_ <= 0) {
    return;
  }
  const int64_t state_bytes = max_cid_cnt_ * static_cast<int64_t>(sizeof(ObIvfCidPerCidState));
  memset(cid_states_, 0, static_cast<size_t>(state_bytes));
  ATOMIC_STORE(&ledger_bytes_total_, 0);
}

void ObIvfCidClusterCache::invalidate_all()
{
  clear_all_fill_gates_();
  ATOMIC_INC(&current_index_epoch_);
  if (OB_NOT_NULL(cid_states_) && max_cid_cnt_ > 0) {
    for (uint64_t cid = 0; cid < static_cast<uint64_t>(max_cid_cnt_); ++cid) {
      if (ATOMIC_LOAD(&cid_states_[cid].cached_bytes_) > 0) {
        (void)ledger_remove_(cid, false);
      }
    }
    reset_all_cid_states_();
  }
  ATOMIC_INC(&stats_.invalidate_cnt_);
}

ObIvfCidClusterCidPhase ObIvfCidClusterCache::get_cid_phase(const uint64_t cid) const
{
  const ObIvfCidPerCidState *st = cid_state_(cid);
  if (OB_ISNULL(st)) {
    return ObIvfCidClusterCidPhase::LEARNING;
  }
  return static_cast<ObIvfCidClusterCidPhase>(ATOMIC_LOAD(&st->phase_));
}

void ObIvfCidClusterCache::record_access_(const uint64_t cid)
{
  ObIvfCidPerCidState *st = cid_state_(cid);
  if (OB_NOT_NULL(st)) {
    ATOMIC_INC(&st->probe_access_);
  }
}

int ObIvfCidClusterCache::try_lookup_hit_(uint64_t cid, ObIvfCidClusterEntry *&entry)
{
  int ret = OB_SUCCESS;
  entry = nullptr;
  const uint64_t epoch = load_index_epoch_();
  ObIvfKvEntryPrep prep;
  const int prep_ret = load_kv_entry_prep_(cid, epoch, prep);
  if (prep_ret != OB_SUCCESS) {
    return prep_ret;
  }
  prep.view_entry_->session_owned_ = true;
  if (OB_FAIL(prep.view_entry_->session_kv_handle_.assign(prep.kv_handle_))) {
    discard_kv_entry_prep_(prep);
    ATOMIC_INC(&stats_.get_miss_cnt_);
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
  if (OB_ISNULL(entry) || !entry->session_owned_) {
    return;
  }
  entry->session_kv_handle_.reset();
  entry->session_owned_ = false;
  OB_DELETE(ObIvfCidClusterEntry, IVF_CID_CLUSTER_CACHE_LABEL, entry);
}

void ObIvfCidClusterCache::discard_kv_entry_prep_(ObIvfKvEntryPrep &prep)
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

int ObIvfCidClusterCache::load_kv_entry_prep_(uint64_t cid, uint64_t index_epoch, ObIvfKvEntryPrep &prep)
{
  int ret = OB_SUCCESS;
  discard_kv_entry_prep_(prep);
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
    LOG_WARN("failed to assign kv handle for kv entry prep", K(ret), K(cid));
    discard_kv_entry_prep_(prep);
    return ret;
  }
  return OB_SUCCESS;
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
    record_access_(cid);
    const int hit_ret = try_lookup_hit_(cid, entry);
    if (OB_SUCCESS == hit_ret) {
      result = ObIvfCidClusterLookupResult::HIT;
      return OB_SUCCESS;
    }
    if (hit_ret != OB_HASH_NOT_EXIST) {
      return hit_ret;
    }
    // Another session is FILLing this CID: read storage (MISS), do not wait or compete for FILL.
    if (cid_fill_in_progress_(cid)) {
      return OB_SUCCESS;
    }
    if (!probe_heat_meets_fill_threshold_(cid)) {
      return OB_SUCCESS;
    }
    if (!has_cache_space_(cid, 0)) {
      return OB_SUCCESS;
    }
    ret = try_acquire_fill_leader_(cid, entry, result, session_stats);
  }
  return ret;
}

bool ObIvfCidClusterCache::cid_fill_in_progress_(const uint64_t cid)
{
  if (!inited_) {
    return false;
  }
  lib::ObMutexGuard fg_guard(fill_gate_shard_lock_(cid));
  ObIvfCidFillGate *gate = nullptr;
  if (OB_SUCCESS == fill_gate_map_shard_(cid).get_refactored(cid, gate) && OB_NOT_NULL(gate)) {
    return gate->filling_;
  }
  return false;
}

int ObIvfCidClusterCache::try_acquire_fill_leader_(
    uint64_t cid,
    ObIvfCidClusterEntry *&entry,
    ObIvfCidClusterLookupResult &result,
    ObIvfCidClusterCacheSessionStats *session_stats)
{
  int ret = OB_SUCCESS;
  UNUSED(session_stats);
  entry = nullptr;
  result = ObIvfCidClusterLookupResult::MISS;
  bool is_fill_leader = false;
  ObIvfCidFillGate *gate = nullptr;
  while (OB_SUCC(ret)) {
    is_fill_leader = false;
    gate = nullptr;
    {
      lib::ObMutexGuard fg_guard(fill_gate_shard_lock_(cid));
      if (OB_FAIL(fill_gate_map_shard_(cid).get_refactored(cid, gate))) {
        if (ret == OB_HASH_NOT_EXIST) {
          ret = OB_SUCCESS;
          void *buf = ob_malloc(sizeof(ObIvfCidFillGate), IVF_CID_CLUSTER_CACHE_LABEL);
          if (OB_ISNULL(buf)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            gate = new (buf) ObIvfCidFillGate(cid);
            if (OB_FAIL(fill_gate_map_shard_(cid).set_refactored(cid, gate))) {
              if (ret == OB_HASH_EXIST) {
                ret = OB_SUCCESS;
                ob_free(gate);
                gate = nullptr;
                (void)fill_gate_map_shard_(cid).get_refactored(cid, gate);
              } else {
                LOG_WARN("failed to set fill gate", K(ret), K(cid));
                ob_free(gate);
                gate = nullptr;
              }
            }
          }
          if (OB_SUCC(ret) && OB_NOT_NULL(gate)) {
            gate->filling_ = true;
            is_fill_leader = true;
          }
        } else {
          LOG_WARN("failed to get fill gate", K(ret), K(cid));
        }
      } else if (OB_NOT_NULL(gate) && gate->filling_) {
        result = ObIvfCidClusterLookupResult::MISS;
        return OB_SUCCESS;
      } else if (OB_NOT_NULL(gate)) {
        (void)fill_gate_map_shard_(cid).erase_refactored(cid);
        ob_free(gate);
      }
    }
    if (is_fill_leader) {
      break;
    }
  }
  if (OB_SUCC(ret) && is_fill_leader) {
    result = ObIvfCidClusterLookupResult::FILL_LEADER;
  }
  return ret;
}

void ObIvfCidClusterCache::finish_cid_fill(uint64_t cid)
{
  ObIvfCidFillGate *gate = nullptr;
  lib::ObMutexGuard fg_guard(fill_gate_shard_lock_(cid));
  if (OB_SUCCESS == fill_gate_map_shard_(cid).get_refactored(cid, gate) && OB_NOT_NULL(gate)) {
    gate->filling_ = false;
    (void)fill_gate_map_shard_(cid).erase_refactored(cid);
    ob_free(gate);
  }
}

int ObIvfCidClusterCache::put(ObIvfCidClusterEntry &entry)
{
  int ret = OB_SUCCESS;
  if (!inited_) {
    ret = OB_NOT_INIT;
  } else if (max_rows_per_cid_ > 0 && entry.row_count_ > max_rows_per_cid_) {
    ret = OB_SIZE_OVERFLOW;
    ATOMIC_INC(&stats_.put_skip_rows_limit_cnt_);
  } else {
    const uint64_t cid = entry.cid_;
    const uint64_t fill_epoch = entry.index_epoch_;
    const uint64_t cur_epoch = load_index_epoch_();
    if (fill_epoch != 0 && fill_epoch != cur_epoch) {
      ATOMIC_INC(&stats_.get_stale_epoch_cnt_);
      return OB_STATE_NOT_MATCH;
    }
    entry.index_epoch_ = cur_epoch;
    if (!has_cache_space_(cid, entry.entry_bytes_)) {
      ATOMIC_INC(&stats_.put_skip_cap_cnt_);
      return OB_BUF_NOT_ENOUGH;
    }
    {
      ObIvfCidPerCidState *st = cid_state_(cid);
      if (OB_NOT_NULL(st)) {
        entry.heat_.access_cnt_ += ATOMIC_LOAD(&st->probe_access_);
        entry.heat_.replay_cnt_ += ATOMIC_LOAD(&st->replay_cnt_);
        ATOMIC_STORE(&st->probe_access_, 0);
        ATOMIC_STORE(&st->replay_cnt_, 0);
      }
    }
    if (entry.heat_.access_cnt_ == 0) {
      entry.heat_.access_cnt_ = 1;
    }
    entry.heat_.fill_cnt_++;
    entry.heat_.last_access_us_ = ObTimeUtility::current_time();
    char *flat_buf = nullptr;
    int64_t flat_len = 0;
    if (OB_NOT_NULL(entry.flat_fill_) && entry.row_count_ > 0) {
      if (OB_FAIL(ivf_cid_flat_fill_finalize(entry, entry.flat_fill_, flat_buf, flat_len))) {
        LOG_WARN("failed to finalize flat fill entry", K(ret), K(cid));
      }
    } else if (OB_FAIL(ivf_cid_flat_encode_entry(entry, flat_buf, flat_len))) {
      LOG_WARN("failed to encode flat entry", K(ret), K(cid));
    }
    if (OB_SUCC(ret)) {
      ObIvfCidClusterKVKey kv_key(mgr_key_, cid);
      ledger_drop_cached_(cid, false);
      if (OB_FAIL(get_ivf_cid_cluster_kv_cache().put_flat(kv_key, flat_buf, flat_len, true))) {
        LOG_WARN("failed to put flat into kv cache", K(ret), K(cid));
      } else {
        ledger_commit_(cid, entry.entry_bytes_);
        ATOMIC_INC(&stats_.put_ok_cnt_);
      }
      ivf_cid_flat_free_buf(flat_buf);
    }
    if (OB_FAIL(ret) && ret != OB_SIZE_OVERFLOW && ret != OB_BUF_NOT_ENOUGH && ret != OB_STATE_NOT_MATCH) {
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
