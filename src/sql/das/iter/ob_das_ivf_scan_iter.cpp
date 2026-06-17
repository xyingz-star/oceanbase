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
#include "sql/das/iter/ob_das_ivf_scan_iter.h"
#include "sql/das/iter/ob_das_ivf_cid_vec_cache_scan_iter.h"
#include "sql/das/iter/ob_das_ivf_per_query_stats.h"
#include "share/vector_index/ob_ivf_cid_cluster_kv_cache.h"
#include "lib/time/ob_time_utility.h"
#include "sql/das/ob_das_scan_op.h"
#include "storage/tx_storage/ob_access_service.h"
#include "src/storage/access/ob_table_scan_iterator.h"
#include "share/vector_type/ob_vector_common_util.h"
#include "sql/engine/expr/ob_expr_vec_ivf_sq8_data_vector.h"
#include "sql/engine/expr/ob_array_expr_utils.h"
#include "share/ob_vec_index_builder_util.h"
#include "sql/das/iter/ob_das_vec_scan_utils.h"
#include "lib/roaringbitmap/ob_rb_memory_mgr.h"
#include "deps/oblib/src/lib/vector/ob_vector_util.h"
#include "share/vector_type/ob_ivf_sq8_latent_decode.h"
#include "share/vector_type/ob_ivf_sq8_latent_fused_distance.h"
#include "lib/file/file_directory_utils.h"
#include "lib/ob_define.h"
#include "lib/random/ob_random.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <type_traits>

namespace oceanbase
{
using namespace common;
using namespace storage;
using namespace transaction;
using namespace share;
using namespace sql;

namespace sql
{

namespace {

/// Remaining ``fine_wall`` not covered by storage_fetch + load + compute (scan open, reuse, heap finalize, etc.).
OB_INLINE int64_t ob_ivf_latency_breakdown_fine_cv_untracked_us(const ObIvfLatencyBreakdown &lat)
{
  const int64_t accounted =
      lat.fine_cv_storage_fetch_us_ + lat.fine_load_us_ + lat.fine_compute_us_;
  const int64_t residual = lat.fine_wall_us_ - accounted;
  return residual > 0 ? residual : 0;
}

OB_INLINE int ob_ivf_lat_create_parent_dirs_for_file(const char *file_path)
{
  int ret = OB_SUCCESS;
  char buf[common::FileDirectoryUtils::MAX_PATH + 1];
  const size_t n = file_path != nullptr ? strlen(file_path) : 0;
  if (OB_ISNULL(file_path) || n == 0 || n >= sizeof(buf)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    MEMCPY(buf, file_path, n + 1);
    char *slash = static_cast<char *>(strrchr(buf, '/'));
    if (slash != nullptr && slash != buf) {
      *slash = '\0';
      ret = common::FileDirectoryUtils::create_full_path(buf);
    }
  }
  return ret;
}

/// Expand OB_IVF_LATENCY_BREAKDOWN_LOG_FILE template: "%t" tid, "%r" dataset rows, "%m" dim,
/// "%c" IVF nlist (centroids), "%%" -> "%".
OB_INLINE bool ob_ivf_lat_expand_breakdown_path_template(
    const char *tmpl,
    const int64_t tid,
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist_centers,
    char *out,
    const size_t out_cap)
{
  bool ok = true;
  size_t o = 0;
  for (const char *p = tmpl; *p != '\0' && ok; ) {
    if (*p == '%' && p[1] != '\0') {
      const char spec = p[1];
      char numbuf[64];
      if (spec == '%') {
        if (o + 1 >= out_cap) {
          ok = false;
        } else {
          out[o++] = '%';
        }
        p += 2;
      } else if (spec == 't' || spec == 'r' || spec == 'm' || spec == 'c') {
        int nw = 0;
        if (spec == 't') {
          nw = snprintf(numbuf, sizeof(numbuf), "%lld", static_cast<long long>(tid));
        } else if (spec == 'r') {
          nw = snprintf(numbuf, sizeof(numbuf), "%lld", static_cast<long long>(dataset_rows));
        } else if (spec == 'm') {
          nw = snprintf(numbuf, sizeof(numbuf), "%lld", static_cast<long long>(dim));
        } else {
          nw = snprintf(numbuf, sizeof(numbuf), "%lld", static_cast<long long>(nlist_centers));
        }
        if (nw <= 0 || o + static_cast<size_t>(nw) >= out_cap) {
          ok = false;
        } else {
          MEMCPY(out + o, numbuf, static_cast<size_t>(nw));
          o += static_cast<size_t>(nw);
        }
        p += 2;
      } else {
        if (o + 2 >= out_cap) {
          ok = false;
        } else {
          out[o++] = '%';
          out[o++] = spec;
        }
        p += 2;
      }
    } else {
      if (o + 1 >= out_cap) {
        ok = false;
      } else {
        out[o++] = *p++;
      }
    }
  }
  if (ok && o < out_cap) {
    out[o] = '\0';
  } else {
    ok = false;
    if (out_cap > 0) {
      out[0] = '\0';
    }
  }
  return ok;
}

OB_INLINE bool ob_ivf_lat_tmpl_has_dataset_placeholders(const char *env_file)
{
  return env_file != nullptr && env_file[0] != '\0' &&
         (strstr(env_file, "%r") != nullptr || strstr(env_file, "%m") != nullptr ||
             strstr(env_file, "%c") != nullptr);
}

/// Inserts ".n<rows>_d<dim>_c<nlist>" immediately before a trailing ".log" suffix.
OB_INLINE bool ob_ivf_lat_inject_rows_dim_nlist_before_dot_log(
    char *path_buf,
    const size_t buf_cap,
    const int64_t dataset_rows,
    const int64_t dim,
    const int64_t nlist_centers)
{
  bool ok = false;
  const char *const dot_log = strrchr(path_buf, '.');
  if (dot_log != nullptr && 0 == strcmp(dot_log, ".log")) {
    const int base_len = static_cast<int>(dot_log - path_buf);
    char tmp[common::FileDirectoryUtils::MAX_PATH + 1];
    const int n = snprintf(tmp,
        sizeof(tmp),
        "%.*s.n%lld_d%lld_c%lld.log",
        base_len,
        path_buf,
        static_cast<long long>(dataset_rows),
        static_cast<long long>(dim),
        static_cast<long long>(nlist_centers));
    if (n > 0 && n < static_cast<int>(sizeof(tmp)) && static_cast<size_t>(n) < buf_cap) {
      MEMCPY(path_buf, tmp, static_cast<size_t>(n) + 1);
      ok = true;
    }
  }
  return ok;
}

/// Also append one line to $HOME/log/ob_ivf_latency_breakdown.n*_d*_c*.<tid>.log
/// (Observer unix user's HOME), or OB_IVF_LATENCY_BREAKDOWN_LOG_FILE if set.
/// Does not change normal observer.log behavior.
OB_INLINE void ob_ivf_latency_breakdown_append_user_log(
    const bool is_vectorized,
    const ObIvfLatencyBreakdown &lat,
    const int64_t misc_us,
    const bool is_brute_force,
    const int64_t cid_vec_scan_rows,
    const int64_t vec_dist_calc_cnt,
    const ObVecIndexType vec_index_type,
    const int64_t dim,
    const int64_t nprobes,
    const int64_t dataset_rows,
    const int64_t nlist_centers,
    const share::ObIvfCidClusterCacheLogSnapshot &cache_snap)
{
  thread_local bool tls_ivf_lat_brk_has_key = false;
  thread_local int64_t tls_ivf_lat_brk_rows = 0;
  thread_local int64_t tls_ivf_lat_brk_dim = 0;
  thread_local int64_t tls_ivf_lat_brk_nlist = 0;
  thread_local char tls_ivf_lat_brk_path[common::FileDirectoryUtils::MAX_PATH + 1];

  const char *path = nullptr;
  char path_buf[common::FileDirectoryUtils::MAX_PATH + 1];

  if (tls_ivf_lat_brk_has_key && dataset_rows == tls_ivf_lat_brk_rows && dim == tls_ivf_lat_brk_dim &&
      nlist_centers == tls_ivf_lat_brk_nlist && tls_ivf_lat_brk_path[0] != '\0') {
    path = tls_ivf_lat_brk_path;
  } else {
    const int64_t tid = GETTID();
    const char *const env_file = ::getenv("OB_IVF_LATENCY_BREAKDOWN_LOG_FILE");
    if (env_file != nullptr && env_file[0] != '\0') {
      if (!ob_ivf_lat_expand_breakdown_path_template(
              env_file, tid, dataset_rows, dim, nlist_centers, path_buf, sizeof(path_buf))) {
        return;
      }
      if (!ob_ivf_lat_tmpl_has_dataset_placeholders(env_file)) {
        (void)ob_ivf_lat_inject_rows_dim_nlist_before_dot_log(
            path_buf, sizeof(path_buf), dataset_rows, dim, nlist_centers);
      }
      const bool orig_has_pct_t = (strstr(env_file, "%t") != nullptr);
      if (!orig_has_pct_t) {
        char tmp[common::FileDirectoryUtils::MAX_PATH + 1];
        const size_t plen = strlen(path_buf);
        if (plen >= sizeof(tmp)) {
          return;
        }
        MEMCPY(tmp, path_buf, plen + 1);
        const char *const ext = strrchr(tmp, '.');
        const bool has_log_ext = (ext != nullptr && 0 == strcmp(ext, ".log"));
        if (has_log_ext) {
          const int base_len = static_cast<int>(ext - tmp);
          const int n = snprintf(path_buf, sizeof(path_buf), "%.*s.%lld.log",
              base_len, tmp, static_cast<long long>(tid));
          if (n <= 0 || n >= static_cast<int>(sizeof(path_buf))) {
            return;
          }
        } else {
          const int n = snprintf(path_buf, sizeof(path_buf), "%s.%lld",
              tmp, static_cast<long long>(tid));
          if (n <= 0 || n >= static_cast<int>(sizeof(path_buf))) {
            return;
          }
        }
      }
      path = path_buf;
      (void)ob_ivf_lat_create_parent_dirs_for_file(path);
    } else {
      const char *home = ::getenv("HOME");
      if (home == nullptr || home[0] == '\0') {
        return;
      }
      {
        char dir_buf[common::FileDirectoryUtils::MAX_PATH + 1];
        const int nd = snprintf(dir_buf, sizeof(dir_buf), "%s/log", home);
        if (nd > 0 && nd < static_cast<int>(sizeof(dir_buf))) {
          (void)common::FileDirectoryUtils::create_full_path(dir_buf);
        }
      }
      const int nf = snprintf(path_buf,
          sizeof(path_buf),
          "%s/log/ob_ivf_latency_breakdown.n%lld_d%lld_c%lld.%lld.log",
          home,
          static_cast<long long>(dataset_rows),
          static_cast<long long>(dim),
          static_cast<long long>(nlist_centers),
          static_cast<long long>(tid));
      if (nf <= 0 || nf >= static_cast<int>(sizeof(path_buf))) {
        return;
      }
      path = path_buf;
    }
    const size_t pn = strlen(path);
    if (pn >= sizeof(tls_ivf_lat_brk_path)) {
      return;
    }
    MEMCPY(tls_ivf_lat_brk_path, path, pn + 1);
    tls_ivf_lat_brk_has_key = true;
    tls_ivf_lat_brk_rows = dataset_rows;
    tls_ivf_lat_brk_dim = dim;
    tls_ivf_lat_brk_nlist = nlist_centers;
    path = tls_ivf_lat_brk_path;
  }
  FILE *fp = ::fopen(path, "ae");
  if (OB_ISNULL(fp)) {
    fp = ::fopen(path, "a");
  }
  if (OB_ISNULL(fp)) {
    return;
  }
  char line[6144];
  const int64_t fine_cv_untracked_us = ob_ivf_latency_breakdown_fine_cv_untracked_us(lat);
  const int64_t lat_cid_cluster_cache_active =
      (cache_snap.valid_ && cache_snap.cache_active_) ? 1 : 0;
  const int64_t lat_query_replay_cache = share::ob_ivf_cid_cluster_cache_query_hit_flag(cache_snap);
  const int nl = snprintf(
      line,
      sizeof(line),
      "[OB_IVF_LATENCY_BREAKDOWN] is_vectorized=%d ivf_total_us=%lld finalize_us=%lld das_body_us=%lld "
      "coarse_wall_us=%lld fine_wall_us=%lld fine_cv_storage_fetch_us=%lld fine_load_us=%lld "
      "fine_compute_us=%lld fine_cv_untracked_us=%lld brute_wall_us=%lld sq8_prep_us=%lld pq_prep_us=%lld "
      "flat_prep_us=%lld misc_us=%lld "
      "is_brute_force=%d cid_vec_scan_rows=%lld vec_dist_calc_cnt=%lld vec_index_type=%d dim=%lld "
      "nprobes=%lld cid_cluster_cache_active=%d query_cid_cache_hit=%d\n",
      static_cast<int>(is_vectorized),
      static_cast<long long>(lat.ivf_total_us_),
      static_cast<long long>(lat.finalize_us_),
      static_cast<long long>(lat.das_body_us_),
      static_cast<long long>(lat.coarse_wall_us_),
      static_cast<long long>(lat.fine_wall_us_),
      static_cast<long long>(lat.fine_cv_storage_fetch_us_),
      static_cast<long long>(lat.fine_load_us_),
      static_cast<long long>(lat.fine_compute_us_),
      static_cast<long long>(fine_cv_untracked_us),
      static_cast<long long>(lat.brute_wall_us_),
      static_cast<long long>(lat.sq8_prep_us_),
      static_cast<long long>(lat.pq_prep_us_),
      static_cast<long long>(lat.flat_prep_us_),
      static_cast<long long>(misc_us),
      static_cast<int>(is_brute_force),
      static_cast<long long>(cid_vec_scan_rows),
      static_cast<long long>(vec_dist_calc_cnt),
      static_cast<int>(vec_index_type),
      static_cast<long long>(dim),
      static_cast<long long>(nprobes),
      static_cast<int>(lat_cid_cluster_cache_active),
      static_cast<int>(lat_query_replay_cache));
  if (nl > 0 && nl < static_cast<int>(sizeof(line))) {
    (void)::fwrite(line, 1, static_cast<size_t>(nl), fp);
  }
  (void)share::ob_ivf_cid_cluster_cache_append_latency_log_file(fp, cache_snap);
  (void)::fflush(fp);
  (void)::fclose(fp);
}

struct ObIvfCoarseWallGuard {
  ObIvfLatencyBreakdown *lat_;
  int64_t start_;
  explicit ObIvfCoarseWallGuard(ObIvfLatencyBreakdown &lat)
      : lat_(lat.enabled_ ? &lat : nullptr), start_(lat_ != nullptr ? ObTimeUtility::current_time() : 0)
  {}
  ~ObIvfCoarseWallGuard()
  {
    if (lat_ != nullptr) {
      lat_->coarse_wall_us_ += ObTimeUtility::current_time() - start_;
    }
  }
};

struct ObIvfFineWallIterGuard {
  ObIvfLatencyBreakdown *lat_;
  int64_t start_;
  explicit ObIvfFineWallIterGuard(ObIvfLatencyBreakdown &lat)
      : lat_(lat.enabled_ ? &lat : nullptr), start_(lat_ != nullptr ? ObTimeUtility::current_time() : 0)
  {}
  ~ObIvfFineWallIterGuard()
  {
    if (lat_ != nullptr) {
      lat_->fine_wall_us_ += ObTimeUtility::current_time() - start_;
    }
  }
};

OB_INLINE bool ivf_sq8_dis_needs_latent_float_scoring(ObExprVectorDistance::ObVecDisType dis_type)
{
  return dis_type != ObExprVectorDistance::ObVecDisType::HAMMING
      && dis_type != ObExprVectorDistance::ObVecDisType::MAX_TYPE;
}

// Default ON: IVF_SQ8 distance uses raw query floats (real_search_vec_) vs SQ8+meta reconstructed candidates.
// Set OB_IVF_SQ8_QUERY_FLOAT_DISTANCE=0 on observer to use legacy path (quantize query to u8 then decode to q_lat).
OB_INLINE bool ivf_sq8_env_use_query_float_for_distance()
{
  const char *const e = ::getenv("OB_IVF_SQ8_QUERY_FLOAT_DISTANCE");
  if (e != nullptr && e[0] == '0' && e[1] == '\0') {
    return false;
  }
  return true;
}

// Default OFF: latent SQ8 decodes to the per-CID reuse buffer first (decode path), then scores.
// Set OB_IVF_SQ8_FUSED_DISTANCE=1 on observer to use fused dequant+distance when eligible (no float[dim] staging).
OB_INLINE bool ivf_sq8_env_use_fused_latent_distance()
{
  const char *const e = ::getenv("OB_IVF_SQ8_FUSED_DISTANCE");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

// Per-probe pq_code scan diagnostics. Set OB_IVF_CID_PROBE_DEBUG=1 on observer.
// Optional OB_IVF_CID_PROBE_DEBUG_LOG_FILE (supports "%t" → thread id); default:
// $HOME/log/ob_ivf_cid_probe_debug.<tid>.log
OB_INLINE bool ob_ivf_cid_probe_debug_enabled()
{
  static int cached = -1;
  if (cached < 0) {
    const char *const env = ::getenv("OB_IVF_CID_PROBE_DEBUG");
    cached = (nullptr != env && env[0] == '1' && '\0' == env[1]) ? 1 : 0;
  }
  return cached != 0;
}

OB_INLINE void ob_ivf_cid_probe_debug_append_line(const char *line)
{
  if (OB_ISNULL(line) || '\0' == line[0]) {
    return;
  }
  thread_local char tls_probe_path[common::FileDirectoryUtils::MAX_PATH + 1];
  thread_local bool tls_probe_path_inited = false;
  if (!tls_probe_path_inited) {
    tls_probe_path_inited = true;
    tls_probe_path[0] = '\0';
    const int64_t tid = GETTID();
    const char *const env_file = ::getenv("OB_IVF_CID_PROBE_DEBUG_LOG_FILE");
    if (nullptr != env_file && '\0' != env_file[0]) {
      const char *const pct = strstr(env_file, "%t");
      if (nullptr != pct) {
        const int prefix_len = static_cast<int>(pct - env_file);
        const int n = snprintf(tls_probe_path,
            sizeof(tls_probe_path),
            "%.*s%lld%s",
            prefix_len,
            env_file,
            static_cast<long long>(tid),
            pct + 2);
        if (n <= 0 || n >= static_cast<int>(sizeof(tls_probe_path))) {
          tls_probe_path[0] = '\0';
        }
      } else {
        const int n = snprintf(tls_probe_path, sizeof(tls_probe_path), "%s", env_file);
        if (n <= 0 || n >= static_cast<int>(sizeof(tls_probe_path))) {
          tls_probe_path[0] = '\0';
        }
      }
    } else {
      const char *home = ::getenv("HOME");
      if (nullptr != home && '\0' != home[0]) {
        char dir_buf[common::FileDirectoryUtils::MAX_PATH + 1];
        const int nd = snprintf(dir_buf, sizeof(dir_buf), "%s/log", home);
        if (nd > 0 && nd < static_cast<int>(sizeof(dir_buf))) {
          (void)common::FileDirectoryUtils::create_full_path(dir_buf);
        }
        const int nf = snprintf(tls_probe_path,
            sizeof(tls_probe_path),
            "%s/log/ob_ivf_cid_probe_debug.%lld.log",
            home,
            static_cast<long long>(tid));
        if (nf <= 0 || nf >= static_cast<int>(sizeof(tls_probe_path))) {
          tls_probe_path[0] = '\0';
        }
      }
    }
    if ('\0' != tls_probe_path[0]) {
      (void)ob_ivf_lat_create_parent_dirs_for_file(tls_probe_path);
    }
  }
  if ('\0' == tls_probe_path[0]) {
    return;
  }
  FILE *fp = ::fopen(tls_probe_path, "ae");
  if (OB_ISNULL(fp)) {
    fp = ::fopen(tls_probe_path, "a");
  }
  if (OB_ISNULL(fp)) {
    return;
  }
  (void)::fputs(line, fp);
  (void)::fclose(fp);
}

OB_INLINE void ob_ivf_cid_probe_debug_log_line(const char *line)
{
  if (!ob_ivf_cid_probe_debug_enabled()) {
    return;
  }
  LOG_INFO("[OB_IVF_CID_PROBE_DEBUG]", K(line));
  ob_ivf_cid_probe_debug_append_line(line);
}

// Storage macro/micro block index uses store_rowkey_cnt (= schema rowkey + MV suffix cols).
// DAS access columns only cover the schema prefix; range must use full store rowkey width.
OB_INLINE int64_t calc_cid_vec_storage_range_rowkey_cnt(const ObDASScanCtDef &cid_vec_ctdef)
{
  return cid_vec_ctdef.table_param_.get_read_info().get_rowkey_count();
}

OB_INLINE bool ob_ivf_cid_range_debug_enabled()
{
  static int cached = -1;
  if (cached < 0) {
    const char *const env = ::getenv("OB_IVF_CID_RANGE_DEBUG");
    cached = (nullptr != env && env[0] == '1' && '\0' == env[1]) ? 1 : 0;
  }
  return cached != 0 || ob_ivf_cid_probe_debug_enabled();
}

OB_INLINE void ob_ivf_cid_range_debug_log_scan_range(
    const ObNewRange &range,
    int64_t schema_rowkey_cnt,
    int64_t store_rowkey_cnt,
    int64_t range_rowkey_cnt,
    uint64_t center_id,
    const char *path)
{
  if (!ob_ivf_cid_range_debug_enabled()) {
    return;
  }
  const int64_t start_cnt = range.start_key_.get_obj_cnt();
  const int64_t end_cnt = range.end_key_.get_obj_cnt();
  const bool is_precise_rowkey = store_rowkey_cnt > 0 && end_cnt == store_rowkey_cnt;
  char line[512];
  const int n = snprintf(
      line,
      sizeof(line),
      "[OB_IVF_CID_RANGE_DEBUG] path=%s center_id=%llu schema_rowkey_cnt=%lld "
      "store_rowkey_cnt=%lld range_rowkey_cnt=%lld start_obj_cnt=%lld end_obj_cnt=%lld "
      "is_precise_rowkey=%d inclusive_start=%d inclusive_end=%d whole_range=%d\n",
      (nullptr != path ? path : "unknown"),
      static_cast<unsigned long long>(center_id),
      static_cast<long long>(schema_rowkey_cnt),
      static_cast<long long>(store_rowkey_cnt),
      static_cast<long long>(range_rowkey_cnt),
      static_cast<long long>(start_cnt),
      static_cast<long long>(end_cnt),
      static_cast<int>(is_precise_rowkey),
      static_cast<int>(range.border_flag_.inclusive_start()),
      static_cast<int>(range.border_flag_.inclusive_end()),
      static_cast<int>(range.is_whole_range()));
  if (n > 0 && n < static_cast<int>(sizeof(line))) {
    ob_ivf_cid_probe_debug_log_line(line);
  }
}

OB_INLINE bool ivf_sq8_latent_fusable_heap_metric(const ObExprVectorDistance::ObVecDisType dt)
{
  switch (dt) {
    case ObExprVectorDistance::ObVecDisType::COSINE:
    case ObExprVectorDistance::ObVecDisType::DOT:
    case ObExprVectorDistance::ObVecDisType::EUCLIDEAN:
    case ObExprVectorDistance::ObVecDisType::MANHATTAN:
    case ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED:
      return true;
    default:
      return false;
  }
}

// Fused cosine/dot(+need_norm) folds candidate L2 norm into the metric; other metrics + need_norm use decode path.
// After COSINE query init, heap metric is DOT — allow DOT together with need_norm_ here.
OB_INLINE bool ivf_sq8_try_fused_latent_distance(const bool need_norm, const ObExprVectorDistance::ObVecDisType dt)
{
  if (!ivf_sq8_env_use_fused_latent_distance()) {
    return false;
  }
  if (!ivf_sq8_latent_fusable_heap_metric(dt)) {
    return false;
  }
  if (need_norm && dt != ObExprVectorDistance::ObVecDisType::COSINE
      && dt != ObExprVectorDistance::ObVecDisType::DOT) {
    return false;
  }
  return true;
}

/// Latent IVF_SQ8 cid_vector holds SQ8 codes; raw `vec.ptr()` is only valid as float* outside latent / after fused or decode.
OB_INLINE bool ivf_sq8_latent_row_may_push_raw_vec(
    const bool latent_sq8_active, const bool fused_or_decode_handled, const float *decoded_candidate)
{
  return !latent_sq8_active || fused_or_decode_handled || decoded_candidate != nullptr;
}

// Decode SQ8 blob to caller-provided float buffer (dim floats). Reuse one buffer per CID scan to avoid per-row arena bump.
OB_INLINE int ivf_sq8_decode_latent_float_to_buf(
    const int64_t dim,
    const float *meta_min,
    const float *meta_step,
    const ObString &blob,
    float *out_lat)
{
  int ret = OB_SUCCESS;
  const int64_t need_u8_bytes = dim * static_cast<int64_t>(sizeof(uint8_t));
  if (OB_ISNULL(meta_min) || OB_ISNULL(meta_step) || OB_ISNULL(out_lat) || OB_ISNULL(blob.ptr())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (blob.length() < need_u8_bytes) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    oceanbase::common::ivf_sq8_legacy_bin_center_u8_decode(
        dim, meta_min, meta_step, reinterpret_cast<const uint8_t *>(blob.ptr()), out_lat);
  }
  return ret;
}

/// IVF_SQ8 latent-float: fused distance if eligible, otherwise decode into `reuse_buf`. Caller sets `ret` on hard errors only.
OB_INLINE void ivf_sq8_latent_float_fused_then_decode(
    ObVectorCenterClusterHelper<float, ObRowkey> &heap,
    const ObRowkey &main_rowkey,
    const ObString &vec,
    const int64_t dim,
    const bool need_norm,
    const float *meta_min,
    const float *meta_step,
    float *reuse_buf,
    bool &sq8_latent_handled,
    float *&lat_sq8_candidate,
    int64_t &vec_dist_calc_cnt,
    int &ret)
{
  if (!OB_SUCC(ret)) {
    return;
  }
  const int64_t need_u8_bytes = dim * static_cast<int64_t>(sizeof(uint8_t));
  if (ivf_sq8_try_fused_latent_distance(need_norm, heap.vec_dis_type()) && vec.length() >= need_u8_bytes) {
    double dist = 0.0;
    const int fr = oceanbase::common::ivf_sq8_latent_fused_distance_vs_query(
        heap.query_vector(),
        dim,
        meta_min,
        meta_step,
        reinterpret_cast<const uint8_t *>(vec.ptr()),
        static_cast<int>(heap.vec_dis_type()),
        need_norm,
        dist);
    if (OB_SUCC(fr)) {
      if (OB_FAIL(heap.push_center(main_rowkey, dist, CenterSaveMode::NOT_SAVE_CENTER_VEC, nullptr))) {
        LOG_WARN("failed to push center (IVF_SQ8 latent fused)", K(ret));
      } else {
        ++vec_dist_calc_cnt;
        sq8_latent_handled = true;
      }
    } else if (fr != OB_ERR_NULL_VALUE) {
      ret = fr;
    }
  }
  if (OB_SUCC(ret) && !sq8_latent_handled) {
    if (OB_ISNULL(reuse_buf)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("IVF SQ8 latent reuse buffer not allocated", K(ret));
    } else if (OB_FAIL(ivf_sq8_decode_latent_float_to_buf(dim, meta_min, meta_step, vec, reuse_buf))) {
      LOG_WARN("failed to IVF SQ8 latent-dequant cid blob", K(ret));
    } else {
      lat_sq8_candidate = reuse_buf;
    }
  }
}

}

void ObDASIvfScanIter::reset_ivf_sq8_latent_float_heap_ctx()
{
  ivf_sq8_cid_u8_score_latent_float_heap_ = false;
  ivf_sq8_meta_min_ = nullptr;
  ivf_sq8_meta_step_ = nullptr;
}

int ObDASIvfBaseScanIter::do_table_scan()
{
  int ret = OB_SUCCESS;

  if (OB_ISNULL(inv_idx_scan_iter_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("inv_idx_scan_iter_ is null", K(ret));
  } else if (OB_FAIL(inv_idx_scan_iter_->do_table_scan())) {
    LOG_WARN("fail to do inv idx table scan.", K(ret));
  }

  return ret;
}

int ObDASIvfBaseScanIter::rescan()
{
  int ret = OB_SUCCESS;

  if (OB_NOT_NULL(inv_idx_scan_iter_) && OB_FAIL(inv_idx_scan_iter_->rescan())) {
    LOG_WARN("failed to rescan inv_idx_scan_iter_", K(ret));
  }

  return ret;
}

void ObDASIvfBaseScanIter::clear_evaluated_flag()
{
  if (OB_NOT_NULL(inv_idx_scan_iter_)) {
    inv_idx_scan_iter_->clear_evaluated_flag();
  }
}

int ObDASIvfBaseScanIter::gen_rowkeys_itr()
{
  int ret = OB_SUCCESS;

  if (OB_NOT_NULL(saved_rowkeys_itr_) && saved_rowkeys_itr_->is_init()) {
  } else {
    uint64_t rowkey_count = saved_rowkeys_.count();
    void *iter_buff = nullptr;
    if (rowkey_count == 0) {
      ret = OB_ITER_END;
      LOG_WARN("no rowkeys found", K(ret));
    } else if (OB_ISNULL(iter_buff = vec_op_alloc_.alloc(sizeof(ObVectorQueryRowkeyIterator)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to allocator rowkey iter.", K(ret));
    } else if (OB_FALSE_IT(saved_rowkeys_itr_ = new (iter_buff) ObVectorQueryRowkeyIterator())) {
    } else if (OB_FAIL(saved_rowkeys_itr_->init(rowkey_count, &saved_rowkeys_))) {
      LOG_WARN("iter init failed.", K(ret));
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::do_table_full_scan(bool is_vectorized,
                                             const ObDASScanCtDef *ctdef,
                                             ObDASScanRtDef *rtdef,
                                             ObDASScanIter *iter,
                                             ObTabletID &tablet_id,
                                             bool &first_scan,
                                             ObTableScanParam &scan_param)
{
  int ret = OB_SUCCESS;

  if (first_scan) {
    ObNewRange scan_range;
    if (OB_FAIL(ObDasVecScanUtils::init_scan_param(ls_id_, tablet_id, ctdef, rtdef, tx_desc_, snapshot_, scan_param,
                                                   false /*is_get*/, &mem_context_->get_arena_allocator()))) {
      LOG_WARN("failed to generate init vec aux scan param", K(ret));
    } else if (OB_FALSE_IT(ObDasVecScanUtils::set_whole_range(scan_range, ctdef->ref_table_id_))) {
    } else if (OB_FAIL(scan_param.key_ranges_.push_back(scan_range))) {
      LOG_WARN("failed to append scan range", K(ret));
    } else if (OB_FALSE_IT(iter->set_scan_param(scan_param))) {
    } else if (OB_FAIL(iter->do_table_scan())) {
      LOG_WARN("failed to do scan", K(ret));
    } else {
      first_scan = false;
    }
  } else {
    const ObTabletID &scan_tablet_id = scan_param.tablet_id_;
    scan_param.need_switch_param_ =
        scan_param.need_switch_param_ || (scan_tablet_id.is_valid() && (tablet_id != scan_tablet_id));
    scan_param.tablet_id_ = tablet_id;
    scan_param.ls_id_ = ls_id_;

    ObNewRange scan_range;
    if (OB_FAIL(iter->reuse())) {
      LOG_WARN("failed to reuse scan iterator.", K(ret));
    } else if (OB_FALSE_IT(ObDasVecScanUtils::set_whole_range(scan_range, ctdef->ref_table_id_))) {
    } else if (OB_FAIL(scan_param.key_ranges_.push_back(scan_range))) {
      LOG_WARN("failed to append scan range", K(ret));
    } else if (OB_FAIL(iter->rescan())) {
      LOG_WARN("failed to rescan scan iterator.", K(ret));
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::do_aux_table_scan(bool &first_scan,
                                            ObTableScanParam &scan_param,
                                            const ObDASScanCtDef *ctdef,
                                            ObDASScanRtDef *rtdef,
                                            ObDASScanIter *iter,
                                            ObTabletID &tablet_id)
{
  int ret = OB_SUCCESS;

  if (first_scan) {
    scan_param.need_switch_param_ = false;
    if (OB_FAIL(ObDasVecScanUtils::init_scan_param(
            ls_id_, tablet_id, ctdef, rtdef, tx_desc_, snapshot_, scan_param, false/*is_get*/, &mem_context_->get_arena_allocator()))) {
      LOG_WARN("failed to init scan param", K(ret));
    } else if (OB_FALSE_IT(iter->set_scan_param(scan_param))) {
    } else if (OB_FAIL(iter->do_table_scan())) {
      LOG_WARN("failed to do scan", K(ret));
    } else {
      first_scan = false;
    }
  } else {
    const ObTabletID &scan_tablet_id = scan_param.tablet_id_;
    scan_param.need_switch_param_ =
        scan_param.need_switch_param_ || (scan_tablet_id.is_valid() && (tablet_id != scan_tablet_id));
    scan_param.tablet_id_ = tablet_id;
    scan_param.ls_id_ = ls_id_;
    if (OB_FAIL(iter->rescan())) {
      LOG_WARN("fail to rescan scan iterator.", K(ret));
    }
  }
  return ret;
}

// Reset memory context and clean up associated resources
void ObDASIvfBaseScanIter::reset_memory_context()
{
  // Clean up HGraph iterative search context and reset memory context
  if (OB_NOT_NULL(hgraph_iter_ctx_)) {
    obvectorutil::delete_iter_ctx(hgraph_iter_ctx_);
    hgraph_iter_ctx_ = nullptr;
  }
  // Clean up HGraph iterative search context and reset memory context before mem_context_ reset
  if (OB_NOT_NULL(hgraph_vsag_alloc_)) {
    hgraph_vsag_alloc_->~ObVsagSearchAlloc();
    hgraph_vsag_alloc_ = nullptr;
  }
  if (nullptr != mem_context_) {
    mem_context_->reset_remain_one_page();
  }

  // Reset HGraph state for next iteration
  hgraph_has_next_center_ = true;
  has_used_hgraph_ = false;
}

// Unified interface for checking if there are more centers available
bool ObDASIvfBaseScanIter::has_next_center()
{
  bool has_next = false;
  if (has_used_hgraph_) {
    has_next = hgraph_has_next_center_;
  } else {
    has_next = iterative_filter_ctx_.has_next_center();
  }
  return has_next;
}

int ObDASIvfBaseScanIter::inner_reuse()
{
  int ret = OB_SUCCESS;

  if (OB_NOT_NULL(inv_idx_scan_iter_) && OB_FAIL(inv_idx_scan_iter_->reuse())) {
    LOG_WARN("failed to reuse inv idx scan iter", K(ret));
  } else if (!centroid_iter_first_scan_ && OB_FAIL(ObDasVecScanUtils::reuse_iter(
                                               ls_id_, centroid_iter_, centroid_scan_param_, centroid_tablet_id_))) {
    LOG_WARN("failed to reuse com aux vec iter", K(ret));
  } else if (!cid_vec_iter_first_scan_ &&
             OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, cid_vec_iter_, cid_vec_scan_param_, cid_vec_tablet_id_))) {
    LOG_WARN("failed to reuse rowkey vid iter", K(ret));
  } else if (!rowkey_cid_iter_first_scan_ &&
             OB_FAIL(ObDasVecScanUtils::reuse_iter(
                 ls_id_, rowkey_cid_iter_, rowkey_cid_scan_param_, rowkey_cid_tablet_id_))) {
    LOG_WARN("failed to reuse vid rowkey iter", K(ret));
  } else if (!brute_first_scan_ && OB_FAIL(ObDasVecScanUtils::reuse_iter(
                                      ls_id_, brute_iter_, brute_scan_param_, brute_tablet_id_))) {
    LOG_WARN("failed to reuse iter", K(ret));
  } else if (!data_filter_iter_first_scan_ && OB_FAIL(ObDasVecScanUtils::reuse_iter(
                                      ls_id_, data_filter_iter_, data_filter_scan_param_, data_filter_tablet_id_))) {
    LOG_WARN("failed to reuse iter", K(ret));
  }

  if (OB_NOT_NULL(saved_rowkeys_itr_)) {
    saved_rowkeys_itr_->reset();
    saved_rowkeys_itr_->~ObVectorQueryRowkeyIterator();
    saved_rowkeys_itr_ = nullptr;
  }
  // Reset memory context and clean up associated resources
  reset_memory_context();

  vec_op_alloc_.reset();
  saved_rowkeys_.reset();
  pre_fileter_rowkeys_.reset();
  center_cache_guard_.reset();
  iterative_filter_ctx_.reuse();
  // adaptive_ctx_.reset();
  return ret;
}

int ObDASIvfBaseScanIter::init_hgraph_search_alloc()
{
  int ret = OB_SUCCESS;
  // Create VSAG allocator if not exists (same lifecycle as hgraph_iter_ctx_)
  if (OB_ISNULL(hgraph_vsag_alloc_)) {
    void *alloc_buf = mem_context_->get_arena_allocator().alloc(sizeof(ObVsagSearchAlloc));
    if (OB_ISNULL(alloc_buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to allocate memory for hgraph vsag allocator", K(ret));
    } else {
      hgraph_vsag_alloc_ = new(alloc_buf) ObVsagSearchAlloc(MTL_ID());
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::inner_init(ObDASIterParam &param)
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(ObDASIterType::DAS_ITER_IVF_SCAN != param.type_)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("invalid das iter param type for ivf scan iter", K(ret), K(param));
  } else {
    ObDASIvfScanIterParam &ivf_scan_param = static_cast<ObDASIvfScanIterParam &>(param);
    ls_id_ = ivf_scan_param.ls_id_;
    tx_desc_ = ivf_scan_param.tx_desc_;
    snapshot_ = ivf_scan_param.snapshot_;

    inv_idx_scan_iter_ = ivf_scan_param.inv_idx_scan_iter_;
    centroid_iter_ = ivf_scan_param.centroid_iter_;
    cid_vec_iter_ = ivf_scan_param.cid_vec_iter_;
    rowkey_cid_iter_ = ivf_scan_param.rowkey_cid_iter_;
    brute_iter_ = ivf_scan_param.brute_iter_;
    data_filter_iter_ = ivf_scan_param.data_filter_iter_;

    vec_aux_ctdef_ = ivf_scan_param.vec_aux_ctdef_;
    vec_aux_rtdef_ = ivf_scan_param.vec_aux_rtdef_;
    sort_ctdef_ = ivf_scan_param.sort_ctdef_;
    sort_rtdef_ = ivf_scan_param.sort_rtdef_;
    data_filter_ctdef_ = ivf_scan_param.data_filter_ctdef_;
    data_filter_rtdef_ = ivf_scan_param.data_filter_rtdef_;

    vec_index_type_ = ivf_scan_param.vec_index_type_;
    vec_idx_try_path_ = ivf_scan_param.vec_idx_try_path_;
    strategy_ = ivf_scan_param.strategy_;

    adaptive_ctx_.reset();
    adaptive_ctx_.selectivity_ = vec_aux_ctdef_->selectivity_;
    adaptive_ctx_.row_count_ = vec_aux_ctdef_->row_count_;
    adaptive_ctx_.is_primary_index_ = ivf_scan_param.is_primary_index_;
    adaptive_ctx_.can_use_vec_pri_opt_ = vec_aux_ctdef_->can_use_vec_pri_opt();

    if (OB_ISNULL(mem_context_)) {
      lib::ContextParam param;
      param.set_mem_attr(MTL_ID(), "IVF", ObCtxIds::DEFAULT_CTX_ID);
      if (OB_FAIL(CURRENT_CONTEXT->CREATE_CONTEXT(mem_context_, param))) {
        LOG_WARN("failed to create vector ivf memory context", K(ret));
      }
    }
    if (OB_FAIL(ret)) {
    } else {
      if (OB_NOT_NULL(sort_ctdef_) && OB_NOT_NULL(sort_rtdef_)) {
        ObExpr *distance_calc = nullptr;
        if (OB_FAIL(
                ObDasVecScanUtils::init_limit(vec_aux_ctdef_, vec_aux_rtdef_, sort_ctdef_, sort_rtdef_, limit_param_))) {
          LOG_WARN("failed to init limit", K(ret), KPC(vec_aux_ctdef_), KPC(vec_aux_rtdef_));
        } else if (OB_FAIL(ObDasVecScanUtils::init_sort(
                      vec_aux_ctdef_, vec_aux_rtdef_, sort_ctdef_, sort_rtdef_, limit_param_, search_vec_, distance_calc))) {
          LOG_WARN("failed to init sort", K(ret), KPC(vec_aux_ctdef_), KPC(vec_aux_rtdef_));
        } else if (OB_FAIL(ObVectorIndexUtil::parser_params_from_string(
                       vec_aux_ctdef_->vec_index_param_, ObVectorIndexType::VIT_IVF_INDEX, vec_index_param_))) {
          LOG_WARN("fail to parse params from string", K(ret), K(vec_aux_ctdef_->vec_index_param_));
        } else if (OB_FAIL(ObDasVecScanUtils::get_real_search_vec(persist_alloc_, sort_rtdef_->eval_ctx_, search_vec_,
                                                                  real_search_vec_))) {
          LOG_WARN("failed to get real search vec", K(ret));
        } else if (OB_FAIL(ObDasVecScanUtils::get_distance_expr_type(*sort_ctdef_->sort_exprs_[0],
                                                                     *sort_rtdef_->eval_ctx_, dis_type_))) {
          LOG_WARN("failed to get distance type.", K(ret));
        } else if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::COSINE) {
          // nomolize serach_vec
          if (OB_FAIL(ObVectorNormalize::L2_normalize_vector(vec_aux_ctdef_->dim_,
                                                             reinterpret_cast<float *>(real_search_vec_.ptr()),
                                                             reinterpret_cast<float *>(real_search_vec_.ptr())))) {
            LOG_WARN("failed to normalize vector", K(ret));
          } else {
            need_norm_ = true;
            dis_type_ = oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT;
          }
        }
        if (OB_SUCC(ret)) {
          if (OB_FAIL(ObVectorIndexParam::build_search_param(vec_aux_ctdef_->vector_index_param_, vec_aux_ctdef_->vec_query_param_, search_param_))) {
            LOG_WARN("build search param fail", K(vec_aux_ctdef_->vector_index_param_), K(vec_aux_ctdef_->vec_query_param_));
          } else {
            LOG_TRACE("search param", K(vec_aux_ctdef_->vector_index_param_), K(vec_aux_ctdef_->vec_query_param_), K(search_param_));
            if (search_param_.similarity_threshold_ > 0) {
              if (OB_FAIL(ObDasVecScanUtils::check_ivf_support_similarity_threshold(*sort_ctdef_->sort_exprs_[0]))) {
                LOG_WARN("check support similarity threshold fail", K(ret));
              } else {
                similarity_threshold_ = search_param_.similarity_threshold_;
              }
            }
          }
        }
        if (OB_FAIL(ret)) {
        } else {
          ObSQLSessionInfo *session = nullptr;
          uint64_t ob_ivf_nprobes = 0;

          if (OB_NOT_NULL(vec_aux_ctdef_) && vec_aux_ctdef_->vec_query_param_.is_set_ivf_nprobes_) {
            nprobes_ = vec_aux_ctdef_->vec_query_param_.ivf_nprobes_;
            LOG_TRACE("use stmt ivf_nprobes", K(nprobes_));
          } else if (OB_ISNULL(session = sort_rtdef_->eval_ctx_->exec_ctx_.get_my_session())) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("failed to get session", K(ret), KPC(session));
          } else if (OB_FAIL(session->get_ob_ivf_nprobes(ob_ivf_nprobes))) {
            LOG_WARN("failed to get ob ob_ivf_nprobes", K(ret));
          } else {
            nprobes_ = ob_ivf_nprobes;
          }
        }
      }
    }

    if (OB_SUCC(ret) && is_adaptive_filter()) {
      ObPhysicalPlanCtx *plan_ctx = GET_PHY_PLAN_CTX(*exec_ctx_);
      ObVecIdxAdaTryPath cur_path = ObVecIdxAdaTryPath::VEC_PATH_UNCHOSEN;
      if (OB_ISNULL(plan_ctx->get_phy_plan())) {
        // remote scan, phy plan is null, do nothing, just use try path in ctdef
        LOG_WARN("plan ctx is null", K(ret), KP(plan_ctx));
      } else if (OB_FALSE_IT(cur_path = static_cast<ObVecIdxAdaTryPath>(plan_ctx->get_phy_plan()->stat_.vec_index_exec_ctx_.cur_path_))) {
      } else if (cur_path != vec_idx_try_path_ &&
                  cur_path > ObVecIdxAdaTryPath::VEC_PATH_UNCHOSEN &&
                  cur_path < ObVecIdxAdaTryPath::VEC_PATH_MAX) {
        LOG_INFO("adaptive filter change path", K(cur_path), K(vec_idx_try_path_));
        vec_idx_try_path_ = cur_path;
      }
    }

    if (OB_SUCC(ret) && OB_NOT_NULL(data_filter_ctdef_)) {
      int64_t main_rowkey_cnt = data_filter_ctdef_->table_param_.get_read_info().get_schema_rowkey_count();
      void *ptr = nullptr;
      if (OB_ISNULL(ptr = persist_alloc_.alloc(sizeof(ObObj) * main_rowkey_cnt))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("allocate memory failed", K(ret), K(main_rowkey_cnt));
      } else {
        ObObj *obj_ptr = new (ptr) ObObj[main_rowkey_cnt];
        tmp_main_rowkey_.assign(obj_ptr, main_rowkey_cnt);
      }
    }

    if (OB_SUCC(ret)) {
      omt::ObTenantConfigGuard tenant_config(TENANT_CONF(MTL_ID()));
      max_scan_vectors_ = tenant_config->_ivf_max_scan_vectors;
      const ObDASBaseRtDef *index_rtdef = vec_aux_rtdef_->get_inv_idx_scan_rtdef();
      if (OB_NOT_NULL(index_rtdef->table_loc_)) {
        scan_tablet_size_ = index_rtdef->table_loc_->get_tablet_locs().size();
      }
    }
  }

    dim_ = vec_aux_ctdef_->dim_;
    selectivity_ = vec_aux_ctdef_->selectivity_;

  return ret;
}

int ObDASIvfBaseScanIter::inner_release()
{
  int ret = OB_SUCCESS;
  int tmp_ret = OB_SUCCESS;

  if (OB_NOT_NULL(inv_idx_scan_iter_) && OB_FAIL(inv_idx_scan_iter_->release())) {
    LOG_WARN("failed to release inv_idx_scan_iter_", K(ret));
    tmp_ret = ret;
    ret = OB_SUCCESS;
  }
  if (OB_NOT_NULL(centroid_iter_) && OB_FAIL(centroid_iter_->release())) {
    LOG_WARN("failed to release centroid_iter_", K(ret));
    tmp_ret = tmp_ret == OB_SUCCESS ? ret : tmp_ret;
    ret = OB_SUCCESS;
  }
  if (OB_NOT_NULL(cid_vec_iter_) && OB_FAIL(cid_vec_iter_->release())) {
    LOG_WARN("failed to release cid_vec_iter_", K(ret));
    tmp_ret = tmp_ret == OB_SUCCESS ? ret : tmp_ret;
    ret = OB_SUCCESS;
  }
  if (OB_NOT_NULL(rowkey_cid_iter_) && OB_FAIL(rowkey_cid_iter_->release())) {
    LOG_WARN("failed to release rowkey_cid_iter_", K(ret));
    tmp_ret = tmp_ret == OB_SUCCESS ? ret : tmp_ret;
    ret = OB_SUCCESS;
  }
  if (OB_NOT_NULL(brute_iter_) && OB_FAIL(brute_iter_->release())) {
    LOG_WARN("failed to release brute_iter_", K(ret));
  }
  if (OB_NOT_NULL(data_filter_iter_) && OB_FAIL(data_filter_iter_->release())) {
    LOG_WARN("failed to release data_filter_iter_", K(ret));
  }

  // return first error code
  if (tmp_ret != OB_SUCCESS) {
    ret = tmp_ret;
  }

  inv_idx_scan_iter_ = nullptr;
  centroid_iter_ = nullptr;
  cid_vec_iter_ = nullptr;
  rowkey_cid_iter_ = nullptr;
  brute_iter_ = nullptr;
  data_filter_iter_ = nullptr;

  if (OB_NOT_NULL(saved_rowkeys_itr_)) {
    saved_rowkeys_itr_->reset();
    saved_rowkeys_itr_->~ObVectorQueryRowkeyIterator();
    saved_rowkeys_itr_ = nullptr;
  }

  saved_rowkeys_.reset();
  pre_fileter_rowkeys_.reset();
  // Reset memory context and clean up associated resources
  reset_memory_context();
  if (nullptr != mem_context_) {
    DESTROY_CONTEXT(mem_context_);
    mem_context_ = nullptr;
  }
  tmp_main_rowkey_.reset();
  vec_op_alloc_.reset();
  persist_alloc_.reset();
  center_cache_guard_.reset();
  iterative_filter_ctx_.reset();
  adaptive_ctx_.reset();
  tx_desc_ = nullptr;
  snapshot_ = nullptr;

  ObDasVecScanUtils::release_scan_param(centroid_scan_param_);
  ObDasVecScanUtils::release_scan_param(cid_vec_scan_param_);
  ObDasVecScanUtils::release_scan_param(rowkey_cid_scan_param_);
  ObDasVecScanUtils::release_scan_param(brute_scan_param_);
  ObDasVecScanUtils::release_scan_param(data_filter_scan_param_);

  vec_aux_ctdef_ = nullptr;
  vec_aux_rtdef_ = nullptr;
  sort_ctdef_ = nullptr;
  sort_rtdef_ = nullptr;
  search_vec_ = nullptr;
  real_search_vec_ = nullptr;
  data_filter_ctdef_ = nullptr;
  data_filter_rtdef_ = nullptr;
  return ret;
}

int ObDASIvfBaseScanIter::inner_get_next_row()
{
  int ret = OB_SUCCESS;
  if (limit_param_.limit_ + limit_param_.offset_ == 0) {
    ret = OB_ITER_END;
  } else if (OB_ISNULL(saved_rowkeys_itr_)) {
    if (OB_FAIL(process_ivf_scan(false/*is_vectorized*/))) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to process ivf scan state", K(ret));
      }
    }
  }

  ObRowkey *rowkey = nullptr;
  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(saved_rowkeys_itr_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to get rowkey iter", K(ret));
  } else if (OB_FAIL(get_next_saved_rowkey())) {
    LOG_WARN("failed to get saved rowkey", K(ret));
  }

  return ret;
}

int ObDASIvfBaseScanIter::inner_get_next_rows(int64_t &count, int64_t capacity)
{
  int ret = OB_SUCCESS;
  if (limit_param_.limit_ + limit_param_.offset_ == 0) {
    ret = OB_ITER_END;
  } else if (OB_ISNULL(saved_rowkeys_itr_)) {
    if (OB_FAIL(process_ivf_scan(true/*is_vectorized*/))) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to process ivf scan state", K(ret));
      }
    }
  }

  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(saved_rowkeys_itr_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("failed to get rowkey iter", K(ret));
  } else if (OB_FALSE_IT(saved_rowkeys_itr_->set_batch_size(capacity))) {
  } else if (OB_FAIL(get_next_saved_rowkeys(count))) {
    if (OB_ITER_END != ret) {
      LOG_WARN("failed to get saved rowkeys", K(ret));
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::do_ivf_scan(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  adaptive_ctx_.reuse();
  if (is_post_filter()) {
    if (OB_FAIL(process_ivf_scan_post(is_vectorized))) {
      LOG_WARN("failed to process ivf_scan post filter", K(ret), K_(pre_fileter_rowkeys), K_(saved_rowkeys));
    }
  } else if (OB_FAIL(process_ivf_scan_pre(mem_context_->get_arena_allocator(), is_vectorized))) {
    LOG_WARN("failed to process ivf_scan pre", K(ret));
  }
  return ret;
}

void ObDASIvfBaseScanIter::ivf_lat_reset()
{
  ivf_lat_.reset();
}

void ObDASIvfBaseScanIter::ivf_lat_log(const bool is_vectorized) const
{
  if (!ivf_lat_.enabled_) {
    return;
  }
  const int64_t fine_cv_untracked_us = ob_ivf_latency_breakdown_fine_cv_untracked_us(ivf_lat_);
  const int64_t misc_us =
      ivf_lat_.das_body_us_ - ivf_lat_.brute_wall_us_ - ivf_lat_.coarse_wall_us_ - ivf_lat_.fine_wall_us_;
  share::ObIvfCidClusterCacheLogSnapshot cache_snap;
  cache_snap.reset();
  if (OB_NOT_NULL(cid_vec_iter_)) {
    (void)ob_das_ivf_try_export_cid_cluster_cache_snapshot(cid_vec_iter_, cache_snap);
  }
  LOG_INFO("[OB_IVF_LATENCY_BREAKDOWN]",
           K(is_vectorized),
           K(ivf_lat_.ivf_total_us_),
           K(ivf_lat_.finalize_us_),
           K(ivf_lat_.das_body_us_),
           K(ivf_lat_.coarse_wall_us_),
           K(ivf_lat_.fine_wall_us_),
           K(ivf_lat_.fine_cv_storage_fetch_us_),
           K(ivf_lat_.fine_load_us_),
           K(ivf_lat_.fine_compute_us_),
           K(fine_cv_untracked_us),
           K(ivf_lat_.brute_wall_us_),
           K(ivf_lat_.sq8_prep_us_),
           K(ivf_lat_.pq_prep_us_),
           K(ivf_lat_.flat_prep_us_),
           K(misc_us),
           K(adaptive_ctx_.is_brute_force_),
           K(adaptive_ctx_.cid_vec_scan_rows_),
           K(adaptive_ctx_.vec_dist_calc_cnt_),
           K(vec_index_type_),
           K(dim_),
           K(nprobes_));
  ob_ivf_latency_breakdown_append_user_log(is_vectorized,
      ivf_lat_,
      misc_us,
      adaptive_ctx_.is_brute_force_,
      adaptive_ctx_.cid_vec_scan_rows_,
      adaptive_ctx_.vec_dist_calc_cnt_,
      vec_index_type_,
      dim_,
      nprobes_,
      adaptive_ctx_.row_count_,
      vec_index_param_.nlist_,
      cache_snap);
}

int ObDASIvfBaseScanIter::process_ivf_scan(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  ivf_lat_reset();
  const bool emit_per_query = ob_ivf_per_query_stats_begin_query(
      adaptive_ctx_.row_count_, dim_, vec_index_param_.nlist_);
  ivf_lat_.enabled_ = emit_per_query && ob_ivf_latency_breakdown_enabled();
  const int64_t t_ivf_all_start = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
  const int64_t t_das_body_start = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;

  can_retry_ = check_if_can_retry();
  if (OB_FAIL(do_ivf_scan(is_vectorized))) {
    LOG_WARN("failed to process ivf_scan pre", K(ret));
  }

  if (OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY == ret && can_retry_) {
    LOG_INFO("index adaptive scan need retry", K(vec_index_type_), K(vec_idx_try_path_), K(adaptive_ctx_));
    if (OB_FAIL(reset_filter_path())) {
      LOG_WARN("failed to reset filter path", K(vec_index_type_), K(vec_idx_try_path_), K(adaptive_ctx_), K(ret));
    } else {
      ob_das_ivf_reset_cid_cluster_cache_session_stats(cid_vec_iter_);
      if (OB_FAIL(do_ivf_scan(is_vectorized))) {
        LOG_WARN("failed to process ivf_scan pre", K(ret));
      }
    }
  }

  if (ivf_lat_.enabled_) {
    ivf_lat_.das_body_us_ = ObTimeUtility::current_time() - t_das_body_start;
  }

  if (OB_SUCC(ret)) {
    LOG_TRACE("ivf scan stat info", K_(vec_index_type), K_(vec_idx_try_path), K(adaptive_ctx_), K(scan_tablet_size_), K(centroid_tablet_id_));
  }

  const int64_t t_finalize_start = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(gen_rowkeys_itr())) {
    if (ret != OB_ITER_END) {
      LOG_WARN("failed to gen rowkeys itr", K(saved_rowkeys_));
    }
  }

  if (ivf_lat_.enabled_) {
    ivf_lat_.finalize_us_ = ObTimeUtility::current_time() - t_finalize_start;
    ivf_lat_.ivf_total_us_ = ObTimeUtility::current_time() - t_ivf_all_start;
    ivf_lat_log(is_vectorized);
  }

  reset_memory_context();

  return ret;
}

int ObDASIvfBaseScanIter::check_iter_filter_need_retry()
{
  int ret = OB_SUCCESS;
  double iter_selectivity = double(adaptive_ctx_.iter_res_row_cnt_) /  double(adaptive_ctx_.iter_filter_row_cnt_);
  double output_row_cnt = iter_selectivity * adaptive_ctx_.row_count_;
  if (adaptive_ctx_.iter_times_ > 2) {
    if (adaptive_ctx_.can_use_vec_pri_opt_) {
      ret = (iter_selectivity <= ObVecIdxExtraInfo::DEFAULT_IVF_PRE_RATE_FILTER_WITH_ROWKEY) ?
            OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY : OB_SUCCESS;
    } else {
      ret = (adaptive_ctx_.vec_dist_calc_cnt_ > ObVecIdxExtraInfo::MAX_IVF_POST_DIST_CALC_CNT
            && output_row_cnt < ObVecIdxExtraInfo::MAX_IVF_PRE_ROW_CNT_WITH_IDX
            && iter_selectivity <= ObVecIdxExtraInfo::DEFAULT_IVF_PRE_RATE_FILTER_WITH_IDX) ?
            OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY : OB_SUCCESS;
    }
  }
  if (OB_FAIL(ret)) {
    LOG_WARN("switch path check iter filter need retry", K(ret), K(adaptive_ctx_), K(iter_selectivity), K(output_row_cnt), K(scan_tablet_size_), K(centroid_tablet_id_), K(strategy_));
  }
  return ret;
}

int ObDASIvfBaseScanIter::check_pre_filter_need_retry()
{
  int ret = OB_SUCCESS;
  double pre_selectivity = double(adaptive_ctx_.pre_scan_row_cnt_) / double(adaptive_ctx_.row_count_);
  if (adaptive_ctx_.can_use_vec_pri_opt_) pre_selectivity =  double(adaptive_ctx_.vec_dist_calc_cnt_) / double(adaptive_ctx_.cid_vec_scan_rows_);
  if (adaptive_ctx_.pre_scan_row_cnt_ < IVF_MAX_BRUTE_FORCE_SIZE) {
    /*do nothing*/
  } else if (adaptive_ctx_.can_use_vec_pri_opt_) {
    ret = pre_selectivity > ObVecIdxExtraInfo::DEFAULT_IVF_PRE_RATE_FILTER_WITH_ROWKEY ?
      OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY : OB_SUCCESS;
  } else if (pre_selectivity > ObVecIdxExtraInfo::DEFAULT_IVF_PRE_RATE_FILTER_WITH_IDX) {
    ret = OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY;
  }
  if (OB_FAIL(ret)) {
    LOG_WARN("switch path check pre filter need retry", K(ret), K(adaptive_ctx_), K(pre_selectivity), K(scan_tablet_size_), K(centroid_tablet_id_), K(strategy_));
  }
  return ret;
}

int ObDASIvfBaseScanIter::reset_filter_path()
{
  int ret = OB_SUCCESS;
  ObPhysicalPlanCtx *plan_ctx = GET_PHY_PLAN_CTX(*exec_ctx_);
  ObPlanStat* plan_stat = nullptr;
  if (strategy_ == ObVecIdxQueryStrategy::LATENCY_FIRST && vec_idx_try_path_ == ObVecIdxAdaTryPath::VEC_INDEX_ITERATIVE_FILTER) {
    // RT first mode, do nothing
  } else if (OB_ISNULL(plan_ctx->get_phy_plan())) {
    // remote scan, phy plan is null, do nothing, just use try path in ctdef
    LOG_WARN("plan ctx is null", K(ret), KP(plan_ctx));
  } else if (OB_FALSE_IT(plan_stat = const_cast<ObPlanStat*>(&(plan_ctx->get_phy_plan()->stat_)))) {
  } else if (vec_idx_try_path_ == ObVecIdxAdaTryPath::VEC_INDEX_PRE_FILTER) {
    vec_idx_try_path_ = ObVecIdxAdaTryPath::VEC_INDEX_ITERATIVE_FILTER;
  } else if (vec_idx_try_path_ == ObVecIdxAdaTryPath::VEC_INDEX_ITERATIVE_FILTER) {
    double iter_selectivity = double(adaptive_ctx_.iter_res_row_cnt_) / double(adaptive_ctx_.iter_filter_row_cnt_);
    adaptive_ctx_.selectivity_ = iter_selectivity;
    vec_idx_try_path_ = ObVecIdxAdaTryPath::VEC_INDEX_PRE_FILTER;
  }

  if (OB_FAIL(ret) || OB_ISNULL(plan_stat)) {
  } else if (OB_FAIL(updata_vec_exec_ctx(plan_stat))) {
    LOG_WARN("failed to updata vec exec ctx", K(ret), K(vec_idx_try_path_));
  } else {
    can_retry_ = false;
    reuse_cid_ctx();
  }
  return ret;
}

int ObDASIvfBaseScanIter::updata_vec_exec_ctx(ObPlanStat* plan_stat)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(plan_stat)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("plan stat is null", K(ret), KP(plan_stat));
  } else {
    int32_t record_count = plan_stat->vec_index_exec_ctx_.record_count_ + 1;
    if (record_count < CHANGE_PATH_WINDOW_SIZE) {
      if (vec_idx_try_path_ == ObVecIdxAdaTryPath::VEC_INDEX_ITERATIVE_FILTER) {
        ATOMIC_INC(&(plan_stat->vec_index_exec_ctx_.iter_filter_chosen_times_));
      } else if (vec_idx_try_path_ == ObVecIdxAdaTryPath::VEC_INDEX_PRE_FILTER) {
        ATOMIC_INC(&(plan_stat->vec_index_exec_ctx_.pre_filter_chosen_times_));
      }
      ATOMIC_INC(&(plan_stat->vec_index_exec_ctx_.record_count_));
    } else {
      double iter_time = plan_stat->vec_index_exec_ctx_.iter_filter_chosen_times_;
      double pre_time = plan_stat->vec_index_exec_ctx_.pre_filter_chosen_times_;
      iter_time = std::log(iter_time) * DECAY_FACTOR;
      pre_time = std::log(pre_time) * DECAY_FACTOR;
      FLOG_INFO("begin to reset plan stat filter path", K(plan_stat->vec_index_exec_ctx_.record_count_),
      K(plan_stat->vec_index_exec_ctx_.cur_path_), K(vec_idx_try_path_), K(adaptive_ctx_), K(ret));
      ATOMIC_STORE(&(plan_stat->vec_index_exec_ctx_.record_count_), 0);
      ATOMIC_STORE(&(plan_stat->vec_index_exec_ctx_.iter_filter_chosen_times_), static_cast<int64_t>(iter_time));
      ATOMIC_STORE(&(plan_stat->vec_index_exec_ctx_.pre_filter_chosen_times_), static_cast<int64_t>(pre_time));
      ATOMIC_STORE(&(plan_stat->vec_index_exec_ctx_.cur_path_), static_cast<uint8_t>(vec_idx_try_path_));
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::build_cid_vec_query_rowkey(const ObString &cid,
                                                     bool is_min,
                                                     int64_t rowkey_cnt,
                                                     common::ObRowkey &rowkey)
{
  int ret = OB_SUCCESS;

  ObObj *obj_ptr = nullptr;
  if (OB_ISNULL(obj_ptr = static_cast<ObObj *>(mem_context_->get_arena_allocator().alloc(sizeof(ObObj) * rowkey_cnt)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("fail to alloc memory for ObObj", K(ret));
  } else if (!cid.empty()) {
    // set_varbinary is shallow; copy into arena so rowkey outlives the caller buffer.
    char *cid_copy = static_cast<char *>(mem_context_->get_arena_allocator().alloc(cid.length()));
    if (OB_ISNULL(cid_copy)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("fail to alloc cid copy for rowkey", K(ret), K(cid.length()));
    } else {
      MEMCPY(cid_copy, cid.ptr(), cid.length());
      ObString cid_owned;
      cid_owned.assign_ptr(cid_copy, static_cast<int32_t>(cid.length()));
      obj_ptr[0].set_varbinary(cid_owned);
    }
  } else {
    obj_ptr[0].set_min_value();
  }
  if (OB_SUCC(ret)) {
    for (int64_t i = 1; i < rowkey_cnt; ++i) {
      if (is_min) {
        obj_ptr[i].set_min_value();
      } else {
        obj_ptr[i].set_max_value();
      }
    }

    rowkey.assign(obj_ptr, rowkey_cnt);
  }
  return ret;
}

int ObDASIvfBaseScanIter::build_cid_vec_query_range(const ObString &cid,
                                                    int64_t rowkey_cnt,
                                                    ObNewRange &cid_pri_key_range)
{
  int ret = OB_SUCCESS;
  ObRowkey cid_rowkey_min;
  ObRowkey cid_rowkey_max;
  if (cid.empty()) {
    cid_pri_key_range.set_whole_range();
  } else {
    if (OB_FAIL(build_cid_vec_query_rowkey(cid, true, rowkey_cnt, cid_rowkey_min))) {
      LOG_WARN("failed to build cid vec query rowkey", K(ret));
    } else if (OB_FAIL(build_cid_vec_query_rowkey(cid, false, rowkey_cnt, cid_rowkey_max))) {
      LOG_WARN("failed to build cid vec query rowkey", K(ret));
    } else {
      cid_pri_key_range.start_key_ = cid_rowkey_min;
      cid_pri_key_range.end_key_ = cid_rowkey_max;
      cid_pri_key_range.border_flag_.set_inclusive_start();
      cid_pri_key_range.border_flag_.set_inclusive_end();
    }
  }

  return ret;
}

void ObDASIvfBaseScanIter::set_related_tablet_ids(const ObDASRelatedTabletID &related_tablet_ids)
{
  centroid_tablet_id_ = related_tablet_ids.centroid_tablet_id_;
  cid_vec_tablet_id_ = related_tablet_ids.cid_vec_tablet_id_;
  rowkey_cid_tablet_id_ = related_tablet_ids.rowkey_cid_tablet_id_;
  sq_meta_tablet_id_ = related_tablet_ids.special_aux_tablet_id_;
  pq_centroid_tablet_id_ = related_tablet_ids.special_aux_tablet_id_;
  brute_tablet_id_ = related_tablet_ids.lookup_tablet_id_;
  data_filter_tablet_id_ = related_tablet_ids.lookup_tablet_id_;
}

int ObDASIvfBaseScanIter::get_next_saved_rowkey()
{
  int ret = OB_SUCCESS;
  if (saved_rowkeys_itr_->is_get_from_scan_iter()) {
    if (OB_FAIL(saved_rowkeys_itr_->get_next_row())) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to get next row from scan iter", K(ret));
      }
    }
  } else {
    ObRowkey rowkey;
    if (OB_FAIL(saved_rowkeys_itr_->get_next_row(rowkey))) {
      if (OB_UNLIKELY(OB_ITER_END != ret)) {
        LOG_WARN("failed to get next next row from adaptor vid iter", K(ret));
      }
    } else {
      const ExprFixedArray& ivf_res_exprs = vec_aux_ctdef_->result_output_;
      ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
      guard.set_batch_idx(0);
      for (int64_t i = 0; OB_SUCC(ret) && i < ivf_res_exprs.count(); ++i) {
        ObExpr *expr = ivf_res_exprs.at(i);
        if (OB_ISNULL(expr)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("should not be null", K(ret));
        } else {
          ObDatum &datum = expr->locate_datum_for_write(*vec_aux_rtdef_->eval_ctx_);
          if (OB_FAIL(datum.from_obj(rowkey.get_obj_ptr()[i]))) {
            LOG_WARN("failed to from obj", K(ret));
          }
        }
      }
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::get_next_saved_rowkeys(int64_t &count)
{
  int ret = OB_SUCCESS;
  if (saved_rowkeys_itr_->is_get_from_scan_iter()) {
    if (OB_FAIL(saved_rowkeys_itr_->get_next_rows(count))) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to get next row from scan iter", K(ret));
      }
    }
  } else {
    ObSEArray<ObRowkey, 8> rowkeys;
    if (OB_FAIL(saved_rowkeys_itr_->get_next_rows(rowkeys, count))) {
      if (OB_UNLIKELY(OB_ITER_END != ret)) {
        LOG_WARN("failed to get next next row from adaptor iter", K(ret));
      }
    } else if (count > 0) {
      const ExprFixedArray& ivf_res_exprs = vec_aux_ctdef_->result_output_;
      ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
      guard.set_batch_size(count);
      for (int64_t idx_exp = 0; OB_SUCC(ret) && idx_exp < ivf_res_exprs.count(); ++idx_exp) {
        ObExpr *expr = ivf_res_exprs.at(idx_exp);
        ObDatum *datum = nullptr;
        if (OB_ISNULL(expr)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("should not be null", K(ret));
        } else if (OB_ISNULL(datum = expr->locate_datums_for_update(*vec_aux_rtdef_->eval_ctx_, count))) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("unexpected error, datums is nullptr", K(ret), KPC(expr));
        } else {
          for (int64_t idx_key = 0; OB_SUCC(ret) && idx_key < count; ++idx_key) {
            guard.set_batch_idx(idx_key);
            if (OB_FAIL(datum[idx_key].from_obj(rowkeys[idx_key].get_obj_ptr()[idx_exp]))) {
              LOG_WARN("fail to from obj", K(ret));
            }
          }
          if (OB_SUCC(ret)) {
            expr->set_evaluated_projected(*vec_aux_rtdef_->eval_ctx_);
          }
        }
      }
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::gen_rowkeys_itr_brute(ObDASIter *scan_iter)
{
  int ret = OB_SUCCESS;

  void *iter_buff = nullptr;
  if (OB_ISNULL(scan_iter) || OB_ISNULL(scan_iter->get_output())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("scan_iter is null.", K(ret), KP(scan_iter));
  } else if (!scan_iter->is_inited()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("scan_iter is not inited.", K(ret));
  } else if (OB_ISNULL(iter_buff = vec_op_alloc_.alloc(sizeof(ObVectorQueryRowkeyIterator)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("failed to allocator rowkey iter.", K(ret));
  } else if (OB_FALSE_IT(saved_rowkeys_itr_ = new (iter_buff) ObVectorQueryRowkeyIterator())) {
  } else {
    if (OB_FAIL(saved_rowkeys_itr_->init(scan_iter))) {
      LOG_WARN("iter init failed.", K(ret));
    }
  }

  return ret;
}

int64_t ObDASIvfBaseScanIter::get_nprobe(const common::ObLimitParam &limit_param, int64_t enlargement_factor /*= 1*/)
{
  int64_t nprobe = INT64_MAX;
  int64_t sum = limit_param.limit_ + limit_param.offset_;

  if (sum > 0 && sum <= (INT64_MAX / enlargement_factor)) {
    nprobe = sum * enlargement_factor;
  }
  return nprobe;
}

int64_t ObDASIvfBaseScanIter::get_heap_size(const int64_t limit_k, const double select_ratio)
{
  int64_t heap_size = limit_k;
  /**
   * The minimum magnification is 2x and the maximum is 10x.
   * so if select_ratio >= 0.5, use 2x
   * if select_ratio < 0.1, use 10x
   * otherwise use limit_k / select_ratio
  */
  if (select_ratio > 0.0 && select_ratio < 1.0) {
    if (select_ratio >= 0.5) heap_size = 2 * limit_k;
    else if (select_ratio <= 0.1) heap_size = 10 * limit_k;
    else heap_size = std::ceil(limit_k / select_ratio);
  }
  return heap_size;
}

int ObDASIvfBaseScanIter::gen_near_cid_heap_from_cache(ObIvfCentCache &cent_cache,
                                                   share::ObVectorCenterClusterHelper<float, ObCenterId> &nearest_cid_heap,
                                                   bool save_center_vec /*= false*/)
{
  int ret = OB_SUCCESS;
  RWLock::RLockGuard guard(cent_cache.get_lock());
  uint64_t capacity = cent_cache.get_count();
  float *cid_vec = nullptr;
  ObString cid_str;
  ObCenterId center_id;
  center_id.tablet_id_ = centroid_tablet_id_.id();
  CenterSaveMode center_save_mode = CenterSaveMode::SHALLOW_COPY_CENTER_VEC;
  // NOTE(liyao): valid center id start from 1
  for (uint64_t i = 1; i <= capacity && OB_SUCC(ret); ++i) {
    if (OB_FAIL(cent_cache.read_centroid(i, cid_vec))) {
      LOG_WARN("fail to read centroid", K(ret), K(i));
    } else if (FALSE_IT(center_id.center_id_ = i)) {
    } else if (OB_FAIL(nearest_cid_heap.push_center(center_id, cid_vec, dim_, center_save_mode))) {
      LOG_WARN("failed to push center.", K(ret));
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::try_write_centroid_cache(
    ObIvfCentCache &cent_cache,
    bool is_vectorized)
{
  int ret = OB_SUCCESS;
  RWLock::WLockGuard guard(cent_cache.get_lock());
  if (!cent_cache.is_writing()) {
    LOG_INFO("other threads already writed centroids cache, skip", K(ret));
  } else {
    ObArenaAllocator tmp_allocator;
    ObCenterId cent_id;
    const ObDASScanCtDef *centroid_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_centroid_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CENTROID_SCAN);
    ObDASScanRtDef *centroid_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_centroid_tbl_idx());
    if (OB_FAIL(do_table_full_scan(is_vectorized,
                                   centroid_ctdef,
                                   centroid_rtdef,
                                   centroid_iter_,
                                   centroid_tablet_id_,
                                   centroid_iter_first_scan_,
                                   centroid_scan_param_))) {
      LOG_WARN("failed to do centroid table scan", K(ret));
    } else if (is_vectorized) {
      IVF_GET_NEXT_ROWS_BEGIN(centroid_iter_)
      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        ObExpr *cid_expr = centroid_ctdef->result_output_[CID_IDX];
        ObExpr *cid_vec_expr = centroid_ctdef->result_output_[CID_VECTOR_IDX];
        ObDatum *cid_datum = cid_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
        ObDatum *cid_vec_datum = cid_vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
        bool has_lob_header = centroid_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();
        uint64_t center_idx = 0;

        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          guard.set_batch_idx(i);
          ObString cid = cid_datum[i].get_string();
          ObString cid_vec = cid_vec_datum[i].get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                          &tmp_allocator,
                          ObLongTextType,
                          CS_TYPE_BINARY,
                          has_lob_header,
                          cid_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(cent_id, cid, ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
            LOG_WARN("fail to get center idx from string", K(ret), KPHEX(cid.ptr(), cid.length()));
          } else if (OB_FAIL(cent_cache.write_centroid(cent_id.center_id_, reinterpret_cast<float*>(cid_vec.ptr()), cid_vec.length()))) {
            LOG_WARN("fail to write centroid", K(ret), K(center_idx), KPHEX(cid_vec.ptr(), cid_vec.length()));
          }
        }
      }
      IVF_GET_NEXT_ROWS_END(centroid_iter_, centroid_scan_param_, centroid_tablet_id_)
    } else {
      centroid_iter_->clear_evaluated_flag();
      ObExpr *cid_expr = centroid_ctdef->result_output_[CID_IDX];
      ObExpr *cid_vec_expr = centroid_ctdef->result_output_[CID_VECTOR_IDX];
      ObDatum &cid_datum = cid_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      ObDatum &cid_vec_datum = cid_vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      bool has_lob_header = centroid_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();

      uint64_t center_idx = 0;
      for (int i = 0; OB_SUCC(ret); ++i) {
        if (OB_FAIL(centroid_iter_->get_next_row())) {
          if (OB_ITER_END != ret) {
            LOG_WARN("failed to scan vid rowkey iter", K(ret));
          }
        } else {
          ObString cid = cid_datum.get_string();
          ObString cid_vec = cid_vec_datum.get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(&tmp_allocator, ObLongTextType, CS_TYPE_BINARY,
                                                                has_lob_header, cid_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(
                         cent_id, cid, ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
            LOG_WARN("fail to get center idx from string", K(ret), KPHEX(cid.ptr(), cid.length()));
          } else if (OB_FAIL(cent_cache.write_centroid(cent_id.center_id_, reinterpret_cast<float *>(cid_vec.ptr()),
                                                       cid_vec.length()))) {
            LOG_WARN("fail to write centroid", K(ret), K(center_idx), KPHEX(cid_vec.ptr(), cid_vec.length()));
          }
        }
      }  // end for
      int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, centroid_iter_, centroid_scan_param_, centroid_tablet_id_))) {
        LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }

    if (OB_SUCC(ret)) {
    if (cent_cache.get_count() > 0) {
      if (cent_cache.get_count() >= ObVecIdxExtraInfo::IVF_CENTERS_HGRAPH_THRESHOLD) {
        ObVectorIndexParam hgraph_param = vec_index_param_;
        if (hgraph_param.dim_ <= 0) {
          hgraph_param.dim_ = dim_;
        }
        if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT && need_norm_) {
          hgraph_param.dist_algorithm_ = ObVectorIndexDistAlgorithm::VIDA_COS;
        } else if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT) {
          hgraph_param.dist_algorithm_ = ObVectorIndexDistAlgorithm::VIDA_IP;
        } else if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN ||
                   dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED) {
          hgraph_param.dist_algorithm_ = ObVectorIndexDistAlgorithm::VIDA_L2;
        }
        if (OB_FAIL(cent_cache.build_hgraph_and_release_centers(hgraph_param))) {
          LOG_WARN("failed to build hgraph when writing centroid cache", K(ret), K(centroid_tablet_id_));
          cent_cache.reuse();
        } else {
          cent_cache.set_completed();
          LOG_INFO("success to write centroid table cache with hgraph", K(centroid_tablet_id_));
        }
      } else {
        cent_cache.set_completed();
        LOG_DEBUG("success to write centroid table cache", K(centroid_tablet_id_));
      }
    } else {
      cent_cache.reuse();
      LOG_DEBUG("Empty centroid table, no need to set cache", K(centroid_tablet_id_));
    }
    }
  }

  if (OB_FAIL(ret)) {
    cent_cache.reuse();
  }

  return ret;
}

int ObDASIvfBaseScanIter::get_centers_cache(bool is_vectorized,
                                        bool is_pq_centers,
                                        ObIvfCacheMgrGuard &cache_guard,
                                        ObIvfCentCache *&cent_cache,
                                        bool &is_cache_usable)
{
  int ret = OB_SUCCESS;
  ObPluginVectorIndexService *vec_index_service = MTL(ObPluginVectorIndexService *);
  ObIvfCacheMgr *cache_mgr = nullptr;
  const ObDASScanCtDef *centroid_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_centroid_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CENTROID_SCAN);

  // pq/flat both use centroid_tablet_id_
  if (! cache_guard.is_valid() && OB_FAIL(vec_index_service->acquire_ivf_cache_mgr_guard(
        ls_id_, centroid_tablet_id_, vec_index_param_, dim_, centroid_ctdef->ref_table_id_, cache_guard))) {
    LOG_WARN("failed to get ObPluginVectorIndexAdapter",
      K(ret), K(ls_id_), K(centroid_tablet_id_), K(vec_index_param_));
  } else if (OB_ISNULL(cache_mgr = cache_guard.get_ivf_cache_mgr())) {
    ret = OB_ERR_NULL_VALUE;
    LOG_WARN("invalid null cache mgr", K(ret));
  } else if (OB_FAIL(cache_mgr->get_or_create_cache_node(
      is_pq_centers ? IvfCacheType::IVF_PQ_CENTROID_CACHE : IvfCacheType::IVF_CENTROID_CACHE, cent_cache))) {
    LOG_WARN("fail to get or create cache node", K(ret), K(is_pq_centers));
    if (ret == OB_ALLOCATE_MEMORY_FAILED) {
      is_cache_usable = false;
      ret = OB_SUCCESS;
    }
  } else if (!cent_cache->is_completed()) {
    if (cent_cache->set_writing_if_idle()) {
      // write cache
      int tmp_ret = is_pq_centers ?
          try_write_pq_centroid_cache(*cent_cache, is_vectorized) :
          try_write_centroid_cache(*cent_cache, is_vectorized);
      if (OB_TMP_FAIL(tmp_ret)) {
        LOG_WARN("fail to try write centroid cache", K(ret), K(is_vectorized), KPC(cent_cache), K(is_pq_centers));
      } else {
        is_cache_usable = cent_cache->is_completed();
      }
    } else {
      LOG_INFO("other threads already writed centroids cache, skip", K(ret));
    }
  } else {
    // read cache
    is_cache_usable = true;
  }
  return ret;
}

template <typename T>
int ObDASIvfBaseScanIter::generate_nearest_cid_heap(
    bool is_vectorized,
    T &nearest_cid_heap,
    bool save_center_vec /*= false*/,
    ObIvfCentCache *cent_cache /*= nullptr*/,
    bool is_cache_usable /*= false*/)
{
  int ret = OB_SUCCESS;

  if (is_cache_usable && OB_NOT_NULL(cent_cache)) {
    if (OB_FAIL(gen_near_cid_heap_from_cache(*cent_cache, nearest_cid_heap, save_center_vec))) {
      LOG_WARN("fail to gen near cid heap from cache", K(ret), K(center_cache_guard_));
    }
  } else {
    if (OB_FAIL(gen_near_cid_heap_from_table(is_vectorized, nearest_cid_heap, save_center_vec))) {
      LOG_WARN("fail to gen near cid heap from table", K(ret), K(center_cache_guard_));
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::gen_near_cid_heap_from_table(
  bool is_vectorized,
  share::ObVectorCenterClusterHelper<float, ObCenterId> &nearest_cid_heap,
  bool save_center_vec /*= false*/)
{
  int ret = OB_SUCCESS;
  const ObDASScanCtDef *centroid_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_centroid_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CENTROID_SCAN);
  ObDASScanRtDef *centroid_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_centroid_tbl_idx());

  // if no have center table cache, need deep copy center vec, else shallol copy
  // todo(wmj): need fit ivf adaptor cache
  CenterSaveMode center_save_mode = save_center_vec ? CenterSaveMode::DEEP_COPY_CENTER_VEC : CenterSaveMode::NOT_SAVE_CENTER_VEC;
  if (OB_FAIL(do_table_full_scan(is_vectorized,
                                  centroid_ctdef,
                                  centroid_rtdef,
                                  centroid_iter_,
                                  centroid_tablet_id_,
                                  centroid_iter_first_scan_,
                                  centroid_scan_param_))) {
    LOG_WARN("failed to do centroid table scan", K(ret));
  } else if (is_vectorized) {
    IVF_GET_NEXT_ROWS_BEGIN(centroid_iter_)
    if (OB_SUCC(ret)) {
      ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
      guard.set_batch_size(scan_row_cnt);
      ObExpr *cid_expr = centroid_ctdef->result_output_[CID_IDX];
      ObExpr *cid_vec_expr = centroid_ctdef->result_output_[CID_VECTOR_IDX];
      ObDatum *cid_datum = cid_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
      ObDatum *cid_vec_datum = cid_vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
      bool has_lob_header = centroid_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();
      ObCenterId center_id;

      for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
        guard.set_batch_idx(i);
        ObString cid = cid_datum[i].get_string();
        ObString cid_vec = cid_vec_datum[i].get_string();

        if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(center_id, cid))) {
          LOG_WARN("failed to get center id from string", K(ret));
        } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                        &mem_context_->get_arena_allocator(),
                        ObLongTextType,
                        CS_TYPE_BINARY,
                        has_lob_header,
                        cid_vec))) {
          LOG_WARN("failed to get real data.", K(ret));
        } else if (OB_FAIL(nearest_cid_heap.push_center(center_id, reinterpret_cast<float *>(cid_vec.ptr()), dim_, center_save_mode))) {
          LOG_WARN("failed to push center.", K(ret));
        }
      }
    }
    IVF_GET_NEXT_ROWS_END(centroid_iter_, centroid_scan_param_, centroid_tablet_id_)
  } else {
    centroid_iter_->clear_evaluated_flag();
    ObExpr *cid_expr = centroid_ctdef->result_output_[CID_IDX];
    ObExpr *cid_vec_expr = centroid_ctdef->result_output_[CID_VECTOR_IDX];
    ObDatum &cid_datum = cid_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
    ObDatum &cid_vec_datum = cid_vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
    bool has_lob_header = centroid_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();

    ObCenterId center_id;
    for (int i = 0; OB_SUCC(ret); ++i) {
      if (OB_FAIL(centroid_iter_->get_next_row())) {
        if (OB_ITER_END != ret) {
          LOG_WARN("failed to scan vid rowkey iter", K(ret));
        }
      } else {
        ObString cid = cid_datum.get_string();
        ObString cid_vec = cid_vec_datum.get_string();
        if (OB_FAIL(ObTextStringHelper::read_real_string_data(&mem_context_->get_arena_allocator(), ObLongTextType,
                                                              CS_TYPE_BINARY, has_lob_header, cid_vec))) {
          LOG_WARN("failed to get real data.", K(ret));
        } else if (cid_vec.empty()) {
          // ignoring null vector.
        } else if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(center_id, cid))) {
          LOG_WARN("failed to get center id from string", K(ret));
        } else if (OB_FAIL(nearest_cid_heap.push_center(center_id, reinterpret_cast<float *>(cid_vec.ptr()), dim_,
                                                        center_save_mode))) {
          LOG_WARN("failed to push center.", K(ret));
        }
      }
    }

    if (ret == OB_ITER_END) {
      if (OB_FAIL(centroid_iter_->reuse())) {
        LOG_WARN("fail to reuse scan iterator.", K(ret));
      }
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::prepare_cid_range(
  const ObDASScanCtDef *cid_vec_ctdef,
  int64_t &cid_vec_column_count,
  int64_t &cid_vec_pri_key_cnt,
  int64_t &rowkey_cnt)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(cid_vec_ctdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef is null", K(ret), KP(cid_vec_ctdef));
  } else if (OB_FALSE_IT(cid_vec_column_count = cid_vec_ctdef->access_column_ids_.count())) {
  } else if (OB_UNLIKELY(cid_vec_column_count <= 0)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid rowkey cnt", K(ret));
  } else {
    cid_vec_pri_key_cnt = cid_vec_column_count - CID_VEC_COM_KEY_CNT; // access_column_ids_ have cid and vector column + rk
    rowkey_cnt = cid_vec_column_count - CID_VEC_COM_KEY_CNT - CID_VEC_FIXED_PRI_KEY_CNT;
  }
  return ret;
}

int ObDASIvfBaseScanIter::try_cid_vec_replay_only_switch(
    const uint64_t cid_num,
    storage::ObTableScanIterator *&cid_vec_scan_iter,
    bool &replay_only_handled)
{
  int ret = OB_SUCCESS;
  replay_only_handled = false;
  ObDASIvfCidVecCacheScanIter *cache_iter =
      ObDASIvfCidVecCacheScanIter::cache_was_active_iter(cid_vec_iter_)
          ? static_cast<ObDASIvfCidVecCacheScanIter *>(cid_vec_iter_)
          : nullptr;
  if (OB_ISNULL(cache_iter) || cid_vec_iter_first_scan_) {
    // First probe or cache off: caller must use full scan_cid_range.
  } else if (OB_FAIL(cache_iter->rescan_for_cid(cid_num))) {
    LOG_WARN("fail to rescan cid vec cache iter", K(ret), K(cid_num));
  } else if (!cache_iter->needs_storage_after_cid_switch()) {
    replay_only_handled = true;
    cid_vec_scan_iter = static_cast<storage::ObTableScanIterator *>(cid_vec_iter_->get_output_result_iter());
    if (OB_ISNULL(cid_vec_scan_iter)
        && !ObDASIvfCidVecCacheScanIter::cid_vec_skips_storage_output_result_iter(cid_vec_iter_)) {
      ret = OB_ERR_NULL_VALUE;
      LOG_WARN("invalid null scan iter", K(ret));
      replay_only_handled = false;
    } else if (ob_ivf_cid_range_debug_enabled()) {
      char line[256];
      const int n = snprintf(
          line,
          sizeof(line),
          "[OB_IVF_CID_RANGE_DEBUG] path=cache_replay_skip_scan_cid_range center_id=%llu\n",
          static_cast<unsigned long long>(cid_num));
      if (n > 0 && n < static_cast<int>(sizeof(line))) {
        ob_ivf_cid_probe_debug_log_line(line);
      }
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::scan_cid_range(
  const ObString &cid,
  int64_t cid_vec_pri_key_cnt,
  const ObDASScanCtDef *cid_vec_ctdef,
  ObDASScanRtDef *cid_vec_rtdef,
  storage::ObTableScanIterator *&cid_vec_scan_iter,
  const ObCenterId *center_id)
{
  int ret = OB_SUCCESS;
  ObNewRange cid_pri_key_range;
  if (OB_ISNULL(cid_vec_ctdef) || OB_ISNULL(cid_vec_rtdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(cid_vec_ctdef), KP(cid_vec_rtdef));
  } else {
    // Scan range must use full store rowkey width (schema rowkey + MV suffix cols).
    const int64_t range_rowkey_cnt = calc_cid_vec_storage_range_rowkey_cnt(*cid_vec_ctdef);
    const int64_t schema_rowkey_cnt =
        cid_vec_ctdef->table_param_.get_read_info().get_schema_rowkey_count();
    const int64_t store_rowkey_cnt = range_rowkey_cnt;
    if (OB_UNLIKELY(range_rowkey_cnt <= 0)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("invalid storage rowkey count for cid vec scan range", K(ret), K(range_rowkey_cnt),
          K(cid_vec_pri_key_cnt), KPC(vec_aux_ctdef_));
    }
    ObDASIvfCidVecCacheScanIter *cache_iter =
        ObDASIvfCidVecCacheScanIter::cache_was_active_iter(cid_vec_iter_)
            ? static_cast<ObDASIvfCidVecCacheScanIter *>(cid_vec_iter_)
            : nullptr;
    ObString cid_for_range = cid;
    if (cid_for_range.empty() && OB_NOT_NULL(center_id)) {
      char *cid_buf = static_cast<char *>(
          mem_context_->get_arena_allocator().alloc(OB_DOC_ID_COLUMN_BYTE_LENGTH));
      if (OB_ISNULL(cid_buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("failed to alloc cid buffer for scan range", K(ret));
      } else {
        ObString tmp;
        tmp.assign_buffer(cid_buf, OB_DOC_ID_COLUMN_BYTE_LENGTH);
        if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(*center_id, tmp))) {
          LOG_WARN("failed to set center_id to string", K(ret), KPC(center_id));
        } else {
          cid_for_range = tmp;
        }
      }
    }
    const bool cache_rescan_fast =
        OB_SUCC(ret) && OB_NOT_NULL(cache_iter) && !cid_vec_iter_first_scan_
        && (!cid_for_range.empty() || OB_NOT_NULL(center_id));
    if (OB_FAIL(ret)) {
    } else if (cache_rescan_fast) {
      uint64_t cid_num = 0;
      if (OB_NOT_NULL(center_id)) {
        cid_num = center_id->center_id_;
      } else if (!cid.empty()) {
        ObCenterId parsed;
        if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(
                parsed, cid, ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
          LOG_WARN("failed to parse cid from string", K(ret), K(cid));
        } else {
          cid_num = parsed.center_id_;
        }
      }
      if (OB_FAIL(ret)) {
      } else {
        bool replay_only = false;
        if (OB_FAIL(try_cid_vec_replay_only_switch(cid_num, cid_vec_scan_iter, replay_only))) {
          LOG_WARN("fail to replay-only switch cid vec cache iter", K(ret), K(cid_num));
        } else if (replay_only) {
          // REPLAY hit: on_cid_switch done; skip build range + storage rescan.
        } else if (cache_iter->needs_storage_after_cid_switch()) {
          if (OB_FAIL(build_cid_vec_query_range(cid_for_range, range_rowkey_cnt, cid_pri_key_range))) {
            LOG_WARN("failed to build cid vec query rowkey", K(ret), K(range_rowkey_cnt));
          } else {
            ob_ivf_cid_range_debug_log_scan_range(
                cid_pri_key_range,
                schema_rowkey_cnt,
                store_rowkey_cnt,
                range_rowkey_cnt,
                OB_NOT_NULL(center_id) ? center_id->center_id_ : 0,
                "cache_storage_rescan");
            cid_vec_scan_param_.key_ranges_.reuse();
            if (OB_FAIL(ObDasVecScanUtils::set_lookup_range(
                    cid_pri_key_range, cid_vec_scan_param_, cid_vec_ctdef->ref_table_id_))) {
              LOG_WARN("failed to append scan range", K(ret));
            } else if (OB_FAIL(cache_iter->complete_storage_rescan())) {
              LOG_WARN("fail to rescan cid vec storage", K(ret));
            }
          }
        }
      }
    } else {
      cid_vec_scan_param_.key_ranges_.reuse();
      if (OB_FAIL(build_cid_vec_query_range(cid_for_range, range_rowkey_cnt, cid_pri_key_range))) {
        LOG_WARN("failed to build cid vec query rowkey", K(ret), K(range_rowkey_cnt));
      } else {
        ob_ivf_cid_range_debug_log_scan_range(
            cid_pri_key_range,
            schema_rowkey_cnt,
            store_rowkey_cnt,
            range_rowkey_cnt,
            OB_NOT_NULL(center_id) ? center_id->center_id_ : 0,
            "storage_scan");
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(ObDasVecScanUtils::set_lookup_range(
                     cid_pri_key_range, cid_vec_scan_param_, cid_vec_ctdef->ref_table_id_))) {
        LOG_WARN("failed to append scan range", K(ret));
      } else if (OB_FAIL(do_aux_table_scan(cid_vec_iter_first_scan_,
                                          cid_vec_scan_param_,
                                          cid_vec_ctdef,
                                          cid_vec_rtdef,
                                          cid_vec_iter_,
                                          cid_vec_tablet_id_))) {
        LOG_WARN("fail to rescan cid vec table scan iterator.", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      cid_vec_scan_iter = static_cast<storage::ObTableScanIterator *>(cid_vec_iter_->get_output_result_iter());
      if (OB_ISNULL(cid_vec_scan_iter)
          && !ObDASIvfCidVecCacheScanIter::cid_vec_skips_storage_output_result_iter(cid_vec_iter_)) {
        ret = OB_ERR_NULL_VALUE;
        LOG_WARN("invalid null scan iter", K(ret));
      }
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::get_rowkey_pre_filter(ObIAllocator& allocator, bool is_vectorized, int64_t max_rowkey_count)
{
  int ret = OB_SUCCESS;
  uint64_t rowkey_count = 0;
  while (OB_SUCC(ret) && rowkey_count < max_rowkey_count) {
    inv_idx_scan_iter_->clear_evaluated_flag();
    if (!is_vectorized) {
      ObRowkey *rowkey = nullptr;
      if (OB_FAIL(inv_idx_scan_iter_->get_next_row())) {
        if (OB_ITER_END != ret) {
          LOG_WARN("failed to get next rowkey", K(ret));
        }
      } else if (OB_FALSE_IT(rowkey_count++)) {
      } else if (OB_FAIL(get_rowkey(allocator, rowkey))) {
        // pre_fileter_rowkeys_ need keep rowkey mem, so use vec_op_alloc_
        LOG_WARN("failed to get rowkey", K(ret));
      } else if (OB_FAIL(pre_fileter_rowkeys_.push_back(*rowkey))) {
        LOG_WARN("failed to push rowkey", K(ret));
      }
    } else {
      int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;

      int64_t scan_row_cnt = 0;
      if (OB_FAIL(inv_idx_scan_iter_->get_next_rows(scan_row_cnt, batch_row_count))) {
        if (OB_ITER_END != ret) {
          LOG_WARN("failed to get next rowkey", K(ret));
        }
      }

      rowkey_count += scan_row_cnt;
      if (OB_FAIL(ret) && OB_ITER_END != ret) {
        LOG_WARN("fail to get next row from inv_idx_scan_iter_", K(ret));
      } else if (scan_row_cnt > 0) {
        ret = OB_SUCCESS;
      }

      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        for (int i = 0; OB_SUCC(ret) && i < scan_row_cnt; i++) {
          guard.set_batch_idx(i);
          ObRowkey *rowkey = nullptr;
          // pre_fileter_rowkeys_ need keep rowkey mem, so use vec_op_alloc_
          if (OB_FAIL(get_rowkey(allocator, rowkey))) {
            LOG_WARN("failed to add rowkey", K(ret), K(i));
          } else if (OB_FAIL(pre_fileter_rowkeys_.push_back(*rowkey))) {
            LOG_WARN("store push rowkey", K(ret));
          }
        }
      }
    }
  }

  if (OB_FAIL(ret) && OB_ITER_END != ret) {
  } else if (OB_ITER_END == ret && rowkey_count > 0) {
    ret = OB_SUCCESS;
  }

  return ret;
}

int ObDASIvfBaseScanIter::get_main_rowkey_from_cid_vec_datum(ObIAllocator& allocator,
                                                             const ObDASScanCtDef *cid_vec_ctdef,
                                                             const int64_t rowkey_cnt,
                                                             ObRowkey &main_rowkey,
                                                             bool need_alloc /* true */)
{
  int ret = OB_SUCCESS;

  ObObj *obj_ptr = nullptr;
  void *buf = nullptr;

  if (OB_ISNULL(cid_vec_ctdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(cid_vec_ctdef));
  } else {
    ObDASScanRtDef *cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
    ObEvalCtx *cid_vec_eval_ctx = OB_NOT_NULL(cid_vec_rtdef) ? cid_vec_rtdef->eval_ctx_ : vec_aux_rtdef_->eval_ctx_;
    if (OB_NOT_NULL(cid_vec_iter_) && OB_NOT_NULL(cid_vec_eval_ctx)) {
      const int lazy_ret = ObDASIvfCidVecCacheScanIter::try_get_lazy_replay_main_rowkey(
          cid_vec_iter_, *cid_vec_eval_ctx, allocator, cid_vec_ctdef, rowkey_cnt, main_rowkey, need_alloc);
      if (lazy_ret == OB_SUCCESS) {
        return OB_SUCCESS;
      } else if (lazy_ret != OB_ENTRY_NOT_EXIST) {
        ret = lazy_ret;
        LOG_WARN("failed to get lazy replay rowkey", K(ret));
        return ret;
      }
    }
    // cid_vec_scan_iter output: [IVF_CID_VEC_CID_COL IVF_CID_VEC_VECTOR_COL ROWKEY]
    // Note: when _enable_defensive_check = 2, cid_vec_out_exprs is [IVF_CID_VEC_CID_COL IVF_CID_VEC_VECTOR_COL ROWKEY
    // DEFENSE_CHECK_COL]
    const ExprFixedArray& cid_vec_out_exprs = cid_vec_ctdef->result_output_;
    if (rowkey_cnt > cid_vec_out_exprs.count() - CID_VEC_COM_KEY_CNT - CID_VEC_FIXED_PRI_KEY_CNT) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("rowkey_cnt is illegal", K(ret), K(rowkey_cnt), K(cid_vec_out_exprs.count()));
    } else if (need_alloc) {
      if (OB_ISNULL(buf = allocator.alloc(sizeof(ObObj) * rowkey_cnt))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("allocate memory failed", K(ret), K(rowkey_cnt));
      } else if (OB_FALSE_IT(obj_ptr = new (buf) ObObj[rowkey_cnt])) {
      }
    } else {
      obj_ptr = main_rowkey.get_obj_ptr();
    }
    if (OB_FAIL(ret)) {
    } else {
      ObDASScanRtDef *cid_vec_rtdef_inner =
          vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
      ObEvalCtx *cid_vec_eval_ctx_inner =
          OB_NOT_NULL(cid_vec_rtdef_inner) ? cid_vec_rtdef_inner->eval_ctx_ : vec_aux_rtdef_->eval_ctx_;
      int rowkey_idx = 0;
      for (int64_t i = 2; OB_SUCC(ret) && i < cid_vec_out_exprs.count() && rowkey_idx < rowkey_cnt; ++i) {
        ObObj tmp_obj;
        ObExpr *expr = cid_vec_out_exprs.at(i);
        if (OB_ISNULL(expr)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("should not be null", K(ret));
        } else if (OB_FAIL(expr->locate_expr_datum(*cid_vec_eval_ctx_inner).to_obj(obj_ptr[rowkey_idx++], expr->obj_meta_, expr->obj_datum_map_))) {
          LOG_WARN("convert datum to obj failed", K(ret));
        }
      }
    }
    if (OB_SUCC(ret) && need_alloc) {
      main_rowkey.assign(obj_ptr, rowkey_cnt);
    }
  }

  return ret;
}

int ObDASIvfBaseScanIter::get_pre_filter_rowkey_batch(ObIAllocator &allocator,
                                                  bool is_vectorized,
                                                  int64_t batch_row_count,
                                                  bool &index_end)
{
  int ret = OB_SUCCESS;
  index_end = false;
  if (!is_vectorized) {
    for (int i = 0; OB_SUCC(ret) && i < batch_row_count && !index_end; ++i) {
      inv_idx_scan_iter_->clear_evaluated_flag();
      ObRowkey *rowkey = nullptr;
      if (OB_FAIL(inv_idx_scan_iter_->get_next_row())) {
        ret = OB_ITER_END == ret ? OB_SUCCESS : ret;
        index_end = true;
      } else if (OB_FAIL(get_rowkey(allocator, rowkey))) {
        // pre_fileter_rowkeys_ need keep rowkey mem, so use vec_op_alloc_
        LOG_WARN("failed to get rowkey", K(ret));
      } else if (OB_FAIL(pre_fileter_rowkeys_.push_back(*rowkey))) {
        LOG_WARN("failed to save rowkey", K(ret));
      }
    }
  } else {
    int64_t scan_row_cnt = 0;
    if (OB_FAIL(inv_idx_scan_iter_->get_next_rows(scan_row_cnt, batch_row_count))) {
      if (OB_ITER_END != ret) {
        LOG_WARN("failed to get next row.", K(ret));
      }
      index_end = true;
    }

    if (OB_FAIL(ret) && OB_ITER_END != ret) {
    } else if (scan_row_cnt > 0) {
      ret = OB_SUCCESS;
    }

    if (OB_SUCC(ret)) {
      ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
      guard.set_batch_size(scan_row_cnt);
      for (int i = 0; OB_SUCC(ret) && i < scan_row_cnt; i++) {
        guard.set_batch_idx(i);
        ObRowkey *rowkey = nullptr;
        // pre_fileter_rowkeys_ need keep rowkey mem, so use vec_op_alloc_
        if (OB_FAIL(get_rowkey(allocator, rowkey))) {
          LOG_WARN("failed to add rowkey", K(ret), K(i));
        } else if (OB_FAIL(pre_fileter_rowkeys_.push_back(*rowkey))) {
          LOG_WARN("failed to save rowkey", K(ret));
        }
      }
    }
  }

  return ret;
}

template <typename T>
int ObDASIvfBaseScanIter::calc_vec_dis(T *a, T *b, int dim, float &dis, ObExprVectorDistance::ObVecDisType dis_type)
{
  int ret = OB_SUCCESS;
  if (dis_type != oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN) {
    double distance = 0;
    if (OB_FAIL(oceanbase::sql::ObExprVectorDistance::DisFunc<T>::distance_funcs[dis_type](a, b, dim, distance))) {
      LOG_WARN("failed to calculate distance", K(ret), K(dis_type), KP(a), KP(b), K(dim));
    } else {
      dis = distance;
    }
  } else {
    dis = ObVectorL2Distance<T>::l2_square_flt_func(a, b, dim);
  }

  return ret;
}

int ObDASIvfBaseScanIter::do_post_filter(bool is_vectorized, ObIVFRowkeyDistMap &rowkey_dist_map, ObSEArray<ObIvfRowkeyDistEntry, 16> &matched_rowkeys)
{
  int ret = OB_SUCCESS;
  if (vec_aux_ctdef_->can_use_vec_pri_opt()) {
    if (OB_FAIL(do_simple_post_filter(rowkey_dist_map, matched_rowkeys))) {
      LOG_WARN("do_simple_post_filter fail", K(ret));
    }
  } else if (OB_FAIL(do_table_post_filter(is_vectorized, rowkey_dist_map, matched_rowkeys))) {
    LOG_WARN("do_table_post_filter fail", K(ret));
  }
  return ret;
}

int ObDASIvfBaseScanIter::do_simple_post_filter(ObIVFRowkeyDistMap &rowkey_dist_map, ObSEArray<ObIvfRowkeyDistEntry, 16> &matched_rowkeys)
{
  int ret = OB_SUCCESS;
  ObIVFRowkeyDistMapIterator rowkey_iter = rowkey_dist_map.begin();
  ObIVFRowkeyDistMapIterator rowkey_end = rowkey_dist_map.end();
  ObDASScanIter* inv_iter = (ObDASScanIter*)inv_idx_scan_iter_;
  ObArray<const ObNewRange *> rk_range;
  const ObRangeArray& key_range = inv_iter->get_scan_param().key_ranges_;
  for (int64_t i = 0; i < key_range.count() && OB_SUCC(ret); i++) {
    const ObNewRange *range = &key_range.at(i);
    if (OB_FAIL(rk_range.push_back(range))) {
      LOG_WARN("fail to push back range", K(ret), K(i));
    }
  }
  while (OB_SUCC(ret) && rowkey_iter != rowkey_end) {
    ObRowkey &rowkey = rowkey_iter->first;
    const ObIvfRowkeyDistEntry &entry = rowkey_iter->second;
    bool is_match = false;
    ObNewRange tmp_range;
    if (OB_FAIL(tmp_range.build_range(rk_range.at(0)->table_id_, rowkey))) {
      LOG_WARN("fail to build tmp range", K(ret));
    }
    // do compare
    for (int64_t i = 0; i < rk_range.count() && !is_match && OB_SUCC(ret); i++) {
      if (rk_range.at(i)->compare_with_startkey2(tmp_range) <= 0 && rk_range.at(i)->compare_with_endkey2(tmp_range) >= 0) {
        is_match = true;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (is_match && OB_FAIL(matched_rowkeys.push_back(entry))) {
      LOG_WARN("failed to push back.", K(ret), K(rowkey), K(entry));
    } else {
      ++rowkey_iter;
    }
  }

  if (OB_SUCC(ret)) {
    adaptive_ctx_.iter_filter_row_cnt_ += rowkey_dist_map.size();
    adaptive_ctx_.iter_res_row_cnt_ += matched_rowkeys.count();
  }
  return ret;
}

int ObDASIvfBaseScanIter::do_table_post_filter(bool is_vectorized, ObIVFRowkeyDistMap &rowkey_dist_map, ObSEArray<ObIvfRowkeyDistEntry, 16> &matched_rowkeys)
{
  int ret = OB_SUCCESS;
  int64_t scan_row_cnt = 0;
  int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  int64_t count = 0;
  int64_t matched_count = 0;
  ObIVFRowkeyDistMapIterator rowkey_iter = rowkey_dist_map.begin();
  ObIVFRowkeyDistMapIterator rowkey_end = rowkey_dist_map.end();
  while (OB_SUCC(ret) && rowkey_iter != rowkey_end) {
    for (int64_t i = 0; OB_SUCC(ret) && i < batch_row_count && rowkey_iter != rowkey_end; ++i) {
      ObRowkey &rowkey = rowkey_iter->first;
      if (OB_FAIL(ObDasVecScanUtils::set_lookup_key(
              rowkey, data_filter_scan_param_, data_filter_ctdef_->ref_table_id_))) {
        LOG_WARN("failed to set lookup key", K(ret), K(i), K(count));
      } else {
        ++count;
        ++rowkey_iter;
        LOG_DEBUG("add filter rowkey", K(i), K(rowkey), K(count));
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(do_aux_table_scan(data_filter_iter_first_scan_,
                                      data_filter_scan_param_,
                                      data_filter_ctdef_,
                                      data_filter_rtdef_,
                                      data_filter_iter_,
                                      data_filter_tablet_id_))) {
      LOG_WARN("failed to do data filter table scan.", K(ret), K(count), K(data_filter_iter_first_scan_));
    } else if (is_vectorized) {
      IVF_GET_NEXT_ROWS_BEGIN(data_filter_iter_)
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          guard.set_batch_idx(i);
          ObRowkey &main_rowkey = tmp_main_rowkey_;
          ObIvfRowkeyDistEntry* entry = nullptr;
          if (OB_FAIL(get_main_rowkey(data_filter_ctdef_, main_rowkey))) {
            LOG_WARN("fail to get main rowkey", K(ret));
          } else if (OB_ISNULL(entry = rowkey_dist_map.get(main_rowkey))) {
            ret = OB_ENTRY_NOT_EXIST;
            LOG_WARN("rowkey not exist", K(ret), K(main_rowkey));
          } else if (OB_FAIL(matched_rowkeys.push_back(*entry))) {
            LOG_WARN("failed to push back.", K(ret), KPC(entry));
          } else {
            ++matched_count;
            LOG_DEBUG("match filter rowkey", K(i), K(main_rowkey), K(matched_count));
          }
        }
      IVF_GET_NEXT_ROWS_END(data_filter_iter_, data_filter_scan_param_, data_filter_tablet_id_)
    } else {
      data_filter_iter_->clear_evaluated_flag();
      for (int i = 0; OB_SUCC(ret) && i < batch_row_count; ++i) {
        ObRowkey &main_rowkey = tmp_main_rowkey_;
        ObIvfRowkeyDistEntry* entry = nullptr;
        if (OB_FAIL(data_filter_iter_->get_next_row())) {
          if (OB_ITER_END != ret) {
            LOG_WARN("failed to scan vid rowkey iter", K(ret));
          }
        } else if (OB_FAIL(get_main_rowkey(data_filter_ctdef_, main_rowkey))) {
          LOG_WARN("fail to get main rowkey", K(ret));
        } else if (OB_ISNULL(entry = rowkey_dist_map.get(main_rowkey))) {
          ret = OB_ENTRY_NOT_EXIST;
          LOG_WARN("rowkey not exist", K(ret), K(main_rowkey));
        } else if (OB_FAIL(matched_rowkeys.push_back(*entry))) {
          LOG_WARN("failed to push back.", K(ret), KPC(entry));
        } else {
          ++matched_count;
          LOG_DEBUG("match filter rowkey", K(i), K(main_rowkey), K(matched_count));
        }
      }
      int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, data_filter_iter_, data_filter_scan_param_, data_filter_tablet_id_))) {
        LOG_WARN("failed to reuse data_filter iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }
  }
  if (OB_SUCC(ret)) {
    adaptive_ctx_.iter_filter_row_cnt_ += count;
    adaptive_ctx_.iter_res_row_cnt_ += matched_count;
  }
  return ret;
}

void ObDASIvfBaseScanIter::reuse_cid_ctx()
{
  iterative_filter_ctx_.reuse();
  probe_rotate_offset_ = 0;
  probe_rotate_count_ = 0;
}

int ObDASIvfBaseScanIter::reuse_cid_vec_iter_after_probe()
{
  int ret = OB_SUCCESS;
  if (ObDASIvfCidVecCacheScanIter::skip_inter_cid_full_reuse(cid_vec_iter_)) {
    static_cast<ObDASIvfCidVecCacheScanIter *>(cid_vec_iter_)->prepare_for_next_cid_probe();
  } else if (OB_FAIL(ObDasVecScanUtils::reuse_iter(
                 ls_id_, cid_vec_iter_, cid_vec_scan_param_, cid_vec_tablet_id_))) {
    LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
  }
  return ret;
}

bool ObDASIvfBaseScanIter::ivf_probe_rotate_enabled()
{
  static int cached = -1;
  if (cached < 0) {
    const char *env = getenv("OB_IVF_PROBE_ROTATE");
    // default on; set OB_IVF_PROBE_ROTATE=0 to disable
    cached = (nullptr == env || '\0' == env[0] || (env[0] == '1' && '\0' == env[1])) ? 1 : 0;
  }
  return cached != 0;
}

void ObDASIvfBaseScanIter::assign_probe_rotate_offset_(const int64_t candidate_cnt)
{
  probe_rotate_count_ = candidate_cnt;
  probe_rotate_offset_ = 0;
  if (candidate_cnt > 1 && ivf_probe_rotate_enabled()) {
    probe_rotate_offset_ = ObRandom::rand(0, candidate_cnt - 1);
  }
}

int64_t ObDASIvfBaseScanIter::rotated_probe_idx_(const int64_t k) const
{
  return probe_rotate_count_ <= 0 ? k : (probe_rotate_offset_ + k) % probe_rotate_count_;
}

int64_t ObDASIvfBaseScanIter::get_cid_vec_batch_count()
{
  int batch_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  if (strategy_ == ObVecIdxQueryStrategy::LATENCY_FIRST && max_scan_vectors_ > 0 && adaptive_ctx_.iter_times_ > 1) {
    batch_count = OB_MIN(max_scan_vectors_ - adaptive_ctx_.cid_vec_scan_rows_ + 1, ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE);
  }
  return batch_count;
}
/*************************** implement ObDASIvfScanIter ****************************/
int ObDASIvfScanIter::inner_init(ObDASIterParam &param)
{
  int ret = ObDASIvfBaseScanIter::inner_init(param);
  if (OB_FAIL(ret)) {
    LOG_WARN("failed to init", K(ret));
  } else if (is_post_filter()) {
    if (OB_FAIL(near_cid_.reserve(nprobes_))) {
      LOG_WARN("failed to reserve nearest cid vec", K(ret), K(vec_index_param_.nlist_));
    }
  } else if (OB_FAIL(near_cid_dist_.allocate_array(persist_alloc_, vec_index_param_.nlist_ + 1))) {
    LOG_WARN("failed to reserve nearest cid vec", K(ret), K(vec_index_param_.nlist_));
  } else {
    memset(near_cid_dist_.get_data(), 0, near_cid_dist_.count() * sizeof(bool));
  }
  return ret;
};

int ObDASIvfScanIter::inner_reuse()
{
  reset_ivf_sq8_latent_float_heap_ctx();
  near_cid_.reuse();
  memset(near_cid_dist_.get_data(), 0, near_cid_dist_.count() * sizeof(bool));
  return ObDASIvfBaseScanIter::inner_reuse();
}

int ObDASIvfScanIter::inner_release()
{
  near_cid_.reset();
  near_cid_dist_.reset();
  return ObDASIvfBaseScanIter::inner_release();
}

// HGraph-based iterative filtering implementations
int ObDASIvfScanIter::get_next_probe_centers_ids_by_hgraph(bool is_vectorized, int64_t next_nprobe)
{
  int ret = OB_SUCCESS;
  ObIvfCacheMgrGuard cache_guard;
  ObIvfCentCache *cent_cache = nullptr;
  bool is_cache_usable = false;

  if (OB_FAIL(get_centers_cache(is_vectorized, false /*is_pq_centers*/, cache_guard, cent_cache, is_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret));
  } else if (is_cache_usable && OB_NOT_NULL(cent_cache) && cent_cache->has_hgraph_index()) {
    if (OB_FAIL(get_nearest_probe_center_ids_with_hgraph(
        is_vectorized, cent_cache, false, !vec_aux_ctdef_->is_post_filter(), next_nprobe))) {
      LOG_WARN("HGraph iterative search failed", K(ret), K(next_nprobe));
    }
  } else {
    // If HGraph was used before but now unavailable, this is an error
    ret = OB_ERR_UNEXPECTED;
    LOG_ERROR("HGraph was used in initial scan but now unavailable for iterative filtering", K(ret),
        K(is_cache_usable), K(OB_NOT_NULL(cent_cache)), K(cent_cache ? cent_cache->has_hgraph_index() : false));
  }

  return ret;
}

int ObDASIvfScanIter::get_next_center_ids(bool is_vectorized, int64_t next_nprobe)
{
  int ret = OB_SUCCESS;
  if (has_used_hgraph_) {
    if (OB_FAIL(get_next_probe_centers_ids_by_hgraph(is_vectorized, next_nprobe))) {
      LOG_WARN("failed to get next probe centers ids by hgraph", K(ret), K(is_vectorized), K(next_nprobe));
    }
  } else {
    if (OB_FAIL(iterative_filter_ctx_.get_next_nearest_probe_center_ids(next_nprobe, near_cid_))) {
      LOG_WARN("failed to get next nearest probe center ids", K(ret), K(next_nprobe));
    }
  }
  if (OB_SUCC(ret) && near_cid_.count() > 0) {
    assign_probe_rotate_offset_(near_cid_.count());
  }
  return ret;
}

int ObDASIvfScanIter::get_nearest_probe_center_ids_with_hgraph(bool is_vectorized, ObIvfCentCache *hgraph_cache,
                                                               bool need_vectors, bool need_distances, int64_t incremental_nprobes)
{
  int ret = OB_SUCCESS;

  if (OB_ISNULL(hgraph_cache)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("hgraph cache is null", K(ret));
  } else {
    common::obvsag::VectorIndexPtr hgraph_index = hgraph_cache->get_hgraph_index();
    if (OB_ISNULL(hgraph_index)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("hgraph index is null", K(ret));
    } else {
      const float* distances = nullptr;
      const int64_t* result_ids = nullptr;
      int64_t result_size = 0;
      float* search_vec = reinterpret_cast<float*>(const_cast<char*>(real_search_vec_.ptr()));
      const char* extra_info = nullptr;
      int64_t centers_total = hgraph_cache->get_count();
      int64_t ef_search = 64;
      // Check if need to initialize HGraph state (incremental_nprobes == -1 means initial search)
      bool need_init = (incremental_nprobes == -1);
      int64_t search_count = need_init ? nprobes_ : incremental_nprobes;
      // Create VSAG allocator if not exists (same lifecycle as hgraph_iter_ctx_)
      if (OB_FAIL(init_hgraph_search_alloc())) {
        LOG_WARN("failed to init hgraph search alloc", K(ret));
      } else if (OB_FAIL(obvectorutil::knn_search(hgraph_index,
                                           search_vec,
                                           dim_,
                                           search_count,  // Request incremental count
                                           distances,
                                           result_ids,
                                           extra_info,
                                           result_size,
                                           ef_search,
                                           nullptr,
                                           false,
                                           false,
                                           1.0f,
                                           hgraph_vsag_alloc_,
                                           false,
                                           hgraph_iter_ctx_,        // Always pass iter context, let HGraph manage it
                                           false))) {               // is_last_search, keep searching
        LOG_WARN("failed to search with hgraph index", K(ret), K(search_count), K(ef_search));
      }

      if (OB_FAIL(ret)) {
      } else if (OB_ISNULL(result_ids)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("hgraph search result is null", K(ret));
      } else if (result_size == 0) {
        ret = OB_ENTRY_NOT_EXIST;
        LOG_WARN("hgraph search result is empty", K(ret));
      } else {
        near_cid_.reset();
        int64_t new_center_cnt = 0;
        for (int64_t i = 0; i < result_size && OB_SUCC(ret); ++i) {
          int64_t center_idx = result_ids[i];
          if (center_idx >= 1 && center_idx <= centers_total) {
            ObCenterId center_id;
            center_id.center_id_ = center_idx;
            center_id.tablet_id_ = centroid_tablet_id_.id();

            if (OB_FAIL(near_cid_.push_back(center_id))) {
              LOG_WARN("failed to push center id", K(ret), K(center_id));
            } else {
              new_center_cnt++;
              // Mark in near_cid_dist_ for check_cid_exist (if needed)
              if (need_distances && center_idx < near_cid_dist_.count()) {
                near_cid_dist_.get_data()[center_idx] = true;
              }
            }
          } else {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("invalid center index from hgraph", K(ret), K(center_idx), K(centers_total));
          }
        }

        LOG_DEBUG("HGraph IVF Flat search completed", K(result_size), K(new_center_cnt),
                 K(incremental_nprobes), K(search_count));

        // Mark that HGraph has reached the end if returned fewer centers than requested
        if (!need_init && new_center_cnt < search_count) {
          hgraph_has_next_center_ = false;
          LOG_INFO("HGraph reached end of centers", K(new_center_cnt), K(search_count));
        }
      }
    }
  }
  return ret;
}

int ObDASIvfScanIter::get_nearest_probe_center_ids(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  ObIvfCoarseWallGuard coarse_guard(ivf_lat_);

  // decide which search strategy to use based on cache type
  bool need_vectors = false;
  bool need_distances = !vec_aux_ctdef_->is_post_filter();
  ObIvfCacheMgrGuard cache_guard;
  ObIvfCentCache *cent_cache = nullptr;
  bool is_cache_usable = false;
  if (OB_FAIL(get_centers_cache(is_vectorized, false /*is_pq_centers*/, cache_guard, cent_cache, is_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret));
  } else if (is_cache_usable && OB_NOT_NULL(cent_cache) && cent_cache->has_hgraph_index()) {
    has_used_hgraph_ = true;  // Record that HGraph was used in initial scan
    if (OB_FAIL(get_nearest_probe_center_ids_with_hgraph(is_vectorized, cent_cache, need_vectors, need_distances))) {
      LOG_WARN("HGraph search failed for IVF scan", K(ret));
    }
  } else {
    share::ObVectorCenterClusterHelper<float, ObCenterId> nearest_cid_heap(
      mem_context_->get_arena_allocator(), reinterpret_cast<const float *>(real_search_vec_.ptr()),
      dis_type_, dim_, nprobes_, 0.0, (is_pre_filter() || is_iter_filter()), &iterative_filter_ctx_.allocator_);
    const int64_t t_gen_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
    if (OB_FAIL(generate_nearest_cid_heap(is_vectorized, nearest_cid_heap, false, cent_cache, is_cache_usable))) {
      LOG_WARN("failed to generate nearest cid heap", K(ret), K(nprobes_), K(dim_), K(real_search_vec_));
    }
    if (ivf_lat_.enabled_) {
      ivf_lat_.coarse_load_us_ += ObTimeUtility::current_time() - t_gen_beg;
    }
    if (OB_SUCC(ret)) {
      const int64_t t_heap_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
      if (nearest_cid_heap.get_center_count() == 0) {
        ret = OB_ENTRY_NOT_EXIST;
      } else if (nearest_cid_heap.is_save_all_center()) {
        if (OB_FAIL(nearest_cid_heap.get_all_centroids(iterative_filter_ctx_.centroids_))) {
          LOG_WARN("init_centroids fail", K(ret));
        } else if (OB_FAIL(iterative_filter_ctx_.get_next_nearest_probe_center_ids(nprobes_, near_cid_))) {
          LOG_WARN("failed to get top n", K(ret), K(iterative_filter_ctx_));
        }
      } else if (OB_FAIL(nearest_cid_heap.get_nearest_probe_center_ids(near_cid_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
      if (ivf_lat_.enabled_) {
        ivf_lat_.coarse_compute_us_ += ObTimeUtility::current_time() - t_heap_beg;
      }
    }
  }
  if (OB_SUCC(ret) && near_cid_.count() > 0) {
    assign_probe_rotate_offset_(near_cid_.count());
  }
  return ret;
}

// for flat/sq, con_key is vector(float/uint8)
int ObDASIvfScanIter::parse_cid_vec_datum(
  ObIAllocator& allocator,
  int64_t cid_vec_column_count,
  const ObDASScanCtDef *cid_vec_ctdef,
  const int64_t rowkey_cnt,
  ObRowkey &main_rowkey,
  ObString &com_key)
{
  int ret = OB_SUCCESS;
  ObExpr *vec_expr = cid_vec_ctdef->result_output_[1];
  ObDASScanRtDef *cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
  ObEvalCtx *cid_vec_eval_ctx = OB_NOT_NULL(cid_vec_rtdef) ? cid_vec_rtdef->eval_ctx_ : vec_aux_rtdef_->eval_ctx_;
  if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(allocator, cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
    LOG_WARN("failed to get main rowkey from cid vec datum", K(ret));
  } else if (OB_FALSE_IT(com_key = vec_expr->locate_expr_datum(*cid_vec_eval_ctx).get_string())) {
  } else if (ObDASIvfCidVecCacheScanIter::skip_payload_lob_read(cid_vec_iter_, 0)) {
    // cache row already has expanded payload in datum
  } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                  &allocator,
                  ObLongTextType,
                  CS_TYPE_BINARY,
                  cid_vec_ctdef->result_output_.at(1)->obj_meta_.has_lob_header(),
                  com_key))) {
    LOG_WARN("failed to get real data.", K(ret));
  }
  return ret;
}

template <typename T>
int ObDASIvfScanIter::get_rowkeys_to_heap(const ObString &cid_str, int64_t cid_vec_pri_key_cnt,
                                          int64_t cid_vec_column_count, int64_t rowkey_cnt, bool is_vectorized,
                                          ObVectorCenterClusterHelper<T, ObRowkey> &nearest_rowkey_heap,
                                          bool &is_first_vec, bool &cid_vec_need_norm, ObIvfPreFilter *prefilter,
                                          const ObCenterId *center_id)
{
  int ret = OB_SUCCESS;
  float *ivf_sq8_latent_reuse_buf = nullptr;
  if constexpr (std::is_same_v<T, float>) {
    if (ivf_sq8_cid_u8_score_latent_float_heap_ && OB_NOT_NULL(ivf_sq8_meta_min_) && OB_NOT_NULL(ivf_sq8_meta_step_)) {
      if (OB_ISNULL(ivf_sq8_latent_reuse_buf = reinterpret_cast<float *>(mem_context_->get_arena_allocator().alloc(
                      static_cast<uint32_t>(sizeof(float) * static_cast<uint32_t>(dim_)))))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("alloc IVF SQ8 latent reuse buffer (one per CID scan) failed", K(ret), K(dim_));
      }
    }
  }
  if (OB_FAIL(ret)) {
    return ret;
  }
  const ObDASScanCtDef *cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CID_VEC_SCAN);
  ObDASScanRtDef *cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
  storage::ObTableScanIterator *cid_vec_scan_iter = nullptr;
  bool replay_only = false;
  if (OB_NOT_NULL(center_id) &&
      OB_FAIL(try_cid_vec_replay_only_switch(center_id->center_id_, cid_vec_scan_iter, replay_only))) {
    LOG_WARN("fail to replay-only switch cid vec cache iter", K(ret), KPC(center_id));
  } else if (!replay_only &&
             OB_FAIL(scan_cid_range(cid_str, cid_vec_pri_key_cnt, cid_vec_ctdef, cid_vec_rtdef, cid_vec_scan_iter, center_id))) {
    LOG_WARN("fail to scan cid range", K(ret), K(cid_str), K(cid_vec_pri_key_cnt));
  }
  if (OB_FAIL(ret)) {
    return ret;
  }
  if (need_norm_) {
    bool cache_unit = false;
    bool cache_known = false;
    if (ObDASIvfCidVecCacheScanIter::get_cached_payloads_l2_unit(cid_vec_iter_, cache_unit, cache_known) && cache_known) {
      cid_vec_need_norm = !cache_unit;
      is_first_vec = false;
    }
  }
  if (is_vectorized) {
    int64_t cid_vec_batch_count = get_cid_vec_batch_count();
    bool index_end = false;
    cid_vec_iter_->clear_evaluated_flag();
    int64_t scan_row_cnt = 0;
    const int64_t batch_row_count = cid_vec_batch_count;
    int64_t cv_sf_batch_idx = 0;
    while (!index_end && OB_SUCC(ret)) {
      const int64_t t_storage_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
      if (OB_FAIL(cid_vec_iter_->get_next_rows(scan_row_cnt, batch_row_count))) {
        if (OB_ITER_END != ret) {
          LOG_WARN("failed to get next row.", K(ret));
        } else {
          index_end = true;
        }
      }
      if (ivf_lat_.enabled_) {
        ivf_lat_.fine_cv_storage_fetch_us_ += ObTimeUtility::current_time() - t_storage_beg;
        ++cv_sf_batch_idx;
      }
      if (OB_FAIL(ret) && OB_ITER_END != ret) {
      } else if (scan_row_cnt > 0) {
        ret = OB_SUCCESS;
        ObEvalCtx *cid_vec_eval_ctx = cid_vec_rtdef->eval_ctx_;
        ObEvalCtx::BatchInfoScopeGuard guard(*cid_vec_eval_ctx);
        guard.set_batch_size(scan_row_cnt);
        bool has_lob_header = cid_vec_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();
        ObExpr *cid_expr = cid_vec_ctdef->result_output_[CID_VECTOR_IDX];
        ObDatum *cid_datum = cid_expr->locate_batch_datums(*cid_vec_eval_ctx);
        adaptive_ctx_.cid_vec_scan_rows_ += scan_row_cnt;
        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
        const int64_t t_vec_row_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
        bool vec_row_load_charged = false;
        guard.set_batch_idx(i);
        ObRowkey main_rowkey;
        ObString vec = cid_datum[i].get_string();
        bool skip = false;
        bool sq8_latent_handled = false;
        float *lat_sq8_candidate = nullptr;
        const bool latent_sq8_heap = std::is_same_v<T, float> && ivf_sq8_cid_u8_score_latent_float_heap_ &&
            OB_NOT_NULL(ivf_sq8_meta_min_) && OB_NOT_NULL(ivf_sq8_meta_step_);
        if (ObDASIvfCidVecCacheScanIter::skip_payload_lob_read(cid_vec_iter_, i)) {
          // cache row already has expanded payload in batch datum
        } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                        mem_context_->get_arena_allocator(),
                        cid_datum[i],
                        cid_expr->datum_meta_,
                        has_lob_header,
                        vec))) {
          LOG_WARN("failed to get real data.", K(ret));
        }
        if (OB_FAIL(ret)) {
        } else if (OB_ISNULL(vec.ptr())) {
          // ignoring null vector.
        } else if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
          LOG_WARN("fail to get main rowkey", K(ret));
        } else if (prefilter != nullptr && !prefilter->test(main_rowkey)) {
          // has been filter, do nothing
          skip = true;
        } else if (!skip && latent_sq8_heap) {
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
            vec_row_load_charged = true;
          }
          const int64_t t_vec_compute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          if constexpr (std::is_same_v<T, float>) {
            ivf_sq8_latent_float_fused_then_decode(
                nearest_rowkey_heap,
                main_rowkey,
                vec,
                dim_,
                need_norm_,
                ivf_sq8_meta_min_,
                ivf_sq8_meta_step_,
                ivf_sq8_latent_reuse_buf,
                sq8_latent_handled,
                lat_sq8_candidate,
                adaptive_ctx_.vec_dist_calc_cnt_,
                ret);
          }
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_compute_beg;
          }
        } else if (!latent_sq8_heap && std::is_same<T, float>::value && need_norm_) {
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
            vec_row_load_charged = true;
          }
          const int64_t t_vec_compute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          const bool cache_zero_copy = ObDASIvfCidVecCacheScanIter::skip_payload_lob_read(cid_vec_iter_, i);
          const bool will_norm = is_first_vec || cid_vec_need_norm;
          const int64_t dim_bytes = vec_aux_ctdef_->dim_ * static_cast<int64_t>(sizeof(float));
          char *norm_scratch = nullptr;
          if (cache_zero_copy && will_norm && vec.length() >= dim_bytes
              && OB_ISNULL(norm_scratch = static_cast<char *>(
                      mem_context_->get_arena_allocator().alloc(dim_bytes)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (cache_zero_copy && will_norm && OB_NOT_NULL(norm_scratch)) {
            MEMCPY(norm_scratch, vec.ptr(), dim_bytes);
            cid_datum[i].set_string(norm_scratch, dim_bytes);
            vec = cid_datum[i].get_string();
          }
          // If the first vec needs do_norm, it means that the vec in the cid_vector table is not normalized.
          if (OB_FAIL(ret)) {
          } else if (is_first_vec) {
            if (OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                    vec_aux_ctdef_->dim_, reinterpret_cast<float *>(vec.ptr()), reinterpret_cast<float *>(vec.ptr()),
                    &cid_vec_need_norm))) {
              LOG_WARN("failed to normalize vector.", K(ret));
            } else {
              is_first_vec = false;
            }
          } else if (cid_vec_need_norm && OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                                              vec_aux_ctdef_->dim_, reinterpret_cast<float *>(vec.ptr()),
                                              reinterpret_cast<float *>(vec.ptr())))) {
            LOG_WARN("failed to normalize vector.", K(ret));
          }
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_compute_beg;
          }
        }
        // Latent decode uses its own branch above; must normalize here so we do not skip L2 when decode succeeds
        // (original code relied on else-if after OB_FAIL(alloc), which does not run when latent branch is taken on success).
        if (OB_SUCC(ret) && !skip && nullptr != lat_sq8_candidate && need_norm_) {
          const int64_t t_vec_norm_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          if (is_first_vec) {
            if (OB_FAIL(ObVectorNormalize::L2_normalize_vector(vec_aux_ctdef_->dim_, lat_sq8_candidate, lat_sq8_candidate,
                    &cid_vec_need_norm))) {
              LOG_WARN("failed to normalize latent SQ8 cid vector.", K(ret));
            } else {
              is_first_vec = false;
            }
          } else if (cid_vec_need_norm && OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                                             vec_aux_ctdef_->dim_, lat_sq8_candidate, lat_sq8_candidate))) {
            LOG_WARN("failed to normalize latent SQ8 cid vector.", K(ret));
          }
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_norm_beg;
          }
        }
        if (OB_FAIL(ret)) {
          LOG_WARN("failed to get rowkey", K(ret));
          if (ivf_lat_.enabled_ && !vec_row_load_charged) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
          }
        } else if (skip) {
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
          }
        } else if (sq8_latent_handled) {
          // distance push done in fused IVF_SQ8 latent path
        } else if (nullptr != lat_sq8_candidate) {
          // get_rowkeys_to_heap is also instantiated for T=uint8; latent CID is float * only when T=float.
          if constexpr (std::is_same_v<T, float>) {
            const int64_t t_vec_push_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
            if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, lat_sq8_candidate, dim_))) {
              LOG_WARN("failed to push center (IVF_SQ8 latent-dequant)", K(ret));
            } else {
              adaptive_ctx_.vec_dist_calc_cnt_ ++;
            }
            if (ivf_lat_.enabled_) {
              ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_push_beg;
            }
          }
        } else if (OB_NOT_NULL(vec.ptr())
                   && ivf_sq8_latent_row_may_push_raw_vec(latent_sq8_heap, sq8_latent_handled, lat_sq8_candidate)) {
          if (ivf_lat_.enabled_ && !vec_row_load_charged) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
            vec_row_load_charged = true;
          }
          const int64_t t_vec_push_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, reinterpret_cast<T *>(vec.ptr()), dim_))) {
            LOG_WARN("failed to push center.", K(ret));
          } else {
            adaptive_ctx_.vec_dist_calc_cnt_ ++;
          }
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_push_beg;
          }
        } else {
          if (ivf_lat_.enabled_ && !vec_row_load_charged) {
            ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
          }
        }
        }
      }
    }
    if (index_end) {
      const int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(reuse_cid_vec_iter_after_probe())) {
        LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }
  } else {
    while (OB_SUCC(ret)) {
      const int64_t t_serial_row_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
      ObRowkey main_rowkey;
      ObString vec;
      bool skip = false;
      bool sq8_latent_handled = false;
      bool latent_sq8_heap = false;
      float *lat_sq8_candidate = nullptr;
      adaptive_ctx_.cid_vec_scan_rows_ ++;
      // cid_vec_iter_ output: [IVF_CID_VEC_CID_COL IVF_CID_VEC_VECTOR_COL ROWKEY]
      if (OB_FAIL(cid_vec_iter_->get_next_row())) {
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
        }
        if (OB_ITER_END != ret) {
          LOG_WARN("failed to scan vid rowkey iter", K(ret));
        }
      } else if (OB_FAIL(parse_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_column_count, cid_vec_ctdef, rowkey_cnt, main_rowkey, vec))) {
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
        }
        LOG_WARN("fail to parse cid vec datum", K(ret), K(cid_vec_column_count), K(rowkey_cnt));
      } else if (prefilter != nullptr && !prefilter->test(main_rowkey)) {
        // has been filter, do nothing
        skip = true;
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
        }
      } else if (OB_ISNULL(vec.ptr())) {
        // ignoring null vector.
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
        }
      } else {
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
        }
        const int64_t t_serial_compute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
        latent_sq8_heap = std::is_same_v<T, float> && ivf_sq8_cid_u8_score_latent_float_heap_ &&
            OB_NOT_NULL(ivf_sq8_meta_min_) && OB_NOT_NULL(ivf_sq8_meta_step_);
        if (!skip && latent_sq8_heap) {
          if constexpr (std::is_same_v<T, float>) {
            ivf_sq8_latent_float_fused_then_decode(
                nearest_rowkey_heap,
                main_rowkey,
                vec,
                dim_,
                need_norm_,
                ivf_sq8_meta_min_,
                ivf_sq8_meta_step_,
                ivf_sq8_latent_reuse_buf,
                sq8_latent_handled,
                lat_sq8_candidate,
                adaptive_ctx_.vec_dist_calc_cnt_,
                ret);
          }
        } else if (!latent_sq8_heap && std::is_same<T, float>::value && need_norm_) {
          const bool cache_zero_copy = ObDASIvfCidVecCacheScanIter::skip_payload_lob_read(cid_vec_iter_, 0);
          const bool will_norm = is_first_vec || cid_vec_need_norm;
          const int64_t dim_bytes = vec_aux_ctdef_->dim_ * static_cast<int64_t>(sizeof(float));
          char *norm_scratch = nullptr;
          if (cache_zero_copy && will_norm && vec.length() >= dim_bytes
              && OB_ISNULL(norm_scratch = static_cast<char *>(
                      mem_context_->get_arena_allocator().alloc(dim_bytes)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (cache_zero_copy && will_norm && OB_NOT_NULL(norm_scratch)) {
            MEMCPY(norm_scratch, vec.ptr(), dim_bytes);
            ObExpr *vec_expr = cid_vec_ctdef->result_output_[1];
            ObEvalCtx *eval_ctx = cid_vec_rtdef->eval_ctx_;
            vec_expr->locate_datum_for_write(*eval_ctx).set_string(norm_scratch, dim_bytes);
            vec = vec_expr->locate_expr_datum(*eval_ctx).get_string();
          }
          // If the first vec needs do_norm, it means that the vec in the cid_vector table is not normalized.
          if (OB_FAIL(ret)) {
          } else if (is_first_vec) {
            if (OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                    vec_aux_ctdef_->dim_, reinterpret_cast<float *>(vec.ptr()), reinterpret_cast<float *>(vec.ptr()),
                    &cid_vec_need_norm))) {
              LOG_WARN("failed to normalize vector.", K(ret));
            } else {
              is_first_vec = false;
            }
          } else if (cid_vec_need_norm &&
                     OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                         vec_aux_ctdef_->dim_, reinterpret_cast<float *>(vec.ptr()), reinterpret_cast<float *>(vec.ptr())))) {
            LOG_WARN("failed to normalize vector.", K(ret));
          }
        }
        if (OB_SUCC(ret) && !skip && nullptr != lat_sq8_candidate && need_norm_) {
          if (is_first_vec) {
            if (OB_FAIL(ObVectorNormalize::L2_normalize_vector(vec_aux_ctdef_->dim_, lat_sq8_candidate, lat_sq8_candidate,
                    &cid_vec_need_norm))) {
              LOG_WARN("failed to normalize latent SQ8 cid vector (serial)", K(ret));
            } else {
              is_first_vec = false;
            }
          } else if (cid_vec_need_norm &&
                     OB_FAIL(ObVectorNormalize::L2_normalize_vector(
                         vec_aux_ctdef_->dim_, lat_sq8_candidate, lat_sq8_candidate))) {
            LOG_WARN("failed to normalize latent SQ8 cid vector (serial)", K(ret));
          }
        }
        if (OB_FAIL(ret)) {
          LOG_WARN("failed to get rowkey", K(ret));
        } else if (skip) {
        } else if (sq8_latent_handled) {
          // distance push done in fused IVF_SQ8 latent path (serial)
        } else if (nullptr != lat_sq8_candidate) {
          if constexpr (std::is_same_v<T, float>) {
            if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, lat_sq8_candidate, dim_))) {
              LOG_WARN("failed to push center (IVF_SQ8 latent-dequant)", K(ret));
            } else {
              adaptive_ctx_.vec_dist_calc_cnt_ ++;
            }
          }
        } else if (OB_NOT_NULL(vec.ptr())
                   && ivf_sq8_latent_row_may_push_raw_vec(latent_sq8_heap, sq8_latent_handled, lat_sq8_candidate)) {
          if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, reinterpret_cast<T *>(vec.ptr()), dim_))) {
            LOG_WARN("failed to push center.", K(ret));
          } else {
            adaptive_ctx_.vec_dist_calc_cnt_ ++;
          }
        }
        if (ivf_lat_.enabled_) {
          ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_serial_compute_beg;
        }
      }
    }

    if (ret == OB_ITER_END) {
      ret = OB_SUCCESS;
      if (OB_FAIL(reuse_cid_vec_iter_after_probe())) {
        LOG_WARN("fail to reuse scan iterator.", K(ret));
      }
    }
  }

  return ret;
}

template <typename T>
int ObDASIvfScanIter::get_nearest_limit_rowkeys_in_cids(
    bool is_vectorized,
    T *search_vec,
    ObSEArray<ObRowkey, 16> &saved_rowkeys,
    ObIvfPreFilter *prefilter)
{
  int ret = OB_SUCCESS;
  // only scan without filter can reach here
  ObExprVectorDistance::ObVecDisType cur_dis_type = dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN ? oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED : dis_type_;
  share::ObVectorCenterClusterHelper<T, ObRowkey> nearest_rowkey_heap(
      vec_op_alloc_, search_vec, cur_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
  if (OB_FAIL(get_nearest_limit_rowkeys_in_cids<T>(is_vectorized, search_vec, nearest_rowkey_heap, prefilter))) {
    LOG_WARN("calc_nearest_limit_rowkeys_in_cids fail", K(ret));
  } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys))) {
    LOG_WARN("failed to get top n", K(ret));
  }
  return ret;
}

template <typename T>
int ObDASIvfScanIter::get_nearest_limit_rowkeys_in_cids(
    bool is_vectorized,
    T *serch_vec,
    share::ObVectorCenterClusterHelper<T, ObRowkey> &nearest_rowkey_heap,
    ObIvfPreFilter *prefilter)
{
  int ret = OB_SUCCESS;
  const ObDASScanCtDef *cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CID_VEC_SCAN);
  ObDASScanRtDef *cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
  int64_t cid_vec_column_count = 0;
  int64_t cid_vec_pri_key_cnt = 0;
  int64_t rowkey_cnt = 0;
  int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
  char *buf = nullptr;
  ObString cid_str;

  if (OB_FAIL(prepare_cid_range(cid_vec_ctdef, cid_vec_column_count, cid_vec_pri_key_cnt, rowkey_cnt))) {
    LOG_WARN("fail to prepare cid range", K(ret));
  } else if (OB_ISNULL(buf = static_cast<char*>(mem_context_->get_arena_allocator().alloc(buf_len)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("failed to alloc cid", K(ret));
  } else {
    cid_str.assign_buffer(buf, buf_len);
  }
  // 3. Obtain nprobes * k rowkeys
  if (near_cid_.count() == 0) {
    // The situation of index creation with table creation
    ObString empty_cid;
    bool is_first_vec = true;
    bool cid_vec_need_norm = true;
    if (OB_FAIL(get_rowkeys_to_heap(empty_cid, cid_vec_pri_key_cnt, cid_vec_column_count, rowkey_cnt, is_vectorized,
                                    nearest_rowkey_heap, is_first_vec, cid_vec_need_norm, prefilter))) {
      LOG_WARN("failed to get rowkeys to heap, when near_cid is empty", K(ret));
    }
  } else {
    // for adaptor old version(< 4353), which cid_vec is not normlized in cosine dis
    bool is_first_vec = true;
    bool cid_vec_need_norm = true;
    for (int64_t k = 0; OB_SUCC(ret) && k < near_cid_.count(); ++k) {
      const int64_t i = rotated_probe_idx_(k);
      const ObCenterId &cur_cid = near_cid_.at(i);
      if (OB_FAIL(get_rowkeys_to_heap(cid_str, cid_vec_pri_key_cnt, cid_vec_column_count, rowkey_cnt,
                  is_vectorized, nearest_rowkey_heap, is_first_vec, cid_vec_need_norm, prefilter, &cur_cid))) {
        LOG_WARN("failed to get rowkeys to heap", K(ret), K(cur_cid));
      }
    }
  }
  return ret;
}

int ObDASIvfScanIter::process_ivf_scan_post(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(do_ivf_scan_post<float>(is_vectorized, reinterpret_cast<float *>(real_search_vec_.ptr())))) {
    LOG_WARN("failed to do post filter", K(ret), K(is_vectorized));
  }
  return ret;
}

template <typename T>
int ObDASIvfScanIter::do_ivf_scan_post(bool is_vectorized, T *search_vec)
{
  int ret = OB_SUCCESS;
  const int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  if (OB_FAIL(get_nearest_probe_center_ids(is_vectorized))) {
    // 1. Scan the centroid table, range: [min] ~ [max]; sort by l2_distance(c_vec, search_vec_); limit nprobes;
    // return the cid column.
    if (ret != OB_ENTRY_NOT_EXIST) {
      LOG_WARN("failed to get nearest probe center ids", K(ret));
    } else if (is_adaptive_filter()) {
      LOG_INFO("nearest probe center ids is empty", K(ret));
      ret = OB_SUCCESS;
      adaptive_ctx_.is_brute_force_= true;
      float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
      ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      bool index_end = false;
      const int64_t t_brute_post_adaptive_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
      while (OB_SUCC(ret) && !index_end) {
        if (OB_FAIL(get_pre_filter_rowkey_batch(mem_context_->get_arena_allocator(), is_vectorized, batch_row_count,
                                                index_end))) {
          if (strategy_ == ObVecIdxQueryStrategy::LATENCY_FIRST && ret == OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY) {
            LOG_INFO("pre-filter timeout and strategy is LATENCY_FIRST, need to response now", K(ret), K(strategy_), K(can_retry_), K(vec_index_type_), K(vec_idx_try_path_));
            ret = OB_SUCCESS;
          } else {
            LOG_WARN("failed to get rowkey batch", K(ret), K(is_vectorized));
          }
        } else if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
          LOG_WARN("failed to get limit rowkey brute", K(ret));
        } else {
          pre_fileter_rowkeys_.reset();
        }
      }
      if (OB_ITER_END == ret) {
        ret = OB_SUCCESS;
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
      if (ivf_lat_.enabled_) {
        ivf_lat_.brute_wall_us_ += ObTimeUtility::current_time() - t_brute_post_adaptive_beg;
      }
    } else {
      LOG_INFO("nearest probe center ids is empty", K(ret));
      ret = OB_SUCCESS;
      adaptive_ctx_.is_brute_force_= true;
      float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
      ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      const int64_t t_brute_post_simple_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
      if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
        LOG_WARN("failed to get limit rowkey brute", K(ret));
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
      if (ivf_lat_.enabled_) {
        ivf_lat_.brute_wall_us_ += ObTimeUtility::current_time() - t_brute_post_simple_beg;
      }
    }
  } else if (is_iter_filter()) {
    adaptive_ctx_.iter_times_ = 0;
    ObSEArray<ObIvfRowkeyDistEntry, 16> near_rowkeys;
    int64_t limit_k = limit_param_.limit_ + limit_param_.offset_;
    int64_t iter_cnt = 0;
    int64_t next_nprobe = 0;
    bool iter_end = false;
    bool no_new_near_rowkeys = false;
    int64_t heap_size = get_heap_size(limit_k, selectivity_);
    share::ObVectorCenterClusterHelper<T, ObRowkey> nearest_rowkey_heap(
        vec_op_alloc_, search_vec, dis_type_, dim_, heap_size, similarity_threshold_);
    ObIVFRowkeyDistMap rowkey_dist_map;
    ObIvfRowkeyDistItemCompare head_cmp(dis_type_);
    ObIvfRowkeyDistHeap rowkey_dist_heap(head_cmp);
    if (OB_FAIL(rowkey_dist_map.create(32, lib::ObMemAttr(MTL_ID(), "IVFMap") ))) {
      LOG_WARN("create rowkey dist map fail", K(ret));
    }
    while (OB_SUCC(ret) && ! iter_end && near_rowkeys.count() < limit_k) {
      ObIvfFineWallIterGuard fine_iter_guard(ivf_lat_);
      rowkey_dist_map.reuse();
      ++adaptive_ctx_.iter_times_;
      int32_t start_idx = -1;
      if (OB_FAIL(get_nearest_limit_rowkeys_in_cids<T>(is_vectorized, search_vec, nearest_rowkey_heap, nullptr))) {
        LOG_WARN("failed to get nearest limit rowkeys in cids", K(near_cid_));
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_centers_dist_map(rowkey_dist_map))) {
        LOG_WARN("get dist map fail", K(ret));
      } else if (OB_FALSE_IT(start_idx = near_rowkeys.count())) {
      } else if (OB_FAIL(do_post_filter(is_vectorized, rowkey_dist_map, near_rowkeys))) {
        LOG_WARN("do post filter fail", K(ret), K(near_rowkeys.count()));
      } else if (OB_FALSE_IT(no_new_near_rowkeys = (near_rowkeys.count() == start_idx))) {
        // there are no new rowkeys in near_rowkeys after post filter
      } else {
        for (int64_t i = start_idx; OB_SUCC(ret) && i < near_rowkeys.count(); ++i) {
          const ObIvfRowkeyDistEntry &entry = near_rowkeys.at(i);
          if (OB_FAIL(rowkey_dist_heap.push(ObIvfRowkeyDistItem(i, entry.distance_)))) {
            LOG_WARN("push rowkey dist fail", K(ret), K(i), K(entry));
          } else {
            LOG_TRACE("push", K(entry), K(i), K(limit_k));
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (near_rowkeys.count() < limit_k) {
        ++iter_cnt;
        const int64_t left_search = limit_k - near_rowkeys.count();
        next_nprobe = OB_MIN(nprobes_, (nprobes_ * ((double)left_search) / limit_k + 1));
        const double select_ratio = double(adaptive_ctx_.iter_res_row_cnt_) /  double(adaptive_ctx_.iter_filter_row_cnt_);
        const int64_t new_heap_size = OB_MAX(get_heap_size(left_search, select_ratio), heap_size);
        LOG_INFO("postfilter does not get enough result", K(limit_k), "near_rowkeys_count", near_rowkeys.count(),
            K(selectivity_), K(left_search), K(next_nprobe), K(iter_cnt), K(nprobes_), K(heap_size),
            K(new_heap_size), K(select_ratio), K(can_retry_), K(strategy_), K(adaptive_ctx_));
        near_cid_.reuse();
        if (similarity_threshold_ != 0 && no_new_near_rowkeys) {
          iter_end = true;
          LOG_INFO("there are no new rowkeys in near_rowkeys after post filter, stop iterative filter", K(ret), K(no_new_near_rowkeys), K(near_rowkeys.count()), K(start_idx));
        } else if (max_scan_vectors_ > 0 && adaptive_ctx_.cid_vec_scan_rows_ > max_scan_vectors_) {
          iter_end = true;
          LOG_INFO("reach max scan vectors, stop iterative filter", K(no_new_near_rowkeys), K(near_rowkeys.count()),
              K(start_idx), K(limit_k), K(left_search), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
        } else if (can_retry_ && OB_FAIL(check_iter_filter_need_retry())) {
          LOG_WARN("ret of check iter filter need retry.", K(ret), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
        } else if (! has_next_center()) {
          iter_end = true;
        } else if (OB_FAIL(get_next_center_ids(is_vectorized, next_nprobe))) {
          LOG_WARN("get next centers failed", K(ret), K(has_used_hgraph_), K(next_nprobe));
        } else if (near_cid_.count() == 0) {
          // Common check: if no centers found after getting next batch, end iteration
          iter_end = true;
          LOG_TRACE("there are no centers left to access", K(ret), K(iterative_filter_ctx_));
        } else if (OB_FAIL(nearest_rowkey_heap.set_nprobe(new_heap_size))) {
          LOG_WARN("set new heap size fail", K(ret), K(new_heap_size), K(heap_size));
        }
      }
    }
    if (OB_SUCC(ret)) {
      for(int64_t cnt = 0; OB_SUCC(ret) && cnt < limit_k && ! rowkey_dist_heap.empty(); ++cnt) {
        const ObIvfRowkeyDistItem &item = rowkey_dist_heap.top();
        if (item.rowkey_idx_ > near_rowkeys.count()) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("out of bound of near_rowkeys", K(ret), K(item), K(near_rowkeys.count()));
        } else if (OB_FAIL(saved_rowkeys_.push_back(near_rowkeys.at(item.rowkey_idx_).rowkey_))) {
          LOG_WARN("push back fail", K(ret), K(item));
        } else if (OB_FAIL(rowkey_dist_heap.pop())) {
          LOG_WARN("rowkey_dist_heap pop fail", K(ret), K(saved_rowkeys_.count()), K(item));
        } else {
          LOG_TRACE("result", K(item), "i", cnt, K(limit_k));
        }
      }
    }
  } else {
    const int64_t t_fine_post_simple_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
    if (OB_FAIL(get_nearest_limit_rowkeys_in_cids<T>(is_vectorized,
                                                                search_vec,
                                                                saved_rowkeys_, nullptr))) {
      // 2. get cidx in (top-nprobes 个 cid)
      //    scan cid_vector table, range: [cidx, min] ~ [cidx, max]; sort by l2_distance(vec, search_vec_); limit
      //    (limit+offset); return rowkey column
      LOG_WARN("failed to get nearest limit rowkeys in cids", K(near_cid_));
    }
    if (ivf_lat_.enabled_) {
      ivf_lat_.fine_wall_us_ += ObTimeUtility::current_time() - t_fine_post_simple_beg;
    }
  }
  return ret;
}

int ObDASIvfScanIter::process_ivf_scan_pre(ObIAllocator &allocator, bool is_vectorized)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(do_ivf_scan_pre<float>(allocator, is_vectorized, reinterpret_cast<float *>(real_search_vec_.ptr())))) {
    LOG_WARN("failed to get rowkey pre filter", K(ret), K(is_vectorized));
  }
  return ret;
}

template <typename T>
int ObDASIvfScanIter::do_ivf_scan_pre(ObIAllocator &allocator, bool is_vectorized, T *search_vec)
{
  int ret = OB_SUCCESS;
  ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
  int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  ObDASScanIter* inv_iter = (ObDASScanIter*)inv_idx_scan_iter_;
  bool is_range_prefilter = vec_aux_ctdef_->can_use_vec_pri_opt();
  const int64_t est_output_row_cnt = adaptive_ctx_.selectivity_ * adaptive_ctx_.row_count_;
  const bool need_check_brute = est_output_row_cnt <= IVF_MAX_BRUTE_FORCE_SIZE * 10 || ! is_range_prefilter;
  ObIvfPreFilter prefilter(MTL_ID());
  const bool track_flat_prep = ivf_lat_.enabled_
      && std::is_same_v<T, float>
      && vec_aux_ctdef_->algorithm_type_ == ObVectorIndexAlgorithmType::VIAT_IVF_FLAT;
  int64_t t_flat_prep_beg = 0;
  if (need_check_brute) {
    t_flat_prep_beg = track_flat_prep ? ObTimeUtility::current_time() : 0;
    if (OB_FAIL(get_rowkey_pre_filter(mem_context_->get_arena_allocator(), is_vectorized, IVF_MAX_BRUTE_FORCE_SIZE))) {
      LOG_WARN("failed to get rowkey pre filter", K(ret), K(is_vectorized));
    } else if (track_flat_prep && t_flat_prep_beg > 0) {
      ivf_lat_.flat_prep_us_ += ObTimeUtility::current_time() - t_flat_prep_beg;
      t_flat_prep_beg = 0;
    }
  }
  if (OB_SUCC(ret) && need_check_brute && pre_fileter_rowkeys_.count() < IVF_MAX_BRUTE_FORCE_SIZE) {
    const int64_t t_brute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
    // do brute search
    adaptive_ctx_.is_brute_force_= true;
    IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, reinterpret_cast<float *>(real_search_vec_.ptr()), raw_dis_type, dim_,
                                      get_nprobe(limit_param_, 1), similarity_threshold_);
    if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
      LOG_WARN("failed to get limit rowkey brute", K(ret));
    } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
      LOG_WARN("failed to get top n", K(ret));
    }
    if (ivf_lat_.enabled_) {
      ivf_lat_.brute_wall_us_ += ObTimeUtility::current_time() - t_brute_beg;
    }
  } else if (OB_SUCC(ret)) {
    if (OB_FAIL(get_nearest_probe_center_ids(is_vectorized))) {
      if (ret == OB_ENTRY_NOT_EXIST) {
        // cid_center table is empty, just do brute search
        ret = OB_SUCCESS;
        adaptive_ctx_.is_brute_force_= true;
        IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, reinterpret_cast<float *>(real_search_vec_.ptr()) /*unused*/, raw_dis_type, dim_,
                                          get_nprobe(limit_param_, 1), similarity_threshold_);
        bool index_end = false;
        const int64_t t_brute_empty_centroid_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
        while (OB_SUCC(ret) && !index_end) {
          if (OB_FAIL(get_pre_filter_rowkey_batch(mem_context_->get_arena_allocator(), is_vectorized, batch_row_count,
                                                  index_end))) {
            LOG_WARN("failed to get rowkey batch", K(ret), K(is_vectorized));
          } else if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
            LOG_WARN("failed to get limit rowkey brute", K(ret));
          } else {
            pre_fileter_rowkeys_.reset();
          }
        }
        if (OB_ITER_END == ret) {
          ret = OB_SUCCESS;
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
          LOG_WARN("failed to get top n", K(ret));
        }
        if (ivf_lat_.enabled_) {
          ivf_lat_.brute_wall_us_ += ObTimeUtility::current_time() - t_brute_empty_centroid_beg;
        }
      } else {
        LOG_WARN("failed to get nearest probe center ids", K(ret));
      }
    } else {
      t_flat_prep_beg = track_flat_prep ? ObTimeUtility::current_time() : 0;
      if (is_range_prefilter) { // rowkey range prefilter
        adaptive_ctx_.pre_scan_row_cnt_ += pre_fileter_rowkeys_.count();
        adaptive_ctx_.is_range_prefilter_ = true;
        ObArray<const ObNewRange *> rk_range;
        const ObRangeArray& key_range = inv_iter->get_scan_param().key_ranges_;
        for (int64_t i = 0; i < key_range.count() && OB_SUCC(ret); i++) {
          const ObNewRange *range = &key_range.at(i);
          if (OB_FAIL(rk_range.push_back(range))) {
            LOG_WARN("fail to push back range", K(ret), K(i));
          }
        }
        if (OB_SUCC(ret)) {
          if (OB_FAIL(prefilter.init(rk_range))) {
            LOG_WARN("fail to init prefilter as range filter", K(ret));
          }
        }
      } else if (OB_FAIL(prefilter.init())) { // rowkey hash bitmap prefilter
        LOG_WARN("fail to init prefilter as roaring bitmap", K(ret));
      } else { // add bitmap for bitmap filter
        adaptive_ctx_.pre_scan_row_cnt_ += pre_fileter_rowkeys_.count();
        for (int i = 0; i < pre_fileter_rowkeys_.count() && OB_SUCC(ret); i++) {
          uint64_t hash_val = hash_val_for_rk(pre_fileter_rowkeys_.at(i));
          if (OB_FAIL(prefilter.add(hash_val))) {
            LOG_WARN("fail to add hash val to prefilter", K(ret));
          }
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(build_rowkey_hash_set(prefilter, is_vectorized, batch_row_count))) {
          LOG_WARN("fail to build rk hash set", K(ret));
        }
      }
      if (track_flat_prep && t_flat_prep_beg > 0) {
        ivf_lat_.flat_prep_us_ += ObTimeUtility::current_time() - t_flat_prep_beg;
      }

      // do for loop until saved_rowkeys_.count() >= limitK
      if (OB_SUCC(ret)) {
        int64_t limit_k = limit_param_.limit_ + limit_param_.offset_;
        int64_t iter_cnt = 0;
        int64_t next_nprobe = 0;
        bool iter_end = false;
        bool is_first_scan = true;
        int32_t start_idx = -1;
        bool no_new_near_rowkeys = false;
        share::ObVectorCenterClusterHelper<T, ObRowkey> nearest_rowkey_heap(
            vec_op_alloc_, search_vec, dis_type_, dim_, limit_k, similarity_threshold_);
        while (OB_SUCC(ret) && ! iter_end && nearest_rowkey_heap.count() < limit_k) {
          ObIvfFineWallIterGuard fine_iter_guard(ivf_lat_);
          if (OB_FALSE_IT(start_idx = nearest_rowkey_heap.count())) {
          } else if (OB_FAIL(get_nearest_limit_rowkeys_in_cids<T>(
              is_vectorized,
              search_vec,
              nearest_rowkey_heap,
              &prefilter))) {
            LOG_WARN("fail to calc nearest limit rowkeys in cids", K(ret), K(dim_));
          } else if (OB_FALSE_IT(no_new_near_rowkeys = (nearest_rowkey_heap.count() == start_idx))) {
            // there are no new rowkeys in near_rowkeys after post filter
          }
          if (OB_FAIL(ret)) {
          } else if (nearest_rowkey_heap.count() < limit_k) {
            ++iter_cnt;
            const int64_t left_search = limit_k - nearest_rowkey_heap.count();
            next_nprobe = OB_MIN(nprobes_, (nprobes_ * ((double)left_search) / limit_k + 1));
            LOG_INFO("prefilter does not get enough result", K(limit_k), "nearest_rowkey_heap_count", nearest_rowkey_heap.count(),
                K(selectivity_), K(left_search), K(next_nprobe), K(iter_cnt), K(nprobes_), K(can_retry_), K(strategy_), K(adaptive_ctx_));
            near_cid_.reuse();
            is_first_scan = false;
            if (similarity_threshold_ != 0 && no_new_near_rowkeys) {
              iter_end = true;
              LOG_INFO("there are no new rowkeys in near_rowkeys after post filter, stop iterative filter", K(ret), K(no_new_near_rowkeys), K(nearest_rowkey_heap.count()), K(start_idx), K(limit_k));
            } else if (max_scan_vectors_ > 0 && adaptive_ctx_.cid_vec_scan_rows_ > max_scan_vectors_) {
              iter_end = true;
              LOG_INFO("reach max scan vectors, stop iterative filter", K(no_new_near_rowkeys), K(nearest_rowkey_heap.count()),
                K(start_idx), K(limit_k), K(left_search), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
            } else if (! has_next_center()) {
              iter_end = true;
            } else if (OB_FAIL(get_next_center_ids(is_vectorized, next_nprobe))) {
              LOG_WARN("get next centers failed", K(ret), K(has_used_hgraph_), K(next_nprobe));
            } else if (near_cid_.count() == 0) {
              // Common check: if no centers found after getting next batch, end iteration
              iter_end = true;
              LOG_TRACE("there are no centers left to access", K(ret), K(iterative_filter_ctx_));
            }
          }
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
          LOG_WARN("get rowkeys from heap fail", K(ret), K(limit_k));
        } else {
          LOG_TRACE("get rowkeys from heap success", K(limit_k), K(saved_rowkeys_.count()), K(iter_cnt), K(next_nprobe), K(nprobes_));
        }
      }
    }
  }

  return ret;
}

int ObDASIvfScanIter::check_cid_exist(const ObString &src_cid, bool &src_cid_exist)
{
  int ret = OB_SUCCESS;
  src_cid_exist = false;
  ObCenterId src_centor_id;
  if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(src_centor_id, src_cid))) {
    LOG_WARN("failed to get center id from string", K(src_cid));
  } else if (OB_UNLIKELY(src_centor_id.center_id_ >= near_cid_dist_.count())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("src_cid is not exist", K(ret), K(src_centor_id), K(near_cid_dist_.count()));
  } else if (near_cid_dist_.at(src_centor_id.center_id_)) {
    src_cid_exist = true;
  }

  return ret;
}

void ObDASIvfScanIter::reuse_cid_ctx()
{
  ObDASIvfBaseScanIter::reuse_cid_ctx();
  near_cid_.reuse();
}
/************************************************** ObDASIvfPQScanIter ******************************************************/

int ObDASIvfPQScanIter::inner_init(ObDASIterParam &param)
{
  int ret = OB_SUCCESS;
  ObVectorIndexParam index_param;
  uint64_t tenant_cluster_version = GET_MIN_CLUSTER_VERSION();
  if (OB_FAIL(ObDASIvfBaseScanIter::inner_init(param))) {
    LOG_WARN("fail to do inner init ", K(ret), K(param));
  } else if (!((tenant_cluster_version >= MOCK_CLUSTER_VERSION_4_3_5_3 &&
                tenant_cluster_version < CLUSTER_VERSION_4_4_0_0) ||
               tenant_cluster_version >= CLUSTER_VERSION_4_4_1_0)) {
    if (OB_FAIL(ObVectorIndexUtil::parser_params_from_string(vec_aux_ctdef_->vec_index_param_,
                                                             ObVectorIndexType::VIT_IVF_INDEX, index_param))) {
      LOG_WARN("fail to parse params from string", K(ret), K(vec_aux_ctdef_->vec_index_param_));
    }
  } else if (OB_FAIL(index_param.assign(vec_aux_ctdef_->get_vec_index_param()))) {
    LOG_WARN("fail to assign params from vec_aux_ctdef_", K(ret));
  }

  if (OB_FAIL(ret)) {
  } else {
    ObDASIvfScanIterParam &ivf_scan_param = static_cast<ObDASIvfScanIterParam &>(param);
    pq_centroid_iter_ = ivf_scan_param.pq_centroid_iter_;
    m_ = vec_index_param_.m_;
    nbits_ = vec_index_param_.nbits_;
    if (dim_ % m_) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("dim should be able to divide m exactly", K(ret), K(dim_), K(m_));
    } else if (nbits_ < 1 || nbits_ > 24) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("nbits should be in range [1,24]", K(ret), K(nbits_));
    }
  }

  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(near_cid_vec_.reserve(nprobes_))) {
    LOG_WARN("failed to reserve nearest cid vec", K(ret), K(vec_index_param_.nlist_));
  } else if (OB_FAIL(near_cid_vec_dis_.reserve(nprobes_))) {
    LOG_WARN("failed to reserve nearest cid vec dis", K(ret), K(vec_index_param_.nlist_));
  } else if (OB_FAIL(near_cid_vec_ptrs_.allocate_array(persist_alloc_, vec_index_param_.nlist_ + 1))) {
    LOG_WARN("failed to reserve nearest cid vec", K(ret), K(vec_index_param_.nlist_));
  } else {
    memset(near_cid_vec_ptrs_.get_data(), 0, near_cid_vec_ptrs_.count() * sizeof(float *));
  }
  return ret;
}

int ObDASIvfPQScanIter::inner_reuse()
{
  int ret = OB_SUCCESS;
  near_cid_vec_.reuse();
  near_cid_vec_dis_.reuse();
  memset(near_cid_vec_ptrs_.get_data(), 0, near_cid_vec_ptrs_.count() * sizeof(float *));

  if (!pq_centroid_first_scan_ && OB_FAIL(ObDasVecScanUtils::reuse_iter(
                                      ls_id_, pq_centroid_iter_, pq_centroid_scan_param_, pq_centroid_tablet_id_))) {
    LOG_WARN("failed to reuse iter", K(ret));
  } else if (OB_FAIL(ObDASIvfBaseScanIter::inner_reuse())) {
    LOG_WARN("fail to do inner reuse", K(ret));
  }
  return ret;
}

int ObDASIvfPQScanIter::inner_release()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(pq_centroid_iter_) && OB_FAIL(pq_centroid_iter_->release())) {
    LOG_WARN("failed to release pq_centroid_iter_", K(ret));
  }
  pq_centroid_iter_ = nullptr;
  ObDasVecScanUtils::release_scan_param(pq_centroid_scan_param_);
  near_cid_vec_.reset();
  near_cid_vec_dis_.reset();
  near_cid_vec_ptrs_.reset();
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObDASIvfBaseScanIter::inner_release())) {
    LOG_WARN("fail to do inner release", K(ret));
  }

  return ret;
}

// for pq, con_key is array(vector(float))
int ObDASIvfPQScanIter::parse_pq_ids_vec_datum(
  ObIAllocator &allocator,
  int64_t cid_vec_column_count,
  const ObDASScanCtDef *cid_vec_ctdef,
  const int64_t rowkey_cnt,
  ObRowkey &main_rowkey,
  ObString &com_key)
{
  int ret = OB_SUCCESS;
  ObExpr *pq_ids_expr = cid_vec_ctdef->result_output_[PQ_IDS_IDX];

  if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(allocator, cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
    LOG_WARN("failed to get main rowkey from cid vec datum", K(ret));
  } else if (pq_ids_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_).is_null()) {
    // do nothing
    com_key.reset();
  } else {
    com_key = pq_ids_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_).get_string();
  }
  return ret;
}

int ObDASIvfPQScanIter::calc_distance_between_pq_ids_by_cache(ObIvfCentCache &cent_cache,
                                                              const ObString &pq_center_ids,
                                                              const ObIArray<float *> &splited_residual,
                                                              float &distance) {
  int ret = OB_SUCCESS;

  RWLock::RLockGuard guard(cent_cache.get_lock());
  float square = 0.0f;
  const uint8_t* pq_ids_ptr = ObVecIVFPQCenterIDS::get_pq_id_ptr(pq_center_ids.ptr());
  PQDecoderGeneric decoder(pq_ids_ptr, nbits_);
  for (int j = 0; OB_SUCC(ret) && j < m_; ++j) {
    // 3.2.1 pq_center_ids[j] is put into ivf_pq_centroid table to find pq_center_vecs[j]
    float *pq_cid_vec = nullptr;
    if (OB_FAIL(cent_cache.read_pq_centroid(j + 1 /*m*/, decoder.decode() + 1, pq_cid_vec))) {
      LOG_WARN("fail to get pq cid vec from cache", K(ret), K(j + 1));
    } else if (OB_ISNULL(pq_cid_vec)) {
      ret = OB_ERR_NULL_VALUE;
      LOG_WARN("invalid null pq_cid_vec",
        K(ret), K(j), KPHEX(pq_center_ids.ptr(), pq_center_ids.length()));
    } else {
      // 3.3.2 Calculate the distance between pq_center_vecs[j] and r(x)[j].
      //       The sum of j = 0 ~ m is the distance from x to rowkey
      float dis = 0.0f;
      if (OB_FAIL(calc_vec_dis<float>(splited_residual.at(j), pq_cid_vec, dim_ / m_, dis, dis_type_))) {
        SHARE_LOG(WARN, "failed to calc vec dis", K(ret));
      } else {
        square += dis;
      }
    }
  } // end for

  if (OB_SUCC(ret)) {
    distance = square;
  }
  return ret;
}

int ObDASIvfPQScanIter::calc_distance_between_pq_ids_by_table(
  bool is_vectorized,
  const ObString &pq_center_ids,
  const ObIArray<float *> &splited_residual,
  int64_t batch_row_count,
  float &distance)
{
  // todo(wmj): need fit ivf adaptor cache，pq_center_ids find pq_center_vec
  int ret = OB_SUCCESS;
  float dis_square = 0.0f;
  const ObDASScanCtDef *pq_cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_pq_id_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_SPECIAL_AUX_SCAN);
  ObDASScanRtDef *pq_cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_pq_id_tbl_idx());
  if (OB_ISNULL(pq_cid_vec_ctdef) || OB_ISNULL(pq_cid_vec_rtdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(pq_cid_vec_ctdef), KP(pq_cid_vec_rtdef));
  }

  int count = 0;
  int pq_cid_idx = 0;
  int64_t pq_cid_count = m_;
  uint64_t tablet_id = ObVecIVFPQCenterIDS::get_tablet_id(pq_center_ids.ptr());
  const uint8_t* pq_ids_ptr = ObVecIVFPQCenterIDS::get_pq_id_ptr(pq_center_ids.ptr());
  PQDecoderGeneric decoder(pq_ids_ptr, nbits_);
  while (OB_SUCC(ret) && count < pq_cid_count) {
    for (int64_t i = 0; OB_SUCC(ret) && i < batch_row_count && count < pq_cid_count; ++i, ++count) {
      ObRowkey pq_cid_rowkey;
      ObString pq_center_id_str;
      ObPqCenterId pq_center_id(tablet_id, count + 1, decoder.decode() + 1);
      if (OB_FAIL(ObVectorClusterHelper::set_pq_center_id_to_string(pq_center_id, pq_center_id_str,
                                                                    &mem_context_->get_arena_allocator()))) {
        LOG_WARN("fail to set pq center id to string", K(ret), K(pq_center_id));
      } else if (OB_FAIL(build_cid_vec_query_rowkey(pq_center_id_str, true /*is_min*/, CENTROID_PRI_KEY_CNT, // first rowkey is pq_cid
                                             pq_cid_rowkey))) {
        LOG_WARN("failed to build cid vec query rowkey", K(ret));
      } else if (OB_FAIL(ObDasVecScanUtils::set_lookup_key(pq_cid_rowkey, pq_centroid_scan_param_,
                                                           pq_cid_vec_ctdef->ref_table_id_))) {
        LOG_WARN("failed to set lookup key", K(ret));
      }
    }

    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(do_aux_table_scan(pq_centroid_first_scan_, pq_centroid_scan_param_, pq_cid_vec_ctdef,
                                         pq_cid_vec_rtdef, pq_centroid_iter_, pq_centroid_tablet_id_))) {
      LOG_WARN("fail to rescan cid vec table scan iterator.", K(ret));
    } else if (is_vectorized) {
      IVF_GET_NEXT_ROWS_BEGIN(pq_centroid_iter_)
      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        bool has_lob_header = pq_cid_vec_ctdef->result_output_.at(PQ_CENTROID_VEC_IDX)->obj_meta_.has_lob_header();
        ObExpr *vec_expr = pq_cid_vec_ctdef->result_output_[PQ_CENTROID_VEC_IDX];
        ObDatum *vec_datum = vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);

        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          float cur_dis = 0.0f;
          guard.set_batch_idx(i);
          ObString vec = vec_datum[i].get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                          &mem_context_->get_arena_allocator(),
                          ObLongTextType,
                          CS_TYPE_BINARY,
                          has_lob_header,
                          vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_ISNULL(vec.ptr())) {
            // ignoring null vector.
            pq_cid_idx++;
          } else if (OB_FAIL(calc_vec_dis<float>(splited_residual.at(pq_cid_idx), reinterpret_cast<float *>(vec.ptr()),
                                                 dim_ / m_, cur_dis, dis_type_))) {
            SHARE_LOG(WARN, "failed to calc vec dis", K(ret));
          } else {
            dis_square += cur_dis;
            pq_cid_idx++;
          }
        }
      }
      IVF_GET_NEXT_ROWS_END(pq_centroid_iter_, pq_centroid_scan_param_, pq_centroid_tablet_id_)
    } else {
      pq_centroid_iter_->clear_evaluated_flag();
      bool has_lob_header = pq_cid_vec_ctdef->result_output_.at(PQ_CENTROID_VEC_IDX)->obj_meta_.has_lob_header();
      ObExpr *vec_expr = pq_cid_vec_ctdef->result_output_[PQ_CENTROID_VEC_IDX];
      ObDatum &vec_datum = vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      for (int i = 0; OB_SUCC(ret) && i < batch_row_count; ++i) {
        float cur_dis = 0.0f;
        if (OB_FAIL(pq_centroid_iter_->get_next_row())) {
          if (OB_ITER_END != ret) {
            LOG_WARN("failed to scan vid rowkey iter", K(ret));
          }
        } else {
          ObString vec = vec_datum.get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                          &mem_context_->get_arena_allocator(),
                          ObLongTextType,
                          CS_TYPE_BINARY,
                          has_lob_header,
                          vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_ISNULL(vec.ptr())) {
            // ignoring null vector.
            pq_cid_idx++;
          } else if (OB_FAIL(calc_vec_dis<float>(splited_residual.at(pq_cid_idx), reinterpret_cast<float *>(vec.ptr()),
                                                 dim_ / m_, cur_dis, dis_type_))) {
            SHARE_LOG(WARN, "failed to calc vec dis", K(ret));
          } else {
            dis_square += cur_dis;
            pq_cid_idx++;
          }
        }
      }
      int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, pq_centroid_iter_, pq_centroid_scan_param_, pq_centroid_tablet_id_))) {
        LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }
  }
  if (pq_cid_idx != pq_cid_count) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected pq cid idx", K(ret), K(pq_cid_idx), K(pq_cid_count));
  } else {
    distance = dis_square;
  }
  return ret;
}

int ObDASIvfPQScanIter::calc_distance_between_pq_ids(
    bool is_vectorized,
    const ObString &pq_center_ids,
    const ObIArray<float *> &splited_residual,
    float &distance)
{
  int ret = OB_SUCCESS;
  ObIvfCacheMgrGuard cache_guard;
  ObIvfCentCache *cent_cache = nullptr;
  bool is_cache_usable = false;
  if (OB_FAIL(get_centers_cache(is_vectorized, true/*is_pq_centers*/, cache_guard, cent_cache, is_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret), K(is_vectorized), KPC(cent_cache));
  } else if (is_cache_usable) {
    if (OB_FAIL(calc_distance_between_pq_ids_by_cache(*cent_cache, pq_center_ids, splited_residual, distance))) {
      LOG_WARN("fail to calc distance between pq ids by cache", K(ret), K(is_vectorized), KPC(cent_cache));
    }
  } else {
    if (OB_FAIL(calc_distance_between_pq_ids_by_table(is_vectorized,
                                                      pq_center_ids,
                                                      splited_residual,
                                                      ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE,
                                                      distance))) {
      LOG_WARN("fail to calc distance between pq ids by table", K(ret), K(is_vectorized));
    }
  }

  return ret;
}

int ObDASIvfPQScanIter::calc_distance_with_precompute(
    ObEvalCtx::BatchInfoScopeGuard &guard,
    int64_t scan_row_cnt,
    int64_t rowkey_cnt,
    ObRowkey& filter_main_rowkey,
    float *sim_table,
    float dis0,
    IvfRowkeyHeap& nearest_rowkey_heap,
    ObIvfPreFilter *prefilter)
{
  int ret = OB_SUCCESS;
  const ObDASScanCtDef *cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CID_VEC_SCAN);
  ObExpr *cid_expr = cid_vec_ctdef->result_output_[PQ_IDS_IDX];
  ObDatum *cid_datum = cid_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
  int counter = 0;
  size_t saved_j[4] = {0, 0, 0, 0};
  for (int64_t j = 0; OB_SUCC(ret) && j < scan_row_cnt; ++j) {
    bool is_skip = false;
    if (cid_datum[j].is_null()) {
      is_skip = true;
    } else if (prefilter != nullptr) {
      // get rowkey and filter first
      guard.set_batch_idx(j);
      if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, filter_main_rowkey, false))) {
        LOG_WARN("fail to get main rowkey", K(ret));
      } else if (!prefilter->test(filter_main_rowkey)) {
        is_skip = true;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (is_skip) {
    } else {
      adaptive_ctx_.vec_dist_calc_cnt_++;
      saved_j[0] = (counter == 0) ? j : saved_j[0];
      saved_j[1] = (counter == 1) ? j : saved_j[1];
      saved_j[2] = (counter == 2) ? j : saved_j[2];
      saved_j[3] = (counter == 3) ? j : saved_j[3];
      counter += 1;
      if (counter == 4) {
        float distance_0 = 0;
        float distance_1 = 0;
        float distance_2 = 0;
        float distance_3 = 0;
        {
          const uint8_t* pq_id_ptr_0 = ObVecIVFPQCenterIDS::get_pq_id_ptr(cid_datum[saved_j[0]].get_string().ptr());
          const uint8_t* pq_id_ptr_1 = ObVecIVFPQCenterIDS::get_pq_id_ptr(cid_datum[saved_j[1]].get_string().ptr());
          const uint8_t* pq_id_ptr_2 = ObVecIVFPQCenterIDS::get_pq_id_ptr(cid_datum[saved_j[2]].get_string().ptr());
          const uint8_t* pq_id_ptr_3 = ObVecIVFPQCenterIDS::get_pq_id_ptr(cid_datum[saved_j[3]].get_string().ptr());
          ObVectorL2Distance<float>::distance_four_codes(
            m_, nbits_, sim_table, pq_id_ptr_0, pq_id_ptr_1, pq_id_ptr_2, pq_id_ptr_3,
            distance_0, distance_1, distance_2, distance_3);
          distance_0 = distance_0 + dis0;
          distance_1 = distance_1 + dis0;
          distance_2 = distance_2 + dis0;
          distance_3 = distance_3 + dis0;
          // 先检查是否需要分配内存，避免不必要的内存分配
          if (nearest_rowkey_heap.should_push_center(distance_0)) {
            ObRowkey main_rowkey;
            {
              guard.set_batch_idx(saved_j[0]);
              {
                if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
                  LOG_WARN("fail to get main rowkey", K(ret));
                } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance_0))) {
                  LOG_WARN("failed to push center.", K(ret));
                }
              }
            }
          }
          if (OB_SUCC(ret) && nearest_rowkey_heap.should_push_center(distance_1)) {
            ObRowkey main_rowkey;
            {
              guard.set_batch_idx(saved_j[1]);
              {
                if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
                  LOG_WARN("fail to get main rowkey", K(ret));
                } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance_1))) {
                  LOG_WARN("failed to push center.", K(ret));
                }
              }
            }
          }
          if (OB_SUCC(ret) && nearest_rowkey_heap.should_push_center(distance_2)) {
            ObRowkey main_rowkey;
            {
              guard.set_batch_idx(saved_j[2]);
              {
                if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
                  LOG_WARN("fail to get main rowkey", K(ret));
                } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance_2))) {
                  LOG_WARN("failed to push center.", K(ret));
                }
              }
            }
          }
          if (OB_SUCC(ret) && nearest_rowkey_heap.should_push_center(distance_3)) {
            ObRowkey main_rowkey;
            {
              guard.set_batch_idx(saved_j[3]);
              {
                if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
                  LOG_WARN("fail to get main rowkey", K(ret));
                } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance_3))) {
                  LOG_WARN("failed to push center.", K(ret));
                }
              }
            }
          }
        }
        counter = 0;
      }
    }
  }
  // process counter left
  for (size_t kk = 0; kk < counter && OB_SUCC(ret); kk++) {
    const uint8_t* pq_id_ptr = ObVecIVFPQCenterIDS::get_pq_id_ptr(cid_datum[saved_j[kk]].get_string().ptr());
    float dis = dis0 + ObVectorL2Distance<float>::distance_one_code(m_, nbits_, sim_table, pq_id_ptr);
    if (nearest_rowkey_heap.should_push_center(dis)) {
      guard.set_batch_idx(saved_j[kk]);
      ObRowkey main_rowkey;
      {
        if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
          LOG_WARN("fail to get main rowkey", K(ret));
        } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, dis))) {
          LOG_WARN("failed to push center.", K(ret));
        }
      }
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::check_can_pre_compute(
    bool is_vectorized,
    ObIvfCacheMgrGuard &pre_cache_guard,
    ObIvfCentCache *&pre_cent_cache,
    bool &pre_compute_table)
{
  int ret = OB_SUCCESS;
  int64_t ksub = 1L << nbits_;
  pre_compute_table = false;
  if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN) {
    if (OB_FAIL(get_pq_precomputetable_cache(is_vectorized, pre_cache_guard, pre_cent_cache, pre_compute_table))) {
      LOG_WARN("fail to get pq precompute table cache", K(ret), K(is_vectorized), KPC(pre_cent_cache));
    }
  } else if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT) {
    // check pq center cache is ok
    ObIvfCacheMgrGuard pq_center_guard;
    ObIvfCentCache *pq_cent_cache = nullptr;
    bool is_pq_cent_cache_usable = false;
    if (OB_FAIL(get_centers_cache(is_vectorized, true/*is_pq_centers*/, pq_center_guard, pq_cent_cache, is_pq_cent_cache_usable))) {
      LOG_WARN("fail to get centers cache", K(ret), K(is_vectorized), KPC(pq_cent_cache));
    } else if (is_pq_cent_cache_usable) {
      pre_compute_table = true;
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::calc_nearest_limit_rowkeys_in_cids(
    bool is_vectorized,
    float *search_vec,
    ObSEArray<ObRowkey, 16> &saved_rowkeys,
    ObIvfPreFilter *prefilter)
{
  int ret = OB_SUCCESS;
  int64_t sub_dim = dim_ / m_;
  // only scan without filter can reach here
  ObExprVectorDistance::ObVecDisType cur_dis_type = dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN ? oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED : dis_type_;
  IvfRowkeyHeap nearest_rowkey_heap(
      vec_op_alloc_, search_vec, cur_dis_type, sub_dim, get_nprobe(limit_param_, 1), similarity_threshold_);
  if (OB_FAIL(calc_nearest_limit_rowkeys_in_cids(is_vectorized, search_vec, nearest_rowkey_heap, prefilter))) {
    LOG_WARN("calc_nearest_limit_rowkeys_in_cids fail", K(ret));
  } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys))) {
    LOG_WARN("failed to get top n", K(ret));
  }
  return ret;
}

int ObDASIvfPQScanIter::calc_nearest_limit_rowkeys_in_cids(
    bool is_vectorized,
    float *search_vec,
    IvfRowkeyHeap &nearest_rowkey_heap,
    ObIvfPreFilter *prefilter)
{
  int ret = OB_SUCCESS;

  int64_t sub_dim = dim_ / m_;
  int64_t ksub = 1L << nbits_;
  const ObDASScanCtDef *cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CID_VEC_SCAN);
  ObDASScanRtDef *cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_cid_vec_tbl_idx());
  int64_t cid_vec_column_count = 0;
  int64_t cid_vec_pri_key_cnt = 0;
  int64_t rowkey_cnt = 0;
  ObArray<float *> splited_residual;
  int64_t buf_len = OB_DOC_ID_COLUMN_BYTE_LENGTH;
  char *buf = nullptr;
  ObString cid_str;
  float *residual = nullptr;
  bool pre_compute_table = false;
  ObIvfCacheMgrGuard pre_cache_guard;
  ObIvfCentCache *pre_cent_cache = nullptr;
  float *sim_table = nullptr;
  float *sim_table_2 = nullptr;
  const float* sim_table_ptrs = nullptr;
  ObRowkey filter_main_rowkey;
  bool is_l2 = (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN);
  if (OB_FAIL(prepare_cid_range(cid_vec_ctdef, cid_vec_column_count, cid_vec_pri_key_cnt, rowkey_cnt))) {
    LOG_WARN("fail to prepare cid range", K(ret));
  } else if (OB_ISNULL(buf = static_cast<char*>(mem_context_->get_arena_allocator().alloc(buf_len)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("failed to alloc cid", K(ret));
  } else if (OB_FALSE_IT(cid_str.assign_buffer(buf, buf_len))) {
  } else {
    // continue with PQ prep below
  }
  const int64_t t_pq_prep_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(check_can_pre_compute(is_vectorized, pre_cache_guard, pre_cent_cache, pre_compute_table))) {
    LOG_WARN("fail to check can use precomputetable", K(ret));
  } else if (!pre_compute_table) {
    char *residual_buf = nullptr;
    if (OB_FAIL(splited_residual.reserve(m_))) {
      LOG_WARN("fail to init splited residual array", K(ret), K(m_));
    } else if (is_l2) { // only L2 need residual
      if (OB_ISNULL(residual_buf = static_cast<char*>(mem_context_->get_arena_allocator().alloc(dim_ * sizeof(float))))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("failed to alloc residual buf", K(ret));
      } else if (OB_FALSE_IT(residual = new(residual_buf) float[dim_])) {
      }
    }
  } else { // scan by precompute table cache
    ObObj *obj_ptr = nullptr;
    sim_table = (float*)mem_context_->get_arena_allocator().alloc(sizeof(float) * (ksub * m_) * 2);
    obj_ptr = (ObObj*)mem_context_->get_arena_allocator().alloc(sizeof(ObObj) * rowkey_cnt);
    if (OB_ISNULL(sim_table) || OB_ISNULL(obj_ptr)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to alloc memory for precompute table", K(ret), K(ksub), K(m_), K(rowkey_cnt),
               "sim_table_size", sizeof(float) * (ksub * m_) * 2,
               "obj_ptr_size", sizeof(ObObj) * rowkey_cnt,
               KP(sim_table), KP(obj_ptr));
    } else {
      filter_main_rowkey.assign(obj_ptr, rowkey_cnt);
      sim_table_2 = sim_table + (ksub * m_);
      if (is_l2 && OB_FAIL(pre_compute_inner_prod_table(search_vec, sim_table_2, is_vectorized))) {
        LOG_WARN("fail to pre compute inner prod table", K(ret));
      } else if (!is_l2 && OB_FAIL(pre_compute_inner_prod_table(search_vec, sim_table, is_vectorized))) {
        LOG_WARN("fail to pre compute inner prod table", K(ret));
      }
    }
  }
  if (ivf_lat_.enabled_) {
    ivf_lat_.pq_prep_us_ += ObTimeUtility::current_time() - t_pq_prep_beg;
  }
  const bool cid_probe_debug = ob_ivf_cid_probe_debug_enabled();
  const int64_t cid_probe_query_scan0 = cid_probe_debug ? adaptive_ctx_.cid_vec_scan_rows_ : 0;
  const int64_t cid_probe_query_dist0 = cid_probe_debug ? adaptive_ctx_.vec_dist_calc_cnt_ : 0;
  if (cid_probe_debug) {
    char begin_line[512];
    const int64_t range_rowkey_cnt = calc_cid_vec_storage_range_rowkey_cnt(*cid_vec_ctdef);
    const int nl = snprintf(
        begin_line,
        sizeof(begin_line),
        "[OB_IVF_CID_PROBE_DEBUG] phase=begin near_cid_cnt=%lld nprobes=%lld dim=%lld "
        "cid_vec_pri_key_cnt=%lld range_rowkey_cnt=%lld is_vectorized=%d cid_vec_tablet_id=%ld\n",
        static_cast<long long>(near_cid_vec_.count()),
        static_cast<long long>(nprobes_),
        static_cast<long long>(dim_),
        static_cast<long long>(cid_vec_pri_key_cnt),
        static_cast<long long>(range_rowkey_cnt),
        static_cast<int>(is_vectorized),
        cid_vec_tablet_id_.id());
    if (nl > 0 && nl < static_cast<int>(sizeof(begin_line))) {
      ob_ivf_cid_probe_debug_log_line(begin_line);
    }
  }
  // 1. for every (cid, cid_vec),
  for (int64_t k = 0; OB_SUCC(ret) && k < near_cid_vec_.count(); ++k) {
    const int64_t i = rotated_probe_idx_(k);
    const ObCenterId &cur_cid = near_cid_vec_.at(i).first;
    float *cur_cid_vec = near_cid_vec_.at(i).second;
    const int64_t probe_scan_before = cid_probe_debug ? adaptive_ctx_.cid_vec_scan_rows_ : 0;
    const int64_t probe_dist_before = cid_probe_debug ? adaptive_ctx_.vec_dist_calc_cnt_ : 0;
    int64_t probe_batches = 0;
    char cid_range_log_buf[OB_DOC_ID_COLUMN_BYTE_LENGTH];
    ObString cid_range_for_log;
    bool cid_str_passed_empty = true;
    if (cid_probe_debug) {
      cid_str_passed_empty = cid_str.empty();
      ObString tmp;
      tmp.assign_buffer(cid_range_log_buf, sizeof(cid_range_log_buf));
      if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(cur_cid, tmp))) {
        LOG_WARN("failed to set center_id to string for probe debug", K(ret), K(cur_cid));
      } else {
        cid_range_for_log = tmp;
      }
    }
    float dis0 = 0.0f;
    if (pre_compute_table) {
      if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN) {
        sim_table_ptrs = pre_cent_cache->get_centroids() + (cur_cid.center_id_ - 1) * ksub * m_;
        ObVectorL2Distance<float>::fvec_madd(m_ * ksub,
                                             sim_table_ptrs,
                                             -2.0,
                                             sim_table_2,
                                             sim_table);
      }
      dis0 = near_cid_vec_dis_.at(i);
    } else {
      // 1.1 Calculate the residual r(x) = x - cid_vec
      //     split r(x) into m parts, the jth part is called r(x)[j]
      splited_residual.reuse();
      if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN) {
        if (OB_FAIL(ObVectorIndexUtil::calc_residual_vector(dim_, search_vec, cur_cid_vec, residual))) {
          LOG_WARN("fail to calc residual vector", K(ret), K(dim_));
        }
      } else {
        dis0 = near_cid_vec_dis_.at(i);
        residual = search_vec; // ip dis = dis0 + search_vec · pq_vec
      }
      if (OB_FAIL(ret)) {
        LOG_WARN("fail to calc residual vector", K(ret), K(dim_));
      } else if (OB_FAIL(ObVectorIndexUtil::split_vector(m_, dim_, residual, splited_residual))) {
        LOG_WARN("fail to split vector", K(ret));
      }
    }
    if (OB_FAIL(ret)) {
    } else {
      // 1.2 cid put the query in the ivf_pq_code table to find (rowkey, pq_center_ids)
      storage::ObTableScanIterator *cid_vec_scan_iter = nullptr;
      bool replay_only = false;
      if (OB_FAIL(try_cid_vec_replay_only_switch(cur_cid.center_id_, cid_vec_scan_iter, replay_only))) {
        LOG_WARN("fail to replay-only switch cid vec cache iter", K(ret), K(cur_cid));
      } else if (!replay_only) {
        if (OB_FALSE_IT(cid_str.assign_buffer(buf, buf_len))) {
        } else if (OB_FAIL(ObVectorClusterHelper::set_center_id_to_string(cur_cid, cid_str))) {
          LOG_WARN("failed to set center_id to string", K(ret), K(cur_cid));
        } else if (OB_FAIL(scan_cid_range(cid_str, cid_vec_pri_key_cnt, cid_vec_ctdef, cid_vec_rtdef, cid_vec_scan_iter, &cur_cid))) {
          LOG_WARN("fail to scan cid range", K(ret), K(cur_cid), K(cid_vec_pri_key_cnt));
        }
      }
      if (OB_FAIL(ret)) {
      } else if (is_vectorized) {
        int64_t cid_vec_batch_count = get_cid_vec_batch_count();
        bool index_end = false;
        cid_vec_iter_->clear_evaluated_flag();
        int64_t scan_row_cnt = 0;
        const int64_t batch_row_count = cid_vec_batch_count;
        int64_t cv_sf_batch_idx = 0;
        while (!index_end && OB_SUCC(ret)) {
          const int64_t t_storage_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          if (OB_FAIL(cid_vec_iter_->get_next_rows(scan_row_cnt, batch_row_count))) {
            if (OB_ITER_END != ret) {
              LOG_WARN("failed to get next row.", K(ret));
            } else {
              index_end = true;
            }
          }
          if (ivf_lat_.enabled_) {
            ivf_lat_.fine_cv_storage_fetch_us_ += ObTimeUtility::current_time() - t_storage_beg;
          }
          if (ivf_lat_.enabled_ || cid_probe_debug) {
            ++cv_sf_batch_idx;
          }
          if (OB_FAIL(ret) && OB_ITER_END != ret) {
          } else if (scan_row_cnt > 0) {
            ret = OB_SUCCESS;
            ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
            guard.set_batch_size(scan_row_cnt);
            ObExpr *cid_expr = cid_vec_ctdef->result_output_[PQ_IDS_IDX];
            ObDatum *cid_datum = cid_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
            adaptive_ctx_.cid_vec_scan_rows_ += scan_row_cnt;

            if (pre_compute_table) {
              const int64_t t_precompute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
              if (OB_FAIL(calc_distance_with_precompute(guard, scan_row_cnt, rowkey_cnt, filter_main_rowkey,
                                                          sim_table, dis0, nearest_rowkey_heap, prefilter))) {
                LOG_WARN("fail to calc distance with pre compute table", K(ret));
              }
              if (ivf_lat_.enabled_) {
                ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_precompute_beg;
              }
            } else {
              for (int64_t j = 0; OB_SUCC(ret) && j < scan_row_cnt; ++j) {
                const int64_t t_vec_row_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
                guard.set_batch_idx(j);
                if (cid_datum[j].is_null() || cid_datum[j].get_string().empty()) {
                  // do nothing
                } else {
                  ObRowkey main_rowkey;
                  ObString pq_center_ids = cid_datum[j].get_string();
                  float distance = 0;
                  if (OB_FAIL(get_main_rowkey_from_cid_vec_datum(mem_context_->get_arena_allocator(), cid_vec_ctdef, rowkey_cnt, main_rowkey))) {
                    LOG_WARN("fail to get main rowkey", K(ret));
                  } else if (prefilter != nullptr && !prefilter->test(main_rowkey)) {
                    // has been filter, do nothing
                  } else {
                    if (ivf_lat_.enabled_) {
                      ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_vec_row_beg;
                    }
                    const int64_t t_vec_compute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
                    if (OB_FAIL(calc_distance_between_pq_ids(is_vectorized, pq_center_ids, splited_residual, distance))) {
                      LOG_WARN("fail to calc distance between pq ids", K(ret));
                    } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance + dis0))) {
                      LOG_WARN("failed to push center.", K(ret));
                    } else {
                      adaptive_ctx_.vec_dist_calc_cnt_++;
                    }
                    if (ivf_lat_.enabled_) {
                      ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_vec_compute_beg;
                    }
                  }
                }
              }
            }
          }
        }
        if (index_end) {
          if (cid_probe_debug) {
            probe_batches = cv_sf_batch_idx;
          }
          int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
          if (OB_FAIL(reuse_cid_vec_iter_after_probe())) {
            LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
          } else {
            ret = tmp_ret;
          }
        }
      } else {
        while (OB_SUCC(ret)) {
          const int64_t t_serial_row_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
          ObRowkey main_rowkey;
          ObString vec_arr_str;
          ObString pq_center_ids;
          float distance = 0.0f;
          adaptive_ctx_.cid_vec_scan_rows_++;
          // cid_vec_iter_ output: [IVF_CID_VEC_CID_COL IVF_CID_VEC_VECTOR_COL ROWKEY]
          if (OB_FAIL(cid_vec_iter_->get_next_row())) {
            if (ivf_lat_.enabled_) {
              ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
            }
            if (OB_ITER_END != ret) {
              LOG_WARN("failed to scan vid rowkey iter", K(ret));
            }
          } else {
            if (cid_probe_debug) {
              ++probe_batches;
            }
            if (OB_FAIL(parse_pq_ids_vec_datum(
              mem_context_->get_arena_allocator(),
              cid_vec_column_count,
              cid_vec_ctdef,
              rowkey_cnt,
              main_rowkey,
              pq_center_ids))) {
            if (ivf_lat_.enabled_) {
              ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
            }
            LOG_WARN("fail to parse cid vec datum", K(ret), K(cid_vec_column_count), K(rowkey_cnt));
          } else if (pq_center_ids.empty()) {
            // ignore null arr
          } else if (prefilter != nullptr && !prefilter->test(main_rowkey)) {
            // has been filter, do nothing
          } else {
            if (ivf_lat_.enabled_) {
              ivf_lat_.fine_load_us_ += ObTimeUtility::current_time() - t_serial_row_beg;
            }
            const int64_t t_serial_compute_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
            if (pre_compute_table) {
              const uint8_t* pq_id_ptr = ObVecIVFPQCenterIDS::get_pq_id_ptr(pq_center_ids.ptr());
              distance = ObVectorL2Distance<float>::distance_one_code(m_, nbits_, sim_table, pq_id_ptr);
            } else if (OB_FAIL(calc_distance_between_pq_ids(is_vectorized, pq_center_ids, splited_residual, distance))) {
              LOG_WARN("fail to calc distance between pq ids", K(ret));
            }
            if (OB_FAIL(ret)) {
            } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance + dis0))) {
              LOG_WARN("failed to push center.", K(ret));
            } else {
              adaptive_ctx_.vec_dist_calc_cnt_++;
            }
            if (ivf_lat_.enabled_) {
              ivf_lat_.fine_compute_us_ += ObTimeUtility::current_time() - t_serial_compute_beg;
            }
          }
          }
        } // end while

        if (ret == OB_ITER_END) {
          ret = OB_SUCCESS;
          if (OB_FAIL(reuse_cid_vec_iter_after_probe())) {
            LOG_WARN("fail to reuse scan iterator.", K(ret));
          }
        }
      }
      if (cid_probe_debug) {
        const int64_t probe_scan_rows = adaptive_ctx_.cid_vec_scan_rows_ - probe_scan_before;
        const int64_t probe_dist_cnt = adaptive_ctx_.vec_dist_calc_cnt_ - probe_dist_before;
        char range_hex[OB_DOC_ID_COLUMN_BYTE_LENGTH * 2 + 1];
        range_hex[0] = '\0';
        if (!cid_range_for_log.empty()) {
          int hex_pos = 0;
          for (int64_t bi = 0; bi < cid_range_for_log.length()
               && hex_pos + 2 < static_cast<int>(sizeof(range_hex)); ++bi) {
            hex_pos += snprintf(range_hex + hex_pos,
                sizeof(range_hex) - static_cast<size_t>(hex_pos),
                "%02X",
                static_cast<unsigned char>(cid_range_for_log.ptr()[bi]));
          }
        }
        char probe_line[768];
        const int pl = snprintf(
            probe_line,
            sizeof(probe_line),
            "[OB_IVF_CID_PROBE_DEBUG] phase=probe k=%lld i=%lld near_cnt=%lld "
            "tablet_id=%llu center_id=%llu cid_str_passed_empty=%d range_len=%lld range_hex=%s "
            "probe_scan_rows=%lld probe_dist_cnt=%lld probe_batches=%lld "
            "cum_scan_rows=%lld cum_dist_cnt=%lld\n",
            static_cast<long long>(k),
            static_cast<long long>(i),
            static_cast<long long>(near_cid_vec_.count()),
            static_cast<unsigned long long>(cur_cid.tablet_id_),
            static_cast<unsigned long long>(cur_cid.center_id_),
            static_cast<int>(cid_str_passed_empty),
            static_cast<long long>(cid_range_for_log.length()),
            range_hex,
            static_cast<long long>(probe_scan_rows),
            static_cast<long long>(probe_dist_cnt),
            static_cast<long long>(probe_batches),
            static_cast<long long>(adaptive_ctx_.cid_vec_scan_rows_),
            static_cast<long long>(adaptive_ctx_.vec_dist_calc_cnt_));
        if (pl > 0 && pl < static_cast<int>(sizeof(probe_line))) {
          ob_ivf_cid_probe_debug_log_line(probe_line);
        }
      }
    }
  } // end for i
  if (cid_probe_debug) {
    char end_line[512];
    const int64_t query_scan_rows = adaptive_ctx_.cid_vec_scan_rows_ - cid_probe_query_scan0;
    const int64_t query_dist_cnt = adaptive_ctx_.vec_dist_calc_cnt_ - cid_probe_query_dist0;
    const int el = snprintf(
        end_line,
        sizeof(end_line),
        "[OB_IVF_CID_PROBE_DEBUG] phase=end near_cid_cnt=%lld query_scan_rows=%lld "
        "query_dist_cnt=%lld cid_vec_scan_rows=%lld vec_dist_calc_cnt=%lld\n",
        static_cast<long long>(near_cid_vec_.count()),
        static_cast<long long>(query_scan_rows),
        static_cast<long long>(query_dist_cnt),
        static_cast<long long>(adaptive_ctx_.cid_vec_scan_rows_),
        static_cast<long long>(adaptive_ctx_.vec_dist_calc_cnt_));
    if (el > 0 && el < static_cast<int>(sizeof(end_line))) {
      ob_ivf_cid_probe_debug_log_line(end_line);
    }
  }
  return ret;
}

// HGraph-based iterative filtering implementations
int ObDASIvfPQScanIter::get_next_probe_centers_by_hgraph(bool is_vectorized, int64_t next_nprobe)
{
  int ret = OB_SUCCESS;
  ObIvfCacheMgrGuard cache_guard;
  ObIvfCentCache *cent_cache = nullptr;
  bool is_cache_usable = false;

  if (OB_FAIL(get_centers_cache(is_vectorized, false /*is_pq_centers*/, cache_guard, cent_cache, is_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret));
  } else if (is_cache_usable && OB_NOT_NULL(cent_cache) && cent_cache->has_hgraph_index()) {
    if (OB_FAIL(get_nearest_probe_centers_with_hgraph(
        is_vectorized, cent_cache, true, true, next_nprobe))) {
      LOG_WARN("HGraph PQ iterative search failed", K(ret), K(next_nprobe));
    }
  } else {
    // If HGraph was used before but now unavailable, this is an error
    ret = OB_ERR_UNEXPECTED;
    LOG_ERROR("HGraph was used in initial scan but now unavailable for iterative filtering", K(ret),
        K(is_cache_usable), K(OB_NOT_NULL(cent_cache)), K(cent_cache ? cent_cache->has_hgraph_index() : false));
  }

  return ret;
}

int ObDASIvfPQScanIter::get_next_centers(bool is_vectorized, int64_t next_nprobe)
{
  int ret = OB_SUCCESS;
  if (has_used_hgraph_) {
    if (OB_FAIL(get_next_probe_centers_by_hgraph(is_vectorized, next_nprobe))) {
      LOG_WARN("failed to get next probe centers by hgraph", K(ret), K(is_vectorized), K(next_nprobe));
    }
  } else {
    if (OB_FAIL(iterative_filter_ctx_.get_next_nearest_probe_centers_vec_dist(next_nprobe, near_cid_vec_, near_cid_vec_dis_))) {
      LOG_WARN("failed to get next nearest probe centers vec dist", K(ret), K(next_nprobe));
    }
  }
  if (OB_SUCC(ret) && near_cid_vec_.count() > 0) {
    assign_probe_rotate_offset_(near_cid_vec_.count());
  }
  return ret;
}

int ObDASIvfPQScanIter::get_nearest_probe_centers_with_hgraph(
    bool is_vectorized,
    ObIvfCentCache *hgraph_cache,
    bool need_vectors,
    bool need_distances,
    int64_t incremental_nprobes)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(hgraph_cache)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("hgraph cache is null", K(ret));
  } else {
    common::obvsag::VectorIndexPtr hgraph_index = hgraph_cache->get_hgraph_index();
    if (OB_ISNULL(hgraph_index)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("hgraph index is null", K(ret));
    } else {
      // use OceanBase's VSAG interface to search
      const float* distances = nullptr;
      const int64_t* result_ids = nullptr;
      int64_t result_size = 0;
      float* search_vec = reinterpret_cast<float*>(const_cast<char*>(real_search_vec_.ptr()));
      const char* extra_info = nullptr;
      int64_t centers_total = hgraph_cache->get_count();
      int64_t ef_search = 64;
      // Check if need to initialize HGraph state (incremental_nprobes == -1 means initial search)
      bool need_init = (incremental_nprobes == -1);
      int64_t search_count = need_init ? nprobes_ : incremental_nprobes;
      // Create VSAG allocator if not exists (same lifecycle as hgraph_iter_ctx_)
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(init_hgraph_search_alloc())) {
        LOG_WARN("failed to init hgraph search alloc", K(ret));
      } else if (OB_FAIL(obvectorutil::knn_search(hgraph_index,
                                            search_vec,
                                            dim_,
                                            search_count,
                                            distances,
                                            result_ids,
                                            extra_info,
                                            result_size,
                                            ef_search,
                                            nullptr,
                                            false,
                                            false,
                                            1.0f,
                                            hgraph_vsag_alloc_,      // Use persistent allocator with same lifecycle as iter_ctx
                                            false,
                                            hgraph_iter_ctx_,        // Always pass iter context, let HGraph manage it
                                            false))) {               // is_last_search, keep searching
        LOG_WARN("failed to search with hgraph index", K(ret), K(search_count), K(ef_search));
      }

      if (OB_FAIL(ret)) {
      } else if (OB_ISNULL(result_ids)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("hgraph search result is null", K(ret));
      } else if (result_size == 0) {
        ret = OB_ENTRY_NOT_EXIST;
        LOG_WARN("hgraph search result is empty", K(ret));
      } else {
        near_cid_vec_.reset();
        near_cid_vec_dis_.reset();
        int64_t new_center_cnt = 0;
        for (int64_t i = 0; i < result_size && OB_SUCC(ret); ++i) {
          int64_t center_idx = result_ids[i];
          if (center_idx >= 1 && center_idx <= centers_total) {
            ObCenterId center_id;
            center_id.center_id_ = center_idx;
            center_id.tablet_id_ = centroid_tablet_id_.id();
            if (need_vectors) {
              float* center_vec = nullptr;
              if (OB_FAIL(hgraph_cache->read_centroid(center_idx, center_vec, true /*deep_copy*/, &persist_alloc_))) {
                LOG_WARN("failed to read centroid for hgraph center vector", K(ret), K(center_idx));
              } else if (OB_ISNULL(center_vec)) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("read centroid returned null pointer", K(ret), K(center_idx));
              } else {
                IvfCidVecPair cid_vec_pair(center_id, center_vec);
                if (OB_FAIL(near_cid_vec_.push_back(cid_vec_pair))) {
                  LOG_WARN("failed to push cid vec pair", K(ret));
                }
              }
            }
            // save distance
            if (OB_SUCC(ret) && need_distances && OB_NOT_NULL(distances)) {
              float final_distance = distances[i];
              if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT && need_norm_) {
                final_distance = 1.0f - distances[i];
              } else if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::DOT) {
                final_distance = 1.0f - distances[i];
              }
              if (OB_FAIL(near_cid_vec_dis_.push_back(final_distance))) {
                LOG_WARN("failed to push distance", K(ret));
              }
            }

            if (OB_SUCC(ret)) {
              new_center_cnt++;
            }
          } else {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("invalid center index from hgraph", K(ret), K(center_idx), K(centers_total));
          }
        }

        LOG_DEBUG("HGraph IVF PQ search completed", K(result_size), K(new_center_cnt),
                 K(incremental_nprobes), K(search_count));

        // Mark that HGraph has reached the end if returned fewer centers than requested
        if (!need_init && new_center_cnt < search_count) {
          hgraph_has_next_center_ = false;
          LOG_INFO("HGraph reached end of centers", K(new_center_cnt), K(search_count));
        }
      }
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::get_nearest_probe_centers(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  ObIvfCoarseWallGuard coarse_guard(ivf_lat_);
  //precompute table use euclidean_squared, so we need to convert to euclidean
  //L2_squared = dis0^2 + precomcute.result
  ObExprVectorDistance::ObVecDisType cur_dis_type = dis_type_;
  if (dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN) {
    cur_dis_type = oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED;
  }
  // decide which search strategy to use based on cache type
  ObIvfCacheMgrGuard cache_guard;
  ObIvfCentCache *cent_cache = nullptr;
  bool is_cache_usable = false;
  if (OB_FAIL(get_centers_cache(is_vectorized, false /*is_pq_centers*/, cache_guard, cent_cache, is_cache_usable))) {
    LOG_WARN("failed to get centers cache", K(ret));
  } else if (is_cache_usable && OB_NOT_NULL(cent_cache) && cent_cache->has_hgraph_index()) {
    has_used_hgraph_ = true;  // Record that HGraph was used in initial scan
    if (OB_FAIL(get_nearest_probe_centers_with_hgraph(is_vectorized, cent_cache, true/*need_vectors*/, true/*need_distances*/))) {
      LOG_WARN("HGraph search failed for PQ scan", K(ret));
    }
  } else {
    share::ObVectorCenterClusterHelper<float, ObCenterId> nearest_cid_heap(
      mem_context_->get_arena_allocator(), reinterpret_cast<const float *>(real_search_vec_.ptr()),
      cur_dis_type, dim_, nprobes_, 0.0, (is_pre_filter() || is_iter_filter()), &iterative_filter_ctx_.allocator_);
    if (OB_FAIL(generate_nearest_cid_heap(is_vectorized, nearest_cid_heap, true/*save_center_vec*/, cent_cache, is_cache_usable))) {
      LOG_WARN("failed to generate nearest cid heap", K(ret), K(nprobes_), K(dim_), K(real_search_vec_));
    } else {
      if (nearest_cid_heap.get_center_count() == 0) {
        ret = OB_ENTRY_NOT_EXIST;
      } else if (nearest_cid_heap.is_save_all_center()) {
        if (OB_FAIL(nearest_cid_heap.get_all_centroids(iterative_filter_ctx_.centroids_))) {
          LOG_WARN("init_centroids fail", K(ret));
        } else if (OB_FAIL(iterative_filter_ctx_.get_next_nearest_probe_centers_vec_dist(nprobes_, near_cid_vec_, near_cid_vec_dis_))) {
          LOG_WARN("failed to get top n", K(ret), K(iterative_filter_ctx_));
        }
      } else { // whatever pre or post, go here
        if (OB_FAIL(nearest_cid_heap.get_nearest_probe_centers_vec_dist(near_cid_vec_, near_cid_vec_dis_))) {
          LOG_WARN("failed to get top n", K(ret));
        }
      }
    }
  }
  if (OB_SUCC(ret) && near_cid_vec_.count() > 0) {
    assign_probe_rotate_offset_(near_cid_vec_.count());
  }
  return ret;
}

int ObDASIvfBaseScanIter::get_rowkey_brute_post(bool is_vectorized, IvfRowkeyHeap& nearest_rowkey_heap)
{
  int ret = OB_SUCCESS;

  float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
  ObString raw_search_vec;
  const ObDASScanCtDef *brute_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_brute_tbl_idx(), ObTSCIRScanType::OB_VEC_COM_AUX_SCAN);
  ObDASScanRtDef *brute_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_brute_tbl_idx());
  int64_t main_rowkey_cnt = 0;
  // prepare norm info
  ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
  ObVectorNormalizeInfo norm_info;
  if (OB_ISNULL(brute_ctdef) || OB_ISNULL(brute_rtdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(brute_ctdef), KP(brute_rtdef));
  } else if (OB_FALSE_IT(main_rowkey_cnt = brute_ctdef->table_param_.get_read_info().get_schema_rowkey_count())) {
  } else if (OB_UNLIKELY(main_rowkey_cnt <= 0)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid rowkey cnt", K(ret));
  } else if (need_norm_) {
    if (OB_FAIL(ObDasVecScanUtils::get_real_search_vec(mem_context_->get_arena_allocator(), sort_rtdef_->eval_ctx_, search_vec_,
                                                       raw_search_vec))) {
      LOG_WARN("failed to get real search vec", K(ret));
    } else if (OB_FALSE_IT(search_vec = reinterpret_cast<float *>(raw_search_vec.ptr()))) {
    }
  }
  int count = 0;
  int batch_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  int search_count = (pre_fileter_rowkeys_.count() > 0) ? pre_fileter_rowkeys_.count() : 1;
  while (OB_SUCC(ret) && count < search_count) {
    if (pre_fileter_rowkeys_.count() > 0) {
      // setup lookup key range
      brute_scan_param_.key_ranges_.reset(); // clean key range array
      for (int i = 0; OB_SUCC(ret) && i < batch_count && count < search_count; i++, count++) {
        ObNewRange rk_range;
        if (OB_FAIL(rk_range.build_range(brute_ctdef->ref_table_id_, pre_fileter_rowkeys_.at(count)))) {
          LOG_WARN("fail to build key range", K(ret));
        } else if (OB_FAIL(ObDasVecScanUtils::set_lookup_range(rk_range, brute_scan_param_, brute_ctdef->ref_table_id_))) {
          LOG_WARN("failed to append scan range", K(ret));
        }
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(do_aux_table_scan(brute_first_scan_,
                                          brute_scan_param_,
                                          brute_ctdef,
                                          brute_rtdef,
                                          brute_iter_,
                                          brute_tablet_id_))) {
        LOG_WARN("fail to rescan brute table scan iterator.", K(ret));
      }
    } else {
      count = search_count;
      if (OB_FAIL(do_table_full_scan(is_vectorized,
                                     brute_ctdef,
                                     brute_rtdef,
                                     brute_iter_,
                                     brute_tablet_id_,
                                     brute_first_scan_,
                                     brute_scan_param_))) {
        LOG_WARN("failed to do centroid table scan", K(ret));
      }
    }
    if (OB_FAIL(ret)) {
    } else if (is_vectorized) {
      IVF_GET_NEXT_ROWS_BEGIN(brute_iter_)
      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        ObExpr *vec_expr = brute_ctdef->result_output_[DATA_VECTOR_IDX];
        ObDatum *vec_datum = vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);

        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          guard.set_batch_idx(i);
          ObString c_vec = vec_datum[i].get_string();
          if (vec_datum[i].is_null()) {
            // do nothing for null vector
          } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                             &mem_context_->get_arena_allocator(),
                             ObLongTextType,
                             CS_TYPE_BINARY,
                             vec_expr->obj_meta_.has_lob_header(),
                             c_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else { // dis for l2
            float distance = 0.0f;
            ObRowkey main_rowkey;
            if (OB_FAIL(calc_vec_dis<float>(reinterpret_cast<float *>(c_vec.ptr()), search_vec, dim_, distance, raw_dis_type))) {
              if (OB_ERR_NULL_VALUE == ret) {
                ret = OB_SUCCESS;
                LOG_TRACE("skip zero-norm vector, cosine distance undefined");
              } else {
                LOG_WARN("fail to calc vec dis", K(ret), K(dim_));
              }
            } else if (OB_FAIL(get_main_rowkey_brute(mem_context_->get_arena_allocator(), brute_ctdef, main_rowkey_cnt, main_rowkey))) {
              LOG_WARN("fail to get main rowkey", K(ret));
            } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance))) {
              LOG_WARN("failed to push center.", K(ret));
            }
          }
        }
      }
      IVF_GET_NEXT_ROWS_END(brute_iter_, brute_scan_param_, brute_tablet_id_)
    } else {
      brute_iter_->clear_evaluated_flag();
      ObExpr *vec_expr = brute_ctdef->result_output_[DATA_VECTOR_IDX];
      ObDatum &vec_datum = vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      while (OB_SUCC(ret)) {
        if (OB_FAIL(brute_iter_->get_next_row())) {
          if (OB_ITER_END != ret) {
            LOG_WARN("get next row failed.", K(ret));
          }
        } else if (vec_datum.is_null()) {
          // do nothing
        } else {
          ObString c_vec = vec_datum.get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                      &mem_context_->get_arena_allocator(),
                      ObLongTextType,
                      CS_TYPE_BINARY,
                      vec_expr->obj_meta_.has_lob_header(),
                      c_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else {
            float distance = 0.0f;
            ObRowkey main_rowkey;
            if (OB_FAIL(calc_vec_dis<float>(reinterpret_cast<float *>(c_vec.ptr()), search_vec, dim_, distance, raw_dis_type))) {
              if (OB_ERR_NULL_VALUE == ret) {
                ret = OB_SUCCESS;
                LOG_TRACE("skip zero-norm vector, cosine distance undefined");
              } else {
                LOG_WARN("fail to calc vec dis", K(ret), K(dim_));
              }
            } else if (OB_FAIL(get_main_rowkey_brute(mem_context_->get_arena_allocator(), brute_ctdef, main_rowkey_cnt, main_rowkey))) {
              LOG_WARN("fail to get main rowkey", K(ret));
            } else if (OB_FAIL(nearest_rowkey_heap.push_center(main_rowkey, distance))) {
              LOG_WARN("failed to push center.", K(ret));
            }
          }
        }
      } // end while
      int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, brute_iter_, brute_scan_param_, brute_tablet_id_))) {
        LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }
  }
  return ret;
}

int ObDASIvfBaseScanIter::get_main_rowkey_brute(
  ObIAllocator &allocator,
  const ObDASScanCtDef *brute_ctdef,
  const int64_t rowkey_cnt,
  ObRowkey &main_rowkey)
{
  int ret = OB_SUCCESS;
  ObObj *obj_ptr = nullptr;
  void *buf = nullptr;

  if (OB_ISNULL(brute_ctdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(brute_ctdef));
  } else {
    // brute_iter output: [DATA_VECTOR ROWKEY]
    // Note: when _enable_defensive_check = 2, cid_vec_out_exprs is [DATA_VECTOR ROWKEY DEFENSE_CHECK_COL]
    const ExprFixedArray& brute_out_exprs = brute_ctdef->result_output_;
    if (rowkey_cnt > brute_out_exprs.count() - 1) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("rowkey_cnt is illegal", K(ret), K(rowkey_cnt), K(brute_out_exprs.count()));
    } else if (OB_ISNULL(buf = allocator.alloc(sizeof(ObObj) * rowkey_cnt))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("allocate memory failed", K(ret), K(rowkey_cnt));
    } else if (OB_FALSE_IT(obj_ptr = new (buf) ObObj[rowkey_cnt])) {
    } else {
      int rowkey_idx = 0;
      for (int64_t i = 1; OB_SUCC(ret) && i < brute_out_exprs.count() && rowkey_idx < rowkey_cnt; ++i) {
        ObObj tmp_obj;
        ObExpr *expr = brute_out_exprs.at(i);
        if (OB_ISNULL(expr)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("should not be null", K(ret));
        } else if (OB_FAIL(expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_).to_obj(tmp_obj, expr->obj_meta_, expr->obj_datum_map_))) {
          LOG_WARN("convert datum to obj failed", K(ret));
        } else if (OB_FALSE_IT(obj_ptr[rowkey_idx++] = tmp_obj)) {
        }
      }
    }
    if (OB_SUCC(ret)) {
      main_rowkey.assign(obj_ptr, rowkey_cnt);
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::process_ivf_scan_post(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  const int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  // 1. Scan the ivf_centroid table, calculate the distance between vec_x and cid_vec,
  //    and get the nearest cluster center (cid 1, cid_vec 1)... (cid n, cid_vec n)
  if (OB_FAIL(get_nearest_probe_centers(is_vectorized))) {
    if (ret != OB_ENTRY_NOT_EXIST) {
      LOG_WARN("failed to get nearest probe center ids", K(ret));
    } else if (is_adaptive_filter()) {
      LOG_INFO("nearest probe center ids is empty", K(ret));
      ret = OB_SUCCESS;
      adaptive_ctx_.is_brute_force_= true;
      float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
      ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      bool index_end = false;
      while (OB_SUCC(ret) && !index_end) {
        if (OB_FAIL(get_pre_filter_rowkey_batch(mem_context_->get_arena_allocator(), is_vectorized, batch_row_count,
                                                index_end))) {
          if (strategy_ == ObVecIdxQueryStrategy::LATENCY_FIRST && ret == OB_VECTOR_INDEX_ADAPTIVE_NEED_RETRY) {
            LOG_INFO("pre-filter timeout and strategy is LATENCY_FIRST, need to response now", K(ret), K(strategy_), K(can_retry_), K(vec_index_type_), K(vec_idx_try_path_));
            ret = OB_SUCCESS;
          } else {
            LOG_WARN("failed to get rowkey batch", K(ret), K(is_vectorized));
          }
        } else if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
          LOG_WARN("failed to get limit rowkey brute", K(ret));
        } else {
          pre_fileter_rowkeys_.reset();
        }
      }
      if (OB_ITER_END == ret) {
        ret = OB_SUCCESS;
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
    } else {
      LOG_INFO("nearest probe center ids is empty", K(ret));
      ret = OB_SUCCESS;
      adaptive_ctx_.is_brute_force_= true;
      float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
      ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
        LOG_WARN("failed to get limit rowkey brute", K(ret));
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
    }
  } else if (is_iter_filter()) {
    adaptive_ctx_.iter_times_ = 0;
    ObSEArray<ObIvfRowkeyDistEntry, 16> near_rowkeys;
    int64_t limit_k = limit_param_.limit_ + limit_param_.offset_;
    int64_t iter_cnt = 0;
    int64_t next_nprobe = 0;
    bool iter_end = false;
    bool no_new_near_rowkeys = false;
    float* search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
    const int64_t sub_dim = dim_ / m_;
    int64_t enlargement_factor = (selectivity_ != 0 && selectivity_ != 1 && is_iter_filter()) ? POST_ENLARGEMENT_FACTOR : 1;
    int64_t heap_size = get_heap_size(limit_k, selectivity_);
    ObExprVectorDistance::ObVecDisType cur_dis_type = dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN ? oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED : dis_type_;
    IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, cur_dis_type, sub_dim, heap_size, similarity_threshold_);
    ObIVFRowkeyDistMap rowkey_dist_map;
    ObIvfRowkeyDistItemCompare head_cmp(cur_dis_type);
    ObIvfRowkeyDistHeap rowkey_dist_heap(head_cmp);
    if (OB_FAIL(rowkey_dist_map.create(32, lib::ObMemAttr(MTL_ID(), "IVFMap") ))) {
      LOG_WARN("create rowkey dist map fail", K(ret));
    }
    while (OB_SUCC(ret) && ! iter_end && near_rowkeys.count() < limit_k) {
      ObIvfFineWallIterGuard fine_iter_guard(ivf_lat_);
      rowkey_dist_map.reuse();
      ++adaptive_ctx_.iter_times_;
      int32_t start_idx = -1;
      if (OB_FAIL(calc_nearest_limit_rowkeys_in_cids(
          is_vectorized,
          search_vec,
          nearest_rowkey_heap,
          nullptr))) {
        LOG_WARN("fail to calc nearest limit rowkeys in cids", K(ret), K(dim_));
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_centers_dist_map(rowkey_dist_map))) {
        LOG_WARN("get dist map fail", K(ret));
      } else if (OB_FALSE_IT(start_idx = near_rowkeys.count())) {
      } else if (OB_FAIL(do_post_filter(is_vectorized, rowkey_dist_map, near_rowkeys))) {
        LOG_WARN("do post filter fail", K(ret), K(near_rowkeys.count()));
      } else if (OB_FALSE_IT(no_new_near_rowkeys = (near_rowkeys.count() == start_idx))) {
        // there are no new rowkeys in near_rowkeys after post filter
      } else {
        for (int64_t i = start_idx; OB_SUCC(ret) && i < near_rowkeys.count(); ++i) {
          const ObIvfRowkeyDistEntry &entry = near_rowkeys.at(i);
          if (OB_FAIL(rowkey_dist_heap.push(ObIvfRowkeyDistItem(i, entry.distance_)))) {
            LOG_WARN("push rowkey dist fail", K(ret), K(i), K(entry));
          } else {
            LOG_TRACE("push", K(entry), K(i), K(limit_k));
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (near_rowkeys.count() < limit_k) {
        ++iter_cnt;
        const int64_t left_search = limit_k - near_rowkeys.count();
        next_nprobe = OB_MIN(nprobes_, (nprobes_ * ((double)left_search) / limit_k + 1));
        const double select_ratio = double(adaptive_ctx_.iter_res_row_cnt_) /  double(adaptive_ctx_.iter_filter_row_cnt_);
        const int64_t new_heap_size = OB_MAX(get_heap_size(left_search, select_ratio), heap_size);
        LOG_INFO("postfilter does not get enough result", K(limit_k), "near_rowkeys_count", near_rowkeys.count(),
            K(selectivity_), K(left_search), K(next_nprobe), K(iter_cnt), K(nprobes_), K(heap_size),
            K(new_heap_size), K(select_ratio), K(can_retry_), K(strategy_), K(adaptive_ctx_));
        near_cid_vec_.reuse();
        near_cid_vec_dis_.reuse();
        if (similarity_threshold_ != 0 && no_new_near_rowkeys) {
          iter_end = true;
          LOG_INFO("there are no new rowkeys in near_rowkeys after post filter, stop iterative filter", K(ret), K(no_new_near_rowkeys), K(near_rowkeys.count()), K(start_idx));
        } else if (max_scan_vectors_ > 0 && adaptive_ctx_.cid_vec_scan_rows_ > max_scan_vectors_) {
          iter_end = true;
          LOG_INFO("reach max scan vectors, stop iterative filter", K(no_new_near_rowkeys), K(near_rowkeys.count()),
              K(start_idx), K(limit_k), K(left_search), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
        } else if (can_retry_ && OB_FAIL(check_iter_filter_need_retry())) {
          LOG_WARN("ret of check iter filter need retry.", K(ret), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
        } else if (! has_next_center()) {
          iter_end = true;
        } else if (OB_FAIL(get_next_centers(is_vectorized, next_nprobe))) {
          LOG_WARN("get next centers failed", K(ret), K(has_used_hgraph_), K(next_nprobe));
        } else if (near_cid_vec_.count() != near_cid_vec_dis_.count()) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("near_cid_vec count is not equal to near_cid_vec_dis count", K(ret),
              "near_cid_vec_count", near_cid_vec_.count(), "near_cid_vec_dis_count", near_cid_vec_dis_.count());
        } else if (near_cid_vec_dis_.count() == 0) {
          // Common check: if no centers found after getting next batch, end iteration
          iter_end = true;
          LOG_TRACE("there are no centers left to access", K(ret), K(iterative_filter_ctx_));
        } else if (OB_FAIL(nearest_rowkey_heap.set_nprobe(new_heap_size))) {
          LOG_WARN("set new heap size fail", K(ret), K(new_heap_size), K(heap_size));
        }
      }
    }
    if (OB_SUCC(ret)) {
      for(int64_t cnt = 0; OB_SUCC(ret) && cnt < limit_k && ! rowkey_dist_heap.empty(); ++cnt) {
        const ObIvfRowkeyDistItem &item = rowkey_dist_heap.top();
        if (item.rowkey_idx_ > near_rowkeys.count()) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("out of bound of near_rowkeys", K(ret), K(item), K(near_rowkeys.count()));
        } else if (OB_FAIL(saved_rowkeys_.push_back(near_rowkeys.at(item.rowkey_idx_).rowkey_))) {
          LOG_WARN("push back fail", K(ret), K(item));
        } else if (OB_FAIL(rowkey_dist_heap.pop())) {
          LOG_WARN("rowkey_dist_heap pop fail", K(ret), K(saved_rowkeys_.count()), K(item));
        } else {
          LOG_TRACE("result", K(item), "i", cnt, K(limit_k));
        }
      }
    }
  } else {
    ObIvfFineWallIterGuard fine_iter_guard(ivf_lat_);
    if (OB_FAIL(calc_nearest_limit_rowkeys_in_cids(
        is_vectorized,
        reinterpret_cast<float *>(real_search_vec_.ptr()),
        saved_rowkeys_,
        nullptr))) {
      // 2. search nearest rowkeys
      LOG_WARN("fail to calc nearest limit rowkeys in cids", K(ret), K(dim_));
    }
  }

  return ret;
}

// NOTICE: shadow copy from expr memory
int ObDASIvfBaseScanIter::get_main_rowkey(
    const ObDASScanCtDef *ctdef,
    ObRowkey &main_rowkey)
{
  int ret = OB_SUCCESS;
  ObObj *obj_ptr = nullptr;
  void *buf = nullptr;

  if (OB_ISNULL(ctdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(ctdef));
  } else {
    const int64_t rowkey_cnt = main_rowkey.get_obj_cnt();
    const ExprFixedArray& rowkey_exprs = ctdef->rowkey_exprs_;
    if (rowkey_exprs.count() != rowkey_cnt) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("out expr is not enough for rowkey", K(ret), K(rowkey_cnt), K(rowkey_exprs.count()));
    } else {
      ObObj *obj_ptr = main_rowkey.get_obj_ptr();
      for (int64_t i = 0; OB_SUCC(ret) && i < rowkey_cnt; ++i) {
        ObExpr *expr = rowkey_exprs.at(i);
        if (OB_ISNULL(expr)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("should not be null", K(ret), K(i));
        } else if (OB_FAIL(expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_).to_obj(obj_ptr[i], expr->obj_meta_, expr->obj_datum_map_))) {
          LOG_WARN("convert datum to obj failed", K(ret), K(i), KPC(expr));
        }
      }
    }
  }
  return ret;
}

// if cid not exist, center_vec is nullptr
int ObDASIvfPQScanIter::check_cid_exist(
    const ObString &src_cid,
    float *&center_vec,
    bool &src_cid_exist)
{
  int ret = OB_SUCCESS;
  src_cid_exist = false;
  ObCenterId src_centor_id;
  center_vec = nullptr;
  if (OB_UNLIKELY(near_cid_vec_ptrs_.empty())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("near_cid_vec_ptrs_ is empty", K(ret));
  } else if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(src_centor_id, src_cid))) {
    LOG_WARN("failed to get center id from string", K(src_cid));
  } else if (OB_UNLIKELY(src_centor_id.center_id_ >= near_cid_vec_ptrs_.count())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("src_cid is not exist", K(ret), K(src_centor_id), K(near_cid_vec_ptrs_.count()));
  } else if (OB_NOT_NULL(near_cid_vec_ptrs_.at(src_centor_id.center_id_))) {
    src_cid_exist = true;
    center_vec = near_cid_vec_ptrs_.at(src_centor_id.center_id_);
  }

  return ret;
}

int ObDASIvfPQScanIter::process_ivf_scan_pre(ObIAllocator &allocator, bool is_vectorized)
{
  int ret = OB_SUCCESS;
  int64_t batch_row_count = ObVectorParamData::VI_PARAM_DATA_BATCH_SIZE;
  float *search_vec = reinterpret_cast<float *>(real_search_vec_.ptr());
  ObExprVectorDistance::ObVecDisType raw_dis_type = !need_norm_ ? dis_type_ : ObExprVectorDistance::ObVecDisType::COSINE;

  ObDASScanIter* inv_iter = (ObDASScanIter*)inv_idx_scan_iter_;
  bool is_range_prefilter = vec_aux_ctdef_->can_use_vec_pri_opt();
  ObIvfPreFilter prefilter(MTL_ID());
  // 1. Scan the ivf_centroid table, calculate the distance between vec_x and cid_vec,
  //    and get the nearest cluster center (cid 1, cid_vec 1)... (cid n, cid_vec n)
  if (OB_FAIL(get_nearest_probe_centers(is_vectorized))) {
    if (ret == OB_ENTRY_NOT_EXIST) {
      // cid_center table is empty, just do brute search
      ret = OB_SUCCESS;
      adaptive_ctx_.is_brute_force_= true;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      bool index_end = false;
      while (OB_SUCC(ret) && !index_end) {
        if (OB_FAIL(get_pre_filter_rowkey_batch(mem_context_->get_arena_allocator(), is_vectorized, batch_row_count,
                                                index_end))) {
          LOG_WARN("failed to get rowkey batch", K(ret), K(is_vectorized));
        } else if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
          LOG_WARN("failed to get limit rowkey brute", K(ret));
        } else {
          pre_fileter_rowkeys_.reset();
        }
      }
      if (OB_ITER_END == ret) {
        ret = OB_SUCCESS;
      }
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
    } else {
      LOG_WARN("failed to get nearest probe center ids", K(ret));
    }
  } else {
    const int64_t est_output_row_cnt = adaptive_ctx_.selectivity_ * adaptive_ctx_.row_count_;
    const bool need_check_brute = est_output_row_cnt <= IVF_MAX_BRUTE_FORCE_SIZE * 10 || ! is_range_prefilter;
    if (need_check_brute && OB_FAIL(get_rowkey_pre_filter(mem_context_->get_arena_allocator(), is_vectorized, IVF_MAX_BRUTE_FORCE_SIZE))) {
      LOG_WARN("failed to get rowkey pre filter", K(ret), K(is_vectorized));
    } else if (need_check_brute && pre_fileter_rowkeys_.count() < IVF_MAX_BRUTE_FORCE_SIZE) {
      // do brute search
      adaptive_ctx_.is_brute_force_= true;
      IvfRowkeyHeap nearest_rowkey_heap(vec_op_alloc_, search_vec/*unused*/, raw_dis_type, dim_, get_nprobe(limit_param_, 1), similarity_threshold_);
      if (OB_FAIL(get_rowkey_brute_post(is_vectorized, nearest_rowkey_heap))) {
        LOG_WARN("failed to get limit rowkey brute", K(ret));
      } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
        LOG_WARN("failed to get top n", K(ret));
      }
    } else { // search with prefilter
      if (is_range_prefilter) { // rowkey range prefilter
        adaptive_ctx_.pre_scan_row_cnt_ += pre_fileter_rowkeys_.count();
        adaptive_ctx_.is_range_prefilter_ = true;
        ObArray<const ObNewRange *> rk_range;
        const ObRangeArray& key_range = inv_iter->get_scan_param().key_ranges_;
        for (int64_t i = 0; i < key_range.count() && OB_SUCC(ret); i++) {
          const ObNewRange *range = &key_range.at(i);
          if (OB_FAIL(rk_range.push_back(range))) {
            LOG_WARN("fail to push back range", K(ret), K(i));
          }
        }
        if (OB_SUCC(ret)) {
          if (OB_FAIL(prefilter.init(rk_range))) {
            LOG_WARN("fail to init prefilter as range filter", K(ret));
          }
        }
      } else if (OB_FAIL(prefilter.init())) { // rowkey hash bitmap prefilter
        LOG_WARN("fail to init prefilter as roaring bitmap", K(ret));
      } else { // add bitmap for bitmap filter
        adaptive_ctx_.pre_scan_row_cnt_ += pre_fileter_rowkeys_.count();
        for (int i = 0; i < pre_fileter_rowkeys_.count() && OB_SUCC(ret); i++) {
          uint64_t hash_val = hash_val_for_rk(pre_fileter_rowkeys_.at(i));
          if (OB_FAIL(prefilter.add(hash_val))) {
            LOG_WARN("fail to add hash val to prefilter", K(ret));
          }
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(build_rowkey_hash_set(prefilter, is_vectorized, batch_row_count))) {
          LOG_WARN("fail to build rk hash set", K(ret));
        }
      }
      // do for loop until saved_rowkeys_.count() >= limitK
      if (OB_SUCC(ret)) {
        int64_t limit_k = limit_param_.limit_ + limit_param_.offset_;
        int64_t iter_cnt = 0;
        int64_t next_nprobe = 0;
        bool iter_end = false;
        bool is_first_scan = true;
        const int64_t sub_dim = dim_ / m_;
        int32_t start_idx = -1;
        bool no_new_near_rowkeys = false;
        ObExprVectorDistance::ObVecDisType cur_dis_type = dis_type_ == oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN ? oceanbase::sql::ObExprVectorDistance::ObVecDisType::EUCLIDEAN_SQUARED : dis_type_;
        IvfRowkeyHeap nearest_rowkey_heap(
            vec_op_alloc_, reinterpret_cast<float *>(real_search_vec_.ptr()), cur_dis_type, sub_dim, limit_k, similarity_threshold_);
        while (OB_SUCC(ret) && ! iter_end && nearest_rowkey_heap.count() < limit_k) {
          ObIvfFineWallIterGuard fine_iter_guard(ivf_lat_);
          if (OB_FALSE_IT(start_idx = nearest_rowkey_heap.count())) {
          } else if (OB_FAIL(calc_nearest_limit_rowkeys_in_cids(
              is_vectorized,
              reinterpret_cast<float *>(real_search_vec_.ptr()),
              nearest_rowkey_heap,
              &prefilter))) {
            LOG_WARN("fail to calc nearest limit rowkeys in cids", K(ret), K(dim_));
          } else if (OB_FALSE_IT(no_new_near_rowkeys = (nearest_rowkey_heap.count() == start_idx))) {
            // there are no new rowkeys in near_rowkeys after post filter
          }
          if (OB_FAIL(ret)) {
          } else if (nearest_rowkey_heap.count() < limit_k) {
            ++iter_cnt;
            const int64_t left_search = limit_k - nearest_rowkey_heap.count();
            next_nprobe = OB_MIN(nprobes_, (nprobes_ * ((double)left_search) / limit_k + 1));
            LOG_INFO("prefilter does not get enough result", K(limit_k), "nearest_rowkey_heap_count", nearest_rowkey_heap.count(),
                K(selectivity_), K(left_search), K(next_nprobe), K(iter_cnt), K(nprobes_), K(can_retry_), K(strategy_), K(adaptive_ctx_));
            near_cid_vec_.reuse();
            near_cid_vec_dis_.reuse();
            is_first_scan = false;
            if (similarity_threshold_ != 0 && no_new_near_rowkeys) {
                iter_end = true;
                LOG_INFO("there are no new rowkeys in near_rowkeys after post filter, stop iterative filter", K(ret), K(no_new_near_rowkeys), K(nearest_rowkey_heap.count()), K(start_idx), K(limit_k));
            } else if (max_scan_vectors_ > 0 && adaptive_ctx_.cid_vec_scan_rows_ > max_scan_vectors_) {
              iter_end = true;
              LOG_INFO("reach max scan vectors, stop iterative filter", K(no_new_near_rowkeys), K(nearest_rowkey_heap.count()),
                K(start_idx), K(limit_k), K(left_search), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
            } else if (! has_next_center()) {
              iter_end = true;
            } else if (OB_FAIL(get_next_centers(is_vectorized, next_nprobe))) {
              LOG_WARN("get next centers failed", K(ret), K(has_used_hgraph_), K(next_nprobe));
            } else if (near_cid_vec_.count() != near_cid_vec_dis_.count()) {
              ret = OB_ERR_UNEXPECTED;
              LOG_WARN("near_cid_vec count is not equal to near_cid_vec_dis count", K(ret),
                "near_cid_vec_count", near_cid_vec_.count(), "near_cid_vec_dis_count", near_cid_vec_dis_.count());
            } else if (near_cid_vec_dis_.count() == 0) {
              // Common check: if no centers found after getting next batch, end iteration
              iter_end = true;
              LOG_TRACE("there are no centers left to access", K(ret), K(iterative_filter_ctx_));
            }
          }
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(nearest_rowkey_heap.get_nearest_probe_center_ids(saved_rowkeys_))) {
          LOG_WARN("get rowkeys from heap fail", K(ret), K(limit_k));
        } else {
          LOG_TRACE("get rowkeys from heap success", K(limit_k), K(saved_rowkeys_.count()), K(iter_cnt), K(next_nprobe), K(nprobes_));
        }
      }
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::try_write_pq_centroid_cache(
    ObIvfCentCache &cent_cache,
    bool is_vectorized)
{
  int ret = OB_SUCCESS;
  RWLock::WLockGuard guard(cent_cache.get_lock());
  if (!cent_cache.is_writing()) {
    LOG_INFO("other threads already writed centroids cache, skip", K(ret));
  } else {
    ObArenaAllocator tmp_allocator;
    const ObDASScanCtDef *pq_cid_vec_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_pq_id_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_SPECIAL_AUX_SCAN);
    ObDASScanRtDef *pq_cid_vec_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_pq_id_tbl_idx());
    ObPqCenterId pq_cent_id;
    if (OB_ISNULL(pq_cid_vec_ctdef) || OB_ISNULL(pq_cid_vec_rtdef)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("ctdef or rtdef is null", K(ret), KP(pq_cid_vec_ctdef), KP(pq_cid_vec_rtdef));
    } else if (OB_FAIL(do_table_full_scan(is_vectorized,
                                    pq_cid_vec_ctdef,
                                    pq_cid_vec_rtdef,
                                    pq_centroid_iter_,
                                    pq_centroid_tablet_id_,
                                    pq_centroid_first_scan_,
                                    pq_centroid_scan_param_))) {
      LOG_WARN("failed to do centroid table scan", K(ret));
    } else if (is_vectorized) {
      IVF_GET_NEXT_ROWS_BEGIN(pq_centroid_iter_)
      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        ObExpr *cid_expr = pq_cid_vec_ctdef->result_output_[CID_IDX];
        ObExpr *cid_vec_expr = pq_cid_vec_ctdef->result_output_[CID_VECTOR_IDX];
        ObDatum *cid_datum = cid_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
        ObDatum *cid_vec_datum = cid_vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);
        bool has_lob_header = pq_cid_vec_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();
        uint64_t center_idx = 0;

        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          guard.set_batch_idx(i);
          ObString cid = cid_datum[i].get_string();
          ObString cid_vec = cid_vec_datum[i].get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                          &tmp_allocator,
                          ObLongTextType,
                          CS_TYPE_BINARY,
                          has_lob_header,
                          cid_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_FAIL(ObVectorClusterHelper::get_pq_center_id_from_string(pq_cent_id, cid,
            ObVectorClusterHelper::IVF_PARSE_M_ID | ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
            LOG_WARN("fail to get center idx from string", K(ret), KPHEX(cid.ptr(), cid.length()));
          } else if (OB_FAIL(cent_cache.write_pq_centroid(
                pq_cent_id.m_id_, pq_cent_id.center_id_, reinterpret_cast<float*>(cid_vec.ptr()), cid_vec.length()))) {
            LOG_WARN("fail to write centroid", K(ret), K(pq_cent_id), KPHEX(cid_vec.ptr(), cid_vec.length()));
          }
        }
      }
      IVF_GET_NEXT_ROWS_END(pq_centroid_iter_, pq_centroid_scan_param_, pq_centroid_tablet_id_)
    } else {
      pq_centroid_iter_->clear_evaluated_flag();
      ObExpr *cid_expr = pq_cid_vec_ctdef->result_output_[CID_IDX];
      ObDatum &cid_datum = cid_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      bool has_lob_header = pq_cid_vec_ctdef->result_output_.at(CID_VECTOR_IDX)->obj_meta_.has_lob_header();
      ObExpr *vec_expr = pq_cid_vec_ctdef->result_output_[CID_VECTOR_IDX];
      ObDatum &vec_datum = vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
      for (int i = 0; OB_SUCC(ret); ++i) {
        if (OB_FAIL(pq_centroid_iter_->get_next_row())) {
          if (OB_ITER_END != ret) {
            LOG_WARN("failed to scan vid rowkey iter", K(ret));
          }
        } else {
          ObString cid_vec = vec_datum.get_string();
          ObString cid = cid_datum.get_string();
          if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                          &mem_context_->get_arena_allocator(),
                          ObLongTextType,
                          CS_TYPE_BINARY,
                          has_lob_header,
                          cid_vec))) {
            LOG_WARN("failed to get real data.", K(ret));
          } else if (OB_FAIL(ObVectorClusterHelper::get_pq_center_id_from_string(pq_cent_id, cid,
            ObVectorClusterHelper::IVF_PARSE_M_ID | ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
            LOG_WARN("fail to get center idx from string", K(ret), KPHEX(cid.ptr(), cid.length()));
          } else if (OB_FAIL(cent_cache.write_pq_centroid(
                pq_cent_id.m_id_, pq_cent_id.center_id_, reinterpret_cast<float*>(cid_vec.ptr()), cid_vec.length()))) {
            LOG_WARN("fail to write centroid", K(ret), K(pq_cent_id), KPHEX(cid_vec.ptr(), cid_vec.length()));
          }
        }
      }
      int tmp_ret = (ret == OB_ITER_END) ? OB_SUCCESS : ret;
      if (OB_FAIL(ObDasVecScanUtils::reuse_iter(ls_id_, pq_centroid_iter_, pq_centroid_scan_param_, pq_centroid_tablet_id_))) {
        LOG_WARN("failed to reuse rowkey cid iter.", K(ret));
      } else {
        ret = tmp_ret;
      }
    }

    if (OB_SUCC(ret)) {
      if (cent_cache.get_count() > 0) {
        cent_cache.set_completed();
        LOG_DEBUG("success to write centroid table cache", K(pq_centroid_tablet_id_), K(cent_cache.get_count()));
      } else {
        cent_cache.reuse();
        LOG_DEBUG("Empty centroid table, no need to set cache", K(pq_centroid_tablet_id_));
      }

    }
  }

  if (OB_FAIL(ret)) {
    cent_cache.reuse();
  }

  return ret;
}

int ObDASIvfPQScanIter::get_pq_precomputetable_cache(
    bool is_vectorized,
    ObIvfCacheMgrGuard &cache_guard,
    ObIvfCentCache *&cent_cache,
    bool &is_cache_usable)
{
  int ret = OB_SUCCESS;
  ObPluginVectorIndexService *vec_index_service = MTL(ObPluginVectorIndexService *);
  ObIvfCacheMgr *cache_mgr = nullptr;
  const ObDASScanCtDef *centroid_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_centroid_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_CENTROID_SCAN);
  // pq/flat both use centroid_tablet_id_
  if (OB_FAIL(vec_index_service->acquire_ivf_cache_mgr_guard(
        ls_id_, centroid_tablet_id_, vec_index_param_, dim_, centroid_ctdef->ref_table_id_, cache_guard))) {
    LOG_WARN("failed to get ObPluginVectorIndexAdapter",
      K(ret), K(ls_id_), K(centroid_tablet_id_), K(vec_index_param_));
  } else if (OB_ISNULL(cache_mgr = cache_guard.get_ivf_cache_mgr())) {
    ret = OB_ERR_NULL_VALUE;
    LOG_WARN("invalid null cache mgr", K(ret));
  } else if (OB_FAIL(cache_mgr->get_or_create_cache_node(IvfCacheType::IVF_PQ_PRECOMPUTE_TABLE_CACHE, cent_cache))) {
    LOG_WARN("fail to get or create cache node", K(ret));
    if (ret == OB_ALLOCATE_MEMORY_FAILED) {
      is_cache_usable = false;
      ret = OB_SUCCESS;
    }
  } else if (!cent_cache->is_completed()) {
    if (cent_cache->set_writing_if_idle()) {
      int tmp_ret = try_write_pq_precompute_table_cache(*cent_cache, is_vectorized);
      if (OB_TMP_FAIL(tmp_ret)) {
        if (tmp_ret != OB_EAGAIN) {
          LOG_WARN("fail to try write centroid cache", K(tmp_ret), K(is_vectorized), KPC(cent_cache));
        } else {
          is_cache_usable = false;
        }
      } else {
        is_cache_usable = cent_cache->is_completed();
      }
    } else {
      LOG_INFO("other threads already writed pq precompute table cache, skip", K(ret));
    }
  } else {
    // read cache
    is_cache_usable = true;
  }
  return ret;
}

int ObDASIvfPQScanIter::try_write_pq_precompute_table_cache(
    ObIvfCentCache &cent_cache,
    bool is_vectorized)
{
  int ret = OB_SUCCESS;
  // write cache
  ObIvfCacheMgrGuard pq_center_guard;
  ObIvfCentCache *pq_cent_cache = nullptr;
  bool is_pq_cent_cache_usable = false;
  ObIvfCacheMgrGuard center_guard;
  ObIvfCentCache *ivf_cent_cache = nullptr;
  bool is_ivf_cent_cache_usable = false;
  if (OB_FAIL(get_centers_cache(is_vectorized, true/*is_pq_centers*/, pq_center_guard, pq_cent_cache, is_pq_cent_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret), K(is_vectorized), KPC(pq_cent_cache));
  } else if (OB_FAIL(get_centers_cache(is_vectorized, false/*is_pq_centers*/, center_guard, ivf_cent_cache, is_ivf_cent_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret), K(is_vectorized), KPC(ivf_cent_cache));
  } else {
    RWLock::WLockGuard guard(cent_cache.get_lock());
    if (!cent_cache.is_writing()) {
      LOG_INFO("other threads already writed centroids cache, skip", K(ret));
    } else {
      if (is_pq_cent_cache_usable && is_ivf_cent_cache_usable) {

          ObArenaAllocator tmp_allocator;
          int64_t ksub = 1L << nbits_;
          int64_t sub_dim = dim_ / m_;
          float *ivf_centers = nullptr;
          float *pq_center = pq_cent_cache->get_centroids(); // m * ksub * sub_dim
          float *precompute_table = cent_cache.get_centroids();
          float *r_norms = (float*)tmp_allocator.alloc(sizeof(float) * m_ * ksub);
          if (ivf_cent_cache->has_hgraph_index()) {
            if (OB_FAIL(ivf_cent_cache->get_centroids(&tmp_allocator, &ivf_centers))) {
              LOG_WARN("fail to get ivf centers", K(ret));
            }
          } else {
            ivf_centers = ivf_cent_cache->get_centroids();
          }

          if (OB_ISNULL(r_norms)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
            LOG_WARN("fail to alloc norms", K(ret), K(m_), K(ksub));
          } else {
            // compute norms
            for (int i = 0; i < m_; i++) {
              for (int j = 0; j < ksub; j++) {
                float* pq_cid_vec = pq_center + (i * ksub + j) * sub_dim;
                r_norms[i * ksub + j] = ObVectorL2Distance<float>::l2_norm_square(pq_cid_vec, sub_dim);
              }
            }
            // nlist
            for (int i = 0; i < ivf_cent_cache->get_capacity(); i++) {
              float* ivf_center = ivf_centers + i * dim_;
              float* tab = &precompute_table[i * m_ * ksub];
              for (int j = 0; j < m_; j++) {
                ObVectorIpDistance<float>::fvec_inner_products_ny(tab + j * ksub,
                                                                  ivf_center + j * sub_dim,
                                                                  pq_center + j * ksub * sub_dim,
                                                                  sub_dim,
                                                                  ksub);
              }
              ObVectorL2Distance<float>::fvec_madd(m_ * ksub, r_norms, 2.0, tab, tab);
            }
          }
        if (OB_SUCC(ret)) {
          cent_cache.set_completed();
        }
        if (OB_FAIL(ret)) {
          cent_cache.reuse();
        }
      } else {
        ret = OB_EAGAIN;
        cent_cache.reuse();
      }
    }
  }
  return ret;
}

int ObDASIvfPQScanIter::pre_compute_inner_prod_table(
    const float* search_vec,
    float* dis_table,
    bool is_vectorized)
{
  int ret = OB_SUCCESS;
  int64_t ksub = 1L << nbits_;
  int64_t sub_dim = dim_ / m_;
  // write cache
  ObIvfCacheMgrGuard pq_center_guard;
  ObIvfCentCache *pq_cent_cache = nullptr;
  bool is_pq_cent_cache_usable = false;
  if (OB_FAIL(get_centers_cache(is_vectorized, true/*is_pq_centers*/, pq_center_guard, pq_cent_cache, is_pq_cent_cache_usable))) {
    LOG_WARN("fail to get centers cache", K(ret), K(is_vectorized), KPC(pq_cent_cache));
  } else if (is_pq_cent_cache_usable) {
    float *pq_center = pq_cent_cache->get_centroids();
    for (int i = 0; i < m_; i++) {
      ObVectorIpDistance<float>::fvec_inner_products_ny(dis_table + i * ksub,
                                                        search_vec + i * sub_dim,
                                                        pq_center + i * ksub * sub_dim,
                                                        sub_dim,
                                                        ksub);
    }
  } else {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("fail to get pq center cache", K(ret));
  }
  return ret;
}

void ObDASIvfPQScanIter::reuse_cid_ctx()
{
  ObDASIvfBaseScanIter::reuse_cid_ctx();
  near_cid_vec_.reuse();
  near_cid_vec_dis_.reuse();
}

uint64_t ObDASIvfBaseScanIter::hash_val_for_rk(const common::ObRowkey& rk)
{
  uint64_t hash_val = 0;
  if (rk.get_obj_cnt() == 1 && ob_is_int_uint_tc(rk.get_obj_ptr()[0].get_type())) {
    hash_val = rk.get_obj_ptr()[0].get_uint64();
  } else {
    for (int i = 0; i < rk.get_obj_cnt(); i++) {
      (void)rk.get_obj_ptr()[i].hash(hash_val, hash_val);
    }
  }
  return hash_val;
}

int ObDASIvfBaseScanIter::build_rowkey_hash_set(
  ObIvfPreFilter &prefilter,
  bool is_vectorized,
  int64_t batch_row_count)
{
  int ret = OB_SUCCESS;
  bool index_end = false;
  // alloc one RowKey
  ObArenaAllocator& allocator = mem_context_->get_arena_allocator();
  int64_t rowkey_cnt = 0;
  const ObDASScanCtDef *ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(vec_aux_ctdef_->get_ivf_rowkey_cid_tbl_idx(),
                                                                        ObTSCIRScanType::OB_VEC_IVF_ROWKEY_CID_SCAN);
  ObDASScanRtDef *rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_rowkey_cid_tbl_idx());
  if (OB_ISNULL(ctdef) || OB_ISNULL(rtdef)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ctdef or rtdef is null", K(ret), KP(ctdef), KP(rtdef));
  } else if (OB_FALSE_IT(rowkey_cnt = ctdef->rowkey_exprs_.count())) {
  } else if (OB_UNLIKELY(rowkey_cnt <= 0)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid rowkey cnt", K(ret));
  } else {
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(MTL_ID(), "IVFRBTMP"));
    const bool is_one_int_rowkey = rowkey_cnt == 1 && ob_is_int_uint_tc(ctdef->rowkey_exprs_.at(0)->obj_meta_.get_type());
    if (!is_vectorized) {
      while (OB_SUCC(ret) && !index_end) {
        inv_idx_scan_iter_->clear_evaluated_flag();
        if (OB_FAIL(inv_idx_scan_iter_->get_next_row())) {
          ret = OB_ITER_END == ret ? OB_SUCCESS : ret;
          index_end = true;
        } else {
          uint64_t hash_val = 0;
          for (int64_t i = 0; OB_SUCC(ret) && i < rowkey_cnt; ++i) {
            ObObj tmp_obj;
            ObExpr *expr = ctdef->rowkey_exprs_.at(i);
            ObDatum &datum = expr->locate_expr_datum(*rtdef->eval_ctx_);
            if (OB_ISNULL(datum.ptr_)) {
              ret = OB_ERR_UNEXPECTED;
              LOG_WARN("get col datum null", K(ret));
            } else if (is_one_int_rowkey) {
              hash_val = datum.get_uint64();
            } else if (OB_FAIL(datum.to_obj(tmp_obj, expr->obj_meta_, expr->obj_datum_map_))) {
              LOG_WARN("convert datum to obj failed", K(ret));
            } else if (OB_FAIL(tmp_obj.hash(hash_val, hash_val))) {
              LOG_WARN("deep copy rowkey value failed", K(ret), K(tmp_obj));
            }
          }
          if (OB_SUCC(ret)) {
            if (OB_FAIL(prefilter.add(hash_val))) {
              LOG_WARN("fail to add hash val to prefilter", K(ret));
            } else {
              adaptive_ctx_.pre_scan_row_cnt_ ++;
            }
          }
        }
        if (OB_SUCC(ret) && can_retry_ && OB_FAIL(check_pre_filter_need_retry())) {
          LOG_WARN("ret of check pre filter need retry.", K(ret), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_), K(strategy_));
        }
      }
      if (ret == OB_ITER_END) {
        ret = OB_SUCCESS;
      }
    } else {
      while (OB_SUCC(ret) && !index_end) {
        int64_t scan_row_cnt = 0;
        if (OB_FAIL(inv_idx_scan_iter_->get_next_rows(scan_row_cnt, batch_row_count))) {
          if (OB_ITER_END != ret) {
            LOG_WARN("failed to get next row.", K(ret));
          }
          index_end = true;
        }
        if (OB_FAIL(ret) && OB_ITER_END != ret) {
        } else if (scan_row_cnt > 0) {
          ret = OB_SUCCESS;
        }
        if (OB_SUCC(ret)) {
          ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
          guard.set_batch_size(scan_row_cnt);
          adaptive_ctx_.pre_scan_row_cnt_ += scan_row_cnt;
          for (int i = 0; OB_SUCC(ret) && i < scan_row_cnt; i++) {
            guard.set_batch_idx(i);
            // pre_fileter_rowkeys_ need keep rowkey mem, so use vec_op_alloc_
            uint64_t hash_val = 0;
            for (int64_t j = 0; OB_SUCC(ret) && j < rowkey_cnt; ++j) {
              ObObj tmp_obj;
              ObExpr *expr = ctdef->rowkey_exprs_.at(j);
              ObDatum &datum = expr->locate_expr_datum(*rtdef->eval_ctx_);
              if (OB_ISNULL(datum.ptr_)) {
                ret = OB_ERR_UNEXPECTED;
                LOG_WARN("get col datum null", K(ret));
              } else if (is_one_int_rowkey) {
                hash_val = datum.get_uint64();
              } else if (OB_FAIL(datum.to_obj(tmp_obj, expr->obj_meta_, expr->obj_datum_map_))) {
                LOG_WARN("convert datum to obj failed", K(ret));
              } else if (OB_FAIL(tmp_obj.hash(hash_val, hash_val))) {
                LOG_WARN("deep copy rowkey value failed", K(ret), K(tmp_obj));
              }
            }
            if (OB_SUCC(ret)) {
              if (OB_FAIL(prefilter.add(hash_val))) {
                LOG_WARN("fail to add hash val to prefilter", K(ret));
              }
            }
          }
        }
        if (OB_SUCC(ret) && can_retry_ && OB_FAIL(check_pre_filter_need_retry())) {
          LOG_WARN("ret of check pre filter need retry.", K(ret), K(can_retry_), K(adaptive_ctx_), K(vec_index_type_), K(vec_idx_try_path_));
        }
      }
      if (ret == OB_ITER_END) {
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

/********************************************************************************************************/

int ObDASIvfSQ8ScanIter::inner_init(ObDASIterParam &param)
{
  int ret = ObDASIvfScanIter::inner_init(param);
  if (OB_SUCC(ret)) {
    ObDASIvfScanIterParam &ivf_scan_param = static_cast<ObDASIvfScanIterParam &>(param);
    sq_meta_iter_ = ivf_scan_param.sq_meta_iter_;
  }

  return ret;
}

int ObDASIvfSQ8ScanIter::inner_release()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(sq_meta_iter_) && OB_FAIL(sq_meta_iter_->release())) {
    LOG_WARN("failed to release inv_idx_scan_iter_", K(ret));
  }
  sq_meta_iter_ = nullptr;
  ObDasVecScanUtils::release_scan_param(sq_meta_scan_param_);
  if (OB_SUCC(ret) && OB_FAIL(ObDASIvfScanIter::inner_release())) {
    LOG_WARN("fail to do ObDASIvfScanIter::inner_release", K(ret));
  }

  return ret;
}

int ObDASIvfSQ8ScanIter::get_real_search_vec_u8(
    bool is_vectorized,
    ObString &real_search_vec_u8,
    ObString *out_sq_meta_min,
    ObString *out_sq_meta_step)
{
  int ret = OB_SUCCESS;
  const ObDASScanCtDef *sq_meta_ctdef = vec_aux_ctdef_->get_vec_aux_tbl_ctdef(
      vec_aux_ctdef_->get_ivf_sq_meta_tbl_idx(), ObTSCIRScanType::OB_VEC_IVF_SPECIAL_AUX_SCAN);
  ObDASScanRtDef *sq_meta_rtdef = vec_aux_rtdef_->get_vec_aux_tbl_rtdef(vec_aux_ctdef_->get_ivf_sq_meta_tbl_idx());
  ObString min_vec;
  ObString step_vec;
  if (OB_FAIL(do_table_full_scan(is_vectorized,
                                 sq_meta_ctdef,
                                 sq_meta_rtdef,
                                 sq_meta_iter_,
                                 sq_meta_tablet_id_,
                                 sq_meta_iter_first_scan_,
                                 sq_meta_scan_param_))) {
    LOG_WARN("failed to do table scan sq_meta", K(ret));
  } else if (is_vectorized) {
    IVF_GET_NEXT_ROWS_BEGIN(sq_meta_iter_)
      if (OB_SUCC(ret)) {
        ObEvalCtx::BatchInfoScopeGuard guard(*vec_aux_rtdef_->eval_ctx_);
        guard.set_batch_size(scan_row_cnt);
        bool has_lob_header = sq_meta_ctdef->result_output_.at(META_VECTOR_IDX)->obj_meta_.has_lob_header();
        ObExpr *meta_vec_expr = sq_meta_ctdef->result_output_[META_VECTOR_IDX];
        ObDatum *meta_vec_datum = meta_vec_expr->locate_batch_datums(*vec_aux_rtdef_->eval_ctx_);

        for (int64_t i = 0; OB_SUCC(ret) && i < scan_row_cnt; ++i) {
          guard.set_batch_idx(i);
          if (i == ObIvfConstant::SQ8_META_MIN_IDX || i == ObIvfConstant::SQ8_META_STEP_IDX) {
            ObString c_vec = meta_vec_datum[i].get_string();
            if (OB_FAIL(ObTextStringHelper::read_real_string_data_with_copy(
                    mem_context_->get_arena_allocator(),
                    meta_vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_),
                    sq_meta_ctdef->result_output_.at(META_VECTOR_IDX)->datum_meta_,
                    has_lob_header,
                    c_vec))) {
              LOG_WARN("failed to get real data.", K(ret));
            } else if (i == ObIvfConstant::SQ8_META_MIN_IDX) {
              // if not has lob, need deepcopy, because datum_row whill reuse
              min_vec = c_vec;
            } else if (i == ObIvfConstant::SQ8_META_STEP_IDX) {
              step_vec = c_vec;
            }
          }
        }
      }
    IVF_GET_NEXT_ROWS_END(sq_meta_iter_, sq_meta_scan_param_, sq_meta_tablet_id_)
  } else {
    // get min_vec max_vec step_vec in sq_meta table
    sq_meta_iter_->clear_evaluated_flag();
    bool has_lob_header = sq_meta_ctdef->result_output_.at(META_VECTOR_IDX)->obj_meta_.has_lob_header();
    ObExpr *meta_vec_expr = sq_meta_ctdef->result_output_[META_VECTOR_IDX];
    ObDatum &meta_vec_datum = meta_vec_expr->locate_expr_datum(*vec_aux_rtdef_->eval_ctx_);
    int row_index = 0;
    while (OB_SUCC(ret)) {
      if (OB_FAIL(sq_meta_iter_->get_next_row())) {
        if (OB_ITER_END != ret) {
          LOG_WARN("get next row failed.", K(ret));
        }
      } else {
        ObString c_vec = meta_vec_datum.get_string();
        if (OB_FAIL(ObTextStringHelper::read_real_string_data_with_copy(
                mem_context_->get_arena_allocator(),
                meta_vec_datum,
                sq_meta_ctdef->result_output_.at(META_VECTOR_IDX)->datum_meta_,
                has_lob_header,
                c_vec))) {
          LOG_WARN("failed to get real data.", K(ret));
        } else if (row_index == ObIvfConstant::SQ8_META_MIN_IDX) {
          // if not has lob, need deepcopy, because datum_row whill reuse
          min_vec = c_vec;
        } else if (row_index == ObIvfConstant::SQ8_META_STEP_IDX) {
          step_vec = c_vec;
        }
      }
      row_index++;
    }
    if (ret == OB_ITER_END) {
      if (OB_FAIL(sq_meta_iter_->reuse())) {
        LOG_WARN("fail to reuse scan iterator.", K(ret));
      }
    }
  }

  if (OB_SUCC(ret)) {
    uint8_t *res_vec = nullptr;
    if (OB_ISNULL(min_vec.ptr()) || OB_ISNULL(step_vec.ptr())) {
      if (OB_ISNULL(res_vec = reinterpret_cast<uint8_t *>(mem_context_->get_arena_allocator().alloc(sizeof(uint8_t) * dim_)))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("failed to allocate memory", K(ret), K(sizeof(uint8_t) * dim_));
      } else {
        MEMSET(res_vec, 0, sizeof(uint8_t) * dim_);
      }
    } else if (OB_FAIL(
                    ObExprVecIVFSQ8DataVector::cal_u8_data_vector(mem_context_->get_arena_allocator(),
                                                                  dim_,
                                                                  reinterpret_cast<float *>(min_vec.ptr()),
                                                                  reinterpret_cast<float *>(step_vec.ptr()),
                                                                  reinterpret_cast<float *>(real_search_vec_.ptr()),
                                                                  res_vec))) {
      LOG_WARN("fail to cal u8 data vector", K(ret), K(dim_));
    }
    if (OB_SUCC(ret)) {
      real_search_vec_u8.assign_ptr(reinterpret_cast<char *>(res_vec), dim_ * sizeof(uint8_t));
    }
    if (OB_SUCC(ret)) {
      if (OB_NOT_NULL(out_sq_meta_min)) {
        *out_sq_meta_min = min_vec;
      }
      if (OB_NOT_NULL(out_sq_meta_step)) {
        *out_sq_meta_step = step_vec;
      }
    }
  }
  return ret;
}

int ObDASIvfSQ8ScanIter::process_ivf_scan_post(bool is_vectorized)
{
  int ret = OB_SUCCESS;
  reset_ivf_sq8_latent_float_heap_ctx();
  ObString real_u8;
  ObString min_sv;
  ObString step_sv;
  if (OB_FAIL(get_real_search_vec_u8(is_vectorized, real_u8, &min_sv, &step_sv))) {
    LOG_WARN("failed to get real search vec u8", K(ret));
  } else {
    const int64_t meta_bytes = dim_ * static_cast<int64_t>(sizeof(float));
    const bool meta_ok = OB_NOT_NULL(min_sv.ptr()) && OB_NOT_NULL(step_sv.ptr())
        && min_sv.length() >= meta_bytes && step_sv.length() >= meta_bytes;
    bool used_sq8_latent_heap = false;
    if (ivf_sq8_dis_needs_latent_float_scoring(dis_type_) && meta_ok) {
      float *min_cp = reinterpret_cast<float *>(mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes)));
      float *step_cp = reinterpret_cast<float *>(mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes)));
      if (OB_ISNULL(min_cp) || OB_ISNULL(step_cp)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("alloc IVF SQ8 latent-float buffer failed", K(ret), K(dim_));
      } else {
        MEMCPY(min_cp, min_sv.ptr(), static_cast<uint32_t>(meta_bytes));
        MEMCPY(step_cp, step_sv.ptr(), static_cast<uint32_t>(meta_bytes));
        ivf_sq8_meta_min_ = min_cp;
        ivf_sq8_meta_step_ = step_cp;
        float *q_for_scan = nullptr;
        float *q_lat = nullptr;
        if (ivf_sq8_env_use_query_float_for_distance()) {
          q_for_scan = reinterpret_cast<float *>(real_search_vec_.ptr());
          if (OB_ISNULL(q_for_scan)) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("IVF_SQ8 OB_IVF_SQ8_QUERY_FLOAT_DISTANCE: real_search_vec_ is null", K(ret), K(dim_));
          }
        } else if (OB_ISNULL(q_lat = reinterpret_cast<float *>(
                                 mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes))))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("alloc IVF SQ8 q_lat failed", K(ret), K(dim_));
        } else {
          oceanbase::common::ivf_sq8_legacy_bin_center_u8_decode(
              dim_,
              min_cp,
              step_cp,
              reinterpret_cast<const uint8_t *>(real_u8.ptr()),
              q_lat);
          q_for_scan = q_lat;
        }
        if (OB_SUCC(ret)) {
          ivf_sq8_cid_u8_score_latent_float_heap_ = true;
          if (OB_FAIL(do_ivf_scan_post<float>(is_vectorized, q_for_scan))) {
            LOG_WARN("failed to do post filter (IVF_SQ8 latent-float)", K(ret), K(is_vectorized));
          }
          used_sq8_latent_heap = true;
          reset_ivf_sq8_latent_float_heap_ctx();
        }
      }
    }
    if (OB_SUCC(ret) && !used_sq8_latent_heap
        && OB_FAIL(do_ivf_scan_post<uint8_t>(is_vectorized, reinterpret_cast<uint8_t *>(real_u8.ptr())))) {
      LOG_WARN("failed to do post filter", K(ret), K(is_vectorized));
    }
  }
  reset_ivf_sq8_latent_float_heap_ctx();
  return ret;
}

int ObDASIvfSQ8ScanIter::process_ivf_scan_pre(ObIAllocator &allocator, bool is_vectorized)
{
  int ret = OB_SUCCESS;
  const int64_t t_sq8_prep_beg = ivf_lat_.enabled_ ? ObTimeUtility::current_time() : 0;
  reset_ivf_sq8_latent_float_heap_ctx();
  ObString real_u8;
  ObString min_sv;
  ObString step_sv;
  if (OB_FAIL(get_real_search_vec_u8(is_vectorized, real_u8, &min_sv, &step_sv))) {
    LOG_WARN("failed to get real search vec u8", K(ret));
  } else {
    const int64_t meta_bytes = dim_ * static_cast<int64_t>(sizeof(float));
    const bool meta_ok = OB_NOT_NULL(min_sv.ptr()) && OB_NOT_NULL(step_sv.ptr())
        && min_sv.length() >= meta_bytes && step_sv.length() >= meta_bytes;
    bool used_sq8_latent_heap = false;
    if (ivf_sq8_dis_needs_latent_float_scoring(dis_type_) && meta_ok) {
      float *min_cp = reinterpret_cast<float *>(mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes)));
      float *step_cp = reinterpret_cast<float *>(mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes)));
      if (OB_ISNULL(min_cp) || OB_ISNULL(step_cp)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("alloc IVF SQ8 latent-float buffer failed (pre)", K(ret), K(dim_));
      } else {
        MEMCPY(min_cp, min_sv.ptr(), static_cast<uint32_t>(meta_bytes));
        MEMCPY(step_cp, step_sv.ptr(), static_cast<uint32_t>(meta_bytes));
        ivf_sq8_meta_min_ = min_cp;
        ivf_sq8_meta_step_ = step_cp;
        float *q_for_scan = nullptr;
        float *q_lat = nullptr;
        if (ivf_sq8_env_use_query_float_for_distance()) {
          q_for_scan = reinterpret_cast<float *>(real_search_vec_.ptr());
          if (OB_ISNULL(q_for_scan)) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("IVF_SQ8 OB_IVF_SQ8_QUERY_FLOAT_DISTANCE (pre): real_search_vec_ is null", K(ret), K(dim_));
          }
        } else if (OB_ISNULL(q_lat = reinterpret_cast<float *>(
                                 mem_context_->get_arena_allocator().alloc(static_cast<int32_t>(meta_bytes))))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
          LOG_WARN("alloc IVF SQ8 q_lat failed (pre)", K(ret), K(dim_));
        } else {
          oceanbase::common::ivf_sq8_legacy_bin_center_u8_decode(
              dim_,
              min_cp,
              step_cp,
              reinterpret_cast<const uint8_t *>(real_u8.ptr()),
              q_lat);
          q_for_scan = q_lat;
        }
        if (OB_SUCC(ret)) {
          ivf_sq8_cid_u8_score_latent_float_heap_ = true;
          if (ivf_lat_.enabled_) {
            ivf_lat_.sq8_prep_us_ += ObTimeUtility::current_time() - t_sq8_prep_beg;
          }
          if (OB_FAIL(do_ivf_scan_pre<float>(allocator, is_vectorized, q_for_scan))) {
            LOG_WARN("failed to get rowkey pre filter (IVF_SQ8 latent-float)", K(ret), K(is_vectorized));
          }
          used_sq8_latent_heap = true;
          reset_ivf_sq8_latent_float_heap_ctx();
        }
      }
    }
    if (OB_SUCC(ret) && !used_sq8_latent_heap) {
      if (ivf_lat_.enabled_) {
        ivf_lat_.sq8_prep_us_ += ObTimeUtility::current_time() - t_sq8_prep_beg;
      }
      if (OB_FAIL(do_ivf_scan_pre<uint8_t>(allocator, is_vectorized, reinterpret_cast<uint8_t *>(real_u8.ptr())))) {
        LOG_WARN("failed to get rowkey pre filter", K(ret), K(is_vectorized));
      }
    }
  }
  reset_ivf_sq8_latent_float_heap_ctx();
  return ret;
}

// ObIvfPreFilter
void ObIvfPreFilter::reset()
{
  // release memory
  if (OB_NOT_NULL(roaring_bitmap_)) {
    if (type_ == FilterType::ROARING_BITMAP) {
      roaring::api::roaring64_bitmap_free(roaring_bitmap_);
    }
  }
  // reset members
  type_ = FilterType::ROARING_BITMAP;
  roaring_bitmap_ = nullptr;
  rk_range_.reset();
}

int ObIvfPreFilter::init()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(roaring_bitmap_)) {
    ret = OB_INIT_TWICE;
    LOG_WARN("init twice", K(ret));
  } else {
    ROARING_TRY_CATCH(roaring_bitmap_ = roaring::api::roaring64_bitmap_create());
    if (OB_SUCC(ret)) {
      type_ = FilterType::ROARING_BITMAP;
    }
  }
  return ret;
}

int ObIvfPreFilter::init(const ObIArray<const ObNewRange*> &range)
{
  int ret = OB_SUCCESS;
  if (rk_range_.count() != 0) {
    ret = OB_INIT_TWICE;
    LOG_WARN("init twice", K(ret), K(rk_range_));
  } else if (range.count() == 0) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid rk range", K(ret), K(range));
  } else if (OB_FAIL(rk_range_.assign(range))) {
    LOG_WARN("fail to assign rt st", K(ret));
  } else {
    type_ = FilterType::SIMPLE_RANGE;
  }
  return ret;
}

int ObIvfPreFilter::add(int64_t id)
{
  int ret = OB_SUCCESS;
  if (type_ == FilterType::ROARING_BITMAP) {
    ROARING_TRY_CATCH(roaring::api::roaring64_bitmap_add(roaring_bitmap_, id));
  } else if (type_ == FilterType::SIMPLE_RANGE) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("simple range not support add", K(ret));
  }
  return ret;
}

bool ObIvfPreFilter::test(const ObRowkey& main_rowkey)
{
  bool bret = false;
  int ret = OB_SUCCESS;
  if (type_ == FilterType::ROARING_BITMAP) {
    uint64_t hash_val = 0;
    if (main_rowkey.get_obj_cnt() == 1 && ob_is_int_uint_tc(main_rowkey.get_obj_ptr()[0].get_type())) {
      hash_val = main_rowkey.get_obj_ptr()[0].get_uint64();
    } else {
      for (int i = 0; i < main_rowkey.get_obj_cnt(); i++) {
        (void)main_rowkey.get_obj_ptr()[i].hash(hash_val, hash_val);
      }
    }
    bret = roaring::api::roaring64_bitmap_contains(roaring_bitmap_, hash_val);
  } else if (type_ == FilterType::SIMPLE_RANGE) {
    ObNewRange tmp_range;
    if (OB_FAIL(tmp_range.build_range(rk_range_.at(0)->table_id_, main_rowkey))) {
      LOG_WARN("fail to build tmp range", K(ret));
    }
    // do compare
    for (int64_t i = 0; i < rk_range_.count() && !bret && OB_SUCC(ret); i++) {
      if (rk_range_.at(i)->compare_with_startkey2(tmp_range) <= 0 && rk_range_.at(i)->compare_with_endkey2(tmp_range) >= 0) {
        bret = true;
      }
    }
  }
  return bret;
}

void ObIvfIterativeFilterContext::reset()
{
  visited_center_cnt_ = 0;
  centroids_.reset();
  allocator_.reset();
}

void ObIvfIterativeFilterContext::reuse()
{
  visited_center_cnt_ = 0;
  centroids_.reuse();
  allocator_.reuse();
}

int ObIvfIterativeFilterContext::get_next_nearest_probe_center_ids(const int64_t nprobe, ObIArray<ObCenterId> &center_ids)
{
  int ret = OB_SUCCESS;
  if (visited_center_cnt_ >= centroids_.count()) {
    ret = OB_ITER_END;
    LOG_WARN("all centers are visited", K(ret), K(visited_center_cnt_), "centroids_count", centroids_.count());
  } else {
    int64_t end = OB_MIN(visited_center_cnt_ + nprobe, centroids_.count());
    for (int64_t i = visited_center_cnt_; OB_SUCC(ret) && i < end; ++i) {
      const ObCentroidQueryInfo<float, ObCenterId> &cur = centroids_.at(i);
      if (OB_FAIL(center_ids.push_back(cur.id_))) {
        LOG_WARN("failed to push center id", K(ret), K(i), K(cur));
      }
    }
    if (OB_SUCC(ret)) {
      visited_center_cnt_ = end;
    }
  }
  return ret;
}

int ObIvfIterativeFilterContext::get_next_nearest_probe_center_ids_dist(const int64_t nprobe, ObArrayWrap<bool> &nearest_cid_dist)
{
  int ret = OB_SUCCESS;
  if (visited_center_cnt_ >= centroids_.count()) {
    ret = OB_ITER_END;
    LOG_WARN("all centers are visited", K(ret), K(visited_center_cnt_), "centroids_count", centroids_.count());
  } else {
    int64_t end = OB_MIN(visited_center_cnt_ + nprobe, centroids_.count());
    for (int64_t i = visited_center_cnt_; OB_SUCC(ret) && i < end; ++i) {
      const ObCentroidQueryInfo<float, ObCenterId> &cur = centroids_.at(i);
      const ObCenterId &center_id = cur.id_;
      if (OB_UNLIKELY(center_id.center_id_ >= nearest_cid_dist.count())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("center_id is not less than nearest_cid_dist", K(ret), K(center_id), K(nearest_cid_dist.count()));
      } else {
        nearest_cid_dist.at(center_id.center_id_) = true;
      }
    }
    if (OB_SUCC(ret)) {
      visited_center_cnt_ = end;
    }
  }
  return ret;
}

int ObIvfIterativeFilterContext::get_next_nearest_probe_centers_vec_dist(const int64_t nprobe, ObIArray<std::pair<ObCenterId, float *>> &center_ids, ObIArray<float> &distances)
{
  int ret = OB_SUCCESS;
  if (visited_center_cnt_ >= centroids_.count()) {
    ret = OB_ITER_END;
    LOG_WARN("all centers are visited", K(ret), K(visited_center_cnt_), "centroids_count", centroids_.count());
  } else {
    int64_t end = OB_MIN(visited_center_cnt_ + nprobe, centroids_.count());
    for (int64_t i = visited_center_cnt_; OB_SUCC(ret) && i < end; ++i) {
      const ObCentroidQueryInfo<float, ObCenterId> &cur = centroids_.at(i);
      if (OB_FAIL(center_ids.push_back(std::make_pair(cur.id_, (float *)cur.vec_)))) {
        LOG_WARN("failed to push center id", K(ret), K(i), K(cur));
      } else if (OB_FAIL(distances.push_back(cur.distance_))) {
        LOG_WARN("failed to push distance", K(ret), K(i), K(cur));
      }
    }
    if (OB_SUCC(ret)) {
      visited_center_cnt_ = end;
    }
  }
  return ret;
}

bool ObIvfIterativeFilterContext::has_next_center() const
{
  LOG_TRACE("next info", K_(visited_center_cnt), "center_count", centroids_.count());
  return visited_center_cnt_ < centroids_.count();
}

}  // namespace sql
}  // namespace oceanbase
