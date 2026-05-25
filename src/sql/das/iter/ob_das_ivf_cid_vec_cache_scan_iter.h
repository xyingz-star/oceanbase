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

#ifndef OBDEV_SRC_SQL_DAS_ITER_OB_DAS_IVF_CID_VEC_CACHE_SCAN_ITER_H_
#define OBDEV_SRC_SQL_DAS_ITER_OB_DAS_IVF_CID_VEC_CACHE_SCAN_ITER_H_

#include "sql/das/iter/ob_das_scan_iter.h"
#include "share/vector_index/ob_ivf_cid_cluster_cache.h"
#include "share/vector_index/ob_vector_index_param.h"

namespace oceanbase
{
namespace sql
{

struct ObDASIvfCidVecCacheScanIterParam : public ObDASScanIterParam
{
  ObDASIvfCidVecCacheScanIterParam()
    : ObDASScanIterParam(),
      cluster_cache_(nullptr),
      algo_(ObVectorIndexAlgorithmType::VIAT_MAX),
      cid_vec_ctdef_(nullptr) {}
  share::ObIvfCidClusterCache *cluster_cache_;
  ObVectorIndexAlgorithmType algo_;
  const ObDASScanCtDef *cid_vec_ctdef_;
};

class ObDASIvfCidVecCacheScanIter : public ObDASScanIter
{
public:
  ObDASIvfCidVecCacheScanIter();
  virtual ~ObDASIvfCidVecCacheScanIter();

  static bool is_row_materialized(ObDASScanIter *cid_vec_iter, int64_t batch_idx = 0);
  /// REPLAY / FILL rows already have expanded payload in datum; do not read_real_string_data as LOB.
  static bool skip_payload_lob_read(ObDASScanIter *cid_vec_iter, int64_t batch_idx = 0);
  /// REPLAY only: FILL recorded whether cluster payloads are already L2 unit (from flat header).
  static bool get_cached_payloads_l2_unit(ObDASScanIter *cid_vec_iter, bool &is_unit, bool &known);
  /// REPLAY serves rows from cache memory; no storage ObTableScanIterator (output_result_iter is null).
  static bool cid_vec_skips_storage_output_result_iter(ObDASScanIter *cid_vec_iter);
  static ObDASIvfCidVecCacheScanIter *get_active_iter() { return active_iter_; }
  void export_log_snapshot(share::ObIvfCidClusterCacheLogSnapshot &out) const;
  /// Reset per-query session counters and cache scan state (e.g. adaptive IVF retry).
  void reset_per_query_session_stats();

  virtual int do_table_scan() override;
  virtual int rescan() override;

protected:
  virtual int inner_init(ObDASIterParam &param) override;
  virtual int inner_reuse() override;
  virtual int inner_release() override;
  virtual int inner_get_next_row() override;
  virtual int inner_get_next_rows(int64_t &count, int64_t capacity) override;

private:
  enum class ScanMode { REPLAY, FILL, MISS };

  int on_cid_switch(uint64_t new_cid);
  int acquire_cid_and_set_mode(uint64_t new_cid);
  int parse_current_cid(uint64_t &cid);
  int flush_building_cluster();
  void clear_materialized_batch();
  void mark_materialized_batch(int64_t count);
  void discard_building_cluster();
  void release_replay_entry();
  int append_fill_row(int64_t batch_idx = 0);
  int replay_one_row();
  int replay_rows(int64_t &count, int64_t capacity);
  int replay_materialize_at(int64_t row_idx, int64_t batch_idx);
  share::ObIvfCidClusterPayloadType payload_type() const;
  int materialize_row_to_eval(const share::ObIvfCidClusterRow &row, int64_t batch_idx);
  /// Open storage scan if result_ is null (after REPLAY), else rescan. Used by FILL and MISS.
  int ensure_storage_scan();
  /// MISS: delegate to ObDASScanIter (same as ENABLED=0 scan path).
  bool scan_delegates_to_base_no_cache() const;
  bool needs_lightweight_cid_switch() const;
  int delegate_base_rescan();
  int delegate_base_do_table_scan();
  int delegate_base_inner_get_next_row();
  int delegate_base_inner_get_next_rows(int64_t &count, int64_t capacity);
  void log_session_stats_if_enabled() const;
  void maybe_log_progress_stats(uint64_t cid) const;
  void finish_fill_leader_if_any();

  share::ObIvfCidClusterCache *cluster_cache_;
  ObVectorIndexAlgorithmType algo_;
  const ObDASScanCtDef *cid_vec_ctdef_;
  ScanMode mode_;
  uint64_t current_cid_;
  share::ObIvfCidClusterEntry *replay_entry_;
  int64_t replay_idx_;
  share::ObIvfCidClusterEntry building_cluster_;
  bool cid_fill_leader_;
  static const int64_t MATERIALIZED_BATCH_CAP = 1024;
  int64_t materialized_batch_cnt_;
  uint8_t materialized_batch_[MATERIALIZED_BATCH_CAP];
  /// Set in inner_init when FILL/REPLAY path is enabled (stable for latency stats).
  bool cache_was_active_;
  share::ObIvfCidClusterCacheSessionStats session_stats_;
  static thread_local ObDASIvfCidVecCacheScanIter *active_iter_;
};

/// Export per-query cid cluster cache stats from cid_vec_iter_ (if it is ObDASIvfCidVecCacheScanIter).
bool ob_das_ivf_try_export_cid_cluster_cache_snapshot(
    ObDASScanIter *cid_vec_iter,
    share::ObIvfCidClusterCacheLogSnapshot &out);
void ob_das_ivf_reset_cid_cluster_cache_session_stats(ObDASScanIter *cid_vec_iter);

} // namespace sql
} // namespace oceanbase

#endif /* OBDEV_SRC_SQL_DAS_ITER_OB_DAS_IVF_CID_VEC_CACHE_SCAN_ITER_H_ */
