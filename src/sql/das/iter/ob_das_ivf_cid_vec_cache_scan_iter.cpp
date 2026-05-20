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

#define USING_LOG_PREFIX SQL_DAS

#include "sql/das/iter/ob_das_ivf_cid_vec_cache_scan_iter.h"
#include "sql/engine/expr/ob_expr_util.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"
#include "lib/utility/ob_macro_utils.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/time/ob_time_utility.h"
#include "share/vector_type/ob_vector_common_util.h"
#include "common/object/ob_object.h"
#include <cstring>

namespace oceanbase
{
using namespace common;
namespace sql
{

namespace {

int64_t ob_ivf_cid_rowkey_storage_bytes(const ObObj *objs, int64_t cnt)
{
  int64_t bytes = sizeof(ObObj) * cnt;
  for (int64_t i = 0; i < cnt; ++i) {
    if (objs[i].is_string_type()) {
      bytes += objs[i].get_string_len();
    }
  }
  return bytes;
}

} // namespace

thread_local ObDASIvfCidVecCacheScanIter *ObDASIvfCidVecCacheScanIter::active_iter_ = nullptr;

ObDASIvfCidVecCacheScanIter::ObDASIvfCidVecCacheScanIter()
  : ObDASScanIter(),
    cluster_cache_(nullptr),
    algo_(ObVectorIndexAlgorithmType::VIAT_MAX),
    cid_vec_ctdef_(nullptr),
    mode_(ScanMode::PASSTHROUGH),
    current_cid_(0),
    replay_entry_(nullptr),
    replay_idx_(0),
    building_cluster_(),
    cid_fill_leader_(false),
    materialized_batch_cnt_(0),
    cache_was_active_(false)
{
  MEMSET(materialized_batch_, 0, sizeof(materialized_batch_));
  session_stats_.reset();
}

ObDASIvfCidVecCacheScanIter::~ObDASIvfCidVecCacheScanIter()
{
  if (active_iter_ == this) {
    active_iter_ = nullptr;
  }
}

bool ObDASIvfCidVecCacheScanIter::is_row_materialized(ObDASScanIter *cid_vec_iter, int64_t batch_idx)
{
  const ObDASIvfCidVecCacheScanIter *cache_iter = dynamic_cast<const ObDASIvfCidVecCacheScanIter *>(cid_vec_iter);
  return OB_NOT_NULL(cache_iter) && batch_idx >= 0 && batch_idx < cache_iter->materialized_batch_cnt_
      && cache_iter->materialized_batch_[batch_idx] != 0;
}

bool ObDASIvfCidVecCacheScanIter::skip_payload_lob_read(ObDASScanIter *cid_vec_iter, int64_t batch_idx)
{
  const ObDASIvfCidVecCacheScanIter *cache_iter = dynamic_cast<const ObDASIvfCidVecCacheScanIter *>(cid_vec_iter);
  if (OB_ISNULL(cache_iter) || !cache_iter->cache_was_active_) {
    return false;
  }
  if (cache_iter->mode_ == ScanMode::REPLAY) {
    return true;
  }
  return is_row_materialized(cid_vec_iter, batch_idx);
}

int ObDASIvfCidVecCacheScanIter::ensure_storage_scan()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(get_output_result_iter())) {
    if (OB_FAIL(ObDASScanIter::rescan())) {
      LOG_WARN("failed to rescan storage", K(ret));
    }
  } else if (OB_FAIL(ObDASScanIter::do_table_scan())) {
    LOG_WARN("failed to do table scan for storage", K(ret));
  }
  return ret;
}

bool ObDASIvfCidVecCacheScanIter::scan_delegates_to_base_no_cache() const
{
  return mode_ == ScanMode::PASSTHROUGH || mode_ == ScanMode::STORAGE_ONLY;
}

bool ObDASIvfCidVecCacheScanIter::needs_lightweight_cid_switch() const
{
  return mode_ == ScanMode::STORAGE_ONLY && building_cluster_.row_count_ <= 0
      && OB_ISNULL(building_cluster_.arena_);
}

int ObDASIvfCidVecCacheScanIter::delegate_base_rescan()
{
  return ObDASScanIter::rescan();
}

int ObDASIvfCidVecCacheScanIter::delegate_base_do_table_scan()
{
  return ObDASScanIter::do_table_scan();
}

int ObDASIvfCidVecCacheScanIter::delegate_base_inner_get_next_row()
{
  const int64_t t0 = (mode_ == ScanMode::STORAGE_ONLY) ? ObTimeUtility::current_time() : 0;
  int ret = ObDASScanIter::inner_get_next_row();
  if (mode_ == ScanMode::STORAGE_ONLY) {
    session_stats_.storage_fetch_us_ += ObTimeUtility::current_time() - t0;
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::delegate_base_inner_get_next_rows(int64_t &count, int64_t capacity)
{
  const int64_t t0 = (mode_ == ScanMode::STORAGE_ONLY) ? ObTimeUtility::current_time() : 0;
  int ret = ObDASScanIter::inner_get_next_rows(count, capacity);
  if (mode_ == ScanMode::STORAGE_ONLY) {
    session_stats_.storage_fetch_us_ += ObTimeUtility::current_time() - t0;
  }
  return ret;
}

bool ObDASIvfCidVecCacheScanIter::cid_vec_skips_storage_output_result_iter(ObDASScanIter *cid_vec_iter)
{
  const ObDASIvfCidVecCacheScanIter *cache_iter = dynamic_cast<const ObDASIvfCidVecCacheScanIter *>(cid_vec_iter);
  return OB_NOT_NULL(cache_iter) && cache_iter->cache_was_active_ && cache_iter->mode_ == ScanMode::REPLAY;
}

void ObDASIvfCidVecCacheScanIter::clear_materialized_batch()
{
  materialized_batch_cnt_ = 0;
}

void ObDASIvfCidVecCacheScanIter::mark_materialized_batch(int64_t count)
{
  materialized_batch_cnt_ = OB_MIN(count, MATERIALIZED_BATCH_CAP);
  if (materialized_batch_cnt_ > 0) {
    MEMSET(materialized_batch_, 1, materialized_batch_cnt_);
  }
}

int ObDASIvfCidVecCacheScanIter::inner_init(ObDASIterParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(ObDASScanIter::inner_init(param))) {
    LOG_WARN("failed to init das scan iter", K(ret));
  } else if (param.type_ != ObDASIterType::DAS_ITER_SCAN) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected param type", K(ret), K(param.type_));
  } else {
    ObDASIvfCidVecCacheScanIterParam &p = static_cast<ObDASIvfCidVecCacheScanIterParam &>(param);
    cluster_cache_ = p.cluster_cache_;
    algo_ = p.algo_;
    cid_vec_ctdef_ = p.cid_vec_ctdef_;
    cache_was_active_ = OB_NOT_NULL(cluster_cache_) && share::is_ivf_cid_cluster_cache_enabled();
    mode_ = cache_was_active_ ? ScanMode::FILL : ScanMode::PASSTHROUGH;
    current_cid_ = 0;
    replay_entry_ = nullptr;
    replay_idx_ = 0;
    cid_fill_leader_ = false;
    active_iter_ = this;
    clear_materialized_batch();
    session_stats_.reset();
    if (cache_was_active_ && share::is_ivf_cid_cluster_cache_stats_enabled() && OB_NOT_NULL(cluster_cache_)) {
      session_stats_.has_cache_snap_begin_ = true;
      cluster_cache_->get_stats(session_stats_.cache_snap_begin_);
    }
  }
  return ret;
}

void ObDASIvfCidVecCacheScanIter::log_session_stats_if_enabled() const
{
  if (!cache_was_active_) {
    return;
  }
  const uint64_t index_epoch = OB_NOT_NULL(cluster_cache_) ? cluster_cache_->get_index_epoch() : 0;
  share::log_ivf_cid_cluster_cache_session_stats(
      session_stats_, index_epoch, static_cast<int64_t>(algo_), cluster_cache_);
}

void ObDASIvfCidVecCacheScanIter::maybe_log_progress_stats(uint64_t cid) const
{
  const int64_t every_n = share::ivf_cid_cluster_cache_stats_every_n_cid();
  if (every_n <= 0 || !cache_was_active_ || !share::is_ivf_cid_cluster_cache_stats_enabled()) {
  } else if (session_stats_.cid_switch_cnt_ % every_n != 0) {
  } else {
    const char *mode_str = "unknown";
    switch (mode_) {
      case ScanMode::REPLAY:
        mode_str = "REPLAY";
        break;
      case ScanMode::FILL:
        mode_str = "FILL";
        break;
      case ScanMode::STORAGE_ONLY:
        mode_str = "STORAGE_ONLY";
        break;
      case ScanMode::PASSTHROUGH:
        mode_str = "PASSTHROUGH";
        break;
      default:
        break;
    }
    share::ObIvfCidClusterCacheLogSnapshot snap;
    export_log_snapshot(snap);
    share::ob_ivf_cid_cluster_cache_write_user_log_file(
        snap, 0, 0, 0, "progress", cid, mode_str);
    LOG_INFO("[OB_IVF_CID_CLUSTER_CACHE_STATS] progress",
             K(cid),
             "mode", mode_str,
             K(session_stats_.cid_switch_cnt_),
             K(session_stats_.storage_only_cid_cnt_),
             K(session_stats_.replay_cid_cnt_),
             K(session_stats_.fill_cid_cnt_),
             K(session_stats_.storage_fetch_us_),
             K(session_stats_.fill_wait_us_),
             K(session_stats_.fill_wait_cnt_));
  }
}

int ObDASIvfCidVecCacheScanIter::inner_reuse()
{
  int ret = OB_SUCCESS;
  if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
    cluster_cache_->finish_cid_fill(current_cid_);
    cid_fill_leader_ = false;
  }
  release_replay_entry();
  discard_building_cluster();
  replay_idx_ = 0;
  clear_materialized_batch();
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(ObDASScanIter::inner_reuse())) {
    LOG_WARN("failed to reuse base scan iter", K(ret));
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::inner_release()
{
  if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
    cluster_cache_->finish_cid_fill(current_cid_);
    cid_fill_leader_ = false;
  }
  release_replay_entry();
  (void)flush_building_cluster();
  log_session_stats_if_enabled();
  if (OB_NOT_NULL(cluster_cache_)) {
    share::release_ivf_cid_cluster_cache(cluster_cache_);
    cluster_cache_ = nullptr;
  }
  if (active_iter_ == this) {
    active_iter_ = nullptr;
  }
  return ObDASScanIter::inner_release();
}

share::ObIvfCidClusterPayloadType ObDASIvfCidVecCacheScanIter::payload_type() const
{
  if (algo_ == ObVectorIndexAlgorithmType::VIAT_IVF_SQ8) {
    return share::IVF_CID_CLUSTER_PAYLOAD_SQ8_U8;
  } else if (algo_ == ObVectorIndexAlgorithmType::VIAT_IVF_PQ) {
    return share::IVF_CID_CLUSTER_PAYLOAD_PQ_IDS;
  }
  return share::IVF_CID_CLUSTER_PAYLOAD_FLAT_FLOAT;
}

int ObDASIvfCidVecCacheScanIter::parse_current_cid(uint64_t &cid)
{
  int ret = OB_SUCCESS;
  cid = 0;
  storage::ObTableScanParam &scan_param = get_scan_param();
  if (scan_param.key_ranges_.count() == 0) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid scan param for cid parse", K(ret));
  } else {
    const ObNewRange &range = scan_param.key_ranges_.at(0);
    if (range.start_key_.get_obj_cnt() == 0) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      ObCenterId center_id;
      const ObString cid_str = range.start_key_.get_obj_ptr()[0].get_string();
      if (OB_FAIL(ObVectorClusterHelper::get_center_id_from_string(
              center_id, cid_str, ObVectorClusterHelper::IVF_PARSE_CENTER_ID))) {
        LOG_WARN("failed to parse cid from range", K(ret), K(cid_str));
      } else {
        cid = center_id.center_id_;
      }
    }
  }
  return ret;
}

void ObDASIvfCidVecCacheScanIter::release_replay_entry()
{
  if (OB_NOT_NULL(cluster_cache_) && OB_NOT_NULL(replay_entry_)) {
    cluster_cache_->unpin_entry(replay_entry_);
  }
  replay_entry_ = nullptr;
  replay_idx_ = 0;
}

int ObDASIvfCidVecCacheScanIter::try_switch_to_replay_after_put_conflict(uint64_t cid)
{
  int ret = OB_SUCCESS;
  bool is_fill_leader = false;
  release_replay_entry();
  bool storage_only = false;
  if (OB_FAIL(cluster_cache_->acquire_cid_cluster(
          cid, replay_entry_, is_fill_leader, storage_only, &session_stats_))) {
    LOG_WARN("failed to acquire cluster for replay after put conflict", K(ret), K(cid));
  } else if (is_fill_leader || storage_only || OB_ISNULL(replay_entry_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("unexpected acquire after put conflict", K(ret), K(cid), K(is_fill_leader), K(storage_only), KP(replay_entry_));
  } else {
    mode_ = ScanMode::REPLAY;
    replay_idx_ = 0;
    session_stats_.replay_cid_cnt_++;
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::flush_building_cluster()
{
  int ret = OB_SUCCESS;
  const uint64_t fill_cid = building_cluster_.cid_;
  const bool was_fill_leader = cid_fill_leader_;
  if (OB_ISNULL(cluster_cache_) || building_cluster_.row_count_ <= 0) {
  } else {
    const int64_t rows = building_cluster_.row_count_;
    if (OB_FAIL(cluster_cache_->put(building_cluster_))) {
      if (ret == OB_EAGAIN) {
        session_stats_.put_cid_skip_pinned_cnt_++;
        discard_building_cluster();
        if (OB_FAIL(try_switch_to_replay_after_put_conflict(fill_cid))) {
          LOG_WARN("failed to switch to replay after put skip", K(ret), K(fill_cid));
        }
        ret = OB_SUCCESS;
      } else if (ret == OB_BUF_NOT_ENOUGH) {
        ret = OB_SUCCESS;
        discard_building_cluster();
      } else {
        LOG_WARN("failed to put cluster entry", K(ret), K(fill_cid), K(building_cluster_.row_count_));
        session_stats_.put_cid_fail_cnt_++;
        discard_building_cluster();
      }
    } else {
      session_stats_.put_cid_ok_cnt_++;
      session_stats_.put_rows_total_ += rows;
      discard_building_cluster();
    }
  }
  if (was_fill_leader && OB_NOT_NULL(cluster_cache_)) {
    cluster_cache_->finish_cid_fill(fill_cid);
    cid_fill_leader_ = false;
  }
  if (building_cluster_.row_count_ <= 0 && (OB_NOT_NULL(building_cluster_.arena_) || building_cluster_.cid_ != 0)) {
    discard_building_cluster();
  }
  return ret;
}

void ObDASIvfCidVecCacheScanIter::discard_building_cluster()
{
  building_cluster_.~ObIvfCidClusterEntry();
  new (&building_cluster_) share::ObIvfCidClusterEntry();
}

int ObDASIvfCidVecCacheScanIter::acquire_cid_and_set_mode(uint64_t new_cid)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(cluster_cache_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    bool is_fill_leader = false;
    bool storage_only = false;
    const int64_t acquire_beg_us = ObTimeUtility::current_time();
    if (OB_FAIL(cluster_cache_->acquire_cid_cluster(
            new_cid, replay_entry_, is_fill_leader, storage_only, &session_stats_))) {
      LOG_WARN("failed to acquire cid cluster", K(ret), K(new_cid));
    }
    session_stats_.acquire_us_ += ObTimeUtility::current_time() - acquire_beg_us;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(replay_entry_)) {
      revert_storage_scan_iter_if_any();
      mode_ = ScanMode::REPLAY;
      session_stats_.replay_cid_cnt_++;
    } else if (storage_only) {
      // Cache miss / not admitted: use base ObDASScanIter path (same as ENABLED=0).
      mode_ = ScanMode::STORAGE_ONLY;
      session_stats_.storage_only_cid_cnt_++;
    } else if (is_fill_leader) {
      mode_ = ScanMode::FILL;
      cid_fill_leader_ = true;
      session_stats_.fill_cid_cnt_++;
      building_cluster_.cid_ = new_cid;
      building_cluster_.index_epoch_ = cluster_cache_->get_index_epoch();
      building_cluster_.heat_.access_cnt_ = 1;
      building_cluster_.heat_.last_access_us_ = ObTimeUtility::current_time();
      void *buf = ob_malloc(sizeof(ObArenaAllocator), ObMemAttr(MTL_ID(), "IvfCidClu"));
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        cluster_cache_->finish_cid_fill(new_cid);
        cid_fill_leader_ = false;
      } else {
        building_cluster_.arena_ = new (buf) ObArenaAllocator(ObMemAttr(MTL_ID(), "IvfCidClu"));
      }
    } else {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("unexpected acquire result", K(ret), K(new_cid));
    }
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::on_cid_switch(uint64_t new_cid)
{
  int ret = OB_SUCCESS;
  if (mode_ == ScanMode::PASSTHROUGH) {
  } else if (!needs_lightweight_cid_switch() && OB_FAIL(flush_building_cluster())) {
    LOG_WARN("failed to flush building cluster", K(ret));
  } else {
    release_replay_entry();
    current_cid_ = new_cid;
    clear_materialized_batch();
    cid_fill_leader_ = false;
    session_stats_.cid_switch_cnt_++;
    if (OB_FAIL(acquire_cid_and_set_mode(new_cid))) {
      LOG_WARN("failed to acquire cid and set mode", K(ret), K(new_cid));
    }
    maybe_log_progress_stats(new_cid);
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::rescan()
{
  int ret = OB_SUCCESS;
  if (mode_ == ScanMode::PASSTHROUGH) {
    ret = delegate_base_rescan();
  } else {
    uint64_t cid = 0;
    if (OB_FAIL(parse_current_cid(cid))) {
      LOG_WARN("failed to parse cid", K(ret));
    } else if (OB_FAIL(on_cid_switch(cid))) {
      LOG_WARN("failed on cid switch", K(ret), K(cid));
    } else if (scan_delegates_to_base_no_cache() || mode_ == ScanMode::FILL) {
      ret = ensure_storage_scan();
      if (OB_FAIL(ret)) {
        LOG_WARN("failed to ensure storage scan", K(ret));
      }
    }
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::do_table_scan()
{
  int ret = OB_SUCCESS;
  if (mode_ == ScanMode::PASSTHROUGH) {
    ret = delegate_base_do_table_scan();
  } else {
    uint64_t cid = 0;
    if (OB_FAIL(parse_current_cid(cid))) {
      LOG_WARN("failed to parse cid", K(ret));
    } else if (OB_FAIL(on_cid_switch(cid))) {
      LOG_WARN("failed on cid switch", K(ret), K(cid));
    } else if (scan_delegates_to_base_no_cache() || mode_ == ScanMode::FILL) {
      ret = ensure_storage_scan();
      if (OB_FAIL(ret)) {
        LOG_WARN("failed to ensure storage scan", K(ret));
      }
    }
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::materialize_row_to_eval(const share::ObIvfCidClusterRow &row, int64_t batch_idx)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(cid_vec_ctdef_) || OB_ISNULL(eval_ctx_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const ExprFixedArray &out = cid_vec_ctdef_->result_output_;
    ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx_);
    guard.set_batch_idx(batch_idx);
    ObDatum &payload_datum = out.at(1)->locate_datum_for_write(*eval_ctx_);
    payload_datum.set_string(row.payload_, row.payload_len_);
    int64_t rowkey_idx = 0;
    for (int64_t i = 2; OB_SUCC(ret) && i < out.count() && rowkey_idx < row.rowkey_.get_obj_cnt(); ++i) {
      ObDatum &d = out.at(i)->locate_datum_for_write(*eval_ctx_);
      if (OB_FAIL(d.from_obj(row.rowkey_.get_obj_ptr()[rowkey_idx++], out.at(i)->obj_datum_map_))) {
        LOG_WARN("failed to set rowkey datum", K(ret));
      }
    }
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::append_fill_row(int64_t batch_idx)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(cid_vec_ctdef_) || OB_ISNULL(eval_ctx_) || OB_ISNULL(building_cluster_.arena_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx_);
    guard.set_batch_idx(batch_idx);
    const ExprFixedArray &out = cid_vec_ctdef_->result_output_;
    share::ObIvfCidClusterRow row;
    row.payload_type_ = payload_type();
    ObExpr *payload_expr = out.at(1);
    ObString payload = payload_expr->locate_expr_datum(*eval_ctx_).get_string();
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(
            building_cluster_.arena_,
            ObLongTextType,
            CS_TYPE_BINARY,
            payload_expr->obj_meta_.has_lob_header(),
            payload))) {
      LOG_WARN("failed to read real string data", K(ret));
    }
    char *payload_buf = nullptr;
    ObObj *obj_buf = nullptr;
    int64_t rowkey_cnt = out.count() - 2;
    if (OB_FAIL(ret)) {
    } else if (payload.length() > 0
               && OB_ISNULL(payload_buf = static_cast<char *>(building_cluster_.arena_->alloc(payload.length())))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (payload.length() > 0) {
      MEMCPY(payload_buf, payload.ptr(), payload.length());
      row.payload_ = payload_buf;
      row.payload_len_ = payload.length();
      ObDatum &payload_datum = payload_expr->locate_datum_for_write(*eval_ctx_);
      payload_datum.set_string(row.payload_, row.payload_len_);
    }
    if (OB_FAIL(ret)) {
    } else if (rowkey_cnt > 0
               && OB_ISNULL(obj_buf = static_cast<ObObj *>(building_cluster_.arena_->alloc(sizeof(ObObj) * rowkey_cnt)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      int64_t rk = 0;
      for (int64_t i = 2; OB_SUCC(ret) && i < out.count(); ++i) {
        if (OB_FAIL(out.at(i)->locate_expr_datum(*eval_ctx_).to_obj(obj_buf[rk++], out.at(i)->obj_meta_, out.at(i)->obj_datum_map_))) {
          LOG_WARN("failed to copy rowkey obj", K(ret));
        }
      }
      if (OB_SUCC(ret)) {
        row.rowkey_.assign(obj_buf, rk);
        if (OB_FAIL(building_cluster_.rows_.push_back(row))) {
          LOG_WARN("failed to push row", K(ret));
        } else {
          building_cluster_.row_count_++;
          building_cluster_.entry_bytes_ += row.payload_len_ + ob_ivf_cid_rowkey_storage_bytes(obj_buf, rk);
          session_stats_.fill_row_cnt_++;
        }
      }
    }
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::replay_one_row()
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(replay_entry_) || replay_idx_ >= replay_entry_->rows_.count()) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(materialize_row_to_eval(replay_entry_->rows_.at(replay_idx_++), 0))) {
    LOG_WARN("failed to materialize replay row", K(ret));
  } else {
    session_stats_.replay_row_cnt_++;
    mark_materialized_batch(1);
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::replay_rows(int64_t &count, int64_t capacity)
{
  int ret = OB_SUCCESS;
  count = 0;
  while (OB_SUCC(ret) && count < capacity) {
    if (OB_ISNULL(replay_entry_) || replay_idx_ >= replay_entry_->rows_.count()) {
      ret = OB_ITER_END;
    } else if (OB_FAIL(materialize_row_to_eval(replay_entry_->rows_.at(replay_idx_++), count))) {
      LOG_WARN("failed to materialize replay row", K(ret));
    } else {
      if (count < MATERIALIZED_BATCH_CAP) {
        materialized_batch_[count] = 1;
        materialized_batch_cnt_ = count + 1;
      }
      count++;
      session_stats_.replay_row_cnt_++;
    }
  }
  if (ret == OB_ITER_END && count > 0) {
    ret = OB_SUCCESS;
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::inner_get_next_row()
{
  int ret = OB_SUCCESS;
  clear_materialized_batch();
  if (scan_delegates_to_base_no_cache()) {
    ret = delegate_base_inner_get_next_row();
  } else if (mode_ == ScanMode::REPLAY) {
    const int64_t t0 = ObTimeUtility::current_time();
    ret = replay_one_row();
    session_stats_.replay_serve_us_ += ObTimeUtility::current_time() - t0;
  } else {
    const int64_t t0 = ObTimeUtility::current_time();
    if (OB_FAIL(ObDASScanIter::inner_get_next_row())) {
      if (ret != OB_ITER_END) {
        LOG_WARN("failed to get next row from storage", K(ret));
      } else if (OB_FAIL(flush_building_cluster())) {
        LOG_WARN("failed to flush cluster on end", K(ret));
      }
    } else {
      const int64_t t1 = ObTimeUtility::current_time();
      if (OB_FAIL(append_fill_row())) {
        LOG_WARN("failed to append fill row", K(ret));
        if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
          cluster_cache_->finish_cid_fill(current_cid_);
          cid_fill_leader_ = false;
        }
      } else {
        mark_materialized_batch(1);
      }
      session_stats_.fill_append_us_ += ObTimeUtility::current_time() - t1;
    }
    session_stats_.storage_fetch_us_ += ObTimeUtility::current_time() - t0;
  }
  return ret;
}

int ObDASIvfCidVecCacheScanIter::inner_get_next_rows(int64_t &count, int64_t capacity)
{
  int ret = OB_SUCCESS;
  clear_materialized_batch();
  if (scan_delegates_to_base_no_cache()) {
    ret = delegate_base_inner_get_next_rows(count, capacity);
  } else if (mode_ == ScanMode::REPLAY) {
    const int64_t t0 = ObTimeUtility::current_time();
    ret = replay_rows(count, capacity);
    session_stats_.replay_serve_us_ += ObTimeUtility::current_time() - t0;
  } else {
    const int64_t t0 = ObTimeUtility::current_time();
    ret = ObDASScanIter::inner_get_next_rows(count, capacity);
    const bool is_end = (ret == OB_ITER_END);
    if (OB_FAIL(ret) && !is_end) {
      LOG_WARN("failed to get next rows from storage", K(ret));
      if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
        cluster_cache_->finish_cid_fill(current_cid_);
        cid_fill_leader_ = false;
      }
    } else {
      // Match replay_rows / storage contract: only promote OB_ITER_END to OB_SUCCESS when
      // this batch has rows. OB_ITER_END with count==0 must reach the parent as end-of-scan.
      if (is_end && count > 0) {
        ret = OB_SUCCESS;
      }
      session_stats_.storage_fetch_us_ += ObTimeUtility::current_time() - t0;
      const int64_t t1 = ObTimeUtility::current_time();
      for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
        if (OB_FAIL(append_fill_row(i))) {
          LOG_WARN("failed to append fill row in batch", K(ret), K(i));
        }
      }
      session_stats_.fill_append_us_ += ObTimeUtility::current_time() - t1;
      if (OB_SUCC(ret) && count > 0) {
        mark_materialized_batch(count);
      }
      if (is_end) {
        if (building_cluster_.row_count_ > 0) {
          if (OB_FAIL(flush_building_cluster())) {
            LOG_WARN("failed to flush cluster on end", K(ret));
          }
        } else if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
          cluster_cache_->finish_cid_fill(current_cid_);
          cid_fill_leader_ = false;
        }
        if (OB_SUCC(ret) && count == 0) {
          ret = OB_ITER_END;
        }
      }
      if (OB_FAIL(ret) && cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
        cluster_cache_->finish_cid_fill(current_cid_);
        cid_fill_leader_ = false;
      }
    }
  }
  return ret;
}

void ObDASIvfCidVecCacheScanIter::reset_per_query_session_stats()
{
  if (cid_fill_leader_ && OB_NOT_NULL(cluster_cache_)) {
    cluster_cache_->finish_cid_fill(current_cid_);
    cid_fill_leader_ = false;
  }
  release_replay_entry();
  discard_building_cluster();
  replay_idx_ = 0;
  clear_materialized_batch();
  session_stats_.reset();
  if (cache_was_active_) {
    mode_ = ScanMode::FILL;
  }
}

void ObDASIvfCidVecCacheScanIter::export_log_snapshot(share::ObIvfCidClusterCacheLogSnapshot &out) const
{
  out.reset();
  out.valid_ = cache_was_active_;
  out.index_epoch_ = OB_NOT_NULL(cluster_cache_) ? cluster_cache_->get_index_epoch() : 0;
  out.algorithm_type_ = static_cast<int64_t>(algo_);
  out.session_ = session_stats_;
  out.cluster_cache_ = cluster_cache_;
  out.cache_active_ = cache_was_active_;
}

void ob_das_ivf_reset_cid_cluster_cache_session_stats(ObDASScanIter *cid_vec_iter)
{
  if (OB_ISNULL(cid_vec_iter)) {
  } else {
    ObDASIvfCidVecCacheScanIter *cache_iter = dynamic_cast<ObDASIvfCidVecCacheScanIter *>(cid_vec_iter);
    if (OB_NOT_NULL(cache_iter)) {
      cache_iter->reset_per_query_session_stats();
    }
  }
}

bool ob_das_ivf_try_export_cid_cluster_cache_snapshot(
    ObDASScanIter *cid_vec_iter,
    share::ObIvfCidClusterCacheLogSnapshot &out)
{
  out.reset();
  if (OB_ISNULL(cid_vec_iter)) {
    return false;
  }
  ObDASIvfCidVecCacheScanIter *cache_iter = dynamic_cast<ObDASIvfCidVecCacheScanIter *>(cid_vec_iter);
  if (OB_ISNULL(cache_iter)) {
    return false;
  }
  cache_iter->export_log_snapshot(out);
  return true;
}

} // namespace sql
} // namespace oceanbase
