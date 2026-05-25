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

#ifndef OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_CACHE_H_
#define OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_CACHE_H_

#include "lib/container/ob_array.h"
#include "lib/hash/ob_hashmap.h"
#include "lib/lock/ob_mutex.h"
#include "share/ob_define.h"
#include "common/ob_tablet_id.h"
#include "share/cache/ob_kvcache_struct.h"
#include "share/vector_index/ob_vector_index_util.h"
#include <cstdio>

namespace oceanbase
{
namespace common
{
class ObKVCacheHandle;
}
}

namespace oceanbase
{
namespace share
{

enum ObIvfCidClusterPayloadType : uint8_t
{
  IVF_CID_CLUSTER_PAYLOAD_FLAT_FLOAT = 0,
  IVF_CID_CLUSTER_PAYLOAD_SQ8_U8 = 1,
  IVF_CID_CLUSTER_PAYLOAD_PQ_IDS = 2,
};

struct ObIvfCidClusterCacheMgrKey
{
  ObIvfCidClusterCacheMgrKey()
    : tenant_id_(OB_INVALID_TENANT_ID),
      index_tablet_id_(),
      cid_vec_table_id_(OB_INVALID_ID),
      data_table_id_(OB_INVALID_ID),
      algorithm_type_(ObVectorIndexAlgorithmType::VIAT_MAX)
  {}
  uint64_t tenant_id_;
  common::ObTabletID index_tablet_id_;
  uint64_t cid_vec_table_id_;
  uint64_t data_table_id_;
  ObVectorIndexAlgorithmType algorithm_type_;
  uint64_t hash() const;
  int hash(uint64_t &hash_val) const { hash_val = hash(); return OB_SUCCESS; }
  bool operator==(const ObIvfCidClusterCacheMgrKey &o) const;
  bool is_valid() const { return tenant_id_ != OB_INVALID_TENANT_ID && index_tablet_id_.is_valid(); }
  TO_STRING_KV(K_(tenant_id), K_(index_tablet_id), K_(cid_vec_table_id), K_(data_table_id), K_(algorithm_type));
};

struct ObIvfCidClusterHeat
{
  ObIvfCidClusterHeat()
    : access_cnt_(0), replay_cnt_(0), fill_cnt_(0), last_access_us_(0) {}
  uint64_t access_cnt_;
  uint64_t replay_cnt_;
  uint64_t fill_cnt_;
  int64_t last_access_us_;
};

struct ObIvfCidClusterRow
{
  ObIvfCidClusterRow()
    : payload_type_(IVF_CID_CLUSTER_PAYLOAD_FLAT_FLOAT), payload_len_(0), payload_(nullptr) {}
  common::ObRowkey rowkey_;
  ObIvfCidClusterPayloadType payload_type_;
  int32_t payload_len_;
  const char *payload_;
  TO_STRING_KV(K_(payload_type), K_(payload_len), K_(rowkey));
};

struct ObIvfCidClusterEntry;
struct ObIvfCidFillGate;
struct ObIvfCidClusterCacheSessionStats;
struct IvfPinFreeFn;
struct IvfPinInvalidateFn;

/// Cache visibility phase (stored in atomic cache_phase_).
enum class ObIvfCidClusterCachePhase : uint8_t
{
  LEARNING = 0,
  REFRESHING = 1,
  SERVING = 2,
};

/// Per-CID lookup outcome exposed to DAS: hit (REPLAY), miss (storage), or fill leader (REFRESHING only).
enum class ObIvfCidClusterLookupResult : int8_t
{
  HIT = 0,
  MISS = 1,
  FILL_LEADER = 2,
};

/// Cumulative stats on one ObIvfCidClusterCache instance (cross-query, thread-safe).
struct ObIvfCidClusterCacheStats
{
  void reset();
  int64_t get_hit_cnt_;
  int64_t get_miss_cnt_;
  int64_t get_stale_epoch_cnt_;
  int64_t put_ok_cnt_;
  int64_t put_fail_cnt_;
  int64_t put_skip_rows_limit_cnt_;
  int64_t put_skip_pinned_cnt_;
  int64_t put_skip_cap_cnt_;
  int64_t fill_wait_cnt_;
  int64_t fill_bypass_cnt_;
  int64_t evict_entry_cnt_;
  int64_t evict_bytes_;
  int64_t invalidate_cnt_;
  int64_t phase_learning_cnt_;
  int64_t phase_refreshing_cnt_;
  int64_t phase_serving_cnt_;
  /// Maintained with ATOMIC_INC/DEC on put/erase; read via get_stats() without lock_.
  int64_t cur_entry_cnt_;
  int64_t cur_bytes_;
};

struct ObIvfCidClusterLedgerRecord
{
  ObIvfCidClusterLedgerRecord() : bytes_(0), heat_() {}
  int64_t bytes_;
  ObIvfCidClusterHeat heat_;
};

class ObIvfCidClusterCache
{
  friend void release_ivf_cid_cluster_cache(ObIvfCidClusterCache *cache);
  friend struct IvfPinFreeFn;
  friend struct IvfPinInvalidateFn;

public:
  ObIvfCidClusterCache();
  ~ObIvfCidClusterCache();
  int init(const ObIvfCidClusterCacheMgrKey &key, int64_t max_bytes, int64_t max_rows_per_cid, int64_t replay_heat_weight);
  void destroy();
  /// Bump index epoch and drop all cluster entries (call after IVF index rebuild).
  void invalidate_all();
  uint64_t get_index_epoch() const { return load_index_epoch_(); }
  int64_t get_max_bytes() const { return max_bytes_; }
  ObIvfCidClusterCachePhase get_cache_phase() const { return load_phase_(); }
  bool is_cache_serving() const { return load_phase_() == ObIvfCidClusterCachePhase::SERVING; }
  /// Query entry: LEARNING/REFRESHING => MISS or FILL_LEADER; SERVING => readonly HIT/MISS.
  int lookup_cid(uint64_t cid,
      ObIvfCidClusterEntry *&entry,
      ObIvfCidClusterLookupResult &result,
      ObIvfCidClusterCacheSessionStats *session_stats = nullptr);
  /// After FILL put during REFRESHING; may transition to SERVING when targets are loaded.
  void notify_refresh_put_done(uint64_t cid);
  /// Leader calls after FILL put (or on error) to wake waiters for this cid.
  void finish_cid_fill(uint64_t cid);
  int put(ObIvfCidClusterEntry &entry);
  /// Release REPLAY view (session-owned or borrowed readonly slot).
  void release_session_entry(ObIvfCidClusterEntry *entry);
  void inc_ref();
  void dec_ref();
  int64_t get_ref() const { return ref_cnt_; }
  void get_stats(ObIvfCidClusterCacheStats &out) const;
  void log_stats(const char *tag) const;

private:
  struct ObIvfCidClusterPinSlot
  {
    ObIvfCidClusterPinSlot() : view_entry_(nullptr), rowkey_objs_(nullptr) {}
    ObIvfCidClusterEntry *view_entry_;
    common::ObObj *rowkey_objs_;
    common::ObKVCacheHandle kv_handle_;
  };
  struct ObIvfKvPinPrep
  {
    ObIvfKvPinPrep()
      : flat_buf_(nullptr),
        flat_len_(0),
        view_entry_(nullptr),
        rowkey_objs_(nullptr),
        rowkey_obj_cnt_(0)
    {}
    const char *flat_buf_;
    int64_t flat_len_;
    ObIvfCidClusterEntry *view_entry_;
    common::ObObj *rowkey_objs_;
    int64_t rowkey_obj_cnt_;
    common::ObKVCacheHandle kv_handle_;
  };
  int load_kv_pin_prep_(uint64_t cid, uint64_t index_epoch, ObIvfKvPinPrep &prep);
  void discard_kv_pin_prep_(ObIvfKvPinPrep &prep);
  void reconcile_kv_pin_miss_locked_(uint64_t cid);
  void release_pin_slot_(ObIvfCidClusterPinSlot *slot);
  int ledger_remove_locked_(uint64_t cid, bool count_evict, bool erase_kv = true);
  int ledger_put_locked_(uint64_t cid, int64_t bytes, const ObIvfCidClusterHeat &heat);
  void bump_probe_heat_locked_(uint64_t cid);
  double heat_score_(const ObIvfCidClusterHeat &heat, int64_t byte_denom) const;
  double prospective_heat_score_locked_(uint64_t cid) const;
  bool hotter_than_coldest_locked_(uint64_t skip_cid, double incoming_score) const;
  void release_dormant_pin_shard_locked_(uint64_t cid, ObIvfCidClusterPinSlot *slot);
  void clear_probe_heat_locked_(uint64_t cid);
  bool find_coldest_evictable_ledger_locked_(uint64_t skip_cid, double &min_score, uint64_t &victim_cid) const;
  int ledger_evict_until_locked_(uint64_t skip_cid, int64_t need_bytes);
  int ensure_ledger_room_locked_(uint64_t skip_cid, int64_t need_bytes, double incoming_score);
  int install_pin_slot_locked_(uint64_t cid,
      ObIvfCidClusterEntry *view_entry,
      common::ObObj *rowkey_objs,
      common::ObKVCacheHandle &kv_handle);
  int warm_pin_slot_(uint64_t cid);
  void clear_all_fill_gates_();
  static const int64_t IVF_CID_PIN_MAP_SHARD_CNT = 64;
  int64_t pin_shard_idx_(uint64_t cid) const;
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterPinSlot *> &pin_map_shard_(uint64_t cid);
  const common::hash::ObHashMap<uint64_t, ObIvfCidClusterPinSlot *> &pin_map_shard_(uint64_t cid) const;
  lib::ObMutex &pin_shard_lock_(uint64_t cid);
  const lib::ObMutex &pin_shard_lock_(uint64_t cid) const;
  int ledger_remove_(uint64_t cid, bool count_evict, bool erase_kv = true);
  int lookup_cid_refreshing_(uint64_t cid,
      ObIvfCidClusterEntry *&entry,
      ObIvfCidClusterLookupResult &result,
      ObIvfCidClusterCacheSessionStats *session_stats);
  int lookup_readonly_serving_(uint64_t cid, ObIvfCidClusterEntry *&entry);
  void record_access_(uint64_t cid, ObIvfCidClusterCachePhase phase);
  void try_begin_refresh_from_learning_();
  void try_begin_refresh_from_serving_();
  bool cas_phase_(ObIvfCidClusterCachePhase expected, ObIvfCidClusterCachePhase desired);
  void store_phase_(ObIvfCidClusterCachePhase phase);
  ObIvfCidClusterCachePhase load_phase_() const;
  int64_t learning_start_threshold_() const;
  void prepare_refresh_targets_(bool use_ledger_heat = false);
  void finalize_refresh_to_serving_();
  int warm_readonly_pin_slots_();
  uint64_t load_index_epoch_() const;
  void try_finalize_refresh_if_complete_();
  bool is_refresh_target_cid_(uint64_t cid) const;
  void mark_refresh_target_done_(uint64_t cid);
  bool inited_;
  ObIvfCidClusterCacheMgrKey mgr_key_;
  uint64_t current_index_epoch_;
  int64_t max_bytes_;
  int64_t max_rows_per_cid_;
  int64_t replay_heat_weight_;
  int64_t ledger_bytes_;
  int64_t ref_cnt_;
  int64_t cache_phase_;
  int64_t total_access_samples_;
  int64_t serving_miss_samples_;
  int64_t refresh_targets_remaining_;
  common::hash::ObHashMap<uint64_t, int8_t> refresh_target_map_;
  ObIvfCidClusterCacheStats stats_;
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterPinSlot *> pin_map_shards_[IVF_CID_PIN_MAP_SHARD_CNT];
  lib::ObMutex *pin_shard_locks_[IVF_CID_PIN_MAP_SHARD_CNT];
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterLedgerRecord> ledger_map_;
  common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *> fill_gates_;
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterHeat> probe_heat_map_;
  mutable lib::ObMutex ledger_lock_;
  mutable lib::ObMutex probe_heat_lock_;
  mutable lib::ObMutex fill_gates_lock_;
};

struct ObIvfCidClusterEntry
{
  ObIvfCidClusterEntry()
    : index_epoch_(0),
      cid_(0),
      row_count_(0),
      entry_bytes_(0),
      heat_(),
      rows_(),
      arena_(nullptr),
      kv_flat_buf_(nullptr),
      rowkey_objs_(nullptr),
      payloads_l2_unit_known_(false),
      payloads_l2_unit_(false),
      session_owned_(false),
      session_borrowed_(false),
      session_kv_handle_()
  {}
  ~ObIvfCidClusterEntry();
  bool is_kv_view() const { return OB_ISNULL(arena_) && OB_NOT_NULL(kv_flat_buf_); }
  uint64_t index_epoch_;
  uint64_t cid_;
  int64_t row_count_;
  int64_t entry_bytes_;
  ObIvfCidClusterHeat heat_;
  common::ObArray<ObIvfCidClusterRow> rows_;
  common::ObArenaAllocator *arena_;
  /// REPLAY view: payload pointers into this KV flat buffer.
  const char *kv_flat_buf_;
  common::ObObj *rowkey_objs_;
  /// FILL-time probe (stored in flat header): skip per-query first-row L2 probe on REPLAY when known.
  bool payloads_l2_unit_known_;
  bool payloads_l2_unit_;
  /// SERVING readonly: query owns KV handle + entry shell.
  bool session_owned_;
  /// SERVING readonly: borrows immutable pin_map view (no unpin).
  bool session_borrowed_;
  common::ObKVCacheHandle session_kv_handle_;
};

/// Per-query stats on ObDASIvfCidVecCacheScanIter (printed in inner_release when enabled).
struct ObIvfCidClusterCacheSessionStats
{
  void reset();
  int64_t fill_cid_cnt_;
  int64_t replay_cid_cnt_;
  int64_t replay_row_cnt_;
  int64_t fill_row_cnt_;
  int64_t put_cid_ok_cnt_;
  int64_t put_cid_fail_cnt_;
  int64_t put_cid_skip_pinned_cnt_;
  int64_t put_rows_total_;
  int64_t storage_only_cid_cnt_;
  /// Total cid switches (on_cid_switch); includes storage_only + replay + fill.
  int64_t cid_switch_cnt_;
  /// Time in lookup_cid (us).
  int64_t acquire_us_;
  /// Time blocked on fill gate (us) and number of cond waits.
  int64_t fill_wait_us_;
  int64_t fill_wait_cnt_;
  /// Storage scan in STORAGE_ONLY / FILL (ObDASScanIter::get_next_* us).
  int64_t storage_fetch_us_;
  /// Serving rows from cache REPLAY (us).
  int64_t replay_serve_us_;
  /// append_fill_row in FILL mode (us); STORAGE_ONLY append is included in storage_fetch path.
  int64_t fill_append_us_;
  /// Snapshot of cache cumulative stats at query start (for per-query delta).
  bool has_cache_snap_begin_;
  ObIvfCidClusterCacheStats cache_snap_begin_;
};

/// Snapshot for one IVF query (paired with OB_IVF_LATENCY_BREAKDOWN log).
struct ObIvfCidClusterCacheLogSnapshot
{
  void reset();
  bool valid_;
  bool cache_active_;
  uint64_t index_epoch_;
  int64_t algorithm_type_;
  ObIvfCidClusterCacheSessionStats session_;
  ObIvfCidClusterCache *cluster_cache_;
};

int acquire_ivf_cid_cluster_cache(const ObIvfCidClusterCacheMgrKey &key, ObIvfCidClusterCache *&cache);
void release_ivf_cid_cluster_cache(ObIvfCidClusterCache *cache);
/// Invalidate all cid cluster caches for one vector index tablet (e.g. after IVF rebuild).
void invalidate_ivf_cid_cluster_cache_for_index_tablet(uint64_t tenant_id, const common::ObTabletID &index_tablet_id);
bool is_ivf_cid_cluster_cache_enabled();
bool is_ivf_cid_cluster_cache_stats_enabled();
int64_t ivf_cid_cluster_cache_fill_min_probe_access();
/// When STATS=1: default 1 (progress every cid + final). EVERY_N_CID=0 or STATS_LIVE=0 disables progress.
int64_t ivf_cid_cluster_cache_stats_every_n_cid();
void log_ivf_cid_cluster_cache_session_stats(
    const ObIvfCidClusterCacheSessionStats &session,
    uint64_t index_epoch,
    int64_t algorithm_type,
    const ObIvfCidClusterCache *cache);
void log_ivf_cid_cluster_cache_snapshot_to_observer(const ObIvfCidClusterCacheLogSnapshot &snap);
/// Append stats to $HOME/log (or OB_IVF_CID_CLUSTER_CACHE_LOG_FILE / OB_IVF_CID_CLUSTER_CACHE_LOG_DIR).
void ob_ivf_cid_cluster_cache_write_user_log_file(
    const ObIvfCidClusterCacheLogSnapshot &snap,
    int64_t dataset_rows = 0,
    int64_t dim = 0,
    int64_t nlist = 0,
    const char *phase = "final",
    uint64_t current_cid = 0,
    const char *current_mode = nullptr);
/// Append [OB_IVF_CID_CLUSTER_CACHE_STATS] line(s) to an open latency log file (same file as breakdown).
int ob_ivf_cid_cluster_cache_append_latency_log_file(
    FILE *fp,
    const ObIvfCidClusterCacheLogSnapshot &snap);

} // namespace share
} // namespace oceanbase

#endif /* OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_CACHE_H_ */
