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
#include "share/vector_index/ob_vector_index_util.h"
#include <cstdio>

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
  /// Maintained with ATOMIC_INC/DEC on put/erase; read via get_stats() without lock_.
  int64_t cur_entry_cnt_;
  int64_t cur_bytes_;
};

class ObIvfCidClusterCache
{
  friend void release_ivf_cid_cluster_cache(ObIvfCidClusterCache *cache);

public:
  struct EntryMapKey
  {
    EntryMapKey() : cid_(0) {}
    explicit EntryMapKey(uint64_t cid) : cid_(cid) {}
    uint64_t cid_;
    uint64_t hash() const { return cid_; }
    int hash(uint64_t &hash_val) const { hash_val = hash(); return OB_SUCCESS; }
    bool operator==(const EntryMapKey &o) const { return cid_ == o.cid_; }
  };

  ObIvfCidClusterCache();
  ~ObIvfCidClusterCache();
  int init(const ObIvfCidClusterCacheMgrKey &key, int64_t max_bytes, int64_t max_rows_per_cid, int64_t replay_heat_weight);
  void destroy();
  /// Bump index epoch and drop all cluster entries (call after IVF index rebuild).
  void invalidate_all();
  uint64_t get_index_epoch() const { return current_index_epoch_; }
  int64_t get_max_bytes() const { return max_bytes_; }
  /// Hit: pins entry. Miss: STORAGE_ONLY by default; FILL leader only if probe access >= min threshold,
  /// cid ranks in global probe top-K, and (cache has room or hotter than coldest evictable entry).
  int acquire_cid_cluster(uint64_t cid,
      ObIvfCidClusterEntry *&entry,
      bool &is_fill_leader,
      bool &storage_only,
      ObIvfCidClusterCacheSessionStats *session_stats = nullptr);
  /// Leader calls after FILL put (or on error) to wake waiters for this cid.
  void finish_cid_fill(uint64_t cid);
  int put(ObIvfCidClusterEntry &entry);
  int unpin_entry(ObIvfCidClusterEntry *entry);
  void inc_ref();
  void dec_ref();
  int64_t get_ref() const { return ref_cnt_; }
  void get_stats(ObIvfCidClusterCacheStats &out) const;
  void log_stats(const char *tag) const;

private:
  int try_pin_ready_entry_(uint64_t cid, ObIvfCidClusterEntry *&entry);
  void bump_probe_heat_locked_(uint64_t cid);
  double calc_probe_heat_score_(const ObIvfCidClusterHeat &heat) const;
  double calc_prospective_heat_score_(uint64_t cid) const;
  /// Among probe-qualified cids, true if fewer than top-K have strictly higher heat than target.
  bool is_probe_top_k_for_fill_locked_(uint64_t cid, double target_score) const;
  /// Returns true if this miss may become FILL leader (after bumping probe heat).
  bool should_admit_fill_locked_(uint64_t cid);
  void clear_probe_heat_locked_(uint64_t cid);
  int evict_until(int64_t need_bytes);
  double calc_heat_score(const ObIvfCidClusterEntry &e) const;
  int erase_entry(const EntryMapKey &key, bool count_evict);
  void clear_all_fill_gates_();
  bool inited_;
  ObIvfCidClusterCacheMgrKey mgr_key_;
  uint64_t current_index_epoch_;
  int64_t max_bytes_;
  int64_t max_rows_per_cid_;
  int64_t replay_heat_weight_;
  int64_t total_bytes_;
  int64_t ref_cnt_;
  ObIvfCidClusterCacheStats stats_;
  common::hash::ObHashMap<EntryMapKey, ObIvfCidClusterEntry *> entry_map_;
  common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *> fill_gates_;
  /// Access heat for cids not yet in entry_map_ (used for fill vs storage-only bypass).
  common::hash::ObHashMap<uint64_t, ObIvfCidClusterHeat> probe_heat_map_;
  mutable lib::ObMutex lock_;
  mutable lib::ObMutex fill_gates_lock_;
};

struct ObIvfCidClusterEntry
{
  ObIvfCidClusterEntry()
    : index_epoch_(0), cid_(0), row_count_(0), entry_bytes_(0), pin_cnt_(0), heat_(), rows_(), arena_(nullptr) {}
  ~ObIvfCidClusterEntry();
  /// Index generation when this cluster snapshot was cached; must match cache current_index_epoch_.
  uint64_t index_epoch_;
  uint64_t cid_;
  int64_t row_count_;
  int64_t entry_bytes_;
  /// REPLAY in progress; evict skips while pin_cnt_ > 0.
  int64_t pin_cnt_;
  ObIvfCidClusterHeat heat_;
  common::ObArray<ObIvfCidClusterRow> rows_;
  common::ObArenaAllocator *arena_;
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
  /// Time in acquire_cid_cluster (us).
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
