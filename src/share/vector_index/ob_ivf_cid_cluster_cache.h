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
struct ObIvfCidFlatFillState;
struct ObIvfCidFillGate;
struct ObIvfCidClusterCacheSessionStats;

/// Per-CID lifecycle (atomic phase_ in cid_states_ after successful put).
enum class ObIvfCidClusterCidPhase : int8_t
{
  LEARNING = 0,
  LOADING = 1,
  READY = 2,
};

/// Per-CID lookup outcome exposed to DAS.
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
  int64_t invalidate_cnt_;
  int64_t cur_entry_cnt_;
  int64_t cur_bytes_;
};

/// Per-CID probe heat + cache footprint; all fields updated with atomics on the hot path.
struct ObIvfCidPerCidState
{
  uint64_t probe_access_;
  uint64_t replay_cnt_;
  int64_t cached_bytes_;
  int64_t phase_;
};

class ObIvfCidClusterCache
{
  friend void release_ivf_cid_cluster_cache(ObIvfCidClusterCache *cache);

public:
  ObIvfCidClusterCache();
  ~ObIvfCidClusterCache();
  int init(const ObIvfCidClusterCacheMgrKey &key, int64_t max_bytes, int64_t max_rows_per_cid, int64_t replay_heat_weight);
  void destroy();
  void invalidate_all();
  uint64_t get_index_epoch() const { return load_index_epoch_(); }
  int64_t get_max_bytes() const { return max_bytes_; }
  ObIvfCidClusterCidPhase get_cid_phase(uint64_t cid) const;
  /// Per-CID: HIT via ObKVCache only; else FILL_LEADER or MISS.
  /// reuse_shell: detached replay view entry to reuse across cid switches (avoids OB_NEW/OB_DELETE per probe).
  int lookup_cid(uint64_t cid,
      ObIvfCidClusterEntry *&entry,
      ObIvfCidClusterLookupResult &result,
      ObIvfCidClusterCacheSessionStats *session_stats = nullptr,
      ObIvfCidClusterEntry *reuse_shell = nullptr);
  void finish_cid_fill(uint64_t cid);
  int put(ObIvfCidClusterEntry &entry);
  void release_session_entry(ObIvfCidClusterEntry *entry);
  /// Drop KV pin but keep session-owned entry shell for lookup_cid(..., reuse_shell).
  void detach_session_replay_entry(ObIvfCidClusterEntry *entry);
  void inc_ref();
  void dec_ref();
  int64_t get_ref() const { return ref_cnt_; }
  void get_stats(ObIvfCidClusterCacheStats &out) const;
  void log_stats(const char *tag) const;

private:
  struct ObIvfKvEntryPrep
  {
    ObIvfKvEntryPrep()
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
  int load_kv_entry_prep_(uint64_t cid, uint64_t index_epoch, ObIvfKvEntryPrep &prep, ObIvfCidClusterEntry *reuse_shell = nullptr);
  void discard_kv_entry_prep_(ObIvfKvEntryPrep &prep);
  void ledger_drop_cached_(uint64_t cid, bool erase_kv = true);
  void ledger_commit_(uint64_t cid, int64_t bytes);
  bool probe_heat_meets_fill_threshold_(uint64_t cid) const;
  bool has_cache_space_(uint64_t cid, int64_t need_bytes) const;
  ObIvfCidPerCidState *cid_state_(uint64_t cid);
  const ObIvfCidPerCidState *cid_state_(uint64_t cid) const;
  void reset_all_cid_states_();
  void clear_all_fill_gates_();
  static const int64_t IVF_CID_FILL_GATE_SHARD_CNT = 64;
  int64_t fill_gate_shard_idx_(uint64_t cid) const;
  common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *> &fill_gate_map_shard_(uint64_t cid);
  const common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *> &fill_gate_map_shard_(uint64_t cid) const;
  lib::ObMutex &fill_gate_shard_lock_(uint64_t cid);
  const lib::ObMutex &fill_gate_shard_lock_(uint64_t cid) const;
  int ledger_remove_(uint64_t cid, bool erase_kv = true);
  int try_lookup_hit_(uint64_t cid, ObIvfCidClusterEntry *&entry, ObIvfCidClusterEntry *reuse_shell = nullptr);
  int try_acquire_fill_leader_(uint64_t cid,
      ObIvfCidClusterEntry *&entry,
      ObIvfCidClusterLookupResult &result,
      ObIvfCidClusterCacheSessionStats *session_stats);
  bool cid_fill_in_progress_(uint64_t cid);
  void record_access_(uint64_t cid);
  uint64_t load_index_epoch_() const;
  bool inited_;
  ObIvfCidClusterCacheMgrKey mgr_key_;
  uint64_t current_index_epoch_;
  int64_t max_bytes_;
  int64_t max_rows_per_cid_;
  int64_t replay_heat_weight_;
  int64_t max_cid_cnt_;
  int64_t ledger_bytes_total_;
  int64_t ref_cnt_;
  ObIvfCidPerCidState *cid_states_;
  ObIvfCidClusterCacheStats stats_;
  common::hash::ObHashMap<uint64_t, ObIvfCidFillGate *> fill_gate_map_shards_[IVF_CID_FILL_GATE_SHARD_CNT];
  lib::ObMutex *fill_gate_shard_locks_[IVF_CID_FILL_GATE_SHARD_CNT];
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
      session_kv_handle_(),
      flat_fill_(nullptr)
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
  const char *kv_flat_buf_;
  common::ObObj *rowkey_objs_;
  bool payloads_l2_unit_known_;
  bool payloads_l2_unit_;
  bool session_owned_;
  common::ObKVCacheHandle session_kv_handle_;
  ObIvfCidFlatFillState *flat_fill_;
};

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
  int64_t cid_switch_cnt_;
  int64_t acquire_us_;
  int64_t storage_fetch_us_;
  int64_t replay_serve_us_;
  int64_t fill_append_us_;
  /// REPLAY cids whose flat header says payloads are L2 unit (cache_unit=1, cid_vec_need_norm=0 at scan start).
  int64_t replay_unit_cid_cnt_;
  /// cache_zero_copy rows with will_norm=false (expected when replay_unit_cid).
  int64_t replay_skip_norm_row_cnt_;
  /// per-row MEMCPY(dim*4) before L2 in cache_zero_copy path.
  int64_t replay_norm_memcpy_row_cnt_;
  /// is_first_vec L2 probe (sets cid_vec_need_norm for rest of query).
  int64_t replay_first_vec_l2_probe_cnt_;
  /// per-row L2_normalize when cid_vec_need_norm (not first-vec probe).
  int64_t replay_norm_l2_row_cnt_;
  /// MEMCPY on a replay_unit_cid (should stay 0 if unit vectors skip re-norm).
  int64_t replay_unit_violation_memcpy_cnt_;
  /// per-row L2 on a replay_unit_cid (should stay 0).
  int64_t replay_unit_violation_l2_cnt_;
  bool has_cache_snap_begin_;
  ObIvfCidClusterCacheStats cache_snap_begin_;
};

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
void invalidate_ivf_cid_cluster_cache_for_index_tablet(uint64_t tenant_id, const common::ObTabletID &index_tablet_id);
bool is_ivf_cid_cluster_cache_enabled();
bool is_ivf_cid_cluster_cache_stats_enabled();
int64_t ivf_cid_cluster_cache_fill_min_probe_access();
int64_t ivf_cid_cluster_cache_max_cid();
int64_t ivf_cid_cluster_cache_stats_every_n_cid();
void log_ivf_cid_cluster_cache_session_stats(
    const ObIvfCidClusterCacheSessionStats &session,
    uint64_t index_epoch,
    int64_t algorithm_type,
    const ObIvfCidClusterCache *cache);
void log_ivf_cid_cluster_cache_snapshot_to_observer(const ObIvfCidClusterCacheLogSnapshot &snap);
void ob_ivf_cid_cluster_cache_write_user_log_file(
    const ObIvfCidClusterCacheLogSnapshot &snap,
    int64_t dataset_rows = 0,
    int64_t dim = 0,
    int64_t nlist = 0,
    const char *phase = "final",
    uint64_t current_cid = 0,
    const char *current_mode = nullptr);
int ob_ivf_cid_cluster_cache_append_latency_log_file(
    FILE *fp,
    const ObIvfCidClusterCacheLogSnapshot &snap);
/// 1 if this query served at least one CID from cluster cache REPLAY; 0 if inactive or all storage/FILL.
int ob_ivf_cid_cluster_cache_query_hit_flag(const ObIvfCidClusterCacheLogSnapshot &snap);

} // namespace share
} // namespace oceanbase

#endif /* OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_CACHE_H_ */
