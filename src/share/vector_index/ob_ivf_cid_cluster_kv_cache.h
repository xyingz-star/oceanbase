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

#ifndef OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_KV_CACHE_H_
#define OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_KV_CACHE_H_

#include "share/cache/ob_kv_storecache.h"
#include "share/vector_index/ob_ivf_cid_cluster_cache.h"

namespace oceanbase
{
namespace share
{

extern const char *IVF_CID_KV_CACHE_NAME;

/// ICFL payload layout (follows ObIvfCidClusterFlatValue in KV memblock; REPLAY reads via FlatValue::buf()).
struct ObIvfCidFlatHeader
{
  static const uint32_t MAGIC = 0x4943464C; // "ICFL"
  static const uint32_t VERSION = 1;
  uint32_t magic_;
  uint32_t version_;
  uint64_t index_epoch_;
  uint64_t cid_;
  int64_t row_count_;
  int64_t entry_bytes_;
  uint64_t access_cnt_;
  uint64_t replay_cnt_;
  uint64_t fill_cnt_;
  int64_t last_access_us_;
  /// Total bytes of this flat blob (set at FILL encode; used on KV GET).
  int64_t flat_buf_len_;
  uint8_t payload_type_;
  /// reserved_[0]: 1 if payloads_l2_unit_ was probed at FILL; reserved_[1]: 1 if first payload is L2 unit.
  uint8_t reserved_[7];
};

class ObIvfCidClusterKVKey : public common::ObIKVCacheKey
{
public:
  ObIvfCidClusterKVKey();
  ObIvfCidClusterKVKey(const ObIvfCidClusterCacheMgrKey &mgr_key, const uint64_t cid);
  virtual bool operator==(const common::ObIKVCacheKey &other) const override;
  virtual uint64_t hash() const override;
  virtual uint64_t get_tenant_id() const override;
  virtual int64_t size() const override;
  virtual int deep_copy(char *buf, const int64_t buf_len, common::ObIKVCacheKey *&key) const override;
  bool is_valid() const;
  TO_STRING_KV(K_(mgr_key), K_(cid));
  ObIvfCidClusterCacheMgrKey mgr_key_;
  uint64_t cid_;
};

/// KV memblock layout: [ ObIvfCidClusterFlatValue | aligned pad | ICFL blob ].
/// buf_ points at ICFL inside the same memblock after deep_copy; stack temps use external buf_ at put only.
class ObIvfCidClusterFlatValue : public common::ObIKVCacheValue
{
public:
  ObIvfCidClusterFlatValue();
  explicit ObIvfCidClusterFlatValue(const char *buf, const int64_t buf_len);
  const char *buf() const { return buf_; }
  int64_t buf_len() const { return buf_len_; }
  virtual int64_t size() const override;
  virtual int deep_copy(char *buf, const int64_t buf_len, common::ObIKVCacheValue *&value) const override;
  TO_STRING_KV(KP_(buf), K_(buf_len));
private:
  const char *buf_;
  int64_t buf_len_;
};

/// Total KV value bytes for an ICFL blob of icfl_len (FlatValue header + aligned ICFL).
int64_t ivf_cid_flat_value_storage_size(const int64_t icfl_len);

class ObIvfCidClusterKVCache : public common::ObKVCache<ObIvfCidClusterKVKey, ObIvfCidClusterFlatValue>
{
public:
  int init_cache();
  int put_flat(const ObIvfCidClusterKVKey &key, const char *flat_buf, const int64_t flat_len, const bool overwrite);
  /// On hit, flat_buf points at ICFL bytes (FlatValue::buf() in memblock; legacy raw ICFL still supported).
  int get_flat(const ObIvfCidClusterKVKey &key,
      const char *&flat_buf,
      int64_t &flat_len,
      common::ObKVCacheHandle &handle);
  int erase_key(const ObIvfCidClusterKVKey &key);
};

ObIvfCidClusterKVCache &get_ivf_cid_cluster_kv_cache();

int ivf_cid_flat_encode_entry(const ObIvfCidClusterEntry &entry, char *&out_buf, int64_t &out_len);
void ivf_cid_flat_free_buf(char *buf);

/// Incremental FILL builder: append rows into one data buffer; finalize does header + one data memcpy.
struct ObIvfCidFlatFillState
{
  ObIvfCidFlatFillState();
  char *data_buf_;
  int64_t data_cap_;
  int64_t data_len_;
  common::ObSEArray<int64_t, 64> payload_off_;
  common::ObSEArray<int32_t, 64> payload_len_;
  common::ObSEArray<int64_t, 64> rk_off_;
  common::ObSEArray<int32_t, 64> rk_len_;
};

int ivf_cid_flat_fill_create(ObIvfCidFlatFillState *&out_state);
void ivf_cid_flat_fill_destroy(ObIvfCidFlatFillState *state);
int ivf_cid_flat_fill_append_row(ObIvfCidFlatFillState *state,
    const ObIvfCidClusterPayloadType payload_type,
    const char *payload,
    const int32_t payload_len,
    const common::ObRowkey &rowkey);
int ivf_cid_flat_fill_finalize(const ObIvfCidClusterEntry &entry,
    const ObIvfCidFlatFillState *state,
    char *&out_buf,
    int64_t &out_len);
bool ivf_cid_flat_fill_probe_first_payload_l2_unit(const ObIvfCidFlatFillState *state, bool &is_unit);

/// Probe first FLAT_FLOAT payload (same ||v||^2 threshold as ObVectorNormalize::L2_normalize_vector).
bool ivf_cid_probe_payloads_l2_unit(const ObIvfCidClusterEntry &entry, bool &is_unit, bool &known);

/// Build a view entry: payload/rowkey payload point into KV flat buffer; rowkey ObObj shells in rowkey_objs_.
int ivf_cid_flat_attach_entry(const char *flat_buf,
    const int64_t flat_len,
    ObIvfCidClusterEntry *&out_entry,
    common::ObObj *&out_rowkey_objs,
    int64_t &out_rowkey_obj_cnt);

/// REPLAY cursor: shell entry only (header fields + kv_flat_buf_; rows_ empty). No bulk attach.
int ivf_cid_flat_open_replay_entry(const char *flat_buf,
    const int64_t flat_len,
    ObIvfCidClusterEntry *&out_entry);

/// Materialize one row from flat blob; vector payload is zero-copy; rowkey decoded into rk_scratch.
int ivf_cid_flat_replay_row_at(const char *flat_buf,
    const int64_t flat_len,
    const int64_t row_idx,
    ObIvfCidClusterRow &out_row,
    common::ObObj *rk_scratch,
    const int64_t rk_scratch_cap);

} // namespace share
} // namespace oceanbase

#endif /* OCEANBASE_SHARE_VECTOR_INDEX_OB_IVF_CID_CLUSTER_KV_CACHE_H_ */
