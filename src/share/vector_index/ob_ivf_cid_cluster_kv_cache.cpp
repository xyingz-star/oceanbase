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

#include "share/vector_index/ob_ivf_cid_cluster_kv_cache.h"
#include "share/table/ob_table_object.h"
#include "share/vector_type/ob_vector_l2_distance.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/utility/ob_macro_utils.h"
#include <cmath>
#include <cstring>

namespace oceanbase
{
namespace share
{

using namespace common;
using namespace table;

const char *IVF_CID_KV_CACHE_NAME = "IVF_CID_CLUSTER_KV_CACHE";

static int64_t ivf_cid_flat_tables_bytes(const int64_t row_count)
{
  return row_count * (sizeof(int64_t) + sizeof(int32_t) + sizeof(int64_t) + sizeof(int32_t));
}

static int64_t ivf_cid_flat_header_and_tables_size(const int64_t row_count)
{
  return sizeof(ObIvfCidFlatHeader) + ivf_cid_flat_tables_bytes(row_count);
}

static int64_t align8(const int64_t v) { return (v + 7) & ~static_cast<int64_t>(7); }

// --- ObIvfCidClusterKVKey ---

ObIvfCidClusterKVKey::ObIvfCidClusterKVKey() : mgr_key_(), cid_(0) {}

ObIvfCidClusterKVKey::ObIvfCidClusterKVKey(const ObIvfCidClusterCacheMgrKey &mgr_key, const uint64_t cid)
  : mgr_key_(mgr_key), cid_(cid)
{}

bool ObIvfCidClusterKVKey::operator==(const ObIKVCacheKey &other) const
{
  const ObIvfCidClusterKVKey *o = static_cast<const ObIvfCidClusterKVKey *>(&other);
  return mgr_key_ == o->mgr_key_ && cid_ == o->cid_;
}

uint64_t ObIvfCidClusterKVKey::hash() const
{
  return murmurhash(&cid_, sizeof(cid_), mgr_key_.hash());
}

uint64_t ObIvfCidClusterKVKey::get_tenant_id() const
{
  return mgr_key_.tenant_id_;
}

int64_t ObIvfCidClusterKVKey::size() const
{
  return static_cast<int64_t>(sizeof(*this));
}

int ObIvfCidClusterKVKey::deep_copy(char *buf, const int64_t buf_len, ObIKVCacheKey *&key) const
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(buf) || buf_len < size()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObIvfCidClusterKVKey *k = new (buf) ObIvfCidClusterKVKey(mgr_key_, cid_);
    key = k;
  }
  return ret;
}

bool ObIvfCidClusterKVKey::is_valid() const
{
  return mgr_key_.is_valid() && cid_ > 0;
}

// --- ObIvfCidClusterFlatValue ---

ObIvfCidClusterFlatValue::ObIvfCidClusterFlatValue() : buf_(nullptr), buf_len_(0) {}

ObIvfCidClusterFlatValue::ObIvfCidClusterFlatValue(const char *buf, const int64_t buf_len)
  : buf_(buf), buf_len_(buf_len)
{}

int64_t ObIvfCidClusterFlatValue::size() const
{
  return buf_len_;
}

int ObIvfCidClusterFlatValue::deep_copy(char *buf, const int64_t buf_len, ObIKVCacheValue *&value) const
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(buf) || buf_len < size() || OB_ISNULL(buf_) || buf_len_ <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // Store only the ICFL flat blob in the memblock. Do not placement-new FlatValue at buf[0]:
    // that overwrites magic_/index_epoch_ and breaks REPLAY (stale epoch, get_hit=0).
    MEMCPY(buf, buf_, buf_len_);
    value = reinterpret_cast<ObIKVCacheValue *>(buf);
  }
  return ret;
}

// --- flat encode / attach ---

static const float IVF_CID_L2_UNIT_ACCURACY = 0.00001f;

bool ivf_cid_probe_payloads_l2_unit(const ObIvfCidClusterEntry &entry, bool &is_unit, bool &known)
{
  known = false;
  is_unit = false;
  if (entry.rows_.count() <= 0) {
  } else {
    const ObIvfCidClusterRow &row = entry.rows_.at(0);
    if (row.payload_type_ != IVF_CID_CLUSTER_PAYLOAD_FLAT_FLOAT || row.payload_len_ <= 0
        || OB_ISNULL(row.payload_)) {
    } else if (row.payload_len_ % static_cast<int32_t>(sizeof(float)) != 0) {
    } else {
      const int64_t dim = row.payload_len_ / static_cast<int64_t>(sizeof(float));
      const float norm_l2_sqr = ObVectorL2Distance<float>::l2_norm_square(
          reinterpret_cast<const float *>(row.payload_), dim);
      known = true;
      is_unit = norm_l2_sqr > 0 && fabsf(1.0f - norm_l2_sqr) <= IVF_CID_L2_UNIT_ACCURACY;
    }
  }
  return known;
}

int ivf_cid_flat_encode_entry(const ObIvfCidClusterEntry &entry, char *&out_buf, int64_t &out_len)
{
  int ret = OB_SUCCESS;
  out_buf = nullptr;
  out_len = 0;
  const int64_t row_count = entry.rows_.count();
  if (row_count <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    int64_t data_bytes = 0;
    ObSEArray<int64_t, 64> rk_serial_sizes;
    for (int64_t i = 0; OB_SUCC(ret) && i < row_count; ++i) {
      const ObIvfCidClusterRow &row = entry.rows_.at(i);
      data_bytes += row.payload_len_;
      const int64_t rk_sz = ObTableSerialUtil::get_serialize_size(row.rowkey_);
      if (OB_FAIL(rk_serial_sizes.push_back(rk_sz))) {
        LOG_WARN("failed to push rk size", K(ret));
      } else {
        data_bytes += rk_sz;
      }
    }
    const int64_t total = align8(ivf_cid_flat_header_and_tables_size(row_count) + data_bytes);
    char *buf = static_cast<char *>(ob_malloc(total, ObMemAttr(OB_SERVER_TENANT_ID, "IvfCidFlat")));
    if (OB_ISNULL(buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMSET(buf, 0, total);
      ObIvfCidFlatHeader *hdr = reinterpret_cast<ObIvfCidFlatHeader *>(buf);
      hdr->magic_ = ObIvfCidFlatHeader::MAGIC;
      hdr->version_ = ObIvfCidFlatHeader::VERSION;
      hdr->index_epoch_ = entry.index_epoch_;
      hdr->cid_ = entry.cid_;
      hdr->row_count_ = row_count;
      hdr->entry_bytes_ = entry.entry_bytes_;
      hdr->access_cnt_ = entry.heat_.access_cnt_;
      hdr->replay_cnt_ = entry.heat_.replay_cnt_;
      hdr->fill_cnt_ = entry.heat_.fill_cnt_;
      hdr->last_access_us_ = entry.heat_.last_access_us_;
      hdr->flat_buf_len_ = total;
      hdr->payload_type_ = static_cast<uint8_t>(entry.rows_.at(0).payload_type_);
      bool payloads_unit = false;
      bool payloads_unit_known = false;
      if (entry.payloads_l2_unit_known_) {
        payloads_unit_known = true;
        payloads_unit = entry.payloads_l2_unit_;
      } else {
        (void)ivf_cid_probe_payloads_l2_unit(entry, payloads_unit, payloads_unit_known);
      }
      hdr->reserved_[0] = payloads_unit_known ? 1 : 0;
      hdr->reserved_[1] = (payloads_unit_known && payloads_unit) ? 1 : 0;
      char *payload_off_tbl = buf + sizeof(ObIvfCidFlatHeader);
      char *payload_len_tbl = payload_off_tbl + row_count * sizeof(int64_t);
      char *rowkey_off_tbl = payload_len_tbl + row_count * sizeof(int32_t);
      char *rowkey_len_tbl = rowkey_off_tbl + row_count * sizeof(int64_t);
      int64_t data_pos = ivf_cid_flat_header_and_tables_size(row_count);
      data_pos = align8(data_pos);
      for (int64_t i = 0; OB_SUCC(ret) && i < row_count; ++i) {
        const ObIvfCidClusterRow &row = entry.rows_.at(i);
        reinterpret_cast<int64_t *>(payload_off_tbl)[i] = data_pos;
        reinterpret_cast<int32_t *>(payload_len_tbl)[i] = row.payload_len_;
        if (row.payload_len_ > 0) {
          MEMCPY(buf + data_pos, row.payload_, row.payload_len_);
          data_pos += row.payload_len_;
        }
        const int64_t rk_sz = rk_serial_sizes.at(i);
        reinterpret_cast<int64_t *>(rowkey_off_tbl)[i] = data_pos;
        reinterpret_cast<int32_t *>(rowkey_len_tbl)[i] = static_cast<int32_t>(rk_sz);
        int64_t pos = data_pos;
        if (OB_FAIL(ObTableSerialUtil::serialize(buf, total, pos, row.rowkey_))) {
          LOG_WARN("failed to serialize rowkey", K(ret), K(i));
        } else {
          data_pos = pos;
        }
      }
      if (OB_SUCC(ret)) {
        out_buf = buf;
        out_len = total;
      } else {
        ob_free(buf);
      }
    }
  }
  return ret;
}

void ivf_cid_flat_free_buf(char *buf)
{
  if (OB_NOT_NULL(buf)) {
    ob_free(buf);
  }
}

int ivf_cid_flat_attach_entry(const char *flat_buf,
    const int64_t flat_len,
    ObIvfCidClusterEntry *&out_entry,
    ObObj *&out_rowkey_objs,
    int64_t &out_rowkey_obj_cnt)
{
  int ret = OB_SUCCESS;
  out_entry = nullptr;
  out_rowkey_objs = nullptr;
  out_rowkey_obj_cnt = 0;
  if (OB_ISNULL(flat_buf) || flat_len < static_cast<int64_t>(sizeof(ObIvfCidFlatHeader))) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    const ObIvfCidFlatHeader *hdr = reinterpret_cast<const ObIvfCidFlatHeader *>(flat_buf);
    if (hdr->magic_ != ObIvfCidFlatHeader::MAGIC || hdr->version_ != ObIvfCidFlatHeader::VERSION) {
      ret = OB_INVALID_DATA;
    } else if (hdr->row_count_ <= 0 || hdr->row_count_ > flat_len) {
      ret = OB_INVALID_DATA;
    } else {
      const int64_t row_count = hdr->row_count_;
      const char *payload_off_tbl = flat_buf + sizeof(ObIvfCidFlatHeader);
      const char *payload_len_tbl = payload_off_tbl + row_count * sizeof(int64_t);
      const char *rowkey_off_tbl = payload_len_tbl + row_count * sizeof(int32_t);
      const char *rowkey_len_tbl = rowkey_off_tbl + row_count * sizeof(int64_t);
      int64_t total_objs = 0;
      ObSEArray<int64_t, 64> row_obj_cnts;
      for (int64_t i = 0; OB_SUCC(ret) && i < row_count; ++i) {
        const int32_t rk_len = reinterpret_cast<const int32_t *>(rowkey_len_tbl)[i];
        const int64_t rk_off = reinterpret_cast<const int64_t *>(rowkey_off_tbl)[i];
        int64_t obj_cnt = 0;
        if (rk_off < 0 || rk_len < 0 || rk_off + rk_len > flat_len) {
          ret = OB_INVALID_DATA;
        } else if (rk_len <= 0) {
          if (OB_FAIL(row_obj_cnts.push_back(0))) {
            LOG_WARN("failed to push row obj cnt", K(ret));
          }
        } else {
          int64_t pos = rk_off;
          if (OB_FAIL(serialization::decode_vi64(flat_buf, flat_len, pos, &obj_cnt))) {
            LOG_WARN("failed to decode rowkey obj cnt", K(ret), K(i));
          } else if (OB_FAIL(row_obj_cnts.push_back(obj_cnt))) {
            LOG_WARN("failed to push row obj cnt", K(ret));
          } else {
            total_objs += obj_cnt;
          }
        }
      }
      ObIvfCidClusterEntry *entry = OB_NEW(ObIvfCidClusterEntry, ObMemAttr(OB_SERVER_TENANT_ID, "IvfCidView"));
      ObObj *rk_objs = nullptr;
      if (OB_ISNULL(entry)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (total_objs > 0
                 && OB_ISNULL(rk_objs = static_cast<ObObj *>(
                         ob_malloc(sizeof(ObObj) * total_objs, ObMemAttr(OB_SERVER_TENANT_ID, "IvfCidRK"))))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        OB_DELETE(ObIvfCidClusterEntry, ObMemAttr(OB_SERVER_TENANT_ID, "IvfCidView"), entry);
        entry = nullptr;
      } else {
        entry->index_epoch_ = hdr->index_epoch_;
        entry->cid_ = hdr->cid_;
        entry->row_count_ = row_count;
        entry->entry_bytes_ = hdr->entry_bytes_;
        entry->heat_.access_cnt_ = hdr->access_cnt_;
        entry->heat_.replay_cnt_ = hdr->replay_cnt_;
        entry->heat_.fill_cnt_ = hdr->fill_cnt_;
        entry->heat_.last_access_us_ = hdr->last_access_us_;
        entry->arena_ = nullptr;
        entry->kv_flat_buf_ = flat_buf;
        entry->rowkey_objs_ = rk_objs;
        if (hdr->reserved_[0] != 0) {
          entry->payloads_l2_unit_known_ = true;
          entry->payloads_l2_unit_ = (hdr->reserved_[1] != 0);
        }
        int64_t obj_cursor = 0;
        for (int64_t i = 0; OB_SUCC(ret) && i < row_count; ++i) {
          ObIvfCidClusterRow row;
          row.payload_type_ = static_cast<ObIvfCidClusterPayloadType>(hdr->payload_type_);
          row.payload_len_ = reinterpret_cast<const int32_t *>(payload_len_tbl)[i];
          const int64_t poff = reinterpret_cast<const int64_t *>(payload_off_tbl)[i];
          row.payload_ = (row.payload_len_ > 0 && poff >= 0 && poff + row.payload_len_ <= flat_len)
              ? flat_buf + poff
              : nullptr;
          const int32_t rk_len = reinterpret_cast<const int32_t *>(rowkey_len_tbl)[i];
          const int64_t rk_off = reinterpret_cast<const int64_t *>(rowkey_off_tbl)[i];
          const int64_t obj_cnt = row_obj_cnts.at(i);
          if (obj_cnt > 0) {
            int64_t pos = rk_off;
            ObRowkey rk;
            rk.assign(rk_objs + obj_cursor, obj_cnt);
            if (OB_FAIL(ObTableSerialUtil::deserialize(flat_buf, flat_len, pos, rk))) {
              LOG_WARN("failed to deserialize rowkey", K(ret), K(i));
            } else {
              obj_cursor += obj_cnt;
              row.rowkey_ = rk;
            }
          }
          if (OB_SUCC(ret) && OB_FAIL(entry->rows_.push_back(row))) {
            LOG_WARN("failed to push row", K(ret));
          }
        }
        if (OB_FAIL(ret)) {
          if (OB_NOT_NULL(rk_objs)) {
            ob_free(rk_objs);
          }
          OB_DELETE(ObIvfCidClusterEntry, ObMemAttr(OB_SERVER_TENANT_ID, "IvfCidView"), entry);
        } else {
          out_entry = entry;
          out_rowkey_objs = rk_objs;
          out_rowkey_obj_cnt = total_objs;
        }
      }
    }
  }
  return ret;
}

// --- KV cache singleton ---

int ObIvfCidClusterKVCache::init_cache()
{
  // Higher priority / mem share than default (1, 10): flat blobs are large (~MB per CID);
  // aggressive wash caused try_pin get_flat miss (get_hit=0) despite put_ok.
  return init(IVF_CID_KV_CACHE_NAME, 5, 20);
}

int ObIvfCidClusterKVCache::put_flat(
    const ObIvfCidClusterKVKey &key,
    const char *flat_buf,
    const int64_t flat_len,
    const bool overwrite)
{
  int ret = OB_SUCCESS;
  if (!key.is_valid() || OB_ISNULL(flat_buf) || flat_len <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObIvfCidClusterFlatValue val(flat_buf, flat_len);
    if (OB_FAIL(put(key, val, overwrite))) {
      LOG_WARN("failed to put ivf cid flat value", K(ret), K(key));
    }
  }
  return ret;
}

int ObIvfCidClusterKVCache::get_flat(
    const ObIvfCidClusterKVKey &key,
    const char *&flat_buf,
    int64_t &flat_len,
    ObKVCacheHandle &handle)
{
  int ret = OB_SUCCESS;
  flat_buf = nullptr;
  flat_len = 0;
  const ObIvfCidClusterFlatValue *stored = nullptr;
  if (!key.is_valid()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(get(key, stored, handle))) {
    if (OB_ENTRY_NOT_EXIST != ret) {
      LOG_WARN("failed to get ivf cid flat value", K(ret), K(key));
    }
  } else if (OB_ISNULL(stored)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    flat_buf = reinterpret_cast<const char *>(stored);
    const ObIvfCidFlatHeader *hdr = reinterpret_cast<const ObIvfCidFlatHeader *>(flat_buf);
    if (hdr->magic_ != ObIvfCidFlatHeader::MAGIC || hdr->version_ != ObIvfCidFlatHeader::VERSION) {
      ret = OB_INVALID_DATA;
    } else if (hdr->flat_buf_len_ <= 0 || hdr->flat_buf_len_ > INT64_MAX / 2) {
      ret = OB_INVALID_DATA;
    } else {
      flat_len = hdr->flat_buf_len_;
    }
  }
  return ret;
}

int ObIvfCidClusterKVCache::erase_key(const ObIvfCidClusterKVKey &key)
{
  return erase(key);
}

ObIvfCidClusterKVCache &get_ivf_cid_cluster_kv_cache()
{
  static ObIvfCidClusterKVCache cache;
  static bool inited = false;
  if (!inited) {
    (void)cache.init_cache();
    inited = true;
  }
  return cache;
}

} // namespace share
} // namespace oceanbase
