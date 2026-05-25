# IVF CID Cluster Cache：设计与实现

本文描述 `ObIvfCidClusterCache`（逻辑层）与 `ObIvfCidClusterKVCache`（物理层）的**当前实现**。

核心思路：**以 CID 为粒度**；查询路径 HIT / MISS / FILL_LEADER；**不做驱逐**；probe 热度与容量账用 **per-CID 原子字段**；HIT 仅经 **ObKVCache**（无逻辑层 pin map）。

---

## 1. 目标与分层

| 层级 | 组件 | 职责 |
|------|------|------|
| DAS | `ObDASIvfCidVecCacheScanIter` | REPLAY / FILL / MISS |
| 逻辑 | `ObIvfCidClusterCache` | per-CID 状态、fill gate、ledger |
| 物理 | `ObIvfCidClusterKVCache` | ICFL flat；memblock = `[FlatValue | ICFL]`，`buf_` 指向块内 ICFL；LRU 迁移可安全 `deep_copy` |

---

## 2. Per-CID 状态（`cid_states_[cid]`）

稠密数组，下标即 CID，大小由 `OB_IVF_CID_CLUSTER_CACHE_MAX_CID`（默认 4096）决定。

```cpp
struct ObIvfCidPerCidState {
  uint64_t probe_access_;   // atomic，lookup 时 ATOMIC_INC
  uint64_t replay_cnt_;     // atomic，put 前合并进 flat heat
  int64_t  cached_bytes_;   // atomic，0 表示未缓存
  int64_t  phase_;          // atomic，READY 表示可 REPLAY
};
int64_t ledger_bytes_total_;  // atomic，全局字节账
```

**准入（无锁）：**

1. `ATOMIC_LOAD(probe_access) >= FILL_MIN_PROBE`（默认 5）
2. `ledger_bytes_total - cached_bytes[cid] + need <= max_bytes_`

满足 → `try_acquire_fill_leader_`（`fill_gate_shard_lock_[cid % 64]`，短持）。

若该 CID 已有 FILL leader（`fill_gate.filling_`），`lookup_cid` 直接 **MISS**。

---

## 3. 查询 / 写入

### lookup_cid

1. `record_access_` → `ATOMIC_INC(probe_access)`
2. `try_lookup_hit_` → **每次** `get_flat` + `ivf_cid_flat_attach_entry`；`get_flat` 经 `FlatValue::buf()` 取 ICFL（REPLAY 仍零拷贝）；`session_kv_handle_` 保 memblock 存活
3. miss → fill gate 判断 → FILL_LEADER 或 MISS

### put / FILL 写入

1. FILL：`ivf_cid_flat_fill_append_row` 直接追加到 `flat_fill_`
2. `put`：`ivf_cid_flat_fill_finalize` → `put_flat` → `ledger_commit_`（**无 warm_pin**）

### release_session_entry

仅 `session_owned_` 路径：reset handle、delete attach 出来的 `ObIvfCidClusterEntry`。

### invalidate_all

epoch++；清 fill gate；对有 `cached_bytes>0` 的 CID 做 `ledger_remove_`；`reset_all_cid_states_`。

---

## 4. 仍需要的锁

| 锁 | 用途 |
|----|------|
| `fill_gate_shard_lock_[cid % 64]` | 同 CID 单 FILL leader |
| ObKVCache 内部 | get_flat / put_flat（hazptr，调用方无需额外 per-CID 锁） |

**已移除：** pin map、`pin_shard_lock_`、`warm_pin_slot_`、`borrow_ref_cnt_`。

---

## 5. 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `OB_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS` | 5 | 允许 FILL 的最小 probe 次数 |
| `OB_IVF_CID_CLUSTER_CACHE_MAX_CID` | 4096 | `cid_states_` 数组长度上界 |
| max_mb（init 参数 / 配置） | 128 | `ledger_bytes_total` 上限 |

---

## 6. 统计

`ObIvfCidClusterCacheStats` 全程 `ATOMIC_*`；`put_skip_pinned_cnt_` 保留字段但 KV-only 路径恒为 0。
