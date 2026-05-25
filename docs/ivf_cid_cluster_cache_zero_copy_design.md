# IVF CID Cluster Cache：设计与实现

本文描述 `ObIvfCidClusterCache`（逻辑层）与 `ObIvfCidClusterKVCache`（物理层）的**当前实现**。

核心思路：**以 CID 为粒度**；查询路径 HIT / MISS / FILL_LEADER；**不做驱逐**；probe 热度与容量账用 **per-CID 原子字段**（无全局 `probe_heat_lock_` / `ledger_lock_`）；`put` 后立刻 `warm_pin_slot_`。

---

## 1. 目标与分层

| 层级 | 组件 | 职责 |
|------|------|------|
| DAS | `ObDASIvfCidVecCacheScanIter` | REPLAY / FILL / MISS |
| 逻辑 | `ObIvfCidClusterCache` | per-CID 原子状态、fill gate、warm pin |
| 物理 | `ObIvfCidClusterKVCache` | ICFL flat；LRU 由 ObKVCache 负责 |

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

若该 CID 已有 FILL leader（`fill_gate.filling_`），`lookup_cid` 直接 **MISS**（走存储），不 wait、不抢 FILL；leader `put` 完成后，后续查询再 `try_lookup_hit` → HIT/REPLAY。

---

## 3. 查询 / 写入

### lookup_cid

1. `record_access_` → `ATOMIC_INC(probe_access)`
2. `try_lookup_hit_` → pin borrow 或 KV attach（零拷贝 payload）
3. miss → 原子准入判断 → FILL_LEADER 或 MISS

### put / FILL 写入

1. **FILL**：`ivf_cid_flat_fill_append_row` 将 payload + 序列化 rowkey **直接追加**到 `flat_fill_` 数据区（不再经 `rows_` + arena 存一份向量）。
2. **put**：`ivf_cid_flat_fill_finalize` 分配最终 ICFL blob，写 header/偏移表，**一次 memcpy** 整块 data 区；`put_flat` 再 deep_copy 进 KV memblock。
3. `ledger_commit_` → `warm_pin_slot_`。

LOB 仍可能先在 arena 物化一次；相对旧路径少掉 **encode 阶段对每行 payload 的第二次 memcpy**。

### invalidate_all

epoch++；清 pin；对有 `cached_bytes>0` 的 CID 做 `ledger_remove_`；`reset_all_cid_states_`。

---

## 4. 仍需要的锁

| 锁 | 用途 |
|----|------|
| `pin_shard_lock_[cid%64]` | warm slot 指针生命周期 |
| `fill_gate_shard_lock_[cid % 64]` | 同分片 FILL gate map；同 CID 单 leader（`filling_` 标记） |
| ObKVCache 内部 | get_flat / put_flat |

**已移除：** `probe_heat_lock_`、`ledger_lock_`、`probe_heat_map_`、`ledger_map_`。

---

## 5. 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `OB_IVF_CID_CLUSTER_CACHE_FILL_MIN_PROBE_ACCESS` | 5 | 允许 FILL 的最小 probe 次数 |
| `OB_IVF_CID_CLUSTER_CACHE_MAX_CID` | 4096 | `cid_states_` 数组长度上界 |
| max_mb（init 参数 / 配置） | 128 | `ledger_bytes_total` 上限 |

---

## 6. 统计

`ObIvfCidClusterCacheStats` 全程 `ATOMIC_*`；`cur_bytes_` = `ledger_bytes_total_`。
