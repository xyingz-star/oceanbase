# IVF CID Cluster Cache：KV LRU 迁移 SIGSEGV（ICFL 裸 blob 与 FlatValue 虚调用不兼容）

本文记录 **2026-05-25 rebuild + conc=80** 压测导致 Observer 崩溃的根因分析，并对比 **KV-only** 与 **pin + KV** 方案是否同样受影响。

相关实现文档：[ivf_cid_cluster_cache_zero_copy_design.md](./ivf_cid_cluster_cache_zero_copy_design.md)

---

## 1. 故障摘要

| 项 | 内容 |
|----|------|
| 场景 | skip-load + rebuild-index + major freeze 后，VectorDBBench conc=80 向量检索 |
| 现象 | ~10s 内 80 路 `2013 Lost connection to MySQL server during query`；Observer 退出，11000 不可连 |
| 信号 | SIGSEGV（sig=11），`sig_addr=0x14943465c`（非法用户态读） |
| 租户/线程 | T1004，`T1004_L0_G0` |
| SQL | `SELECT ... ORDER BY cosine_distance(embedding, ...) APPROXIMATE LIMIT 100` |
| 二进制 | `4.6.0.0_1-64fe0aec660ada49...`（含 debug_info） |
| core | `RLIMIT_CORE=0`，未落盘 |

---

## 2. 调用栈（符号化）

`observer.log` 中 `CRASH ERROR` 的 `lbt`（由内向外）：

```
ObDASIvfCidVecCacheScanIter::rescan()
  → on_cid_switch()
    → acquire_cid_and_set_mode()
      → ObIvfCidClusterCache::lookup_cid()
        → try_lookup_hit_()
          → load_kv_entry_prep_()          // KV-only 路径；pin 时代为 load_kv_pin_prep_()
            → ObIvfCidClusterKVCache::get_flat()
              → ObKVCache::get()
                → ObKVCacheMap::get()
                  → ObKVCacheMap::internal_data_move()   // LRU 热 entry 提升
                    → ObIKVCacheStore::store()           // ← 实际 fault IP (+0x3c)
                      → ObIKVCacheValue::deep_copy()    // 虚调用（推断）
```

**要点**：崩溃发生在 **ObKVCache 全局 map 的 LRU 节点迁移**，不是 DAS materialize、也不是 SQL 表达式层。

---

## 3. 根因机制

### 3.1 ICFL 在 memblock 里是「裸字节」，不是 C++ 对象

`put_flat` 时 `ObIvfCidClusterFlatValue::deep_copy` **刻意**只 memcpy 原始 ICFL flat blob 进 KV memblock，不在头部 placement-new `FlatValue`（避免覆盖 magic / epoch）：

```cpp
// ob_ivf_cid_cluster_kv_cache.cpp
MEMCPY(buf, buf_, buf_len_);
value = reinterpret_cast<ObIKVCacheValue *>(buf);  // value 指向 ICFL header 起始
```

因此 hash map 节点里的 `iter->value_` 实际指向 **ICFL header**，不是带 vtable 的 `ObIvfCidClusterFlatValue`。

### 3.2 `get_flat` 为何多数时候能工作

`get_flat` 把 `stored` 强转为 `(const char *)` 读 magic / `flat_buf_len_`，与真实布局一致，**只要不走 LRU 迁移**：

```cpp
flat_buf = reinterpret_cast<const char *>(stored);
const ObIvfCidFlatHeader *hdr = reinterpret_cast<const ObIvfCidFlatHeader *>(flat_buf);
```

### 3.3 LRU 迁移路径触发虚调用 → 读野指针

`ObKVCacheMap::get` 命中且 `LRU == mb_policy && need_modify_cache(...)` 时调用 `internal_data_move`，对新节点执行 `store(*key, *value, ...)`：

```cpp
// ob_kvcache_map.cpp
store_->store(*old_iter->inst_, *old_iter->key_, *old_iter->value_, ...);

// ob_kvcache_store.cpp
value.deep_copy(...);  // const ObIKVCacheValue &，虚函数分派
```

此时 `value` 在内存布局上是 **ICFL header**，却被当作 `ObIvfCidClusterFlatValue`：

| FlatValue 字段（对象语义） | memblock 实际内容（ICFL） |
|---------------------------|---------------------------|
| `buf_`（offset 0，指针） | `magic_` + `version_` 被当成指针 |
| `buf_len_`（offset 8） | `index_epoch_` 等被当成 memcpy 长度 |

`deep_copy` 会 `MEMCPY(buf, buf_, buf_len_)`，从「假指针」读源地址 → **SIGSEGV**。  
观测到的 `sig_addr=0x14943465c` 与把 header 中某个 64 位字段（如 `index_epoch_` / `cid_`）误当作 `buf_` 的量级一致。

### 3.4 为何 conc=80 + rebuild 后更易触发

1. **大量并行 `get_flat`**：每个 IVF 查询对多个 CID 做 `lookup_cid` → HIT 时必调 KV `get`。
2. **热 CID 反复命中**：`iter_get_cnt` / `mb_get_cnt` 上升，满足 `need_modify_cache`，LRU → LFU 的 `internal_data_move` 频繁触发。
3. **大 value（~MB/CID）**：IVF KV cache `init(5, 20)` 高优先级；大块 entry 的 policy / 统计行为更易进入上述分支。
4. **与 invalidate / rebuild 的关系**：`invalidate_all` 只 bump epoch、不 erase KV；rebuild 后首轮仍可能命中**旧 epoch blob**（逻辑 stale，通常返回 miss），或在重新 put 后对**新 blob** 大量 HIT——**不直接造成 SIGSEGV**，但会放大 get 次数与 LRU 迁移机会。

---

## 4. 与「去掉 pin」的关系

### 4.1 结论（先说）

| 问题 | KV-only | pin + KV（commit `208bd87775` 一类实现） |
|------|---------|------------------------------------------|
| **根因 bug 是否存在** | **是** | **是**（同一 `get_flat` → `ObKVCacheMap::get` 路径） |
| **能否从机制上消除** | 否 | 否（pin 未改 KV 存储格式与 `internal_data_move`） |
| **触发概率** | **更高** | **较低**（有 mitigations，见下） |
| **16:14 全量 load 成功** | 可解释 | 同样可解释（LRU 迁移为概率/热度触发，非必现） |

**去掉 pin 没有引入这个 KV 层 bug**；bug 来自 **ICFL 裸 blob 存储约定** 与 **ObKVCacheMap LRU 迁移实现** 的不兼容。  
去掉 pin 后 **并发 `get_flat` 不再被 per-CID pin 锁串行化**，热 CID 上 LRU 迁移更密集，**更容易复现**。

### 4.2 pin 方案路径回顾（git `208bd87775`）

**HIT 两阶段：**

1. **Pin 快路径**（持 `pin_shard_lock_(cid)`）：若 pin slot 存在且 `index_epoch` 匹配 → `borrow_ref_cnt_++`，**不调用 `get_flat`**。
2. **Pin 慢路径**（仍持锁）：`load_kv_pin_prep_()` → **`get_flat`** → attach → 返回 session-owned entry（或 `warm_pin` 后装入 slot）。

**put 后：** `warm_pin_slot_()` 再次 **`get_flat`** 并 `install_pin_slot_locked_`。

**put 覆盖：** `borrow_ref_cnt_ > 0` 时 **skip `put_flat`**（`put_skip_pinned`），避免覆盖正在被 REPLAY 的 KV entry。

### 4.3 pin 为何「有同样 bug」但仍可能不崩

pin **不能修复** `internal_data_move` 对裸 ICFL 的虚调用；以下路径 **仍会** 调用 `get_flat`，仍可能 SIGSEGV：

- 某 CID **第一次** HIT（pin slot 不存在）
- `put` 成功后 **`warm_pin_slot_`**
- `invalidate_all` 清 pin 后再次 HIT
- pin slot epoch 不匹配时的 slow path

pin **降低触发率**的原因：

1. **热 CID 稳态 REPLAY 走 pin 快路径**，不再每次 `get_flat` → 大幅减少 LRU `internal_data_move` 机会。
2. **`pin_shard_lock_(cid)` 覆盖整个 `try_lookup_hit_`**（含 slow path 的 `get_flat`），同 CID 上 **串行** KV get，避免 KV-only 下 80 线程同时 `get_flat` 同一热 key。
3. **`put_skip_pinned`** 在有活跃 borrow 时跳过 `put_flat`，减少 KV 链表的 put/overwrite 与 get 交错。

因此：**pin 方案是「缓解触发条件」，不是「修复根因」**。在 conc=80、56 个 CID 全冷启动、或 invalidate 后全员 slow path 的场景下，**pin 方案同样可以崩**。

### 4.4 KV-only 相对 pin 的额外风险

| 维度 | KV-only | pin + KV |
|------|---------|----------|
| HIT 是否每次 `get_flat` | 是（`try_lookup_hit_` 无 pin 快路径） | 稳态 often 否 |
| 同 CID 并发 get | 无 per-CID 锁，80 路可并行 | `pin_shard_lock_` 串行 |
| put 与 REPLAY 并发 | `put_flat(overwrite=true)` 仍可执行 | borrow 时 skip put |
| 逻辑层 pin map | 无 | 有（与 KV hazard handle 正交） |

KV-only 依赖 **`ObKVCacheHandle` hazptr** 保证 blob 生命周期；这在 **正常 get + hold** 下足够，但 **不能避免** LRU 迁移时对 value 的错误 `deep_copy`。

---

## 5. 其他曾考虑的假设（优先级低于 P0）

| 假设 | 说明 | 与本次栈关系 |
|------|------|--------------|
| invalidate epoch 竞态 | bump epoch 后仍可能短暂 HIT 旧 blob | 多导致 stale 语义；crash 栈在 get/store，非 attach 后 materialize |
| pin 移除 → put 覆盖 UAF | overwrite 与 handle 并发 | hazard 应 pin memblock；栈指向 store 而非 use-after-free 读 payload |
| cache mgr `ObIvfCidClusterCache` UAF | ref_cnt 误减 destroy | 栈在 KV 层，优先级低 |
| REPLAY `row.payload_` 悬空 | release handle 后仍 materialize | 栈在 lookup/get，未到 materialize |

---

## 6. 修复方向（建议）

### P0 — 消除虚调用对裸 ICFL 的误用（必做）

任选或组合：

1. **禁止 IVF CID KV 走 LRU 提升**：初始化/policy 保证该 cache 上 `mb_policy != LRU`，或 `get` 成功后不对 ICFL entry 调用 `internal_data_move`。
2. **专用 move 拷贝**：`internal_data_move` 对 IVF CID 识别 raw blob，按 `ObIvfCidFlatHeader::flat_buf_len_` memcpy，不走 `FlatValue::deep_copy`。
3. **恢复可安全 deep_copy 的 value 布局**（成本高）：例如在 memblock 内保留 indirection，且不与 ICFL magic 冲突。

### P1 — 逻辑层增强（可选，非替代 P0）

- 恢复 **per-CID 串行 get** 或 **warm pin** 仅作为性能/稳定性优化，**不能**单独当作 crash fix。
- `invalidate` 时 `erase_key`（与 epoch 一致化），减少 stale blob 命中。

### 验证

```bash
ulimit -c unlimited
# 复现：rebuild-index + conc=80
gdb -batch -ex 'bt full' /path/to/observer core
# 期望看到：FlatValue::deep_copy 或 ObIKVCacheStore::store ← internal_data_move ← ObKVCacheMap::get
```

---

## 7. 时间线对照

| 时间 | 路径 | 结果 | 与本 bug 关系 |
|------|------|------|---------------|
| 16:14 | 全量 drop+load，conc=80 | QPS ~911 成功 | LRU 迁移未触发或次数少，**非「KV-only 无 bug」** |
| 16:25 | rebuild | IVF_CLEAN hung | 不同问题（异步任务/等待） |
| 16:56 | rebuild + freeze + conc=80 | Observer SIGSEGV | **本 bug** |
| 16:57 | （dmesg） | 同 IP 族 segfault | 同根因，负载较低未立即打挂 |

---

## 8. 总结

1. **根因**：KV memblock 存 **ICFL 裸 blob**，但 LRU `internal_data_move` 按 **`ObIvfCidClusterFlatValue` 对象** 做 `deep_copy` 虚调用 → 野指针读 → SIGSEGV。
2. **与 pin 无关的代码缺陷**：在 `ObKVCacheMap` / value 存储契约层；**pin 与 KV-only 共用同一 `get_flat` 实现**。
3. **pin 方案同样具有此 bug**；其 pin 快路径、per-CID 锁、`put_skip_pinned` 仅 **降低** `get_flat`/LRU 迁移频率，**不能**从机制上消除。
4. **KV-only 在 conc=80 下更易复现**，因每次 HIT 都 `get_flat` 且无 per-CID 串行化。
5. **正确修复**应在 KV 层（policy / move / deep_copy 契约），而非恢复 pin map。
