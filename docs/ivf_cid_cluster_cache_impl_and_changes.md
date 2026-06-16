# IVF CID Cluster Cache — 设计说明

## 1. 背景与范围

### 1.1 IVF 查询的两阶段

IVF 查询在 OceanBase 中分为 **coarse** 与 **fine** 两阶段。

**Coarse（粗筛）** — 确定 probe 哪些 cluster：

- 扫描 **centroid 表**（共 `nlist` 个聚类中心）
- 计算 query 到各 center 的距离，取 top **`nprobes`** 个 CID
- 工作量 ≈ O(nlist × dim)，仅涉及 nlist 行

**Fine（精扫）** — 在选中 cluster 内找最近邻：

- 对 coarse 得到的每个 CID，扫描 **`cid_vec` 表** 中该 cluster 的 **全部向量行**
- 逐行计算精确距离并维护 top-k heap
- 工作量 ≈ O(nprobes × 每 cluster 向量数 × dim)，数据量远大于 coarse

```
Query
  ├─ coarse: centroid 表 → 选出 nprobes 个 CID
  └─ fine:   对每个 CID，扫描 cid_vec[cid] 全部行 → 距离 + heap
```

本 cache **仅优化 fine 阶段读取 `cid_vec` 的路径**；coarse、heap、距离公式均不变。

### 1.2 Storage scan 的开销构成

No-cache 的 fine 阶段，每个 probe CID 对 `cid_vec` 开 CID 范围 scan（`scan_cid_range`），再循环 `get_next_rows` 直到该簇扫完。主要开销在 **取数**，而非距离计算。

| 环节            | 说明 |
| --------------- | ---- |
| Scan 打开/rescan | 每个 CID 切换时的 iter setup（nprobes 次） |
| Storage 读行    | 全簇需循环 `get_next_rows`；上层 batch 上限 1000，storage 按 micro block 出批（1536D/50K 约十余行/次，大簇数百～数千次） |
| LOB 物化        | 768D 行内 LOB：只改指针；1536D 行外 LOB：读 payload 并拷贝到 query 内存 |
| 距离 + heap     | 精确距离与 top-k；非 cache 主要优化点 |

因此，cache 的设计目标是：用 **内存中已缓存的 cid_vec 行** 替代对热 CID 的 **重复 storage scan**。

### 1.3 本 cache 的定位

**调用链（fine 阶段读 `cid_vec` 的部分）**：

```
ObDASIvfScanIter          ← IVF 检索总控：coarse 阶段、fine 阶段、算距离、维护 heap
    └─ cid_vec_iter       ← 表扫描迭代器：fine 阶段按 CID 从 cid_vec 表取每一行向量
           ↑
    本 cache 替换/包装这一层（接口仍是 get_next_rows()）
```

- **`ObDASIvfScanIter`**：IVF 总控。**coarse 阶段**选 CID；**fine 阶段**经 `cid_vec_iter_` 逐 batch 取行、算距离、更 heap；不区分底下是 storage 还是 cache。
- **本 cache 替换 `cid_vec_iter_`**：接口仍是 `get_next_rows()`。未命中读 storage；命中则从 ObKVCache 的 ICFL blob 供行。

---

## 2. 自顶向下设计

### 2.1 系统上下文

```
┌─────────────────────────────────────────┐
│         ObDASIvfScanIter                │
│  fine：取行 + 算距离 + 维护 heap          │
└──────────────────┬──────────────────────┘
                   │ get_next_rows()
┌──────────────────▼──────────────────────┐
│    ObDASIvfCidVecCacheScanIter          │
│    命中 / 填充 / 未命中（见下）           │
└──────────────────┬──────────────────────┘
         ┌─────────┴─────────┐
         ▼                   ▼
  ObIvfCidClusterCache   ObDASScanIter
  逻辑 cache：准入与状态    读 cid_vec 表
         │
         ▼
  ObIvfCidClusterKVCache（ICFL blob）
```

切换 CID 时，`ObDASIvfCidVecCacheScanIter` 处于以下三种模式之一（代码枚举名 REPLAY / FILL / MISS）：

| 模式 | 含义 |
| ---- | ---- |
| **命中（REPLAY）** | 该 CID 已在 ObKVCache；从 ICFL blob 供 batch，不读 storage |
| **填充（FILL）** | 未命中，本 session 成为 fill leader；读 storage 建 ICFL 并写入 cache |
| **未命中（MISS）** | 未命中且未填充；只读 storage，累计 probe，等待后续可 FILL |

| 代码类                               | 做什么 |
| ------------------------------------ | ------ |
| `ObDASIvfScanIter`                   | coarse 选 CID；fine 取 `cid_vec` 行、算距离、更 heap |
| `ObDASIvfCidVecCacheScanIter`        | `get_next_rows()`；按命中/填充/未命中选内存 blob 或 storage |
| `ObIvfCidClusterCache`               | probe 计数、FILL 门控、容量记账 |
| `ObDASScanIter`                      | 未命中或填充时读 `cid_vec` 表 |
| `ObIvfCidClusterKVCache`             | 整簇 ICFL 编解码；ObKVCache 存取 |
| `create_das_ivf_cid_vec_cache_scan_iter` | 开关打开创建 cache iter，关闭则用 `ObDASScanIter` |

`ObDASIvfScanIter` 只依赖 `get_next_rows()` 接口，不感知 ICFL 布局；`get_rowkeys_to_heap` 对 cache on/off 共用同一 per-row 循环。

### 2.2 数据模型

**逻辑 entry**（每个 CID 一条；FILL 时在内存构建，finalize 后序列化为 ICFL 写入 KV）：

```
CidEntry {
  cid              : uint64    // 聚类中心编号（0 .. nlist-1）
  rows[]           : Row[]     // FILL 时从 storage 逐行 append：向量 payload + 主表 rowkey
  index_epoch      : uint64    // 索引版本号；rebuild 后 invalidate，epoch 不匹配的 entry 丢弃
  payloads_l2_unit : bool      // 该 cluster 向量是否已 L2 归一化；REPLAY 起始可据此跳过逐行 norm
}
```

**物理 blob（ICFL）**：FILL 结束时一次性序列化，写入 ObKVCache；REPLAY 时用 offset 表按行号定位 payload（无需从头顺序扫 blob），datum 零拷贝指向 blob 内内存。

```
ICFL = Header（cid、行数、payload 类型、是否 L2 单位向量）
     + OffsetTable（每行 payload / rowkey 在 DataSection 中的偏移与长度）
     + DataSection（按行顺序紧挨存放，无分隔符）：
         第0行: [向量 bytes][rowkey bytes]
         第1行: [向量 bytes][rowkey bytes]
         …
```

**Cache 实例 key**（一个 IVF 向量索引对应一个 cache 实例；同索引上所有查询共享 probe、ledger、KV，非 per-query / per-CID）：

```
CacheMgrKey {
  tenant           : uint64         // 租户
  centroid_tablet  : ObTabletID     // centroid 辅助表所在 tablet
  cid_vec_table    : uint64         // _ivf_cid_vector 辅助表 id
  data_table       : uint64         // 主向量表 id
  algorithm        : AlgorithmType  // IVF_FLAT / IVF_PQ / IVF_SQ8
}
```

实例 key 定位 cache 对象；其内每个 cluster 的 entry 为 **实例 key + cid**。

### 2.3 准入与状态

切换 CID 时调用 `lookup_cid`：先累计 probe 计数，再按下列顺序判定结果（见下图）。

```
                         lookup(cid)
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           KV 已有 entry   可成为 fill leader   其余
                │             │             │
           命中（REPLAY）  填充（FILL）    未命中（MISS）
```

**未命中（MISS）** — 未在 KV 中命中 entry，且下列任一成立时，本次只读 storage：

- probe 次数低于 FILL 阈值（默认 5）
- ledger 剩余容量不足（已达 `max_bytes` 上限）
- 该 CID 正由其他 session 填充
- probe 与容量均满足，但本 session 未抢到 fill leader

**填充（FILL）** — probe 达阈值且 ledger 有空间时，一个 session 通过分片 fill gate（64 分片）竞争成为 **fill leader**，独占该 CID 的首次构建；cluster 扫完后 `put` 写入 KV，entry 进入 **READY**，同 CID 后续查询可走命中（REPLAY）。

**失效与容量** — 索引 rebuild 调用 `invalidate_all`，`index_epoch` 递增，旧 entry 全部作废；运行期不做按 entry 的 LRU 驱逐，新 FILL 仅受 `max_bytes`（默认 10GB）硬限制，满则拒绝写入。

### 2.4 单查询生命周期

对一个 probe CID，fine 阶段依次经历：

```
1. 切换 CID（rescan_for_cid）
      lookup 判定模式 → 命中 / 填充 / 未命中
      若命中：跳过 storage scan 打开（不再 scan_cid_range）

2. 循环取 batch（get_next_rows）
      命中：不读 storage；只填 batch 元数据（向量挂 ICFL 指针），向量 bytes 不拷贝
      填充：读 storage，逐行写入 builder；该 CID 扫完后 put 进 KV
      未命中：读 storage，仅累加 probe

3. 算距离（get_rowkeys_to_heap）
      对 batch 中每行：解析 rowkey → 必要时 norm → 算距离 → 更新 heap
      （与无 cache 路径相同）
```

**性能归因**：收益来自消除 repeated storage fetch；距离计算次数在相同 nprobes 下不变。

---

## 3. 运行参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `OB_IVF_CID_CLUSTER_CACHE_ENABLED` | on | 总开关 |
| fill probe 阈值 | 5 | 冷 CID 不 FILL |
| `max_bytes` | 10 GB | 全局容量上限 |
| `max_cid` | 4096 | center_id 范围 0～max_cid−1（即 nlist 上限）；entry 个数由 `max_bytes` 限制，且 ≤ nlist |

---

## 4. 伪代码 — 设计到实现的映射

以下伪代码概括最终版实现；命名与源码一致，省略错误处理与资源管理细节。

### 4.1 装配（DAS 工厂）

```
function create_cid_vec_iter(plan):
    if not is_ivf_cid_cluster_cache_enabled():
        return ObDASScanIter(plan)
    cache = acquire_ivf_cid_cluster_cache(mgr_key)   // 按索引对象定位实例
    return ObDASIvfCidVecCacheScanIter(plan, cache)
```

### 4.2 CID 切换（IVF scan × cache iter）

```
function get_rowkeys_to_heap(cid, center_id, heap):
    replay_only = false
    if center_id present:
        try_cid_vec_replay_only_switch(center_id, replay_only)
        // replay_only → 跳过 scan_cid_range（无 storage rescan）

    if not replay_only:
        scan_cid_range(cid)                            // 打开 storage scan

    if need_norm and cache knows l2_unit for this CID:
        cid_vec_need_norm = not l2_unit                // cluster 级，非逐行

    while not end:
        get_next_rows(batch)

        for row in batch:
            load_rowkey(row)
            norm_if_needed(row)
            push_center(row, heap)
```

```
function try_cid_vec_replay_only_switch(cid, out replay_only):
    if cache_iter inactive or first_scan:
        return
    cache_iter.rescan_for_cid(cid)                     // 内部 lookup + 设 mode
    if cache_iter.mode == REPLAY:
        replay_only = true                             // needs_storage_after_cid_switch == false
```

### 4.3 Lookup 决策（逻辑 cache）

```
function lookup_cid(cid) -> HIT | FILL_LEADER | MISS:
    record_access(cid)                               // probe_access[cid]++

    if try_lookup_hit(cid):                          // KV get_flat + pin entry
        return HIT

    if cid_fill_in_progress(cid):                    // 他 session 正在 FILL
        return MISS

    if probe_access[cid] < FILL_MIN_PROBE:           // 默认 5
        return MISS

    if not ledger.has_space(estimate_bytes(cid)):
        return MISS

    if try_acquire_fill_leader(cid):                 // 分片 fill gate
        return FILL_LEADER

    return MISS
```

### 4.4 Mode 切换（cache iter）

```
function rescan_for_cid(new_cid):
    flush_building_cluster()                         // 收尾上一个 FILL
    result = cluster_cache.lookup_cid(new_cid)

    switch result:
        HIT:
            mode = REPLAY; replay_idx = 0
            revert_storage_scan_iter()               // 释放 storage 资源
        FILL_LEADER:
            mode = FILL; init building_cluster(new_cid)
        MISS:
            mode = MISS
```

### 4.5 Batch 供给（cache iter）

```
function get_next_rows(out count, capacity):
    switch mode:
        REPLAY:
            return replay_rows(count, capacity)
        FILL | MISS:
            rows = base_scan.get_next_rows(capacity)
            if mode == FILL:
                for row in rows: append_fill_row(row)
            if end_of_cid:
                flush_building_cluster()             // FILL → put
            return rows
```

```
function replay_rows(count, capacity):
    while count < capacity and replay_idx < entry.row_count:
        row = icfl_decode_at(entry.blob, replay_idx++)
        materialize_row_to_eval(row)                 // payload: datum ← blob ptr（零拷贝）
        count++
```

### 4.6 FILL 落盘（逻辑 cache × KV）

```
function flush_building_cluster():
    if building_cluster.row_count == 0: return

    blob = ivf_cid_flat_fill_finalize(building_cluster)   // encode once
    ok = cluster_cache.put(cid, blob)                     // ledger 记账 + put_flat
    finish_cid_fill(cid)                                  // 释放 fill gate
```

```
function put(cid, entry):
    kv_key = (cache_instance_key, cid, index_epoch)
    kv_cache.put_flat(kv_key, entry.icfl_blob)
    phase[cid] = READY
    ledger.commit(entry.bytes)
```

### 4.7 Per-row 消费（IVF scan，与数据来源无关）

```
function process_batch_row(row):
    if skip_payload_lob_read:                        // REPLAY/FILL 已物化
        payload = row.payload_datum.ptr                // 无 LOB 读
    else:
        payload = read_lob(row)                        // no-cache 路径

    rowkey = extract_rowkey(row)
    vec = norm_and_prepare(payload)                    // 可能 MEMCPY 到 arena
    heap.push_center(vec, rowkey, distance(query, vec))
```
