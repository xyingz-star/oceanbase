#!/usr/bin/env bash
#
# IVF 配置遍历：与 vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh 相同（数据集 × nlist 倍数 × NMBKM div 等），
# 但默认在「建索引（CREATE INDEX / optimize）」结束后即结束，不跑串行检索与并发检索。
#
# 实现：向子脚本传入 VectorDBBench 的 --skip-search-serial --skip-search-concurrent（见 vectordbbench_oceanbaseivf_single.sh）。
# 默认 export VDB_OB_SKIP_POST_CREATE_INDEX=1：CREATE INDEX 后跳过 MAJOR FREEZE / 等待合并 / gather_schema_stats（纯测建索引+k-means）。
#   需要与生产一致的后处理时：VDB_OB_SKIP_POST_CREATE_INDEX=0 bash 本脚本
# 仍会执行：drop/load（或 skip-load + rebuild-index）以及插入后的建索引；不会执行 recall/ndcg/QPS 等检索压测。
#
# 若需临时恢复完整压测，可在同一 shell 中：
#   VDB_SKIP_SEARCH_SERIAL=0 VDB_SKIP_SEARCH_CONCURRENT=0 bash .../vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh
# 或直接调用原 sweep 脚本。
#
# 其余环境变量、ONLY_DIRS、SWEEP_*、OB_*、VEC_DATA_ROOT 等与父脚本完全一致。
# 父脚本默认仅扫数据集 1536D500K；多数据集 / 全盘发现见 vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh 头部说明。
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${SCRIPT_DIR}/vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh"

if [[ ! -f "${SWEEP}" ]]; then
  echo "ERROR: missing ${SWEEP}" >&2
  exit 1
fi

export VDB_SKIP_SEARCH_SERIAL="${VDB_SKIP_SEARCH_SERIAL:-1}"
export VDB_SKIP_SEARCH_CONCURRENT="${VDB_SKIP_SEARCH_CONCURRENT:-1}"
export VDB_OB_SKIP_POST_CREATE_INDEX="${VDB_OB_SKIP_POST_CREATE_INDEX:-1}"

exec bash "${SWEEP}"
