#!/bin/bash
#
# 多次运行 vectordbbench_oceanbaseivf_single.sh（例如做重复基准减小方差）。
#
# 用法：
#   ./vectordbbench_oceanbaseivf_single_repeat.sh           # 默认 3 次；内层默认 VDB_LINK_LOCAL_ALIAS=1 复用本地 parquet
#   NUM_RUNS=5 ./vectordbbench_oceanbaseivf_single_repeat.sh
#   ./vectordbbench_oceanbaseivf_single_repeat.sh 5       # 第一个参数 = 运行次数
#
# 其它环境变量会原样传给内层脚本，例如：
#   CASE_TYPE=Performance768D1M NLIST=2000 ./vectordbbench_oceanbaseivf_single_repeat.sh 3
#
# CONTINUE_ON_ERROR=1 时某次失败后仍继续跑完所有轮次，最后若有失败则退出码为 1。
#
# 每轮会自动设置 VDB_TASK_LABEL，使 VectorDBBench 落盘为：
#   result_{日期}_{task_label}_oceanbase.json
# 形如 repeat_r1of3、myexp_r2of5。可自行加前缀：
#   VDB_RUN_LABEL_PREFIX=ob_nmbkm_v5 ./vectordbbench_oceanbaseivf_single_repeat.sh 3
# 单次手动跑也可用：VDB_TASK_LABEL=mytag ./vectordbbench_oceanbaseivf_single.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="${SCRIPT_DIR}/vectordbbench_oceanbaseivf_single.sh"

NUM_RUNS="${NUM_RUNS:-3}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_RUNS="$1"
  shift
fi

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# 与单次脚本一致：默认链简写目录到官方路径，避免重复下数据集
export VDB_LINK_LOCAL_ALIAS="${VDB_LINK_LOCAL_ALIAS:-1}"

if [[ ! -x "${INNER}" ]] && [[ ! -f "${INNER}" ]]; then
  echo "ERROR: inner script not found: ${INNER}" >&2
  exit 1
fi

failed=0
_label_prefix="${VDB_RUN_LABEL_PREFIX:-repeat}"
_label_prefix="${_label_prefix// /_}"

for ((i = 1; i <= NUM_RUNS; i++)); do
  _task_label="${_label_prefix}_r${i}of${NUM_RUNS}"
  echo "========================================"
  echo "Run ${i}/${NUM_RUNS}  start: $(date -Iseconds)  VDB_TASK_LABEL=${_task_label}"
  echo "========================================"
  set +e
  env \
    VDB_LINK_LOCAL_ALIAS="${VDB_LINK_LOCAL_ALIAS}" \
    VDB_TASK_LABEL="${_task_label}" \
    bash "${INNER}" "$@"
  rc=$?
  set -e
  echo "Run ${i}/${NUM_RUNS}  end:   $(date -Iseconds)  exit=${rc}"
  echo
  if [[ "${rc}" -ne 0 ]]; then
    failed=1
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "ERROR: run ${i} failed, stopping (set CONTINUE_ON_ERROR=1 to run all)." >&2
      exit "${rc}"
    fi
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: one or more runs failed (see logs above)." >&2
  exit 1
fi

echo "All ${NUM_RUNS} run(s) completed successfully."
exit 0
