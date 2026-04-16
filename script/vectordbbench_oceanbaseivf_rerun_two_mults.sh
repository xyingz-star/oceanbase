#!/usr/bin/env bash
#
# 仅重跑两组 IVF 实验（与全量 sweep 中「缩放比」定义一致）：
#   1) 1536D50K  · 缩放比 mult = 0.25  →  base_nlist=round(sqrt(N))，nlist=round(base_nlist×0.25)
#   2) 768D1M    · 缩放比 mult = 4    →  同上，mult=4
# 其中 N 为 VectorDBBench 训练集规模（50K / 1M），spn=floor(N/nlist)。
#
# 依赖同目录下的 vectordbbench_oceanbaseivf_single.sh（与 vectordbbench_oceanbaseivf_single_repeat.sh 相同）。
#
# 用法：
#   ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#   # 每组重复 3 次（与 repeat 脚本类似，用于减小方差）
#   ./vectordbbench_oceanbaseivf_rerun_two_mults.sh 3
#   NUM_RUNS=5 ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#
# OceanBase 连接（必须先 export，示例如下）：
#   export OB_HOST=127.0.0.1
#   export OB_PORT=11000        # 以本机 observer 实际 SQL 端口为准（常见非 2881）
#   export OB_USER='root@mysql_tenant'
#   export OB_DATABASE=single   # 租户下须存在；或让 VectorDBBench 在 1049 时自动 CREATE DATABASE
#   export OB_PASSWORD=''       # 空密码可留空；有密码则填写真实值
#
# 数据目录（与其它脚本一致，可按需覆盖）：
#   VEC_DATA_ROOT=/data/zhuxueying.zxy/vec_data \
#   ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#
# 与历史结果对齐时，检索探针数可固定为 56（与汇总表 ivf_nprobes 列一致）：
#   IVF_NPROBES=56 ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#
# 某组失败后仍跑下一组：
#   CONTINUE_ON_ERROR=1 ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#
# 长任务后台（stdout/stderr 进日志，父进程立刻退出）：
#   SWEEP_DAEMONIZE=1 SWEEP_DAEMONIZE_LOG=~/log/rerun_two_mults.log \
#     ./vectordbbench_oceanbaseivf_rerun_two_mults.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="${SCRIPT_DIR}/vectordbbench_oceanbaseivf_single.sh"
SELF_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

NUM_RUNS="${NUM_RUNS:-1}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_RUNS="$1"
  shift
fi

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
VEC_DATA_ROOT="${VEC_DATA_ROOT:-/data/zhuxueying.zxy/vec_data}"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-${VEC_DATA_ROOT}}"
IVF_NPROBES_MAX="${IVF_NPROBES_MAX:-100}"
# 未显式设置 IVF_NPROBES 时：与常见 sweep 一致，探针不超过 nlist，大 nlist 时 cap 在 IVF_NPROBES_MAX
export VDB_LINK_LOCAL_ALIAS="${VDB_LINK_LOCAL_ALIAS:-1}"
export VDB_NUM_CONCURRENCY="${VDB_NUM_CONCURRENCY:-80}"

if [[ ! -f "${INNER}" ]]; then
  echo "ERROR: inner script not found: ${INNER}" >&2
  exit 1
fi

require_ob_connection_env() {
  local missing=()
  [[ -z "${OB_USER:-}" ]] && missing+=(OB_USER)
  [[ -z "${OB_DATABASE:-}" ]] && missing+=(OB_DATABASE)
  [[ -z "${OB_PORT:-}" ]] && missing+=(OB_PORT)
  if ((${#missing[@]} > 0)); then
    echo "ERROR: 未设置 OceanBase 连接变量: ${missing[*]}" >&2
    echo "请在当前 shell 中 export 后再运行，例如：" >&2
    echo "  export OB_HOST=127.0.0.1" >&2
    echo "  export OB_PORT=11000" >&2
    echo "  export OB_USER='root@你的租户'" >&2
    echo "  export OB_DATABASE=你的库名   # 不存在时 bench 可自动 CREATE DATABASE（需账号有建库权限）" >&2
    echo "  export OB_PASSWORD=...        # 有密码则填写；空密码可省略" >&2
    exit 1
  fi
}
require_ob_connection_env

# 传给内层 single.sh（子进程继承 export）
export OB_HOST="${OB_HOST:-127.0.0.1}"
export OB_PASSWORD="${OB_PASSWORD:-}"

# ---------- 可选：整脚本后台 ----------
if [[ "${SWEEP_DAEMONIZE:-0}" == "1" ]] && [[ -z "${SWEEP_DAEMONIZE_CHILD:-}" ]]; then
  _daemon_log="${SWEEP_DAEMONIZE_LOG:-${HOME}/log/vdb_ivf_sweep/rerun_two_mults_$(date +%Y%m%dT%H%M%S).log}"
  mkdir -p "$(dirname "${_daemon_log}")"
  nohup bash -c "
    export SWEEP_DAEMONIZE_CHILD=1
    unset SWEEP_DAEMONIZE
    exec >>\"${_daemon_log}\" 2>&1
    exec bash \"${SELF_SCRIPT}\" \"${NUM_RUNS}\"
  " </dev/null &
  echo "SWEEP_DAEMONIZE: PID=$! log=${_daemon_log}" >&2
  echo "hint: tail -f ${_daemon_log}" >&2
  exit 0
fi

resolve_case_type() {
  case "$1" in
    1536D50K)  echo "Performance1536D50K" ;;
    768D1M)     echo "Performance768D1M" ;;
    *)          echo "" ;;
  esac
}

dataset_train_vector_count_for_case() {
  case "$1" in
    Performance1536D50K) echo 50000 ;;
    Performance768D1M)   echo 1000000 ;;
    *) echo 0 ;;
  esac
}

nlist_sqrt_n() {
  awk -v n="$1" 'BEGIN {
    if (n < 1) { print 1; exit }
    k = int(sqrt(n) + 0.5); if (k < 1) k = 1; if (k > n) k = n; print k
  }'
}

nlist_from_base_and_mult() {
  local base="$1"
  local n="$2"
  local mult="$3"
  awk -v b="$base" -v nn="$n" -v m="$mult" 'BEGIN {
    if (nn < 1) { print 1; exit }
    if (m <= 0) { print 1; exit }
    k = int(b * m + 0.5)
    if (k < 1) k = 1
    if (k > nn) k = nn
    print k
  }'
}

effective_ivf_nprobes() {
  local nlist="$1"
  if [[ -n "${IVF_NPROBES+x}" ]] && [[ -n "${IVF_NPROBES}" ]]; then
    echo "${IVF_NPROBES}"
  elif (( nlist <= IVF_NPROBES_MAX )); then
    echo "${nlist}"
  else
    echo "${IVF_NPROBES_MAX}"
  fi
}

# 两组：(数据目录名, 缩放比 mult)
RUNS=(
  "1536D50K:0.25"
  "768D1M:4"
)

failed=0

for _spec in "${RUNS[@]}"; do
  IFS=':' read -r _base _mult <<< "${_spec}"
  ct="$(resolve_case_type "${_base}")"
  if [[ -z "${ct}" ]]; then
    echo "ERROR: unknown dataset base '${_base}'" >&2
    exit 1
  fi
  _ds_n="$(dataset_train_vector_count_for_case "${ct}")"
  if (( _ds_n <= 0 )); then
    echo "ERROR: cannot resolve N for CASE_TYPE=${ct}" >&2
    exit 1
  fi

  base_nlist="$(nlist_sqrt_n "${_ds_n}")"
  nlist="$(nlist_from_base_and_mult "${base_nlist}" "${_ds_n}" "${_mult}")"
  spn=$((_ds_n / nlist))
  if (( spn < 1 )); then
    echo "ERROR: spn<1 for ${_base} mult=${_mult} nlist=${nlist} N=${_ds_n}" >&2
    exit 1
  fi
  probes="$(effective_ivf_nprobes "${nlist}")"

  mult_tag="${_mult//./p}"
  if [[ "${_mult}" == "1" ]]; then
    label="sweep_${_base}_sqrtn_nlist${nlist}_spn${spn}_N${_ds_n}"
  else
    label="sweep_${_base}_sqrtn_m${mult_tag}_nlist${nlist}_spn${spn}_N${_ds_n}"
  fi

  echo "========================================"
  echo "dataset=${_base}  CASE_TYPE=${ct}  N=${_ds_n}"
  echo "base_nlist=round(sqrt(N))=${base_nlist}  mult=${_mult}  nlist=${nlist}  SAMPLE_PER_NLIST=${spn}  IVF_NPROBES=${probes}"
  echo "VDB_TASK_LABEL=${label}"
  echo "start: $(date -Iseconds)"
  echo "========================================"

  export OB_TABLE_NAME="vdb_$(echo "${_base}" | tr '[:upper:]' '[:lower:]')"
  export DATASET_LOCAL_DIR
  export CASE_TYPE="${ct}"
  export NLIST="${nlist}"
  export SAMPLE_PER_NLIST="${spn}"
  export IVF_NPROBES="${probes}"

  for ((i = 1; i <= NUM_RUNS; i++)); do
    if (( NUM_RUNS > 1 )); then
      _suf="_r${i}of${NUM_RUNS}"
    else
      _suf=""
    fi
    _p="${VDB_RUN_LABEL_PREFIX:-}"
    _p="${_p// /_}"
    if [[ -n "${_p}" ]]; then
      export VDB_TASK_LABEL="${_p}_${label}${_suf}"
    else
      export VDB_TASK_LABEL="${label}${_suf}"
    fi

    echo "--- run ${i}/${NUM_RUNS}  VDB_TASK_LABEL=${VDB_TASK_LABEL} ---"
    set +e
    bash "${INNER}" "$@"
    rc=$?
    set -e
    echo "run ${i}/${NUM_RUNS} end: $(date -Iseconds) exit=${rc}"
    if [[ "${rc}" -ne 0 ]]; then
      failed=1
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        echo "ERROR: run failed, stopping (CONTINUE_ON_ERROR=1 to continue)." >&2
        exit "${rc}"
      fi
    fi
  done
  echo
done

if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: one or more runs failed." >&2
  exit 1
fi

echo "All rerun step(s) completed (${#RUNS[@]} configs × ${NUM_RUNS} run(s) each)."
exit 0
