#!/usr/bin/env bash
#
# IVF sweep（仅标准 / GPU 外部 K-means 路径）：与 vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh 相同地
# 遍历 数据集 × nlist 倍数（SWEEP_NLIST_MULTS），但不做 NMBKM div（OB_NMBKM_MIN_N_SCALE）遍历。
#
# 每轮 bench 前默认删除 SWEEP_NMBKM_DIV_FILE（默认 /tmp/ob_nmbkm_min_n_scale），避免历史 sweep 写入的
# div 仍被 observer 读取，从而误走 NMBKM 相关逻辑；便于专注验证「全量 / 外部 GPU」K-means。
# 关闭删除： SWEEP_CLEAR_NMBKM_DIV_FILE=0
#
# Observer 侧需已按你们环境打开外部 GPU K-means（及 worker 等）；本脚本只负责 VectorDBBench 侧参数与清 div 文件。
#
# 其余行为（自动表阶段、daemon、日志目录、ONLY_DIRS、SWEEP_SKIP_DIRS 等）与原 sweep 一致。
# 默认日志目录： ~/log/vdb_ivf_sweep_gpu_km（可用 SWEEP_LOG_DIR 覆盖）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ "${SWEEP_DAEMONIZE:-0}" == "1" ]] && [[ -z "${SWEEP_DAEMONIZE_CHILD:-}" ]]; then
  _daemon_log="${SWEEP_DAEMONIZE_LOG:-${HOME}/log/vdb_ivf_sweep_gpu_km/sweep_daemon_$(date +%Y%m%dT%H%M%S).log}"
  mkdir -p "$(dirname "${_daemon_log}")"
  nohup bash -c "
    export SWEEP_DAEMONIZE_CHILD=1
    unset SWEEP_DAEMONIZE
    exec >>\"${_daemon_log}\" 2>&1
    exec bash \"${SELF_SCRIPT}\"
  " </dev/null &
  echo "SWEEP_DAEMONIZE: PID=$! log=${_daemon_log}" >&2
  echo "hint: tail -f ${_daemon_log}" >&2
  exit 0
fi

INNER="${SCRIPT_DIR}/vectordbbench_oceanbaseivf_single.sh"
VEC_DATA_ROOT="${VEC_DATA_ROOT:-/data/zhuxueying.zxy/vec_data}"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-${VEC_DATA_ROOT}}"
IVF_NPROBES_MAX="${IVF_NPROBES_MAX:-100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SWEEP_LOG_ENABLE="${SWEEP_LOG_ENABLE:-1}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${HOME}/log/vdb_ivf_sweep_gpu_km}"
SWEEP_LOG_TIMESTAMP="${SWEEP_LOG_TIMESTAMP:-1}"
SWEEP_NLIST_MULTS="${SWEEP_NLIST_MULTS:-0.25 0.5 1 2 4}"
# 与 observer 读取路径一致；仅用于 rm，本脚本不再写入 div
SWEEP_NMBKM_DIV_FILE="${SWEEP_NMBKM_DIV_FILE:-/tmp/ob_nmbkm_min_n_scale}"
# 1=每轮 bench 前删除 div 文件，避免 NMBKM 配置残留
SWEEP_CLEAR_NMBKM_DIV_FILE="${SWEEP_CLEAR_NMBKM_DIV_FILE:-1}"
SWEEP_SKIP_DIRS="${SWEEP_SKIP_DIRS:-768D10M}"
export VDB_NUM_CONCURRENCY="${VDB_NUM_CONCURRENCY:-80}"
SWEEP_AUTO_OB_STAGES="${SWEEP_AUTO_OB_STAGES:-1}"

DATASET_ORDER_SMALL_FIRST=(
  1536D50K 1536D500K 768D1M cohere openai 1536D5M 768D100K 768D10M
)

[[ -f "${INNER}" ]] || { echo "ERROR: missing ${INNER}" >&2; exit 1; }

require_ob_connection_env() {
  local missing=()
  [[ -z "${OB_USER:-}" ]] && missing+=(OB_USER)
  [[ -z "${OB_DATABASE:-}" ]] && missing+=(OB_DATABASE)
  [[ -z "${OB_PORT:-}" ]] && missing+=(OB_PORT)
  if ((${#missing[@]} > 0)); then
    echo "ERROR: 未设置 OceanBase 连接变量: ${missing[*]}" >&2
    echo "请先在同一 shell 中 export，再运行本脚本，例如：" >&2
    echo "  export OB_HOST=127.0.0.1" >&2
    echo "  export OB_PORT=11000" >&2
    echo "  export OB_USER='root@你的租户'" >&2
    echo "  export OB_DATABASE=你的库名" >&2
    echo "  export OB_PASSWORD=...        # 空密码可省略或 export OB_PASSWORD=''" >&2
    exit 1
  fi
}
require_ob_connection_env
export OB_HOST="${OB_HOST:-127.0.0.1}"
export OB_PASSWORD="${OB_PASSWORD:-}"

sweep_table_for_dataset() {
  local base="$1"
  echo "vdb_$(echo "${base}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_')"
}

_ob_mysql_raw() {
  local sql="$1"
  export MYSQL_PWD="${OB_PASSWORD:-}"
  local out
  if ! out="$(mysql -h "${OB_HOST}" -P "${OB_PORT}" --protocol=TCP -u "${OB_USER}" -N -s -e "${sql}" "${OB_DATABASE}" 2>/dev/null)"; then
    unset MYSQL_PWD
    return 1
  fi
  unset MYSQL_PWD
  printf '%s' "${out}"
}

_ob_table_exists() {
  local tbl="$1"
  local n
  n="$(_ob_mysql_raw "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND LOWER(table_name) = LOWER('${tbl}');")" || return 1
  n="${n//$'\r'/}"
  n="${n// /}"
  [[ "${n:-0}" =~ ^[0-9]+$ ]] && [[ "${n}" -gt 0 ]]
}

_ob_table_row_count() {
  local tbl="$1"
  local n
  n="$(_ob_mysql_raw "SELECT COUNT(*) FROM \`${tbl//\`/\`\`}\`;")" || return 1
  n="${n//$'\r'/}"
  n="${n// /}"
  [[ "${n}" =~ ^[0-9]+$ ]] || return 1
  echo "${n}"
}

sweep_auto_export_vdb_stages() {
  local tbl="$1" expect_n="$2"
  if [[ "${SWEEP_AUTO_OB_STAGES:-1}" != "1" ]]; then
    echo "### AUTO_OFF: VDB_SKIP_DROP_OLD=${VDB_SKIP_DROP_OLD:-0} VDB_SKIP_LOAD=${VDB_SKIP_LOAD:-0} VDB_REBUILD_INDEX=${VDB_REBUILD_INDEX:-0}"
    return 0
  fi
  if ! command -v mysql &>/dev/null; then
    echo "### AUTO: mysql 客户端未找到 → 全量 drop-old + load（请安装 mysql 或设 SWEEP_AUTO_OB_STAGES=0 自管）" >&2
    export VDB_SKIP_DROP_OLD=0 VDB_SKIP_LOAD=0 VDB_REBUILD_INDEX=0
    return 0
  fi
  if ! _ob_table_exists "${tbl}"; then
    echo "### AUTO: 表 ${tbl} 不存在 → drop-old + load + optimize（新建）"
    export VDB_SKIP_DROP_OLD=0 VDB_SKIP_LOAD=0 VDB_REBUILD_INDEX=0
    return 0
  fi
  local cnt
  cnt="$(_ob_table_row_count "${tbl}")" || {
    echo "### AUTO: 无法 COUNT ${tbl}，退化为全量" >&2
    export VDB_SKIP_DROP_OLD=0 VDB_SKIP_LOAD=0 VDB_REBUILD_INDEX=0
    return 0
  }
  if (( cnt == 0 )); then
    echo "### AUTO: 表 ${tbl} 为空 → skip-drop-old + load"
    export VDB_SKIP_DROP_OLD=1 VDB_SKIP_LOAD=0 VDB_REBUILD_INDEX=0
    return 0
  fi
  if (( cnt >= expect_n )); then
    echo "### AUTO: 表 ${tbl} 行数=${cnt} >= N=${expect_n} → skip-load + rebuild-index（建索引前删 idx1 由 bench 执行）"
    export VDB_SKIP_DROP_OLD=1 VDB_SKIP_LOAD=1 VDB_REBUILD_INDEX=1
    return 0
  fi
  echo "### AUTO: 表 ${tbl} 行数=${cnt} < N=${expect_n} → drop-old + load 刷新"
  export VDB_SKIP_DROP_OLD=0 VDB_SKIP_LOAD=0 VDB_REBUILD_INDEX=0
}

resolve_case_type() {
  case "$1" in
    1536D500K) echo Performance1536D500K ;;
    1536D50K)  echo Performance1536D50K ;;
    768D1M|cohere) echo Performance768D1M ;;
    *) echo "" ;;
  esac
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

dataset_train_vector_count_for_case() {
  case "$1" in
    Performance1536D50K) echo 50000 ;;
    Performance1536D500K) echo 500000 ;;
    Performance1536D5M) echo 5000000 ;;
    Performance768D1M) echo 1000000 ;;
    Performance768D10M) echo 10000000 ;;
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

order_vec_data_dirs_small_first() {
  local root="$1"
  shift
  local -a candidates=("$@")
  local -A want=() seen=()
  local name x
  for name in "${candidates[@]}"; do [[ -n "${name}" ]] && want["$name"]=1; done
  local -a out=() rest=()
  for x in "${DATASET_ORDER_SMALL_FIRST[@]}"; do
    if [[ -n "${want[$x]:-}" ]] && [[ -d "${root}/${x}" ]]; then
      out+=("$x"); seen["$x"]=1
    fi
  done
  for name in "${candidates[@]}"; do
    [[ -n "${name}" ]] || continue
    [[ -n "${seen[$name]:-}" ]] && continue
    [[ -d "${root}/${name}" ]] || continue
    rest+=("$name")
  done
  ((${#rest[@]})) && mapfile -t rest < <(printf '%s\n' "${rest[@]}" | LC_ALL=C sort)
  for x in "${rest[@]}"; do
    out+=("$x")
  done
  printf '%s\n' "${out[@]}"
}

mkdir -p "${DATASET_LOCAL_DIR}"
if [[ -n "${ONLY_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  mapfile -t _raw < <(printf '%s\n' ${ONLY_DIRS})
else
  mapfile -t _raw < <(find "${VEC_DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')
fi
mapfile -t _discovered < <(order_vec_data_dirs_small_first "${VEC_DATA_ROOT}" "${_raw[@]}")

failed=0
swept_performance_768d1m=0
for base in "${_discovered[@]}"; do
  [[ -n "${base}" ]] || continue
  [[ -d "${VEC_DATA_ROOT}/${base}" ]] || { echo "WARN: missing ${VEC_DATA_ROOT}/${base}" >&2; continue; }

  _skip=0
  for _s in ${SWEEP_SKIP_DIRS}; do
    [[ "${base}" == "${_s}" ]] && _skip=1 && break
  done
  if ((_skip)); then
    echo "SKIP '${base}' (SWEEP_SKIP_DIRS)" >&2
    continue
  fi

  ct="$(resolve_case_type "${base}")"
  if [[ -z "${ct}" ]]; then
    echo "SKIP '${base}' (no CaseType / unsupported)" >&2
    continue
  fi
  if [[ "${ct}" == "Performance768D1M" ]] && (( swept_performance_768d1m )); then
    echo "SKIP '${base}' (768D1M already run)" >&2
    continue
  fi

  _ds_n="$(dataset_train_vector_count_for_case "${ct}")"
  (( _ds_n > 0 )) || { echo "WARN: no N for ${ct}" >&2; continue; }

  base_nlist="$(nlist_sqrt_n "${_ds_n}")"

  [[ "${ct}" == "Performance768D1M" ]] && swept_performance_768d1m=1

  for mult in ${SWEEP_NLIST_MULTS}; do
    nlist="$(nlist_from_base_and_mult "${base_nlist}" "${_ds_n}" "${mult}")"

    spn=$((_ds_n / nlist))
    (( spn >= 1 )) || { echo "SKIP '${base}' mult=${mult} spn<1 nlist=${nlist}" >&2; continue; }

    probes="$(effective_ivf_nprobes "${nlist}")"
    mult_tag="${mult}"
    mult_tag="${mult_tag//./p}"

    if [[ "${mult}" == "1" ]]; then
      label="sweep_${base}_gpu_km_sqrtn_nlist${nlist}_spn${spn}_N${_ds_n}"
    else
      label="sweep_${base}_gpu_km_sqrtn_m${mult_tag}_nlist${nlist}_spn${spn}_N${_ds_n}"
    fi

    echo "==== ${base} ${ct} N=${_ds_n} base_nlist=${base_nlist} mult=${mult} nlist=${nlist} spn=${spn} probes=${probes} GPU_KM_SWEEP clear_nmbkm_div=${SWEEP_CLEAR_NMBKM_DIV_FILE} conc=${VDB_NUM_CONCURRENCY} $(date -Iseconds) ===="

    if [[ "${SWEEP_CLEAR_NMBKM_DIV_FILE}" == "1" ]]; then
      rm -f "${SWEEP_NMBKM_DIV_FILE}"
    fi

    export OB_TABLE_NAME="$(sweep_table_for_dataset "${base}")"
    sweep_auto_export_vdb_stages "${OB_TABLE_NAME}" "${_ds_n}"
    export DATASET_LOCAL_DIR CASE_TYPE="${ct}" NLIST="${nlist}" SAMPLE_PER_NLIST="${spn}" IVF_NPROBES="${probes}" VDB_TASK_LABEL="${label}"

    _sweep_log=""
    if [[ "${SWEEP_LOG_ENABLE}" == "1" ]]; then
      mkdir -p "${SWEEP_LOG_DIR}"
      if [[ "${SWEEP_LOG_TIMESTAMP}" == "1" ]]; then
        _sweep_log="${SWEEP_LOG_DIR}/${label}_$(date +%Y%m%dT%H%M%S)_km.log"
      else
        _sweep_log="${SWEEP_LOG_DIR}/${label}_km.log"
      fi
    fi

    set +e
    if [[ -n "${_sweep_log}" ]]; then
      {
        echo "### ${label} $(date -Iseconds)"
        echo "### table=${OB_TABLE_NAME} AUTO_STAGES=${SWEEP_AUTO_OB_STAGES} SKIP_DROP=${VDB_SKIP_DROP_OLD:-0} SKIP_LOAD=${VDB_SKIP_LOAD:-0} REBUILD_IDX=${VDB_REBUILD_INDEX:-0}"
        echo "### conc=${VDB_NUM_CONCURRENCY} SERIAL_SKIP=${VDB_SKIP_SEARCH_SERIAL:-0}"
        echo "### SWEEP_NLIST_MULTS base_nlist=${base_nlist} mult=${mult} (GPU/std KM sweep; no NMBKM div)"
        echo "### SWEEP_CLEAR_NMBKM_DIV_FILE=${SWEEP_CLEAR_NMBKM_DIV_FILE} SWEEP_NMBKM_DIV_FILE=${SWEEP_NMBKM_DIV_FILE}"
        echo
      } >"${_sweep_log}"
      bash "${INNER}" 2>&1 | tee -a "${_sweep_log}"
      rc=${PIPESTATUS[0]:-1}
    else
      bash "${INNER}"
      rc=$?
    fi
    set -e
    echo "end rc=${rc} ${_sweep_log:+log=${_sweep_log}}"
    if [[ "${rc}" -ne 0 ]]; then
      failed=1
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit "${rc}"
    fi
  done
done

(( failed == 0 )) || { echo "ERROR: sweep had failures" >&2; exit 1; }
echo "Done."
