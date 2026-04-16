#!/usr/bin/env bash
#
# IVF sweep：base_nlist=round(sqrt(N))，nlist=round(base_nlist×mult)，默认 mult 为 0.25 0.5 1 2 4（可用 SWEEP_NLIST_MULTS 覆盖）。
# 每个 (数据集 × 每个 mult × 每个 NMBKM scale) 各跑一轮；spn=floor(N/nlist)。不再按 NMBKM 样本数门槛跳过组合（仅保留 spn>=1 等基本检查）。
#
# NMBKM scale（SWEEP_OB_NMBKM_SCALES，默认 4 64）对应 K-means 除数 div（b0=n/div）。
# observer 每次建索引前会读取（优先级）：环境变量 OB_NMBKM_MIN_N_SCALE → OB_NMBKM_MIN_N_SCALE_FILE →
# /tmp/ob_nmbkm_min_n_scale（首行一个正整数）→ 默认 16。本脚本在每轮 bench 前写入 SWEEP_NMBKM_DIV_FILE（默认即 /tmp/...），
# 与 observer 同机且可读即可，无需改代码重编、无需重启 observer。
# 1536D5M 在 resolve_case_type 中注释；768D10M 默认由 SWEEP_SKIP_DIRS 排除（不跑、不依赖 CaseType）。
# 检索：串行阶段保留（recall/ndcg）；并发阶段仅一档 VDB_NUM_CONCURRENCY（默认 80）。不要串行可设 VDB_SKIP_SEARCH_SERIAL=1。
# 常用：ONLY_DIRS="1536D500K"  SWEEP_DAEMONIZE=1  VEC_DATA_ROOT=...
#
# ---------- 自动表 / 数据 / 索引策略（默认开，无需再 export VDB_SKIP_* / VDB_REBUILD_*）----------
# 仅需：OB_USER OB_DATABASE OB_PORT（及 OB_HOST、OB_PASSWORD）。表名固定规则：vdb_<数据集目录小写>（如 vdb_1536d50k）。
# 每轮用 mysql 客户端连库判断：
#   - 表不存在 → 全量：--drop-old --load（建新表+灌数+建索引）
#   - 表空 → --skip-drop-old --load（保留/建空表后灌数）
#   - 行数 >= 该 Case 训练规模 N → --skip-drop-old --skip-load --rebuild-index（只删 idx1 再建索引+检索；由 bench 内完成）
#   - 0 < 行数 < N → 全量刷新（drop-old + load）
# 需本机有 mysql 命令（或 MariaDB 客户端）。关闭自动： SWEEP_AUTO_OB_STAGES=0 后自行 export VDB_SKIP_*。
#
# ---------- 后台运行（关掉终端也不断；全量日志进文件）----------
# 设 SWEEP_DAEMONIZE=1：父进程立刻退出，子进程在后台跑，stdout/stderr 重定向到 daemon 日志。
#   SWEEP_DAEMONIZE=1 bash /path/to/vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh
# 默认日志：~/log/vdb_ivf_sweep/sweep_daemon_<时间戳>.log
# 指定 daemon 总日志：
#   SWEEP_DAEMONIZE=1 SWEEP_DAEMONIZE_LOG=~/log/vdb_ivf_sweep/my_run.log bash .../vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh
# 终端只看到一行： SWEEP_DAEMONIZE: PID=<后台进程号> log=<路径>
# 每条数据集 bench 另有 SWEEP_LOG_DIR 下按 task-label 命名的 *_km.log（与前台相同）。
# 建议长任务同时： CONTINUE_ON_ERROR=1（某一组合失败仍继续后面组合）。
# 查进度： tail -f <SWEEP_DAEMONIZE_LOG 或 tail -f ~/log/vdb_ivf_sweep/*_km.log
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ "${SWEEP_DAEMONIZE:-0}" == "1" ]] && [[ -z "${SWEEP_DAEMONIZE_CHILD:-}" ]]; then
  _daemon_log="${SWEEP_DAEMONIZE_LOG:-${HOME}/log/vdb_ivf_sweep/sweep_daemon_$(date +%Y%m%dT%H%M%S).log}"
  mkdir -p "$(dirname "${_daemon_log}")"
  # nohup + 脱离 stdin：关 SSH/终端时不易被 SIGHUP 打断；全量输出进 _daemon_log
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
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${HOME}/log/vdb_ivf_sweep}"
SWEEP_LOG_TIMESTAMP="${SWEEP_LOG_TIMESTAMP:-1}"
# 对 base_nlist=round(sqrt(N)) 依次乘以如下系数（空格分隔），得到本轮 nlist 并 clamp 到 [1,N]
SWEEP_NLIST_MULTS="${SWEEP_NLIST_MULTS:-0.25 0.5 1 2 4}"
# K-means NMBKM：除数 div。空格分隔。
# 未设置 SWEEP_OB_NMBKM_SCALES 时默认 4 64；若 export SWEEP_OB_NMBKM_SCALES= 为空字符串则只跑一轮且不写 div 文件（由已有 /tmp 文件或 observer 默认决定）。
SWEEP_OB_NMBKM_SCALES="${SWEEP_OB_NMBKM_SCALES-4 64}"
# 每轮写入该路径（observer 内同路径读取）；改路径需与 C++ 中 DEFAULT 一致或设 OB_NMBKM_MIN_N_SCALE_FILE 启动 observer
SWEEP_NMBKM_DIV_FILE="${SWEEP_NMBKM_DIV_FILE:-/tmp/ob_nmbkm_min_n_scale}"
# 发现到的数据目录名中，这些不参与 sweep（空格分隔）。清空可恢复 768D10M 等（需同时在 resolve_case_type 里映射 CaseType）。
SWEEP_SKIP_DIRS="${SWEEP_SKIP_DIRS:-768D10M}"
export VDB_NUM_CONCURRENCY="${VDB_NUM_CONCURRENCY:-80}"
# 1=按表行数自动设置 VDB_SKIP_DROP_OLD / VDB_SKIP_LOAD / VDB_REBUILD_INDEX；0=沿用环境变量
SWEEP_AUTO_OB_STAGES="${SWEEP_AUTO_OB_STAGES:-1}"

# 遍历顺序：先小数据再大数据；其余目录按名字排序接在后面（不再对每个未列名目录打 WARN）
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

# 表名仅含 [a-z0-9_]，避免 SQL 注入
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

# 根据表是否存在、行数是否已达到本 Case 的 N，导出 VDB_SKIP_DROP_OLD / VDB_SKIP_LOAD / VDB_REBUILD_INDEX
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
    # 以下规模大，默认不参与 sweep；需要时取消注释
    # 1536D5M)   echo Performance1536D5M ;;
    # 768D10M)   echo Performance768D10M ;;
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

# base_nlist = round(sqrt(N)), nlist = round(base_nlist * mult) clamped to [1, N]
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

    # 内层：NMBKM div（OB_NMBKM_MIN_N_SCALE）。未配置 SWEEP_OB_NMBKM_SCALES 时跑一轮且不 export。
    _scale_list=()
    if [[ -n "${SWEEP_OB_NMBKM_SCALES// }" ]]; then
      # shellcheck disable=SC2206
      _scale_list=(${SWEEP_OB_NMBKM_SCALES})
    else
      _scale_list=("")
    fi

    for scale in "${_scale_list[@]}"; do
      if [[ -n "${scale}" ]]; then
        printf '%s\n' "${scale}" >"${SWEEP_NMBKM_DIV_FILE}"
        chmod a+r "${SWEEP_NMBKM_DIV_FILE}" 2>/dev/null || true
      else
        : # 不写文件：沿用盘上已有 /tmp 文件或 observer 默认 16
      fi

      scale_tag=""
      if [[ -n "${scale}" ]]; then
        scale_tag="_div${scale}"
      fi

      if [[ "${mult}" == "1" ]]; then
        label="sweep_${base}_sqrtn${scale_tag}_nlist${nlist}_spn${spn}_N${_ds_n}"
      else
        label="sweep_${base}_sqrtn_m${mult_tag}${scale_tag}_nlist${nlist}_spn${spn}_N${_ds_n}"
      fi

      echo "==== ${base} ${ct} N=${_ds_n} base_nlist=${base_nlist} mult=${mult} nlist=${nlist} spn=${spn} probes=${probes} nmbkm_div=${scale:-<sweep_default>} file=${SWEEP_NMBKM_DIV_FILE} conc=${VDB_NUM_CONCURRENCY} $(date -Iseconds) ===="

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
          echo "### SWEEP_NLIST_MULTS base_nlist=${base_nlist} mult=${mult}"
          echo "### SWEEP_OB_NMBKM_SCALES scale=${scale:-} wrote ${SWEEP_NMBKM_DIV_FILE}=$(cat "${SWEEP_NMBKM_DIV_FILE}" 2>/dev/null || echo '?')"
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
done

(( failed == 0 )) || { echo "ERROR: sweep had failures" >&2; exit 1; }
echo "Done."
