#!/usr/bin/env bash
#
# 单次 OceanBase IVF 压测：封装 `vectordbbench oceanbaseivf`，供 sweep / rerun 脚本调用。
#
# 必填环境变量（与 vectordbbench_oceanbaseivf_rerun_two_mults.sh 一致）：
#   CASE_TYPE          例如 Performance1536D50K、Performance768D1M
#   NLIST              聚类数
#   SAMPLE_PER_NLIST   IVF 训练采样倍数
#   IVF_NPROBES        查询时探测的 list 数（字符串即可，如 56）
#   VDB_TASK_LABEL     结果 JSON 的 task_label
#
# OceanBase 连接（必填）：
#   OB_USER, OB_DATABASE, OB_PORT
#   OB_HOST              默认 127.0.0.1
#   OB_PASSWORD          未设置或空 = 使用空密码连接（VectorDBBench 已允许空 password）
#
# 表名（不同数据集用不同表；传给 CLI --table-name）：
#   OB_TABLE_NAME        优先；未设时用 VDB_TABLE_NAME；再未设默认 items
#   --skip-drop-old 时：表已存在则不建表；不存在则自动 CREATE 空表（须再跑 --load 灌数）
#
# 可选：
#   VECTORDDBENCH_ROOT   VectorDBBench 源码根目录（用于 PYTHONPATH）
#                        未设置时自动在若干候选路径中选取「含 vectordb_bench/cli/cli.py」的完整仓库
#                        （test/VectorDBBench 可能不完整，会跳过并选用 ~/VectorDBBench 等）
#   INDEX_TYPE           默认 ivf_flat（可选 ivf_sq8、ivf_pq）；export INDEX_TYPE=ivf_flat 可切回 FLAT
#   IVF_PQ_SUBDIM        仅 ivf_pq：每段 sub-vector 维数，默认 8 → m=dim/subdim（1536D→m=192，768D→m=96）
#   IVF_PQ_M / IVF_M     仅 ivf_pq：显式指定 m（覆盖 IVF_PQ_SUBDIM 推导；须整除 dim）
#   IVF_PQ_NBITS / IVF_NBITS  仅 ivf_pq：PQ 码本位数，默认 8（OceanBase 支持 1–24）
#   DATASET_LOCAL_DIR    数据集根目录，默认 ${VEC_DATA_ROOT:-/data/zhuxueying.zxy/vec_data}
#   VDB_NUM_CONCURRENCY  默认 80（逗号列表会传给 --num-concurrency）
#   VDB_SKIP_DROP_OLD=1  等价 --skip-drop-old（复用已有表结构，不 DROP）
#   VDB_SKIP_LOAD=1      等价 --skip-load（不重导数据）
#   VDB_REBUILD_INDEX=1  等价 --rebuild-index：须与 VDB_SKIP_LOAD=1 同用；删旧向量索引 idx1、按当前 nlist 建索引，
#                        并记录 optimize_duration（CREATE INDEX 耗时，见 OceanBase 客户端）；表数据不变。
#   若由 vectordbbench_oceanbaseivf_sweep_vec_data_nlist.sh 调用：默认 SWEEP_AUTO_OB_STAGES=1 会自动设置上述三项，
#                        一般无需再 export（除非 SWEEP_AUTO_OB_STAGES=0）。
#   NMBKM div：每轮建索引前可写 /tmp/ob_nmbkm_min_n_scale（一行正整数），observer 内 K-means 会读取（sweep 已自动写）。
#   VDB_OB_SKIP_POST_CREATE_INDEX=1：CREATE VECTOR INDEX 成功后不再执行 MAJOR FREEZE、等待合并、dbms_stats（仅实验/测 k-means 耗时；默认不设置则与原先一致）。
#   VDB_SKIP_SEARCH_SERIAL / VDB_SKIP_SEARCH_CONCURRENT
#   PYTHON               默认 python3.11
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 必须包含 vectordb_bench/cli/cli.py，否则会出现 ModuleNotFoundError: vectordb_bench.cli.cli
_vdb_cli_mark="vectordb_bench/cli/cli.py"
resolve_vectordbbench_root() {
  local d
  # 注意：若 oceanbase 通过符号链接指到 /data/...，SCRIPT_DIR 会落在 /data/.../oceanbase/script，
  # 此时 ../../VectorDBBench 指向 /data/.../test/VectorDBBench，可能与 $HOME/test/VectorDBBench 不同步。
  for d in "${VECTORDDBENCH_ROOT:-}" \
    "${SCRIPT_DIR}/../../VectorDBBench" \
    "${HOME}/test/VectorDBBench" \
    "${HOME}/VectorDBBench" \
    "/home/zhuxueying.zxy/test/VectorDBBench" \
    "/home/zhuxueying.zxy/VectorDBBench"; do
    [[ -z "${d}" ]] && continue
    if [[ -f "${d}/${_vdb_cli_mark}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

if ! VECTORDDBENCH_ROOT="$(resolve_vectordbbench_root)"; then
  echo "ERROR: 未找到完整的 VectorDBBench 源码树（需要存在 ${_vdb_cli_mark}）。" >&2
  echo "请设置: export VECTORDDBENCH_ROOT=/path/to/VectorDBBench" >&2
  exit 1
fi

export PYTHONPATH="${VECTORDDBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python3.11}"

_missing=()
[[ -z "${CASE_TYPE:-}" ]] && _missing+=(CASE_TYPE)
[[ -z "${NLIST:-}" ]] && _missing+=(NLIST)
[[ -z "${SAMPLE_PER_NLIST:-}" ]] && _missing+=(SAMPLE_PER_NLIST)
[[ -z "${IVF_NPROBES:-}" ]] && _missing+=(IVF_NPROBES)
[[ -z "${VDB_TASK_LABEL:-}" ]] && _missing+=(VDB_TASK_LABEL)
[[ -z "${OB_USER:-}" ]] && _missing+=(OB_USER)
[[ -z "${OB_DATABASE:-}" ]] && _missing+=(OB_DATABASE)
[[ -z "${OB_PORT:-}" ]] && _missing+=(OB_PORT)
if ((${#_missing[@]} > 0)); then
  echo "ERROR: 缺少环境变量: ${_missing[*]}" >&2
  echo "参见本脚本头部注释；OceanBase 连接需: OB_USER, OB_DATABASE, OB_PORT（及可选 OB_HOST、OB_PASSWORD）。" >&2
  exit 1
fi

OB_HOST="${OB_HOST:-127.0.0.1}"
# 未设置 OB_PASSWORD 时保持为空（真正无密码），勿再默认成空格以免 1045
OB_PASSWORD="${OB_PASSWORD:-}"

normalize_index_type() {
  local t="${1,,}"
  case "${t}" in
    ivf_flat | ivfflat) echo "IVF_FLAT" ;;
    ivf_sq8 | ivfsq8) echo "IVF_SQ8" ;;
    ivf_pq | ivfpq) echo "IVF_PQ" ;;
    ivf_*) echo "${1^^}" ;;
    *) echo "${1}" ;;
  esac
}

case_type_vector_dim() {
  local ct="$1"
  if [[ "${ct}" =~ ([0-9]+)[dD] ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

ivf_pq_m_for_subdim() {
  local dim="$1"
  local subdim="$2"
  if [[ -z "${dim}" ]] || [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: cannot derive IVF_PQ m from CASE_TYPE=${CASE_TYPE:-}; set IVF_PQ_M or use Performance*D* CaseType" >&2
    return 1
  fi
  if [[ ! "${subdim}" =~ ^[0-9]+$ ]] || (( subdim < 1 )); then
    echo "ERROR: IVF_PQ_SUBDIM must be a positive integer, got: ${subdim}" >&2
    return 1
  fi
  if (( dim % subdim != 0 || dim < subdim )); then
    echo "ERROR: dim=${dim} not divisible by IVF_PQ_SUBDIM=${subdim} (need m=dim/subdim integer)" >&2
    return 1
  fi
  echo $((dim / subdim))
}

INDEX_TYPE="$(normalize_index_type "${INDEX_TYPE:-ivf_flat}")"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-${VEC_DATA_ROOT:-/data/zhuxueying.zxy/vec_data}}"
export DATASET_LOCAL_DIR

IVF_PQ_EXTRA_ARGS=()
if [[ "${INDEX_TYPE}" == "IVF_PQ" ]]; then
  _pq_dim="$(case_type_vector_dim "${CASE_TYPE:-}" 2>/dev/null || true)"
  IVF_PQ_SUBDIM="${IVF_PQ_SUBDIM:-8}"
  IVF_PQ_M="${IVF_PQ_M:-${IVF_M:-}}"
  if [[ -z "${IVF_PQ_M}" ]]; then
    IVF_PQ_M="$(ivf_pq_m_for_subdim "${_pq_dim}" "${IVF_PQ_SUBDIM}")" || exit 1
  fi
  IVF_PQ_NBITS="${IVF_PQ_NBITS:-${IVF_NBITS:-8}}"
  if [[ -n "${_pq_dim}" ]] && (( _pq_dim % IVF_PQ_M == 0 )); then
    IVF_PQ_SUBDIM=$((_pq_dim / IVF_PQ_M))
  fi
  if [[ ! "${IVF_PQ_M}" =~ ^[0-9]+$ ]] || (( IVF_PQ_M < 1 )); then
    echo "ERROR: IVF_PQ_M must be a positive integer, got: ${IVF_PQ_M}" >&2
    exit 1
  fi
  if [[ -n "${_pq_dim}" ]] && (( _pq_dim % IVF_PQ_M != 0 || _pq_dim < IVF_PQ_M )); then
    echo "ERROR: IVF_PQ_M=${IVF_PQ_M} invalid for dim=${_pq_dim} (must divide dim and be <= dim)" >&2
    exit 1
  fi
  if [[ ! "${IVF_PQ_NBITS}" =~ ^[0-9]+$ ]] || (( IVF_PQ_NBITS < 1 || IVF_PQ_NBITS > 24 )); then
    echo "ERROR: IVF_PQ_NBITS must be in [1,24], got: ${IVF_PQ_NBITS}" >&2
    exit 1
  fi
  IVF_PQ_EXTRA_ARGS=(--m "${IVF_PQ_M}" --nbits "${IVF_PQ_NBITS}")
fi

TABLE_NAME="${OB_TABLE_NAME:-${VDB_TABLE_NAME:-items}}"

NUM_CONCURRENCY_STR="${VDB_NUM_CONCURRENCY:-80}"
if [[ "${NUM_CONCURRENCY_STR}" != *","* ]]; then
  NUM_CONCURRENCY_STR="${NUM_CONCURRENCY_STR}"
fi

DROP_ARGS=(--drop-old)
[[ "${VDB_SKIP_DROP_OLD:-0}" == "1" ]] && DROP_ARGS=(--skip-drop-old)

LOAD_ARGS=(--load)
[[ "${VDB_SKIP_LOAD:-0}" == "1" ]] && LOAD_ARGS=(--skip-load)

SER_ARGS=(--search-serial)
[[ "${VDB_SKIP_SEARCH_SERIAL:-0}" == "1" ]] && SER_ARGS=(--skip-search-serial)

CONC_ARGS=(--search-concurrent)
[[ "${VDB_SKIP_SEARCH_CONCURRENT:-0}" == "1" ]] && CONC_ARGS=(--skip-search-concurrent)

REBUILD_ARGS=()
[[ "${VDB_REBUILD_INDEX:-0}" == "1" ]] && REBUILD_ARGS=(--rebuild-index)

echo "### vectordbbench_oceanbaseivf_single $(date -Iseconds)"
echo "### PYTHONPATH=${VECTORDDBENCH_ROOT}"
echo "### CASE_TYPE=${CASE_TYPE} NLIST=${NLIST} SAMPLE_PER_NLIST=${SAMPLE_PER_NLIST} IVF_NPROBES=${IVF_NPROBES} INDEX_TYPE=${INDEX_TYPE}"
if [[ "${INDEX_TYPE}" == "IVF_PQ" ]]; then
  echo "### IVF_PQ dim=${_pq_dim:-?} subdim=${IVF_PQ_SUBDIM} m=${IVF_PQ_M} nbits=${IVF_PQ_NBITS}"
fi
echo "### VDB_TASK_LABEL=${VDB_TASK_LABEL}"
echo "### OB ${OB_USER}@${OB_HOST}:${OB_PORT}/${OB_DATABASE} table=${TABLE_NAME}"
echo "### stages: ${DROP_ARGS[*]} ${LOAD_ARGS[*]} ${REBUILD_ARGS[*]} ${SER_ARGS[*]} ${CONC_ARGS[*]}"

exec "${PYTHON}" -m vectordb_bench.cli.vectordbbench oceanbaseivf \
  "${DROP_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  "${REBUILD_ARGS[@]}" \
  "${SER_ARGS[@]}" \
  "${CONC_ARGS[@]}" \
  --case-type "${CASE_TYPE}" \
  --task-label "${VDB_TASK_LABEL}" \
  --host "${OB_HOST}" \
  --user "${OB_USER}" \
  --password "${OB_PASSWORD}" \
  --database "${OB_DATABASE}" \
  --port "${OB_PORT}" \
  --table-name "${TABLE_NAME}" \
  --index-type "${INDEX_TYPE}" \
  --nlist "${NLIST}" \
  --sample_per_nlist "${SAMPLE_PER_NLIST}" \
  --ivf_nprobes "${IVF_NPROBES}" \
  "${IVF_PQ_EXTRA_ARGS[@]}" \
  --num-concurrency "${NUM_CONCURRENCY_STR}" \
  "$@"
