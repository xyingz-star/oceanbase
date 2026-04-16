#!/bin/bash
#
# 向量表 768D1M 导入脚本
# 流程：建外部表 -> 建内部向量表 -> 从外部表导入 -> 校验行数
#
# 使用前请按实际环境修改下方变量。
#

set -e

# ---------- 连接与路径配置 ----------
HOST="${OB_HOST:-11.162.217.149}"
PORT="${OB_PORT:-2442}"
USER_TENANT="${OB_USER_TENANT:-root@mysql_tenant}"
DB="${OB_DB:-test}"

# 外部数据路径（OceanBase 节点上可访问的路径）
DATA_LOCATION="${OB_VEC_DATA_LOCATION:-file:///data/zhuxueying.zxy/vec_data/}"
FILE_PATTERN="${OB_VEC_FILE_PATTERN:-shuffle_train.parquet}"

# 并行度等
PARALLEL_DEGREE="${OB_INSERT_PARALLEL:-2}"
LIMIT_ROWS="${OB_INSERT_LIMIT:-1000000}"

# ---------- 辅助：执行 SQL ----------
mysql_cmd() {
  mysql -h"${HOST}" -P"${PORT}" -u"${USER_TENANT}" -D"${DB}" -A -c "$@"
}

mysql_cmd_quiet() {
  mysql -h"${HOST}" -P"${PORT}" -u"${USER_TENANT}" -D"${DB}" -A -c -N "$@"
}

# ---------- 1. 外部表 ----------
echo "[1/5] DROP EXTERNAL TABLE IF EXISTS ex_vec_768d1m ..."
mysql_cmd -e "DROP TABLE IF EXISTS ex_vec_768d1m;"

echo "[2/5] CREATE EXTERNAL TABLE ex_vec_768d1m ..."
mysql_cmd -e "
  CREATE EXTERNAL TABLE ex_vec_768d1m (
    id bigint,
    emb ARRAY(DOUBLE)
  )
  location='${DATA_LOCATION}'
  format (
    type = 'PARQUET'
  )
  PATTERN = '${FILE_PATTERN}';
"

# ---------- 2. 内部向量表 ----------
echo "[3/5] DROP TABLE IF EXISTS vec_768d1m ..."
mysql_cmd -e "DROP TABLE IF EXISTS vec_768d1m;"

echo "[4/5] CREATE TABLE vec_768d1m ..."
mysql_cmd -e "
  CREATE TABLE vec_768d1m (
    id bigint,
    embedding VECTOR(768)
  ) partition by hash(id) partitions 16;
"

# ---------- 3. 导入并校验 ----------
echo "[5/5] 查询外部表行数 -> 导入 -> 校验内部表行数 ..."
source_rows=$(mysql_cmd_quiet -e "SELECT COUNT(*) FROM ex_vec_768d1m")
echo "  ex_vec_768d1m 行数: ${source_rows}"

mysql_cmd -e "
  INSERT /*+ parallel(${PARALLEL_DEGREE}) append enable_parallel_dml */
  INTO vec_768d1m (id, embedding)
  SELECT id, emb
  FROM ex_vec_768d1m
  ORDER BY id
  LIMIT ${LIMIT_ROWS};
"

dest_rows=$(mysql_cmd_quiet -e "SELECT COUNT(*) FROM vec_768d1m")
echo "  vec_768d1m 行数: ${dest_rows}"

echo "done."
