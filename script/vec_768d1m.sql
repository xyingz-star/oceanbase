-- 向量表 768D1M 建表与导入用 SQL（需在 OceanBase test 库执行）
-- 外部数据路径、PATTERN 等需按实际环境修改后执行

-- 1. 外部表
DROP TABLE IF EXISTS ex_vec_768d1m;

CREATE EXTERNAL TABLE ex_vec_768d1m (
  id bigint,
  emb ARRAY(DOUBLE)
)
location='file:///data/zhuxueying.zxy/vec_data/'
format (
  type = 'PARQUET'
)
PATTERN = 'shuffle_train.parquet';

-- 2. 内部向量表
DROP TABLE IF EXISTS vec_768d1m;

CREATE TABLE vec_768d1m (
  id bigint,
  embedding VECTOR(768)
) partition by hash(id) partitions 16;

-- 3. 导入（在客户端或脚本里执行，便于加 hint 和 LIMIT）
-- INSERT /*+ parallel(2) append enable_parallel_dml */ INTO vec_768d1m (id, embedding)
-- SELECT id, emb FROM ex_vec_768d1m ORDER BY id LIMIT 1000000;

-- 4. 校验
-- SELECT COUNT(*) FROM ex_vec_768d1m;
-- SELECT COUNT(*) FROM vec_768d1m;
