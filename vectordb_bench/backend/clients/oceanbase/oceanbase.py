import logging
import os
import struct
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mysql.connector as mysql
from mysql.connector.errors import ProgrammingError

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import OceanBaseConfigDict, OceanBaseHNSWConfig

log = logging.getLogger(__name__)

OCEANBASE_DEFAULT_LOAD_BATCH_SIZE = 256


def _quote_sql_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


class OceanBase(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: OceanBaseConfigDict,
        db_case_config: OceanBaseHNSWConfig,
        collection_name: str = "items",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "OceanBase"
        self.dim = dim
        self.db_config = db_config
        self.db_case_config = db_case_config
        self.table_name = collection_name
        self.load_batch_size = OCEANBASE_DEFAULT_LOAD_BATCH_SIZE
        self._index_name = "vidx"
        self._primary_field = "id"
        self._vector_field = "embedding"

        log.info(
            f"{self.name} initialized with config:\nDatabase: {self.db_config}\nCase Config: {self.db_case_config}"
        )

        self._conn = None
        self._cursor = None

        try:
            self._connect()
            if drop_old:
                self._drop_table()
                self._create_table()
            else:
                self._ensure_table_exists()
        finally:
            self._disconnect()

    def _connect(self):
        try:
            pwd = self.db_config["password"]
            # 连接层使用真实空串；勿把占位空格当作密码发给服务端
            if pwd == " ":
                pwd = ""
            dbname = self.db_config["database"]
            common = dict(
                host=self.db_config["host"],
                user=self.db_config["user"],
                port=self.db_config["port"],
                password=pwd,
            )
            try:
                self._conn = mysql.connect(database=dbname, **common)
            except ProgrammingError as e:
                # 1049: unknown database — 开发环境常见未预先建库
                if e.errno != 1049:
                    log.exception("Failed to connect to the database")
                    raise
                log.warning(
                    "Database %r does not exist; creating with CREATE DATABASE IF NOT EXISTS",
                    dbname,
                )
                admin = mysql.connect(**common)
                try:
                    cur = admin.cursor()
                    try:
                        cur.execute(f"CREATE DATABASE IF NOT EXISTS {_quote_sql_ident(dbname)}")
                        admin.commit()
                    finally:
                        cur.close()
                finally:
                    admin.close()
                self._conn = mysql.connect(database=dbname, **common)
            self._cursor = self._conn.cursor()
        except mysql.Error:
            log.exception("Failed to connect to the database")
            raise

    def _disconnect(self):
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        try:
            self._connect()
            self._cursor.execute("SET autocommit=1")

            if self.db_case_config.index in {IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ}:
                self._cursor.execute(
                    f"SET ob_hnsw_ef_search={(self.db_case_config.search_param())['params']['ef_search']}"
                )
            else:
                self._cursor.execute(
                    f"SET ob_ivf_nprobes={(self.db_case_config.search_param())['params']['ivf_nprobes']}"
                )
            yield
        finally:
            self._disconnect()

    def _drop_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        # DROP IF EXISTS：表本就不存在时也是同一条路径，避免误解为「一定删了有数据的表」
        log.info(
            "DROP TABLE IF EXISTS %s (remove old bench table if present; no-op if absent)",
            self.table_name,
        )
        self._cursor.execute(f"DROP TABLE IF EXISTS {_quote_sql_ident(self.table_name)}")

    def _table_exists(self) -> bool:
        assert self._cursor
        self._cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND LOWER(table_name) = LOWER(%s)",
            (self.table_name,),
        )
        row = self._cursor.fetchone()
        return bool(row and row[0] > 0)

    def _ensure_table_exists(self) -> None:
        """With --skip-drop-old: do not drop; create empty table only if missing."""
        if self._table_exists():
            log.info("Table %s already exists; skip CREATE TABLE", self.table_name)
            return
        log.info("Table %s not found; creating", self.table_name)
        self._create_table()

    def _create_table(self):
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        log.info(f"Creating table {self.table_name}")
        t = _quote_sql_ident(self.table_name)
        create_table_query = f"""
        CREATE TABLE {t} (
            id INT PRIMARY KEY,
            embedding VECTOR({self.dim})
        );
        """
        self._cursor.execute(create_table_query)

    def _drop_vector_index_if_exists(self) -> None:
        """Remove built-in bench index name so CREATE VECTOR INDEX can run again."""
        idx = "idx1"
        t = _quote_sql_ident(self.table_name)
        idxq = _quote_sql_ident(idx)
        # OceanBase/MySQL errno 1091: index does not exist — normal on first CREATE or after manual drop.
        for sql in (
            f"ALTER TABLE {t} DROP INDEX {idxq}",
            f"DROP INDEX {idxq} ON {t}",
        ):
            try:
                self._cursor.execute(sql)
                log.info("Dropped vector index %s on %s", idx, self.table_name)
                return
            except mysql.Error as e:
                if getattr(e, "errno", None) == 1091:
                    continue
                log.warning("Drop index %s on %s: %s", idx, self.table_name, e)
                return
        log.info(
            "No vector index %s on %s yet (1091); first build or already dropped — continue to CREATE",
            idx,
            self.table_name,
        )

    def optimize(self, data_size: int):
        self._drop_vector_index_if_exists()

        index_params = self.db_case_config.index_param()
        index_args = ", ".join(f"{k}={v}" for k, v in index_params["params"].items())
        t = _quote_sql_ident(self.table_name)
        index_query = (
            f"CREATE /*+ PARALLEL(18) */ VECTOR INDEX idx1 "
            f"ON {t}(embedding) "
            f"WITH (distance={self.db_case_config.parse_metric()}, "
            f"type={index_params['index_type']}, lib={index_params['lib']}, {index_args}"
        )

        if self.db_case_config.index in {IndexType.HNSW, IndexType.HNSW_SQ, IndexType.HNSW_BQ}:
            index_query += ", extra_info_max_size=32"

        index_query += ")"

        log.info("Create index query: %s", index_query)

        try:
            log.info("Creating index...")
            t0 = time.perf_counter()
            self._cursor.execute(index_query)
            # Bench metric optimize_duration: CREATE INDEX only (not major freeze / compaction / stats).
            self.reported_optimize_duration_s = time.perf_counter() - t0
            log.info(f"Index created in {self.reported_optimize_duration_s:.2f} seconds")

            # After CREATE INDEX, default: major freeze + wait compaction + gather stats (production-like).
            # Set VDB_OB_SKIP_POST_CREATE_INDEX=1 to skip these (e.g. k-means / index build experiments only).
            _skip_post = os.environ.get("VDB_OB_SKIP_POST_CREATE_INDEX", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if _skip_post:
                log.info(
                    "Skipping ALTER SYSTEM MAJOR FREEZE / compaction wait / gather_schema_stats "
                    "(VDB_OB_SKIP_POST_CREATE_INDEX=1)"
                )
            else:
                log.info("Performing major freeze...")
                self._cursor.execute("ALTER SYSTEM MAJOR FREEZE;")
                time.sleep(10)
                self._wait_for_major_compaction()

                log.info("Gathering schema statistics...")
                self._cursor.execute("CALL dbms_stats.gather_schema_stats('test', degree => 96);")
        except mysql.Error:
            log.exception("Failed to optimize index")
            raise

    def need_normalize_cosine(self) -> bool:
        if self.db_case_config.index == IndexType.HNSW_BQ:
            log.info("current HNSW_BQ only supports L2, cosine dataset need normalize.")
            return True

        return False

    def _wait_for_major_compaction(self):
        while True:
            self._cursor.execute(
                "SELECT IF(COUNT(*) = COUNT(STATUS = 'IDLE' OR NULL), 'TRUE', 'FALSE') "
                "AS all_status_idle FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION;"
            )
            all_status_idle = self._cursor.fetchone()[0]
            if all_status_idle == "TRUE":
                break
            time.sleep(10)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        insert_count = 0
        t = _quote_sql_ident(self.table_name)
        try:
            for batch_start in range(0, len(embeddings), self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, len(embeddings))
                batch = [(metadata[i], embeddings[i]) for i in range(batch_start, batch_end)]
                values = ", ".join(f"({item_id}, '[{','.join(map(str, embedding))}]')" for item_id, embedding in batch)
                self._cursor.execute(
                    f"INSERT /*+ ENABLE_PARALLEL_DML PARALLEL(32) */ INTO {t} VALUES {values}"  # noqa: S608
                )
                insert_count += len(batch)
        except mysql.Error:
            log.exception("Failed to insert embeddings")
            raise

        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"WHERE id >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"WHERE id == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Oceanbase - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        if not self._cursor:
            raise ValueError("Cursor is not initialized")

        packed = struct.pack(f"<{len(query)}f", *query)
        hex_vec = packed.hex()
        t = _quote_sql_ident(self.table_name)
        query_str = (
            f"SELECT id FROM {t} "  # noqa: S608
            f"{self.expr} ORDER BY "
            f"{self.db_case_config.parse_metric_func_str()}(embedding, X'{hex_vec}') "
            f"APPROXIMATE LIMIT {k}"
        )

        try:
            self._cursor.execute(query_str)
            return [row[0] for row in self._cursor.fetchall()]
        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
