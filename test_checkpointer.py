import json
from typing import Any, Iterator, List, Optional, Tuple, Dict
from contextlib import contextmanager
import logging
import pyodbc
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    PendingWrite,
    get_checkpoint_id,
)

class AzureSQLSaver(BaseCheckpointSaver):
    """Azure SQL Server-based checkpoint saver implementation."""

    connection_string: str

    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self._ensure_tables_exist()

    def _get_connection(self):
        return pyodbc.connect(self.connection_string)

    def _ensure_tables_exist(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='checkpoints_test' AND xtype='U')
            CREATE TABLE checkpoints_test (
                thread_id NVARCHAR(255),
                checkpoint_ns NVARCHAR(255),
                checkpoint_id NVARCHAR(255),
                checkpoint_data VARBINARY(MAX),
                checkpoint_type NVARCHAR(255),
                metadata VARBINARY(MAX),
                parent_checkpoint_id NVARCHAR(255),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """)
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='checkpoint_writes' AND xtype='U')
            CREATE TABLE checkpoint_writes (
                thread_id NVARCHAR(255),
                checkpoint_ns NVARCHAR(255),
                checkpoint_id NVARCHAR(255),
                task_id NVARCHAR(255),
                idx INT,
                channel NVARCHAR(255),
                value_type NVARCHAR(255),
                value NVARCHAR(MAX),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            )
            """)
            conn.commit()

    @classmethod
    @contextmanager
    def from_conn_info(cls, *, server: str, database: str, username: str, password: str) -> Iterator["AzureSQLSaver"]:
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}"
        )
        saver = AzureSQLSaver(connection_string)
        try:
            yield saver
        finally:
            pass

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id", "")

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            MERGE checkpoints_test AS target
            USING (SELECT ? AS thread_id, ? AS checkpoint_ns, ? AS checkpoint_id) AS source
            ON target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns AND target.checkpoint_id = source.checkpoint_id
            WHEN MATCHED THEN
                UPDATE SET checkpoint_data = ?, checkpoint_type = ?, metadata = ?, parent_checkpoint_id = ?
            WHEN NOT MATCHED THEN
                INSERT (thread_id, checkpoint_ns, checkpoint_id, checkpoint_data, checkpoint_type, metadata, parent_checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                thread_id, checkpoint_ns, checkpoint_id,
                serialized_checkpoint, type_, serialized_metadata, parent_checkpoint_id,
                thread_id, checkpoint_ns, checkpoint_id,
                serialized_checkpoint, type_, serialized_metadata, parent_checkpoint_id
            ))
            conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx, (channel, value) in enumerate(writes):
                type_, serialized_value = self.serde.dumps_typed(value)
                cursor.execute("""
                MERGE checkpoint_writes AS target
                USING (SELECT ? AS thread_id, ? AS checkpoint_ns, ? AS checkpoint_id, ? AS task_id, ? AS idx) AS source
                ON target.thread_id = source.thread_id AND target.checkpoint_ns = source.checkpoint_ns AND target.checkpoint_id = source.checkpoint_id AND target.task_id = source.task_id AND target.idx = source.idx
                WHEN MATCHED THEN
                    UPDATE SET channel = ?, value_type = ?, value = ?
                WHEN NOT MATCHED THEN
                    INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value_type, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                    channel, type_, serialized_value,
                    thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                    channel, type_, serialized_value
                ))
            conn.commit()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if checkpoint_id:
                cursor.execute("""
                SELECT checkpoint_id, checkpoint_data, checkpoint_type, metadata, parent_checkpoint_id
                FROM checkpoints_test
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """, (thread_id, checkpoint_ns, checkpoint_id))
            else:
                cursor.execute("""
                SELECT TOP 1 checkpoint_id, checkpoint_data, checkpoint_type, metadata, parent_checkpoint_id
                FROM checkpoints_test
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC
                """, (thread_id, checkpoint_ns))

            row = cursor.fetchone()
            if not row:
                return None

            checkpoint = self.serde.loads_typed((row[2], row[1]))
            metadata = self.serde.loads(row[3])

            return CheckpointTuple(
                checkpoint=checkpoint,
                metadata=metadata,
                config={"configurable": {"thread_id": f"{thread_id}"}}
            )

    def _load_pending_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> List[PendingWrite]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT task_id, idx, channel, value_type, value
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY idx
            """, (thread_id, checkpoint_ns, checkpoint_id))

            rows = cursor.fetchall()
            return [
                PendingWrite(task_id=row[0], channel=row[2], value=self.serde.loads_typed((row[3], row[4])))
                for row in rows
            ]

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT checkpoint_id, checkpoint_data, checkpoint_type, metadata, parent_checkpoint_id
                FROM checkpoints_test
                WHERE thread_id = ? AND checkpoint_ns = ?
            """
            params = [thread_id, checkpoint_ns]

            if before and before.get("configurable", {}).get("checkpoint_id"):
                query += " AND checkpoint_id < ?"
                params.append(before["configurable"]["checkpoint_id"])

            query += " ORDER BY checkpoint_id DESC"
            if limit:
                query = query.replace("SELECT ", f"SELECT TOP {limit} ")

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            for row in rows:
                checkpoint = self.serde.loads_typed((row[2], row[1]))
                metadata = self.serde.loads(row[3])
                pending_writes = self._load_pending_writes(thread_id, checkpoint_ns, row[0])
                yield CheckpointTuple(
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_checkpoint_id=row[4] if row[4] else None,
                    pending_writes=pending_writes
                )
