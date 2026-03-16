from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Iterable
from uuid import uuid4

from memory.models import MemoryRecord, MemorySummary, MemoryType, parse_memory_type, utc_now


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_json_load(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


@dataclass(slots=True)
class KeywordSearchHit:
    memory_id: str
    content: str
    memory_type: MemoryType
    keyword_score: float
    importance_score: float
    timestamp: datetime


@dataclass(slots=True)
class GraphEdge:
    edge_id: int
    source_memory_id: str
    target_memory_id: str
    relation: str
    weight: float


class MemoryStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.initialize_schema()

    def close(self) -> None:
        self._conn.close()

    def initialize_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance_score REAL NOT NULL DEFAULT 0.5,
                recency_score REAL NOT NULL DEFAULT 1.0,
                semantic_tags TEXT NOT NULL DEFAULT '[]',
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memory_events(memory_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_summaries (
                summary_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                compression_ratio REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memory_events(memory_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_graph (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                FOREIGN KEY(source_memory_id) REFERENCES memory_events(memory_id) ON DELETE CASCADE,
                FOREIGN KEY(target_memory_id) REFERENCES memory_events(memory_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memory_metadata (
                memory_id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(memory_id) REFERENCES memory_events(memory_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_events_type_time
                ON memory_events(memory_type, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_events_time
                ON memory_events(timestamp DESC);
            """
        )
        self._conn.commit()

    def add_memory(
        self,
        memory_type: MemoryType | str,
        content: str,
        importance_score: float = 0.5,
        semantic_tags: list[str] | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        memory_id = str(uuid4())
        mem_type = parse_memory_type(memory_type)
        ts = timestamp or utc_now()
        tags = semantic_tags or []
        meta = metadata or {}
        clamped_importance = max(0.0, min(1.0, importance_score))

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_events (
                    memory_id, memory_type, content, importance_score, recency_score, semantic_tags, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    mem_type.value,
                    content,
                    clamped_importance,
                    1.0,
                    json.dumps(tags, ensure_ascii=True),
                    _to_iso(ts),
                ),
            )
            if embedding is not None:
                self._conn.execute(
                    "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(embedding, ensure_ascii=True)),
                )
            self._conn.execute(
                "INSERT INTO memory_metadata (memory_id, metadata) VALUES (?, ?)",
                (memory_id, json.dumps(meta, ensure_ascii=True)),
            )

        return memory_id

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            memory_type=parse_memory_type(row["memory_type"]),
            content=row["content"],
            importance_score=float(row["importance_score"]),
            recency_score=float(row["recency_score"]),
            semantic_tags=_safe_json_load(row["semantic_tags"], []),
            timestamp=_from_iso(row["timestamp"]),
            metadata=_safe_json_load(row["metadata"], {}),
            embedding=_safe_json_load(row["embedding"], None),
        )

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        cursor = self._conn.execute(
            """
            SELECT e.*, em.embedding, md.metadata
            FROM memory_events e
            LEFT JOIN memory_embeddings em ON em.memory_id = e.memory_id
            LEFT JOIN memory_metadata md ON md.memory_id = e.memory_id
            WHERE e.memory_id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def list_memory(
        self,
        memory_type: MemoryType | str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        params: list[Any] = []
        where = ""
        if memory_type is not None:
            mem_type = parse_memory_type(memory_type)
            where = "WHERE e.memory_type = ?"
            params.append(mem_type.value)

        params.extend([max(1, limit), max(0, offset)])
        query = f"""
            SELECT e.*, em.embedding, md.metadata
            FROM memory_events e
            LEFT JOIN memory_embeddings em ON em.memory_id = e.memory_id
            LEFT JOIN memory_metadata md ON md.memory_id = e.memory_id
            {where}
            ORDER BY e.timestamp DESC
            LIMIT ? OFFSET ?
        """
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_memory_by_types(
        self,
        memory_types: Iterable[MemoryType | str] | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        if not memory_types:
            return self.list_memory(limit=limit)

        normalized = [parse_memory_type(mem_type).value for mem_type in memory_types]
        placeholders = ",".join("?" for _ in normalized)
        query = f"""
            SELECT e.*, em.embedding, md.metadata
            FROM memory_events e
            LEFT JOIN memory_embeddings em ON em.memory_id = e.memory_id
            LEFT JOIN memory_metadata md ON md.memory_id = e.memory_id
            WHERE e.memory_type IN ({placeholders})
            ORDER BY e.timestamp DESC
            LIMIT ?
        """
        rows = self._conn.execute(query, [*normalized, max(1, limit)]).fetchall()
        return [self._row_to_record(row) for row in rows]

    def query_keyword(
        self,
        query: str,
        limit: int = 10,
        memory_types: Iterable[MemoryType | str] | None = None,
    ) -> list[KeywordSearchHit]:
        terms = [token.lower() for token in query.split() if token.strip()]
        candidates = self.list_memory_by_types(memory_types=memory_types, limit=500)
        scored: list[KeywordSearchHit] = []

        for record in candidates:
            lowered = record.content.lower()
            keyword_score = 0.0
            for term in terms:
                if term in lowered:
                    keyword_score += lowered.count(term)
            if terms:
                keyword_score = keyword_score / len(terms)
            if terms and keyword_score == 0:
                continue
            scored.append(
                KeywordSearchHit(
                    memory_id=record.memory_id,
                    content=record.content,
                    memory_type=record.memory_type,
                    keyword_score=keyword_score,
                    importance_score=record.importance_score,
                    timestamp=record.timestamp,
                )
            )

        scored.sort(
            key=lambda item: (item.keyword_score, item.importance_score, item.timestamp),
            reverse=True,
        )
        return scored[: max(1, limit)]

    def upsert_summary(self, memory_id: str, summary: str, compression_ratio: float) -> str:
        summary_id = str(uuid4())
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_summaries (summary_id, memory_id, summary, compression_ratio, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    memory_id,
                    summary,
                    compression_ratio,
                    _to_iso(utc_now()),
                ),
            )
        return summary_id

    def list_summaries(self, limit: int = 20) -> list[MemorySummary]:
        rows = self._conn.execute(
            """
            SELECT summary_id, memory_id, summary, compression_ratio, created_at
            FROM memory_summaries
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()

        return [
            MemorySummary(
                summary_id=row["summary_id"],
                memory_id=row["memory_id"],
                summary=row["summary"],
                compression_ratio=float(row["compression_ratio"]),
                created_at=_from_iso(row["created_at"]),
            )
            for row in rows
        ]

    def add_graph_edge(self, source_memory_id: str, target_memory_id: str, relation: str, weight: float = 1.0) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO memory_graph (source_memory_id, target_memory_id, relation, weight)
                VALUES (?, ?, ?, ?)
                """,
                (source_memory_id, target_memory_id, relation, weight),
            )
        return int(cursor.lastrowid)

    def list_graph_edges(self, memory_id: str, limit: int = 50) -> list[GraphEdge]:
        rows = self._conn.execute(
            """
            SELECT edge_id, source_memory_id, target_memory_id, relation, weight
            FROM memory_graph
            WHERE source_memory_id = ? OR target_memory_id = ?
            ORDER BY weight DESC, edge_id DESC
            LIMIT ?
            """,
            (memory_id, memory_id, max(1, limit)),
        ).fetchall()
        return [
            GraphEdge(
                edge_id=int(row["edge_id"]),
                source_memory_id=row["source_memory_id"],
                target_memory_id=row["target_memory_id"],
                relation=row["relation"],
                weight=float(row["weight"]),
            )
            for row in rows
        ]

    def graph_weight_sum(self, memory_id: str) -> float:
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(weight), 0) AS total_weight
            FROM memory_graph
            WHERE source_memory_id = ? OR target_memory_id = ?
            """,
            (memory_id, memory_id),
        ).fetchone()
        if not row:
            return 0.0
        return float(row["total_weight"])

    def search_by_metadata(self, key: str, value: str, limit: int = 50) -> list[MemoryRecord]:
        if not key:
            return []
        # SQLite JSON functions are not guaranteed in all environments;
        # we scan recent records and filter in Python for portability.
        candidates = self.list_memory(limit=max(100, limit * 10))
        filtered = [item for item in candidates if str(item.metadata.get(key)) == str(value)]
        filtered.sort(key=lambda item: item.timestamp, reverse=True)
        return filtered[: max(1, limit)]

    def list_older_than(
        self,
        cutoff: datetime,
        memory_type: MemoryType | str | None = None,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        params: list[Any] = [_to_iso(cutoff)]
        where = "WHERE e.timestamp < ?"
        if memory_type is not None:
            mem_type = parse_memory_type(memory_type)
            where += " AND e.memory_type = ?"
            params.append(mem_type.value)
        params.append(max(1, limit))
        rows = self._conn.execute(
            f"""
            SELECT e.*, em.embedding, md.metadata
            FROM memory_events e
            LEFT JOIN memory_embeddings em ON em.memory_id = e.memory_id
            LEFT JOIN memory_metadata md ON md.memory_id = e.memory_id
            {where}
            ORDER BY e.timestamp ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        with self._conn:
            cursor = self._conn.execute("DELETE FROM memory_events WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0
