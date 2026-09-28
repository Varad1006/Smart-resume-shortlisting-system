"""SQLite persistence for shortlist runs.

Stdlib ``sqlite3`` executed in worker threads so the event loop never blocks. Results are
stored as JSON; full resume text is never persisted (only scores and short evidence
snippets), which keeps the database small and limits personal data at rest.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

from resume_shortlister.application.dto import (
    RunStatus,
    RunSummary,
    ShortlistOptions,
    ShortlistRun,
    Stage,
)
from resume_shortlister.domain.models import ShortlistResult

T = TypeVar("T")

_RESULT = TypeAdapter(ShortlistResult)
_OPTIONS = TypeAdapter(ShortlistOptions)
_UNFINISHED = (RunStatus.QUEUED.value, RunStatus.RUNNING.value)

SCHEMA = """
CREATE TABLE IF NOT EXISTS shortlist_runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    status            TEXT NOT NULL,
    stage             TEXT NOT NULL,
    progress_done     INTEGER NOT NULL DEFAULT 0,
    progress_total    INTEGER NOT NULL DEFAULT 0,
    title             TEXT NOT NULL,
    job_description   TEXT NOT NULL,
    options_json      TEXT NOT NULL,
    document_count    INTEGER NOT NULL,
    shortlisted_count INTEGER,
    top_candidate     TEXT,
    top_score         REAL,
    result_json       TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_shortlist_runs_created_at ON shortlist_runs (created_at DESC);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteShortlistRepository:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    # ----------------------------------------------------------------- plumbing
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commit on success, roll back on error
                yield connection
        finally:
            connection.close()

    async def _run(self, function: Callable[..., T], *args: Any) -> T:
        return await asyncio.to_thread(function, *args)

    async def initialize(self) -> None:
        await self._run(self._initialize)

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes
    async def add(self, run: ShortlistRun) -> None:
        await self._run(self._add, run)

    def _add(self, run: ShortlistRun) -> None:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO shortlist_runs (id, created_at, updated_at, status, stage,
                       progress_done, progress_total, title, job_description, options_json,
                       document_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    run.status.value,
                    run.stage.value,
                    run.progress_done,
                    run.progress_total,
                    run.title,
                    run.job_description,
                    _OPTIONS.dump_json(run.options).decode(),
                    run.document_count,
                ),
            )

    async def update_progress(self, run_id: str, stage: Stage, done: int, total: int) -> None:
        await self._run(self._update_progress, run_id, stage, done, total)

    def _update_progress(self, run_id: str, stage: Stage, done: int, total: int) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE shortlist_runs
                   SET status = ?, stage = ?, progress_done = ?, progress_total = ?,
                       updated_at = ?
                   WHERE id = ? AND status IN (?, ?)""",
                (RunStatus.RUNNING.value, stage.value, done, total, _now(), run_id, *_UNFINISHED),
            )

    async def complete(self, run_id: str, result: ShortlistResult) -> None:
        await self._run(self._complete, run_id, result)

    def _complete(self, run_id: str, result: ShortlistResult) -> None:
        shortlisted = result.shortlisted
        top = shortlisted[0] if shortlisted else None
        with self._transaction() as connection:
            connection.execute(
                """UPDATE shortlist_runs
                   SET status = ?, stage = ?, progress_done = progress_total, updated_at = ?,
                       result_json = ?, shortlisted_count = ?, top_candidate = ?, top_score = ?
                   WHERE id = ? AND status IN (?, ?)""",
                (
                    RunStatus.COMPLETED.value,
                    Stage.DONE.value,
                    _now(),
                    _RESULT.dump_json(result).decode(),
                    len(shortlisted),
                    top.filename if top else None,
                    top.final_score if top else None,
                    run_id,
                    *_UNFINISHED,
                ),
            )

    async def fail(self, run_id: str, error: str) -> None:
        await self._run(self._fail, run_id, error)

    def _fail(self, run_id: str, error: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE shortlist_runs SET status = ?, error = ?, updated_at = ?
                   WHERE id = ? AND status IN (?, ?)""",
                (RunStatus.FAILED.value, error, _now(), run_id, *_UNFINISHED),
            )

    async def fail_unfinished(self, error: str) -> int:
        return await self._run(self._fail_unfinished, error)

    def _fail_unfinished(self, error: str) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                """UPDATE shortlist_runs SET status = ?, error = ?, updated_at = ?
                   WHERE status IN (?, ?)""",
                (RunStatus.FAILED.value, error, _now(), *_UNFINISHED),
            )
            return cursor.rowcount

    async def delete(self, run_id: str) -> bool:
        return await self._run(self._delete, run_id)

    def _delete(self, run_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM shortlist_runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------- reads
    async def get(self, run_id: str) -> ShortlistRun | None:
        return await self._run(self._get, run_id)

    def _get(self, run_id: str) -> ShortlistRun | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM shortlist_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ShortlistRun(
            id=row["id"],
            status=RunStatus(row["status"]),
            stage=Stage(row["stage"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            job_description=row["job_description"],
            options=_OPTIONS.validate_json(row["options_json"]),
            document_count=row["document_count"],
            progress_done=row["progress_done"],
            progress_total=row["progress_total"],
            result=_RESULT.validate_json(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
        )

    async def list(self, limit: int, offset: int) -> tuple[list[RunSummary], int]:
        return await self._run(self._list, limit, offset)

    def _list(self, limit: int, offset: int) -> tuple[list[RunSummary], int]:
        with self._transaction() as connection:
            rows = connection.execute(
                """SELECT id, status, stage, created_at, title, document_count,
                          shortlisted_count, top_candidate, top_score
                   FROM shortlist_runs ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM shortlist_runs").fetchone()[0]
        summaries = [
            RunSummary(
                id=row["id"],
                status=RunStatus(row["status"]),
                stage=Stage(row["stage"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                title=row["title"],
                document_count=row["document_count"],
                shortlisted_count=row["shortlisted_count"],
                top_candidate=row["top_candidate"],
                top_score=row["top_score"],
            )
            for row in rows
        ]
        return summaries, total
