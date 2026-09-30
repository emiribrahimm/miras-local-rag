"""SQLite veri katmanı: dokümanlar, parçalar ve embedding vektörleri tek dosyada."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    fingerprint TEXT NOT NULL          -- içerik + parçalama ayarlarının özeti
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    section   TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL            -- float32, L2-normalize edilmiş
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class StoredChunk:
    id: int
    filename: str
    title: str
    source_url: str
    section: str
    text: str


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- meta -------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # --- yazma ------------------------------------------------------------
    def fingerprints(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT filename, fingerprint FROM documents"))

    def delete_document(self, filename: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        self.conn.commit()

    def clear(self) -> None:
        self.conn.executescript("DELETE FROM documents; DELETE FROM meta;")
        self.conn.commit()

    def add_document(self, filename: str, title: str, source_url: str, fingerprint: str,
                     sections: list[str], texts: list[str], embeddings: np.ndarray) -> None:
        with self.conn:  # tek işlem: yarım kalmış doküman oluşmaz
            self.conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            doc_id = self.conn.execute(
                "INSERT INTO documents(filename, title, source_url, fingerprint) VALUES(?, ?, ?, ?)",
                (filename, title, source_url, fingerprint)).lastrowid
            self.conn.executemany(
                "INSERT INTO chunks(doc_id, position, section, text, embedding) VALUES(?, ?, ?, ?, ?)",
                [(doc_id, i, sections[i], texts[i], embeddings[i].astype(np.float32).tobytes())
                 for i in range(len(texts))])

    # --- okuma ------------------------------------------------------------
    def counts(self) -> tuple[int, int]:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return docs, chunks

    def load_all(self) -> tuple[list[StoredChunk], np.ndarray]:
        rows = self.conn.execute(
            "SELECT c.id, d.filename, d.title, d.source_url, c.section, c.text, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY d.filename, c.position"
        ).fetchall()
        chunks = [StoredChunk(*row[:6]) for row in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        return chunks, np.stack([np.frombuffer(row[6], dtype=np.float32) for row in rows])
