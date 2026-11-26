"""SQLite veri katmanı: dokümanlar, parçalar ve embedding vektörleri tek dosyada."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    fingerprint TEXT NOT NULL,         -- içerik + parçalama ayarlarının özeti
    aliases     TEXT NOT NULL DEFAULT ''  -- alanın başka adları, virgülle ayrılmış
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
CREATE TABLE IF NOT EXISTS answer_cache (
    key     TEXT PRIMARY KEY,          -- model, ayarlar ve prompt mesajlarının özeti
    answer  TEXT NOT NULL,             -- doğrulanmamış ham model çıktısı
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    key            TEXT PRIMARY KEY,   -- oturum + tur; oy değişirse aynı satır güncellenir
    created        REAL NOT NULL,
    rating         INTEGER NOT NULL,   -- 1 = beğendi, -1 = beğenmedi
    question       TEXT NOT NULL,      -- kullanıcının yazdığı soru
    carried        TEXT NOT NULL,      -- önceki turdan taşınan konu, virgülle ayrılmış
    answer         TEXT NOT NULL,
    refusal_reason TEXT,
    sources        TEXT NOT NULL,      -- json: getirilen parçalar (dosya, bölüm, benzerlik, atıf)
    chat_model     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    id             INTEGER PRIMARY KEY,
    created        REAL NOT NULL,
    question       TEXT NOT NULL,      -- kullanıcının yazdığı soru
    carried        TEXT NOT NULL,      -- önceki turdan taşınan konu, virgülle ayrılmış
    docs           TEXT NOT NULL,      -- arama bu dosyalarla sınırlandıysa, virgülle ayrılmış
    refusal_reason TEXT,
    top_score      REAL NOT NULL,      -- en iyi parçanın ham benzerliği
    cited          TEXT NOT NULL,      -- atıf yapılan dosyalar, virgülle ayrılmış
    cached         INTEGER NOT NULL,
    chat_model     TEXT NOT NULL
);
"""

# Önbellekte tutulan en fazla cevap; aşılınca en eskiler silinir.
MAX_CACHED_ANSWERS = 1000
# Soru günlüğünde tutulan en fazla kayıt; aşılınca en eskiler silinir.
MAX_LOGGED_QUERIES = 5000


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class StoredChunk:
    id: int
    filename: str
    title: str
    source_url: str
    section: str
    text: str
    aliases: tuple[str, ...] = ()


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Web arayüzü açıkken CLI'dan ingest/oy yazılabilsin: WAL'da okuyanlar yazanı bekletmez,
        # kısa süren kilitler için de hata yerine 10 sn beklenir.
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self.conn.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        # Takma adlardan önce oluşturulmuş veritabanlarına kolon eklenir.
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(documents)")}
        if "aliases" not in columns:
            self.conn.execute("ALTER TABLE documents ADD COLUMN aliases TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

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
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            self._touch_index()

    def _touch_index(self) -> None:
        """İndeks değişti: başka süreçlerdeki aramalar (web arayüzü, API) bunu görüp yenilenir."""
        self.conn.execute("INSERT INTO meta(key, value) VALUES('index_version', ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (str(time.time_ns()),))

    def index_version(self) -> str:
        return self.get_meta("index_version") or ""

    def clear(self) -> None:
        self.conn.executescript("DELETE FROM documents; DELETE FROM meta; DELETE FROM answer_cache;")
        self.conn.commit()

    def add_document(self, filename: str, title: str, source_url: str, fingerprint: str,
                     sections: list[str], texts: list[str], embeddings: np.ndarray,
                     aliases: tuple[str, ...] = ()) -> None:
        with self.conn:  # tek işlem: yarım kalmış doküman oluşmaz
            self.conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            doc_id = self.conn.execute(
                "INSERT INTO documents(filename, title, source_url, fingerprint, aliases) "
                "VALUES(?, ?, ?, ?, ?)",
                (filename, title, source_url, fingerprint, ", ".join(aliases))).lastrowid
            self.conn.executemany(
                "INSERT INTO chunks(doc_id, position, section, text, embedding) VALUES(?, ?, ?, ?, ?)",
                [(doc_id, i, sections[i], texts[i], embeddings[i].astype(np.float32).tobytes())
                 for i in range(len(texts))])
            self._touch_index()

    # --- cevap önbelleği ---------------------------------------------------
    def cached_answer(self, key: str) -> str | None:
        row = self.conn.execute("SELECT answer FROM answer_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def cache_answer(self, key: str, answer: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO answer_cache(key, answer, created) VALUES(?, ?, ?)",
                              (key, answer, time.time()))
            self.conn.execute("DELETE FROM answer_cache WHERE key NOT IN (SELECT key FROM answer_cache "
                              "ORDER BY created DESC, rowid DESC LIMIT ?)", (MAX_CACHED_ANSWERS,))

    def cache_size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]

    def clear_answer_cache(self, older_than: float | None = None) -> int:
        """Önbelleği boşaltır, silinen cevap sayısını döner. İndeks ve oylar kalır.

        `older_than` (Unix zamanı) verilirse yalnızca o andan önce kaydedilen cevaplar silinir.
        """
        with self.conn:
            if older_than is None:
                return self.conn.execute("DELETE FROM answer_cache").rowcount
            return self.conn.execute("DELETE FROM answer_cache WHERE created < ?", (older_than,)).rowcount

    # --- geri bildirim -------------------------------------------------------
    # İndeksten bağımsız kullanıcı verisi: `clear()` (ingest --rebuild) bunlara dokunmaz.
    def save_feedback(self, key: str, rating: int, question: str, carried: tuple[str, ...],
                      answer: str, refusal_reason: str | None, sources: list[dict],
                      chat_model: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO feedback(key, created, rating, question, carried, answer, "
                "refusal_reason, sources, chat_model) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, time.time(), rating, question, ", ".join(carried), answer, refusal_reason,
                 json.dumps(sources, ensure_ascii=False), chat_model))

    def delete_feedback(self, key: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM feedback WHERE key = ?", (key,))

    def feedback_rows(self, rating: int | None = None) -> list[dict]:
        """Oylar, yeniden eskiye; `rating` verilirse yalnızca o oy."""
        query = ("SELECT key, created, rating, question, carried, answer, refusal_reason, sources, "
                 "chat_model FROM feedback")
        params: tuple = ()
        if rating is not None:
            query += " WHERE rating = ?"
            params = (rating,)
        cursor = self.conn.execute(query + " ORDER BY created DESC, rowid DESC", params)
        names = [d[0] for d in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor]
        for row in rows:
            row["carried"] = tuple(t.strip() for t in row["carried"].split(",") if t.strip())
            row["sources"] = json.loads(row["sources"])
        return rows

    # --- soru günlüğü -------------------------------------------------------
    # Oylar gibi kullanıcı verisi: `clear()` (ingest --rebuild) bunlara dokunmaz.
    def log_query(self, question: str, carried: tuple[str, ...], docs: tuple[str, ...],
                  refusal_reason: str | None, top_score: float, cited: list[str], cached: bool,
                  chat_model: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO queries(created, question, carried, docs, refusal_reason, top_score, "
                "cited, cached, chat_model) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), question, ", ".join(carried), ", ".join(docs), refusal_reason,
                 top_score, ", ".join(cited), int(cached), chat_model))
            self.conn.execute("DELETE FROM queries WHERE id NOT IN (SELECT id FROM queries "
                              "ORDER BY id DESC LIMIT ?)", (MAX_LOGGED_QUERIES,))

    def query_rows(self, refused: bool = False, limit: int | None = None,
                   since: float | None = None) -> list[dict]:
        """Günlükteki sorular, yeniden eskiye; `refused` ise yalnızca reddedilenler, `since`
        (Unix zamanı) verilirse yalnızca o andan sonrakiler."""
        query = ("SELECT created, question, carried, docs, refusal_reason, top_score, cited, cached, "
                 "chat_model FROM queries WHERE created >= ?")
        if refused:
            query += " AND refusal_reason IS NOT NULL"
        cursor = self.conn.execute(query + " ORDER BY id DESC LIMIT ?",
                                   (since or 0.0, -1 if limit is None else limit))
        names = [d[0] for d in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor]
        for row in rows:
            row["carried"], row["docs"], row["cited"] = (
                _split(row["carried"]), _split(row["docs"]), _split(row["cited"]))
            row["cached"] = bool(row["cached"])
        return rows

    def query_counts(self) -> tuple[int, int]:
        """(günlükteki soru sayısı, reddedilenler)."""
        return self.conn.execute(
            "SELECT COUNT(*), COUNT(refusal_reason) FROM queries").fetchone()

    def clear_queries(self) -> int:
        with self.conn:
            return self.conn.execute("DELETE FROM queries").rowcount

    # --- kullanıcı verisi yedeği ---------------------------------------------
    USER_TABLES = {"feedback": ("key", "created", "rating", "question", "carried", "answer",
                                "refusal_reason", "sources", "chat_model"),
                   "queries": ("created", "question", "carried", "docs", "refusal_reason", "top_score",
                               "cited", "cached", "chat_model")}

    def export_user_data(self) -> dict[str, list[dict]]:
        """Oylar ve soru günlüğü, tablodaki ham halleriyle; indeks ve önbellek yeniden üretilebilir."""
        result = {}
        for table, columns in self.USER_TABLES.items():
            cursor = self.conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY rowid")
            result[table] = [dict(zip(columns, row)) for row in cursor]
        return result

    def import_user_data(self, data: dict[str, list[dict]]) -> dict[str, int]:
        """Yedeği geri yükler, tablo başına eklenen satır sayısını döner. Zaten olan satırlar
        (aynı oy anahtarı; aynı an ve soru) atlanır, yani aynı yedek iki kez yüklenebilir."""
        added = {}
        with self.conn:
            for table, columns in self.USER_TABLES.items():
                rows = data.get(table, [])
                missing = [c for c in columns if any(c not in row for row in rows)]
                if missing:
                    raise ValueError(f"Yedekteki {table} satırlarında eksik alan: {', '.join(missing)}")
                before = self.conn.total_changes
                for row in rows:
                    if table == "queries" and self.conn.execute(
                            "SELECT 1 FROM queries WHERE created = ? AND question = ?",
                            (row["created"], row["question"])).fetchone():
                        continue
                    self.conn.execute(
                        f"INSERT OR IGNORE INTO {table}({', '.join(columns)}) "
                        f"VALUES({', '.join('?' * len(columns))})", [row[c] for c in columns])
                added[table] = self.conn.total_changes - before
        return added

    # --- okuma ------------------------------------------------------------
    def counts(self) -> tuple[int, int]:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return docs, chunks

    def documents(self) -> list[dict]:
        """İndeksteki dokümanlar başlığa göre: dosya, başlık, kaynak, takma adlar, parça sayısı."""
        rows = self.conn.execute(
            "SELECT d.filename, d.title, d.source_url, d.aliases, COUNT(c.id) FROM documents d "
            "LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.title, d.filename").fetchall()
        return [{"filename": f, "title": t, "source_url": u, "chunks": n,
                 "aliases": tuple(a.strip() for a in aliases.split(",") if a.strip())}
                for f, t, u, aliases, n in rows]

    def load_all(self) -> tuple[list[StoredChunk], np.ndarray]:
        rows = self.conn.execute(
            "SELECT c.id, d.filename, d.title, d.source_url, c.section, c.text, d.aliases, "
            "c.embedding FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "ORDER BY d.filename, c.position"
        ).fetchall()
        chunks = [StoredChunk(*row[:6], tuple(a.strip() for a in row[6].split(",") if a.strip()))
                  for row in rows]
        if not rows:
            return chunks, np.zeros((0, 0), dtype=np.float32)
        return chunks, np.stack([np.frombuffer(row[7], dtype=np.float32) for row in rows])
