"""Kurulum kontrolleri: `doctor` komutu. Modelleri yüklemez, internete çıkmaz.

Her kontrol bir `Check` döner: "ok", "warn" (çalışır ama bakılmalı) ya da "error" (çalışmaz).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .ingest import _doc_files, pending_changes
from .store import Store

# Foundry Local SDK modelleri buraya indiriyor (FoundryBackend, app_name="miras-rag").
MODEL_CACHE = Path.home() / ".miras-rag" / "cache" / "models"
# İki model birlikte yaklaşık bu kadar yer kaplıyor.
MODELS_GB = 7


@dataclass(frozen=True)
class Check:
    level: str    # "ok" | "warn" | "error"
    name: str
    message: str


def check_python() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 12):
        return Check("error", "python", f"Python {version}; en az 3.12 gerekli.")
    return Check("ok", "python", f"Python {version}")


def check_sdk() -> Check:
    if importlib.util.find_spec("foundry_local_sdk") is None:
        return Check("error", "foundry-local-sdk", "Kurulu değil: `uv sync` çalıştırın.")
    return Check("ok", "foundry-local-sdk", "kurulu")


def check_models() -> Check:
    if MODEL_CACHE.is_dir() and any(MODEL_CACHE.iterdir()):
        return Check("ok", "modeller", f"indirilmiş: {MODEL_CACHE}")
    free_gb = shutil.disk_usage(MODEL_CACHE.parent if MODEL_CACHE.parent.exists() else Path.home()).free / 1e9
    level = "error" if free_gb < MODELS_GB else "warn"
    return Check(level, "modeller", f"henüz indirilmemiş; ilk çalıştırmada ~{MODELS_GB} GB inecek "
                                    f"(boş alan {free_gb:.0f} GB).")


def check_docs(settings: Settings) -> list[Check]:
    if not settings.docs_dir.is_dir():
        return [Check("error", "dokümanlar", f"{settings.docs_dir} yok (MIRAS_DOCS_DIR).")]
    files = _doc_files(settings.docs_dir)
    if not files:
        return [Check("error", "dokümanlar", f"{settings.docs_dir} içinde .md/.txt dosya yok.")]
    unreadable = []
    for path in files:
        try:
            path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            unreadable.append(path.name)
    checks = [Check("ok", "dokümanlar", f"{len(files)} dosya: {settings.docs_dir}")]
    if unreadable:
        checks.append(Check("warn", "kodlama", "UTF-8 olmayan, ingest'in atlayacağı dosyalar: "
                                              + ", ".join(unreadable)))
    return checks


def check_index(settings: Settings) -> list[Check]:
    # Store dosyayı yoksa oluşturur; kontrol veritabanı yaratmasın.
    if not Path(settings.db_path).exists():
        return [Check("error", "indeks", f"{settings.db_path} yok: `python main.py ingest` çalıştırın.")]
    store = Store(settings.db_path)
    try:
        docs, chunks = store.counts()
        indexed_with = store.get_meta("embed_model")
        if not chunks:
            return [Check("error", "indeks", "Boş: `python main.py ingest` çalıştırın.")]
        if indexed_with != settings.embed_model:
            return [Check("error", "indeks", f"'{indexed_with}' ile kurulmuş, ayarlardaki model "
                                             f"'{settings.embed_model}': `ingest --rebuild` gerekli.")]
        checks = [Check("ok", "indeks", f"{docs} doküman, {chunks} parça ({indexed_with})")]
        if settings.docs_dir.is_dir():
            pending = pending_changes(store, settings)
            if pending:
                names = pending.changed + [f"{name} (silinmiş)" for name in pending.missing]
                checks.append(Check("warn", "indeks", "İndekslenmemiş değişiklik: " + ", ".join(names)
                                    + " (`ingest`)"))
        return checks
    finally:
        store.close()


def check_settings(settings: Settings) -> list[Check]:
    checks = []
    if settings.min_similarity == 0:
        checks.append(Check("warn", "ayarlar", "min_similarity 0: kapı kapalı, konu dışı sorular da modele gider."))
    if settings.followup_min_similarity >= settings.min_similarity > 0:
        checks.append(Check("warn", "ayarlar", "followup_min_similarity eşikten büyük ya da eşit; "
                                               "eşiği geçemeyecek takip soruları da konuyu kaybeder."))
    if settings.chat_device == "gpu":
        checks.append(Check("warn", "ayarlar", "chat_device gpu: test edilmedi, sorun olursa cpu deneyin."))
    return checks or [Check("ok", "ayarlar", f"top_k {settings.top_k}, eşik {settings.min_similarity}, "
                                             f"takip tabanı {settings.followup_min_similarity}")]


def run_checks(settings: Settings) -> list[Check]:
    return [check_python(), check_sdk(), check_models(), *check_docs(settings), *check_index(settings),
            *check_settings(settings)]
