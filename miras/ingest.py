"""Veri alma hattı: dokümanları oku -> parçala -> embedding üret -> SQLite'a yaz."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .backends import Backend
from .chunking import chunk_document, parse_document
from .config import Settings
from .htmltext import html_to_markdown
from .store import Store

# HTML dosyaları parçalanmadan önce Markdown'a çevrilir; klasörde oldukları gibi kalırlar.
HTML_SUFFIXES = {".html", ".htm"}
DOC_SUFFIXES = {".md", ".txt"} | HTML_SUFFIXES
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@dataclass
class IngestReport:
    added: list[str]
    unchanged: list[str]
    removed: list[str]
    chunks_embedded: int
    # Okunamayan dosyalar ("x.md: UTF-8 olarak okunamadı"); indeksteki eski halleri korunur.
    skipped: list[str] = field(default_factory=list)


def _fingerprint(raw: str, settings: Settings) -> str:
    """İçerik ve parçalama ayarlarının özeti; ikisinden biri değişince dosya yeniden gömülür."""
    return hashlib.sha256(
        f"{settings.chunk_target_chars}|{settings.chunk_max_chars}|{raw}".encode("utf-8")
    ).hexdigest()


def _document(path: Path, raw: str):
    """Dosya metnini başlık ve bölümlerine ayırır; HTML önce Markdown'a çevrilir."""
    if path.suffix.lower() in HTML_SUFFIXES:
        raw = html_to_markdown(raw)
    return parse_document(raw, fallback_title=path.stem)


def _doc_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        return []
    return sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in DOC_SUFFIXES)


def ingest(store: Store, backend: Backend, settings: Settings, *, rebuild: bool = False,
           on_progress: Callable[[str], None] = lambda _msg: None,
           on_fraction: Callable[[float], None] = lambda _done: None) -> IngestReport:
    """Yalnızca değişen dosyaları yeniden işler (içerik özetine göre).

    `on_fraction` her dosya gömüldükten sonra gömülen parça oranıyla (0–1) çağrılır; süre
    parça sayısıyla orantılı olduğu için web arayüzündeki ilerleme çubuğu bunu gösterir.
    """
    # Farklı bir embedding modeliyle üretilmiş vektörler karşılaştırılamaz: baştan kur.
    if rebuild or store.get_meta("embed_model") not in (None, backend.embed_model):
        store.clear()

    files = _doc_files(settings.docs_dir)
    if not files:
        raise FileNotFoundError(f"{settings.docs_dir} içinde .md/.txt doküman yok.")

    known = store.fingerprints()
    report = IngestReport([], [], [], 0)
    # Önce neyin gömüleceği çıkarılır; toplam parça sayısı ilerlemeyi göstermek için gerekli.
    todo = []
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            report.skipped.append(f"{path.name}: UTF-8 olarak okunamadı, atlandı.")
            continue
        fingerprint = _fingerprint(raw, settings)
        if known.get(path.name) == fingerprint:
            report.unchanged.append(path.name)
            continue
        doc = _document(path, raw)
        chunks = chunk_document(doc, settings.chunk_target_chars, settings.chunk_max_chars)
        if not chunks:
            # Metni silinmiş dosyanın eski parçaları aramada kalmasın.
            if path.name in known:
                store.delete_document(path.name)
                report.removed.append(path.name)
            continue
        todo.append((path.name, doc, fingerprint, chunks))

    total = sum(len(chunks) for *_, chunks in todo)
    for number, (filename, doc, fingerprint, chunks) in enumerate(todo, start=1):
        on_progress(f"{filename}: {len(chunks)} parça gömülüyor ({number}/{len(todo)})...")
        embeddings = backend.embed_documents([c.embedding_text for c in chunks])
        store.add_document(filename, doc.title, doc.source_url, fingerprint,
                           [c.section for c in chunks], [c.text for c in chunks], embeddings,
                           doc.aliases)
        report.added.append(filename)
        report.chunks_embedded += len(chunks)
        on_fraction(report.chunks_embedded / total)

    for filename in sorted(set(known) - {p.name for p in files}):
        store.delete_document(filename)
        report.removed.append(filename)

    store.set_meta("embed_model", backend.embed_model)
    return report


@dataclass
class PendingChanges:
    """Doküman klasörüyle indeks arasındaki fark; `ingest` çalışınca kapanır."""
    changed: list[str]  # yeni ya da içeriği değişmiş dosyalar
    missing: list[str]  # indekste olup klasörde artık olmayan dosyalar

    def __bool__(self) -> bool:
        return bool(self.changed or self.missing)


def pending_changes(store: Store, settings: Settings) -> PendingChanges:
    """Klasördeki dosyaları indeksle karşılaştırır; embedding üretmez, model gerektirmez.

    Okunamayan ve parça çıkmayan yeni dosyalar sayılmaz; ingest onları zaten atlıyor.
    """
    known = store.fingerprints()
    files = _doc_files(settings.docs_dir)
    changed = []
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if known.get(path.name) == _fingerprint(raw, settings):
            continue
        if path.name not in known and not chunk_document(
                _document(path, raw), settings.chunk_target_chars, settings.chunk_max_chars):
            continue
        changed.append(path.name)
    return PendingChanges(changed, sorted(set(known) - {p.name for p in files}))


def _valid_doc_name(name: str) -> str | None:
    """Klasör dışına çıkamayan, gizli olmayan bir .md/.txt dosya adı; uygun değilse None."""
    filename = Path(name).name  # "../x.md" gibi yollar klasör dışına yazamasın
    if filename != name or filename.startswith(".") or Path(filename).suffix.lower() not in DOC_SUFFIXES:
        return None
    return filename


def remove_document(store: Store, docs_dir: Path, filename: str) -> None:
    """Dokümanı klasörden siler ve parçalarını indeksten kaldırır; yeniden gömme gerekmez.

    Dosya diskte yoksa yalnızca indeks temizlenir.
    """
    if _valid_doc_name(filename) is None:
        raise ValueError(f"{filename}: geçersiz doküman adı.")
    (docs_dir / filename).unlink(missing_ok=True)
    store.delete_document(filename)


def save_uploads(docs_dir: Path, uploads: list[tuple[str, bytes]]) -> tuple[list[str], list[str]]:
    """Arayüzden yüklenen dosyaları doküman klasörüne yazar; (kaydedilenler, hatalar) döner.

    Aynı adlı dosyanın üzerine yazılır; içerik değişmediyse ingest onu zaten atlar.
    """
    saved: list[str] = []
    errors: list[str] = []
    for name, data in uploads:
        filename = _valid_doc_name(Path(name).name)
        if filename is None:
            errors.append(f"{name}: yalnızca .md ve .txt dosyaları eklenebilir.")
            continue
        if not data.strip():
            errors.append(f"{filename}: dosya boş.")
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{filename}: dosya 2 MB'tan büyük.")
            continue
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"{filename}: UTF-8 olarak okunamadı.")
            continue
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / filename).write_text(text, encoding="utf-8")
        saved.append(filename)
    return saved, errors
