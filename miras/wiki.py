"""Türkçe Vikipedi maddelerini indirip üst bilgili Markdown dokümanına çevirir.

Asistanın kendisi çevrimdışı çalışır; buradaki tek ağ çağrısı kullanıcı bir madde eklemek
istediğinde (kenar çubuğu, `add-wiki`, `scripts/fetch_corpus.py`) yapılır.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .retrieval import _ASCII, _fold

API = "https://tr.wikipedia.org/w/api.php"
HOSTS = {"tr.wikipedia.org", "tr.m.wikipedia.org"}

# Soru-cevap için bilgi taşımayan bölümler atılır.
SKIP_SECTIONS = {
    "kaynakça", "dış bağlantılar", "ayrıca bakınız", "notlar", "galeri",
    "konuyla ilgili yayınlar", "dipnotlar", "kaynaklar", "bibliyografya",
    "daha fazla okuma", "popüler kültürde", "resimler",
}


class WikiError(RuntimeError):
    """Madde bulunamadığında ya da Vikipedi'ye ulaşılamadığında fırlatılır."""


def parse_query(query: str) -> str:
    """Madde adı ya da tr.wikipedia.org bağlantısı -> madde başlığı."""
    query = query.strip()
    if re.match(r"https?://", query):
        parts = urllib.parse.urlsplit(query)
        if parts.hostname not in HOSTS or not parts.path.startswith("/wiki/"):
            raise WikiError("Yalnızca tr.wikipedia.org madde bağlantıları eklenebilir.")
        query = urllib.parse.unquote(parts.path[len("/wiki/"):])
    title = " ".join(query.replace("_", " ").split())
    if not title:
        raise WikiError("Bir madde adı ya da bağlantısı yazın.")
    return title


def slug(title: str) -> str:
    """"Nemrut Dağı" -> "nemrut-dagi"; data/docs içindeki dosya adlarıyla aynı kural."""
    return re.sub(r"[^a-z0-9]+", "-", _fold(title).translate(_ASCII)).strip("-") or "madde"


def fetch(title: str) -> dict:
    """API'nin `query` cevabı (sayfa ve varsa yönlendirmeler)."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|revisions|pageprops", "explaintext": "1", "exsectionformat": "wiki",
        "rvprop": "ids|timestamp", "ppprop": "disambiguation", "titles": title,
    }
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "miras-rag/0.1 (egitim projesi)"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)["query"]
        except urllib.error.HTTPError as err:
            # Vikipedi art arda isteklerde 429 döndürebiliyor; bekleyip yeniden dene.
            if err.code != 429 or attempt == 4:
                raise WikiError(f"Vikipedi isteği başarısız: {err}") from err
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as err:
            raise WikiError(f"Vikipedi'ye ulaşılamadı: {err}") from err
    raise AssertionError("unreachable")


def to_markdown(extract: str) -> str:
    """'== Başlık ==' biçimini markdown başlıklarına çevirir, gereksiz bölümleri atar."""
    lines: list[str] = []
    skip_level: int | None = None
    for raw in extract.splitlines():
        line = raw.strip()
        m = re.fullmatch(r"(={2,6})\s*(.+?)\s*\1", line)
        if m:
            level = len(m.group(1))
            if skip_level is not None and level > skip_level:
                continue
            skip_level = level if m.group(2).lower() in SKIP_SECTIONS else None
            if skip_level is None:
                lines += ["", f"{'#' * level} {m.group(2)}", ""]
            continue
        if skip_level is None and line:
            lines += [line, ""]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def merge_aliases(title: str, *groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Takma adları sırasıyla birleştirir; başlığın kendisi ve tekrarlar (harf farkı dahil) atılır."""
    seen = {_fold(title)}
    merged: list[str] = []
    for name in (n.strip() for group in groups for n in group):
        if name and _fold(name) not in seen:
            seen.add(_fold(name))
            merged.append(name)
    return tuple(merged)


def article(query: str, aliases: tuple[str, ...] = (), filename: str | None = None) -> tuple[str, str]:
    """Maddeyi indirir; (dosya adı, üst bilgili Markdown metni) döner.

    Dosya adı verilmezse madde başlığından üretilir ("Nemrut Dağı" -> nemrut-dagi.md).
    Yönlendirmeyle gelinen ad ("Truva" -> Troya) takma adlara eklenir; kullanıcı alanı o
    adla soracaktır.
    """
    result = fetch(parse_query(query))
    page = result["pages"][0]
    if page.get("missing") or page.get("invalid") or not page.get("extract"):
        raise WikiError(f"Vikipedi'de madde bulunamadı: {parse_query(query)}")
    if "disambiguation" in page.get("pageprops", {}):
        raise WikiError(f"'{page['title']}' bir anlam ayrımı sayfası; daha belirli bir madde adı "
                        "yazın (ör. 'Selimiye Camii').")
    aliases = merge_aliases(page["title"], aliases,
                            [r["from"] for r in result.get("redirects", [])])
    rev = page["revisions"][0]
    url = "https://tr.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"))
    header = (
        "---\n"
        f"title: {page['title']}\n"
        + (f"aliases: {', '.join(aliases)}\n" if aliases else "")
        + f"source: {url}\n"
        f"revision: {rev['revid']} ({rev['timestamp']})\n"
        "license: CC BY-SA 4.0 (Vikipedi katkıcıları)\n"
        "---\n\n"
    )
    body = to_markdown(page["extract"])
    return filename or f"{slug(page['title'])}.md", f"{header}# {page['title']}\n\n{body}\n"
