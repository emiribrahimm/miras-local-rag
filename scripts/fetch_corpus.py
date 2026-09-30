"""Bilgi tabanını oluşturan Türkçe Vikipedi maddelerini indirir.

Yalnızca derlemi (data/docs) bir kez hazırlamak için internete ihtiyaç duyar;
asistanın kendisi tamamen çevrimdışı çalışır. Çıktı dosyaları repoya eklidir,
bu betiği tekrar çalıştırmak gerekmez.

Kullanım: uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://tr.wikipedia.org/w/api.php"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

# dosya adı -> Vikipedi madde başlığı
ARTICLES = {
    "gobeklitepe": "Göbeklitepe",
    "efes": "Efes",
    "hattusa": "Hattuşa",
    "nemrut-dagi": "Nemrut Dağı",
    "troya": "Truva",
    "catalhoyuk": "Çatalhöyük",
    "safranbolu": "Safranbolu",
    "divrigi-ulu-camii": "Divriği Ulu Camii ve Darüşşifası",
    "selimiye-camii": "Selimiye Camii",
    "ani": "Ani",
    "afrodisias": "Afrodisias",
    "pamukkale": "Pamukkale",
}

# Soru-cevap için bilgi taşımayan bölümler atılır.
SKIP_SECTIONS = {
    "kaynakça", "dış bağlantılar", "ayrıca bakınız", "notlar", "galeri",
    "konuyla ilgili yayınlar", "dipnotlar", "kaynaklar", "bibliyografya",
    "daha fazla okuma", "popüler kültürde", "resimler",
}


def fetch(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|revisions", "explaintext": "1", "exsectionformat": "wiki",
        "rvprop": "ids|timestamp", "titles": title,
    }
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "miras-rag/0.1 (egitim projesi)"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                page = json.load(resp)["query"]["pages"][0]
            break
        except urllib.error.HTTPError as err:
            # Vikipedi art arda isteklerde 429 döndürebiliyor; bekleyip yeniden dene.
            if err.code != 429 or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    if page.get("missing") or not page.get("extract"):
        raise RuntimeError(f"Madde bulunamadı: {title}")
    return page


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, title in ARTICLES.items():
        page = fetch(title)
        rev = page["revisions"][0]
        url = "https://tr.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"))
        header = (
            "---\n"
            f"title: {page['title']}\n"
            f"source: {url}\n"
            f"revision: {rev['revid']} ({rev['timestamp']})\n"
            "license: CC BY-SA 4.0 (Vikipedi katkıcıları)\n"
            "---\n\n"
        )
        body = to_markdown(page["extract"])
        (OUT_DIR / f"{slug}.md").write_text(f"{header}# {page['title']}\n\n{body}\n", encoding="utf-8")
        time.sleep(1)
        print(f"{slug}.md  {page['title']!r}  {len(body):>6} karakter")


if __name__ == "__main__":
    main()
