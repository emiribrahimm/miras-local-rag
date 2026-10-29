"""Bilgi tabanını oluşturan Türkçe Vikipedi maddelerini indirir.

Yalnızca derlemi (data/docs) bir kez hazırlamak için internete ihtiyaç duyar;
asistanın kendisi tamamen çevrimdışı çalışır. Çıktı dosyaları repoya eklidir,
bu betiği tekrar çalıştırmak gerekmez.

Kullanım: uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from miras.wiki import article  # noqa: E402

OUT_DIR = ROOT / "data" / "docs"

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

# dosya adı -> alanın başka adları (başlık eşleştirmesinde kullanılır)
ALIASES = {
    "troya": "Truva, Troia, İlion",
    "efes": "Ephesos, Ephesus",
    "hattusa": "Hattuşaş, Hattusas",
    "gobeklitepe": "Göbekli Tepe",
    "catalhoyuk": "Çatal Höyük",
    "afrodisias": "Aphrodisias",
    "pamukkale": "Hierapolis",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, title in ARTICLES.items():
        aliases = tuple(a.strip() for a in ALIASES.get(slug, "").split(",") if a.strip())
        filename, text = article(title, aliases, filename=f"{slug}.md")
        (OUT_DIR / filename).write_text(text, encoding="utf-8")
        time.sleep(1)
        print(f"{filename}  {title!r}  {len(text):>6} karakter")


if __name__ == "__main__":
    main()
