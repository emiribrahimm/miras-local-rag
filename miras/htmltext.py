"""HTML dokümanları parçalamadan önce Markdown'a çevirir (yalnızca standart kütüphane).

Amaç tam bir dönüştürücü değil: başlıklar bölüm yolunu, paragraflar parçaları belirlesin;
menüler, betikler ve alt bilgi gibi içerik olmayan kısımlar indekse girmesin.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# İçeriği tamamen atılan etiketler.
SKIPPED = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "template"}
# Yeni paragraf başlatan etiketler.
BLOCKS = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "dl", "dt", "dd",
          "blockquote", "pre", "figure", "figcaption", "li"}
HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "###", "h5": "###", "h6": "###"}
# Başlık yolunu bozmaması için atılan tek satırlık ön bilgi işaretleri.
_FRONT_MATTER = re.compile(r"^-{3,}$")


class _Converter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.title = ""
        self.heading: str | None = None
        self.first_h1 = ""

    def _flush(self) -> None:
        text = " ".join("".join(self.current).split())
        self.current = []
        if not text or _FRONT_MATTER.match(text):
            return
        if self.heading:
            if self.heading == "#" and not self.first_h1:
                self.first_h1 = text
            text = f"{self.heading} {text}"
        self.blocks.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIPPED:
            self.skip_depth += 1
        elif self.skip_depth:
            return
        elif tag == "title":
            self.in_title = True
        elif tag in HEADINGS:
            self._flush()
            self.heading = HEADINGS[tag]
        elif tag in BLOCKS:
            self._flush()
            if tag == "li":
                self.current.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif self.skip_depth:
            return
        elif tag == "title":
            self.in_title = False
        elif tag in HEADINGS:
            self._flush()
            self.heading = None
        elif tag in BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.in_title:
            self.title += data
        else:
            self.current.append(data)


def html_to_markdown(raw: str) -> str:
    """HTML metnini `parse_document`'in anladığı Markdown'a çevirir.

    Başlık ilk `<h1>`'den, yoksa `<title>`'dan gelir ve üst bilgiye yazılır; `<h2>` bölüm, `<h3>` ve
    altı alt bölüm olur. Liste öğeleri "- " ile başlar.
    """
    converter = _Converter()
    converter.feed(raw)
    converter.close()
    converter._flush()
    title = converter.first_h1 or " ".join(converter.title.split())
    body = "\n\n".join(converter.blocks)
    if not title:
        return body + "\n"
    return f"---\ntitle: {title}\n---\n\n{body}\n"
