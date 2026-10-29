import io
import json
import urllib.error

import pytest

from miras import wiki
from miras.chunking import parse_document
from miras.wiki import WikiError, article, parse_query, slug, to_markdown

EXTRACT = ("Sümela Manastırı, Trabzon'un Maçka ilçesinde bir manastırdır.\n\n"
           "== Tarihçe ==\nManastır 4. yüzyılda kurulmuştur.\n\n"
           "== Kaynakça ==\nBir kaynak.\n\n=== Alt kaynak ===\nAtılır.\n\n"
           "== Mimari ==\nKayaya oyulmuştur.")


def fake_query(title="Sümela Manastırı", extract=EXTRACT, **extra):
    page = {"title": title, "extract": extract,
            "revisions": [{"revid": 123, "timestamp": "2026-10-01T10:00:00Z"}]}
    return {"pages": [page], **extra}


def test_parse_query_accepts_titles_and_tr_links():
    assert parse_query("  Nemrut   Dağı ") == "Nemrut Dağı"
    assert parse_query("https://tr.wikipedia.org/wiki/Nemrut_Da%C4%9F%C4%B1") == "Nemrut Dağı"
    assert parse_query("https://tr.m.wikipedia.org/wiki/Ani#Tarih") == "Ani"
    for bad in ("https://en.wikipedia.org/wiki/Ani", "https://tr.wikipedia.org/w/index.php", "  "):
        with pytest.raises(WikiError):
            parse_query(bad)


def test_slug_matches_existing_file_names():
    assert slug("Nemrut Dağı") == "nemrut-dagi"
    assert slug("Çatalhöyük") == "catalhoyuk"
    assert slug("İshak Paşa Sarayı") == "ishak-pasa-sarayi"
    assert slug("Hattuşa") == "hattusa"


def test_to_markdown_drops_reference_sections():
    md = to_markdown(EXTRACT)
    assert "## Tarihçe" in md and "## Mimari" in md
    assert "Kaynakça" not in md and "Atılır" not in md


def test_article_builds_header_the_parser_reads(monkeypatch):
    monkeypatch.setattr(wiki, "fetch", lambda title: fake_query())
    filename, text = article("https://tr.wikipedia.org/wiki/S%C3%BCmela_Manast%C4%B1r%C4%B1",
                             ("Sumela", "Meryem Ana"))
    assert filename == "sumela-manastiri.md"
    doc = parse_document(text, fallback_title="x")
    assert doc.title == "Sümela Manastırı" and doc.aliases == ("Sumela", "Meryem Ana")
    assert doc.source_url == "https://tr.wikipedia.org/wiki/S%C3%BCmela_Manast%C4%B1r%C4%B1"
    assert "revision: 123 (2026-10-01T10:00:00Z)" in text
    assert article("Sümela Manastırı", filename="sumela.md")[0] == "sumela.md"


def test_redirect_source_becomes_an_alias(monkeypatch):
    query = fake_query("Troya", "Troya antik bir kenttir.",
                       redirects=[{"from": "Truva", "to": "Troya"}, {"from": "troya", "to": "Troya"}])
    monkeypatch.setattr(wiki, "fetch", lambda title: query)
    _, text = article("Truva", ("Troia", "truva"))
    assert parse_document(text, "x").aliases == ("Troia", "truva")
    _, text = article("Truva")
    assert parse_document(text, "x").aliases == ("Truva",)


def test_updating_an_existing_doc_keeps_its_aliases(monkeypatch, tmp_path):
    (tmp_path / "troya.md").write_text("---\ntitle: Troya\naliases: Truva, İlion\n---\n\nEski.\n",
                                       encoding="utf-8")
    monkeypatch.setattr(wiki, "fetch", lambda title: fake_query("Troya", "Troya antik bir kenttir."))
    filename, text = article("Troya", ("Troia",), docs_dir=tmp_path)
    assert filename == "troya.md"
    assert parse_document(text, "x").aliases == ("Truva", "İlion", "Troia")


def test_disambiguation_page_is_rejected(monkeypatch):
    query = fake_query("Selimiye", "Selimiye şunlara işaret edebilir:")
    query["pages"][0]["pageprops"] = {"disambiguation": ""}
    monkeypatch.setattr(wiki, "fetch", lambda title: query)
    with pytest.raises(WikiError, match="anlam ayrımı"):
        article("Selimiye")


def test_missing_article_is_an_error(monkeypatch):
    monkeypatch.setattr(wiki, "fetch", lambda title: {"pages": [{"title": title, "missing": True}]})
    with pytest.raises(WikiError, match="bulunamadı"):
        article("Böyle bir madde yok")


def test_fetch_wraps_network_errors(monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("ağ yok")

    monkeypatch.setattr(wiki.urllib.request, "urlopen", offline)
    with pytest.raises(WikiError, match="ulaşılamadı"):
        wiki.fetch("Ani")


def test_fetch_returns_query(monkeypatch):
    body = json.dumps({"query": fake_query()}).encode()
    monkeypatch.setattr(wiki.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    assert wiki.fetch("Sümela Manastırı")["pages"][0]["title"] == "Sümela Manastırı"
