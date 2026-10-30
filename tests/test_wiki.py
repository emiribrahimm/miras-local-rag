import io
import json
import urllib.error

import pytest

from miras import wiki
from miras.chunking import parse_document
from miras.wiki import (WikiError, article, latest_revisions, local_revisions, outdated,
                        parse_query, slug, to_markdown)

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


def wiki_doc(title, revision, aliases=""):
    url = "https://tr.wikipedia.org/wiki/" + title.replace(" ", "_")
    return (f"---\ntitle: {title}\n" + (f"aliases: {aliases}\n" if aliases else "")
            + f"source: {url}\nrevision: {revision} (2026-08-08T16:08:20Z)\n---\n\n# {title}\n\nEski metin.\n")


def test_local_revisions_reads_wikipedia_headers_only(tmp_path):
    (tmp_path / "nemrut-dagi.md").write_text(wiki_doc("Nemrut Dağı", 111), encoding="utf-8")
    (tmp_path / "eski.md").write_text("---\ntitle: Eski\nsource: https://tr.wikipedia.org/wiki/Eski\n---\n\nx\n",
                                      encoding="utf-8")
    (tmp_path / "kendi.md").write_text("---\ntitle: Kendi\nsource: https://example.org/k\n---\n\nx\n",
                                       encoding="utf-8")
    (tmp_path / "notlar.md").write_text("# Notlar\n\nÜst bilgi yok.\n", encoding="utf-8")
    docs = local_revisions(tmp_path)
    assert [(d.filename, d.title, d.revision) for d in docs] == [
        ("eski.md", "Eski", 0), ("nemrut-dagi.md", "Nemrut Dağı", 111)]


def test_latest_revisions_maps_back_normalized_and_redirected_titles(monkeypatch):
    calls = []

    def fake_get(params):
        calls.append(params["titles"])
        return {"normalized": [{"from": "efes", "to": "Efes"}],
                "redirects": [{"from": "Truva", "to": "Troya"}],
                "pages": [{"title": "Efes", "revisions": [{"revid": 5}]},
                          {"title": "Troya", "revisions": [{"revid": 7}]},
                          {"title": "Yok", "missing": True}]}

    monkeypatch.setattr(wiki, "_get", fake_get)
    assert latest_revisions(["efes", "Truva", "Yok"]) == {"efes": 5, "Truva": 7}
    assert calls == ["efes|Truva|Yok"]
    calls.clear()
    latest_revisions([f"M{i}" for i in range(120)])
    assert [len(c.split("|")) for c in calls] == [50, 50, 20]


def test_outdated_lists_changed_and_missing_articles(monkeypatch, tmp_path):
    for name, title, rev in [("efes.md", "Efes", 10), ("ani.md", "Ani", 20), ("troya.md", "Troya", 30)]:
        (tmp_path / name).write_text(wiki_doc(title, rev), encoding="utf-8")
    monkeypatch.setattr(wiki, "latest_revisions", lambda titles: {"Efes": 10, "Ani": 25})
    updates = {u.doc.filename: u.latest for u in outdated(tmp_path)}
    assert updates == {"ani.md": 25, "troya.md": None}


def test_add_wiki_command_writes_and_indexes(store, backend, settings, monkeypatch, capsys):
    import dataclasses

    from miras import cli

    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(
        cli.SETTINGS, docs_dir=settings.docs_dir, db_path=settings.db_path))
    monkeypatch.setattr(cli, "_backend", lambda s: backend)
    monkeypatch.setattr(wiki, "fetch", lambda title: fake_query())
    assert cli.main(["add-wiki", "Sümela Manastırı", "--aliases", "Sumela"]) == 0
    out = capsys.readouterr().out
    assert "sumela-manastiri.md" in out and "Eklenen/güncellenen: 1" in out
    assert "aliases: Sumela" in (settings.docs_dir / "sumela-manastiri.md").read_text(encoding="utf-8")
    assert store.counts()[0] == 3

    assert cli.main(["add-wiki", "https://en.wikipedia.org/wiki/Ani"]) == 1
    assert "tr.wikipedia.org" in capsys.readouterr().err


def test_update_wiki_command_refreshes_changed_articles(store, backend, settings, monkeypatch, capsys):
    import dataclasses

    from miras import cli

    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(
        cli.SETTINGS, docs_dir=settings.docs_dir, db_path=settings.db_path))
    monkeypatch.setattr(cli, "_backend", lambda s: backend)
    (settings.docs_dir / "troya.md").write_text(wiki_doc("Troya", 30, "Truva, İlion"), encoding="utf-8")
    (settings.docs_dir / "ani.md").write_text(wiki_doc("Ani", 20), encoding="utf-8")
    monkeypatch.setattr(wiki, "latest_revisions", lambda titles: {"Troya": 31})
    monkeypatch.setattr(wiki, "fetch", lambda title: fake_query("Troya", "Troya yeni metin."))

    assert cli.main(["update-wiki", "--check"]) == 0
    out = capsys.readouterr().out
    assert "troya.md" in out and "30 -> 31" in out and "bulunamadı" in out
    assert "Eski metin" in (settings.docs_dir / "troya.md").read_text(encoding="utf-8")

    assert cli.main(["update-wiki"]) == 0
    assert "Eklenen/güncellenen: 2" in capsys.readouterr().out  # ikisi de ilk kez indeksleniyor
    text = (settings.docs_dir / "troya.md").read_text(encoding="utf-8")
    assert "Troya yeni metin." in text and "aliases: Truva, İlion" in text
    assert "Eski metin" in (settings.docs_dir / "ani.md").read_text(encoding="utf-8")

    monkeypatch.setattr(wiki, "local_revisions", lambda docs_dir: [])
    assert cli.main(["update-wiki"]) == 0
    assert "Tüm maddeler güncel." in capsys.readouterr().out
