"""Komut satırı komutları, gerçek model yerine FakeBackend ile."""

import dataclasses

import pytest

from miras import cli


@pytest.fixture
def run(store, backend, settings, monkeypatch, capsys):
    # İndeks FakeBackend ile kuruldu; ayarlar da onu göstersin ki `status` uyarı vermesin.
    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(settings, embed_model=backend.embed_model))
    monkeypatch.setattr(cli, "_backend", lambda _settings: backend)

    def run(*argv: str) -> tuple[int, str, str]:
        code = cli.main(list(argv))
        out, err = capsys.readouterr()
        return code, out, err

    return run


def test_cache_command_counts_and_clears(run, store):
    store.cache_answer("a", "Cevap [K1].")
    store.cache_answer("b", "Başka [K1].")
    assert run("cache")[1].startswith("Önbellekte 2 cevap var")
    assert run("cache", "--clear")[1] == "Önbellekten 2 cevap silindi.\n"
    assert store.cache_size() == 0


def test_status_reports_index_and_pending_changes(run, settings):
    code, out, _ = run("status")
    assert code == 0
    assert "İndeks: 2 doküman, 3 parça (fake-hash-embedding)" in out
    assert "ile indeks aynı" in out
    (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    (settings.docs_dir / "efes.md").unlink()
    out = run("status")[1]
    assert "  + ani.md" in out and "  - efes.md (klasörde yok)" in out


def test_status_warns_when_embedding_model_changed(run, settings, monkeypatch):
    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(settings, embed_model="baska"))
    assert "`ingest --rebuild` gerekli" in run("status")[1]


def test_docs_command_lists_indexed_documents(run):
    out = run("docs")[1]
    assert out.splitlines()[:3] == ["Efes  (efes.md, 1 parça)", "     takma adlar: Ephesos, Ephesus",
                                    "Göbeklitepe  (gobeklitepe.md, 2 parça)"]
    assert "     https://example.org/g" in out and out.endswith("2 doküman, 3 parça\n")


def test_remove_doc_deletes_file_and_chunks(run, store, settings):
    code, out, _ = run("remove-doc", "efes.md")
    assert code == 0 and out == "Silindi: efes.md. Veritabanı: 1 doküman, 2 parça\n"
    assert not (settings.docs_dir / "efes.md").exists()


@pytest.mark.parametrize("name, message", [
    ("yok.md", "ne indekste"),
    ("../efes.md", "ne indekste"),
])
def test_remove_doc_rejects_unknown_files(run, name, message):
    code, _, err = run("remove-doc", name)
    assert code == 1 and message in err


def test_last_doc_cannot_be_removed(run, settings):
    assert run("remove-doc", "efes.md")[0] == 0
    code, _, err = run("remove-doc", "gobeklitepe.md")
    assert code == 1 and "En az bir doküman kalmalı." in err
    assert (settings.docs_dir / "gobeklitepe.md").exists()


def test_search_shows_chunks_and_gate_without_chat_model(run, backend):
    code, out, _ = run("search", "Göbeklitepe hangi yıl listeye girdi?", "-k", "1")
    assert code == 0 and backend.chat_calls == []
    lines = out.splitlines()
    assert lines[0] == "Soruda geçen alanlar: Göbeklitepe (sıralamada +0.1)"
    assert lines[1].startswith("[K1] ") and "Göbeklitepe › Koruma  (gobeklitepe.md)" in lines[1]
    assert "[K2]" not in out and "bu parçalar modele gider" in out


def test_search_reports_refusal_below_threshold(run):
    out = run("search", "Pizza hamuru tarifi nedir?")[1]
    assert "Soruda geçen" not in out and "model çağrılmaz, soru reddedilir" in out


def test_search_rejects_bad_k(run):
    code, _, err = run("search", "Efes nerede?", "-k", "0")
    assert code == 1 and "top_k en az 1" in err
