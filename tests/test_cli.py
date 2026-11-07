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
