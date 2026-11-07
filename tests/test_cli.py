"""Komut satırı komutları, gerçek model yerine FakeBackend ile."""

import pytest

from miras import cli


@pytest.fixture
def run(store, backend, settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "SETTINGS", settings)
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
