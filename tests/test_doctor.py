"""Kurulum kontrolleri; model ve internet gerekmez."""

import dataclasses

import pytest

from miras import doctor


@pytest.fixture
def indexed(store, backend, settings):
    # İndeks FakeBackend ile kuruldu; ayarlar da onu göstersin.
    return dataclasses.replace(settings, embed_model=backend.embed_model)


def levels(checks, name):
    return [c.level for c in checks if c.name == name]


def test_healthy_setup_has_no_errors(indexed, tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "qwen").mkdir(parents=True)
    monkeypatch.setattr(doctor, "MODEL_CACHE", models)
    checks = doctor.run_checks(indexed)
    assert {c.level for c in checks if c.name not in ("foundry-local-sdk",)} == {"ok"}
    index = next(c for c in checks if c.name == "indeks")
    assert index.message == "2 doküman, 3 parça (fake-hash-embedding)"


def test_missing_index_is_an_error_and_is_not_created(settings, tmp_path):
    settings = dataclasses.replace(settings, db_path=tmp_path / "yok" / "miras.db")
    checks = doctor.check_index(settings)
    assert checks[0].level == "error" and "ingest" in checks[0].message
    assert not settings.db_path.exists()


def test_stale_index_and_pending_files_are_reported(indexed):
    stale = doctor.check_index(dataclasses.replace(indexed, embed_model="baska"))
    assert stale[0].level == "error" and "--rebuild" in stale[0].message
    (indexed.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    (indexed.docs_dir / "eski.md").write_bytes("# Eski\n\nçok eski".encode("latin-1"))
    assert levels(doctor.check_index(indexed), "indeks") == ["ok", "warn"]
    docs = doctor.check_docs(indexed)
    assert docs[1].level == "warn" and docs[1].message.endswith("eski.md")


def test_docs_dir_problems(settings, tmp_path):
    assert doctor.check_docs(dataclasses.replace(settings, docs_dir=tmp_path / "yok"))[0].level == "error"
    empty = tmp_path / "bos"
    empty.mkdir()
    assert "dosya yok" in doctor.check_docs(dataclasses.replace(settings, docs_dir=empty))[0].message


def test_models_not_downloaded_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "MODEL_CACHE", tmp_path / "models")
    check = doctor.check_models()
    assert check.level in ("warn", "error") and "indirilmemiş" in check.message


def test_settings_that_disable_safeguards_are_flagged(settings):
    assert levels(doctor.check_settings(settings), "ayarlar") == ["ok"]
    messages = [c.message for c in doctor.check_settings(
        dataclasses.replace(settings, min_similarity=0.0, chat_device="gpu"))]
    assert any("kapı kapalı" in m for m in messages) and any("gpu" in m for m in messages)
    high_floor = doctor.check_settings(dataclasses.replace(settings, followup_min_similarity=0.5,
                                                           min_similarity=0.5))
    assert "followup_min_similarity" in high_floor[0].message


def test_files_without_text_are_a_warning(settings):
    (settings.docs_dir / "menu.html").write_text("<nav>Ana sayfa</nav><script>x()</script>", encoding="utf-8")
    checks = doctor.check_docs(settings)
    assert checks[-1] == doctor.Check("warn", "boş dosya", "İçinden metin çıkmayan dosyalar: menu.html")


def test_config_file_is_reported(tmp_path, monkeypatch):
    from miras import config

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "miras.toml")
    monkeypatch.setattr(config, "FILE_VALUES", {})
    assert doctor.check_config_file().message.endswith("yok; varsayılanlar ve MIRAS_* geçerli")
    (tmp_path / "miras.toml").write_text("# top_k = 4\n", encoding="utf-8")
    assert "boş" in doctor.check_config_file().message
    monkeypatch.setattr(config, "FILE_VALUES", {"top_k": "6", "bm25_weight": "0.2"})
    assert doctor.check_config_file().message.endswith("miras.toml: bm25_weight, top_k")
