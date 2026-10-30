"""Streamlit arayüzünü gerçek model olmadan uçtan uca çalıştırır."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from miras import config
from miras.backends import FakeBackend
from miras.prompts import ABSTAIN_SENTENCE


@pytest.fixture
def app(store, settings, monkeypatch):
    import miras.backends

    fake = FakeBackend("Efes İzmir'dedir [K1].")
    monkeypatch.setattr(config, "SETTINGS", settings)
    monkeypatch.setattr(miras.backends, "FoundryBackend", lambda *args, **kwargs: fake)
    # Asistan süreç boyunca önbellekte kalır; her test kendi veritabanı ve sahte modeliyle başlasın.
    st.cache_resource.clear()
    app = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30).run()
    app.fake = fake
    return app


def test_answer_and_sources_are_rendered(app):
    assert not app.exception
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert app.chat_message[1].markdown[0].value == "Efes İzmir'dedir [K1]."
    assert "[K1] Efes" in app.expander[0].label


def test_supporting_sentence_is_highlighted(app):
    app.fake.reply = "Efes İzmir ilinin Selçuk ilçesindedir [K1]."
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert app.expander[0].markdown[0].value == (
        ":orange-background[Efes İzmir ilinin Selçuk ilçesinde bulunan antik bir kenttir.]")


def test_repeated_question_is_answered_from_cache(app):
    app.chat_input[0].set_value("Efes nerede?").run()
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert len(app.fake.chat_calls) == 1
    assert app.chat_message[3].markdown[0].value == "Efes İzmir'dedir [K1]."
    assert any("önbellekten" in c.value for c in app.chat_message[3].caption)
    assert not any("önbellekten" in c.value for c in app.chat_message[1].caption)


def test_out_of_scope_question_shows_refusal(app):
    app.chat_input[0].set_value("Pizza hamuru tarifi nedir?").run()
    assert app.chat_message[1].markdown[0].value == ABSTAIN_SENTENCE
    assert "model çağrılmadı" in app.chat_message[1].caption[0].value


def test_follow_up_question_shows_carried_topic(app):
    app.chat_input[0].set_value("Efes nerede?").run()
    app.chat_input[0].set_value("Peki hangi ilçesinde?").run()
    assert not app.exception
    captions = [c.value for c in app.chat_message[3].caption]
    assert "↪ Önceki sorudan devam: Efes" in captions
    assert not any("devam" in c.value for c in app.chat_message[1].caption)


def test_uploaded_doc_is_indexed_and_searchable(app, settings):
    doc = "---\ntitle: Ani\n---\n\n# Ani\n\nAni ören yeri Kars ilinde bulunan bir Orta Çağ kentidir.\n"
    app.sidebar.file_uploader[0].set_value([("ani.md", doc.encode(), "text/markdown"),
                                            ("bos.md", b"", "text/markdown")])
    next(b for b in app.sidebar.button if b.label == "İndeksle").click().run()
    assert not app.exception
    assert (settings.docs_dir / "ani.md").exists()
    assert "İndekslendi: ani.md" in app.sidebar.success[0].value
    assert "bos.md: dosya boş." in app.sidebar.error[0].value
    assert "- Ani" in app.sidebar.markdown[1].value

    app.chat_input[0].set_value("Ani hangi ilde?").run()
    assert "[K1] Ani" in app.expander[0].label


def test_download_button_follows_history(app):
    button = app.sidebar.get("download_button")[0]
    assert button.proto.label == "Sohbeti indir (.md)" and button.proto.disabled
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert not app.sidebar.get("download_button")[0].proto.disabled


def test_deleted_doc_leaves_list_and_search(app, settings):
    assert "- Efes (1 parça)" in app.sidebar.markdown[1].value
    app.sidebar.selectbox[0].set_value("efes.md")
    next(b for b in app.sidebar.button if b.label == "Sil").click().run()
    assert not app.exception
    assert not (settings.docs_dir / "efes.md").exists()
    assert "Silindi: efes.md" in app.sidebar.success[0].value
    assert "Efes" not in app.sidebar.markdown[1].value

    app.chat_input[0].set_value("Efes nerede?").run()
    assert not any("efes.md" in e.label or "[K1] Efes" in e.label for e in app.expander)


def test_last_doc_cannot_be_deleted(app, settings):
    for name in ("efes.md", "gobeklitepe.md"):
        app.sidebar.selectbox[0].set_value(name)
        next(b for b in app.sidebar.button if b.label == "Sil").click().run()
    assert not app.exception
    assert "En az bir doküman kalmalı." in app.sidebar.error[0].value
    assert (settings.docs_dir / "gobeklitepe.md").exists()


def test_votes_are_saved_changed_and_reset_with_chat(app, settings):
    from miras.store import Store

    app.chat_input[0].set_value("Efes nerede?").run()
    app.chat_input[0].set_value("Pizza hamuru tarifi nedir?").run()
    assert len(app.feedback) == 2
    app.feedback[0].set_value(1).run()
    app.feedback[1].set_value(0).run()
    assert not app.exception
    store = Store(settings.db_path)
    rows = {r["question"]: r for r in store.feedback_rows()}
    assert rows["Efes nerede?"]["rating"] == 1
    assert rows["Efes nerede?"]["sources"][0]["cited"] is True
    assert rows["Pizza hamuru tarifi nedir?"]["rating"] == -1
    assert rows["Pizza hamuru tarifi nedir?"]["refusal_reason"] == "low_similarity"

    app.feedback[0].set_value(0).run()
    app.feedback[1].set_value(None).run()
    assert [(r["question"], r["rating"]) for r in store.feedback_rows()] == [("Efes nerede?", -1)]

    next(b for b in app.sidebar.button if b.label == "Sohbeti temizle").click().run()
    app.chat_input[0].set_value("Efes nerede?").run()
    assert app.feedback[0].value is None
    assert len(store.feedback_rows()) == 1


def test_wikipedia_article_is_added_and_searchable(app, settings, monkeypatch):
    import miras.wiki

    page = {"title": "Ani", "extract": "Ani ören yeri Kars ilinde bulunan bir Orta Çağ kentidir.",
            "revisions": [{"revid": 1, "timestamp": "2026-10-01T00:00:00Z"}]}
    monkeypatch.setattr(miras.wiki, "fetch", lambda title: {"pages": [page]})
    app.sidebar.text_input[0].set_value("https://tr.wikipedia.org/wiki/Ani")
    app.sidebar.text_input[1].set_value("Ani Harabeleri, Ani Ören Yeri")
    next(b for b in app.sidebar.button if b.label == "İndir ve indeksle").click().run()
    assert not app.exception
    text = (settings.docs_dir / "ani.md").read_text(encoding="utf-8")
    assert "aliases: Ani Harabeleri, Ani Ören Yeri" in text and "source: https://tr.wikipedia.org/wiki/Ani" in text
    assert "İndekslendi: ani.md" in app.sidebar.success[0].value
    assert "- Ani (1 parça)" in app.sidebar.markdown[1].value

    app.chat_input[0].set_value("Ani hangi ilde?").run()
    assert "[K1] Ani" in app.expander[0].label


def test_bad_wikipedia_link_shows_error(app, settings):
    app.sidebar.text_input[0].set_value("https://en.wikipedia.org/wiki/Ani")
    next(b for b in app.sidebar.button if b.label == "İndir ve indeksle").click().run()
    assert not app.exception
    assert "tr.wikipedia.org" in app.sidebar.error[0].value
    assert not (settings.docs_dir / "ani.md").exists()


def test_wikipedia_updates_are_checked_and_applied(app, settings, monkeypatch):
    import miras.wiki

    (settings.docs_dir / "ani.md").write_text(
        "---\ntitle: Ani\naliases: Ani Harabeleri\nsource: https://tr.wikipedia.org/wiki/Ani\n"
        "revision: 1 (2026-08-01T00:00:00Z)\n---\n\n# Ani\n\nEski metin.\n", encoding="utf-8")
    monkeypatch.setattr(miras.wiki, "latest_revisions", lambda titles: {"Ani": 2})
    page = {"title": "Ani", "extract": "Ani ören yeri Kars ilinde bulunan bir Orta Çağ kentidir.",
            "revisions": [{"revid": 2, "timestamp": "2026-10-01T00:00:00Z"}]}
    monkeypatch.setattr(miras.wiki, "fetch", lambda title: {"pages": [page]})

    next(b for b in app.sidebar.button if b.label == "Güncellemeleri denetle").click().run()
    assert not app.exception
    assert any("Ani: 1 → 2" in m.value for m in app.sidebar.markdown)
    next(b for b in app.sidebar.button if b.label == "Güncelle (1)").click().run()
    assert not app.exception
    text = (settings.docs_dir / "ani.md").read_text(encoding="utf-8")
    assert "revision: 2" in text and "aliases: Ani Harabeleri" in text and "Eski metin" not in text
    assert "İndekslendi: ani.md" in app.sidebar.success[0].value
    assert not any(b.label.startswith("Güncelle (") for b in app.sidebar.button)

    next(b for b in app.sidebar.button if b.label == "Güncellemeleri denetle").click().run()
    assert any("Tüm maddeler güncel." in c.value for c in app.sidebar.caption)


def test_wikipedia_update_check_offline_shows_error(app, settings, monkeypatch):
    import miras.wiki

    def offline(params):
        raise miras.wiki.WikiError("Vikipedi'ye ulaşılamadı: ağ yok")

    monkeypatch.setattr(miras.wiki, "_get", offline)
    (settings.docs_dir / "ani.md").write_text(
        "---\ntitle: Ani\nsource: https://tr.wikipedia.org/wiki/Ani\n---\n\nx\n", encoding="utf-8")
    next(b for b in app.sidebar.button if b.label == "Güncellemeleri denetle").click().run()
    assert not app.exception
    assert "ulaşılamadı" in app.sidebar.error[0].value
    assert not any(b.label.startswith("Güncelle (") for b in app.sidebar.button)
