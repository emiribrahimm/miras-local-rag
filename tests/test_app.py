"""Streamlit arayüzünü gerçek model olmadan uçtan uca çalıştırır."""

import pytest
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
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30).run()


def test_answer_and_sources_are_rendered(app):
    assert not app.exception
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert app.chat_message[1].markdown[0].value == "Efes İzmir'dedir [K1]."
    assert "[K1] Efes" in app.expander[0].label


def test_out_of_scope_question_shows_refusal(app):
    app.chat_input[0].set_value("Pizza hamuru tarifi nedir?").run()
    assert app.chat_message[1].markdown[0].value == ABSTAIN_SENTENCE
    assert "model çağrılmadı" in app.chat_message[1].caption[0].value


def test_follow_up_question_shows_carried_topic(app):
    app.chat_input[0].set_value("Efes nerede?").run()
    app.chat_input[0].set_value("Peki hangi ilçede?").run()
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
