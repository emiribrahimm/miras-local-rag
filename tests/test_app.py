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
