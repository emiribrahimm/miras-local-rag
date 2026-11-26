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
    json_button = app.sidebar.get("download_button")[1]
    assert json_button.proto.label == "Sohbeti indir (.json)" and not json_button.proto.disabled


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


def test_files_changed_outside_the_app_can_be_indexed(app, settings):
    (settings.docs_dir / "ani.md").write_text("# Ani\n\nAni ören yeri Kars ilinde bulunan bir kenttir.\n",
                                              encoding="utf-8")
    (settings.docs_dir / "efes.md").unlink()
    app.run()
    assert "ani.md, efes.md (silinmiş)" in app.sidebar.warning[0].value
    next(b for b in app.sidebar.button if b.label == "İndeksi güncelle").click().run()
    assert not app.exception
    assert "İndekslendi: ani.md" in app.sidebar.success[0].value
    assert not app.sidebar.warning
    assert "- Ani (1 parça)" in app.sidebar.markdown[1].value and "Efes" not in app.sidebar.markdown[1].value


def test_clearing_cache_sends_repeated_question_to_model(app):
    assert not any(b.label.startswith("Önbelleği temizle") for b in app.sidebar.button)
    app.chat_input[0].set_value("Efes nerede?").run()
    next(b for b in app.sidebar.button if b.label == "Önbelleği temizle (1 cevap)").click().run()
    assert not app.exception
    assert not any(b.label.startswith("Önbelleği temizle") for b in app.sidebar.button)
    app.chat_input[0].set_value("Efes nerede?").run()
    assert len(app.fake.chat_calls) == 2


def test_last_answer_can_be_regenerated_past_the_cache(app, settings):
    from miras.store import Store

    app.fake.reply = "Efes Ankara'dadır [K1]."
    app.chat_input[0].set_value("Efes nerede?").run()
    app.feedback[0].set_value(0).run()
    app.fake.reply = "Efes İzmir'dedir [K1]."
    next(b for b in app.button if b.label == "Yeniden üret").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2 and len(app.fake.chat_calls) == 2
    assert app.chat_message[1].markdown[0].value == "Efes İzmir'dedir [K1]."
    assert app.feedback[0].value is None
    # Eski cevabın oyu durur, yeni cevap önbellekte eskisinin yerini alır.
    assert [r["answer"] for r in Store(settings.db_path).feedback_rows()] == ["Efes Ankara'dadır [K1]."]
    app.chat_input[0].set_value("Efes nerede?").run()
    assert app.chat_message[3].markdown[0].value == "Efes İzmir'dedir [K1]."
    assert len(app.fake.chat_calls) == 2


def test_refused_answer_has_no_regenerate_button(app):
    app.chat_input[0].set_value("Pizza hamuru tarifi nedir?").run()
    assert not any(b.label == "Yeniden üret" for b in app.button)


def test_document_chunks_can_be_inspected(app):
    app.sidebar.selectbox(key="inspect").set_value("gobeklitepe.md").run()
    assert not app.exception
    assert any(c.value.startswith("2 parça · [kaynak](https://example.org/g)") for c in app.sidebar.caption)
    labels = [m.value for m in app.sidebar.markdown]
    assert "**1. Göbeklitepe**" in labels and "**2. Göbeklitepe › Koruma**" in labels
    assert any("2018 yılında Dünya Mirası" in t.value for t in app.sidebar.text)

    app.sidebar.selectbox(key="inspect").set_value("efes.md").run()
    assert any("takma adlar: Ephesos, Ephesus" in c.value for c in app.sidebar.caption)


def test_search_can_be_limited_from_the_sidebar(app):
    app.sidebar.multiselect(key="only_docs").set_value(["efes.md"]).run()
    assert not app.exception
    app.chat_input[0].set_value("Göbeklitepe hangi yıl listeye girdi?").run()
    assert app.chat_message[1].markdown[0].value == ABSTAIN_SENTENCE and app.fake.chat_calls == []
    assert "🔎 Yalnızca şu dokümanlarda arandı: efes.md" in [c.value for c in app.chat_message[1].caption]

    app.sidebar.multiselect(key="only_docs").set_value([]).run()
    app.chat_input[0].set_value("Göbeklitepe hangi yıl listeye girdi?").run()
    assert len(app.fake.chat_calls) == 1
    assert not any("arandı" in c.value for c in app.chat_message[3].caption)


def test_deleted_doc_drops_out_of_the_filter(app, settings):
    app.sidebar.multiselect(key="only_docs").set_value(["efes.md", "gobeklitepe.md"]).run()
    app.sidebar.selectbox[0].set_value("efes.md")
    next(b for b in app.sidebar.button if b.label == "Sil").click().run()
    assert not app.exception
    assert app.sidebar.multiselect(key="only_docs").value == ["gobeklitepe.md"]


def test_search_settings_apply_to_this_session_only(app, settings):
    app.sidebar.slider(key="min_similarity").set_value(0.99).run()
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert app.chat_message[1].markdown[0].value == ABSTAIN_SENTENCE and app.fake.chat_calls == []
    answer = app.session_state["history"][0][1]
    assert answer.search["min_similarity"] == 0.99 and answer.refusal_reason == "low_similarity"
    captions = [c.value for c in app.chat_message[1].caption]
    assert "eşik: 0.99)" in captions[0] and "⚙️ Bu oturumun arama ayarları: min_similarity 0.99" in captions

    app.sidebar.slider(key="min_similarity").set_value(settings.min_similarity)
    app.sidebar.slider(key="top_k").set_value(1).run()
    app.chat_input[0].set_value("Efes nerede?").run()
    answer = app.session_state["history"][1][1]
    assert len(answer.hits) == 1 and not answer.refused
    assert any(c.value.endswith("top_k 1") for c in app.chat_message[3].caption)
    assert not any("⚙️" in c.value for c in app.chat_message[1].caption if "top_k" in c.value)


def test_refused_questions_are_listed_in_the_sidebar(app):
    app.chat_input[0].set_value("Pizza hamuru tarifi nedir?").run()
    app.chat_input[0].set_value("Efes nerede?").run()
    assert not app.exception
    assert any(c.value.startswith("Günlükte 2 soru, 1 tanesi reddedildi.") for c in app.sidebar.caption)
    refused = [m.value for m in app.sidebar.markdown if "Pizza hamuru" in m.value]
    assert len(refused) == 1 and "_(kapıda, " in refused[0]
    assert not any("Efes nerede?" in m.value for m in app.sidebar.markdown)
    assert any(b.label == "Eval adayı olarak indir (.json)" for b in app.sidebar.get("download_button"))


def test_index_changed_elsewhere_is_picked_up(app, store, backend, settings):
    from miras.ingest import ingest

    (settings.docs_dir / "ani.md").write_text("---\ntitle: Ani\n---\n\nAni Kars ilindedir.", encoding="utf-8")
    ingest(store, backend, settings)  # arayüz açıkken CLI'dan
    app.run()
    assert not app.exception
    assert "- Ani (1 parça)" in app.sidebar.markdown[1].value
    app.chat_input[0].set_value("Ani hangi ilde?").run()
    assert any("ani.md" in e.label or "[K1] Ani" in e.label for e in app.expander)
