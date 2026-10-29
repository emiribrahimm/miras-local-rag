import pytest

from miras.pipeline import refers_back
from miras.prompts import (ABSTAIN_SENTENCE, EMPTY_QUESTION_MESSAGE, build_messages, cited_indices,
                           is_abstention, strip_invalid_citations)


def test_grounded_answer_keeps_valid_citations(assistant, backend):
    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    answer = assistant.ask("Göbeklitepe hangi yıl listeye girdi?")
    assert not answer.refused
    assert answer.cited == [0]
    assert answer.sources[0].chunk.filename == "gobeklitepe.md"
    assert answer.support == {0: ["Göbeklitepe 2018 yılında Dünya Mirası listesine girdi."]}


def test_prompt_contains_labelled_context_and_question(assistant, backend):
    assistant.ask("Efes nerede?")
    system, user = backend.chat_calls[0]
    assert ABSTAIN_SENTENCE in system["content"]
    assert "[K1] (Efes)" in user["content"] and "SORU: Efes nerede?" in user["content"]


def test_low_similarity_refuses_without_calling_model(assistant, backend):
    answer = assistant.ask("Pizza hamuru tarifi nedir?")
    assert answer.refused and answer.refusal_reason == "low_similarity"
    assert answer.text == ABSTAIN_SENTENCE
    assert backend.chat_calls == []


def test_empty_question_is_rejected_without_any_model_call(assistant, backend):
    answer = assistant.ask("   \n ")
    assert answer.refusal_reason == "empty" and answer.text == EMPTY_QUESTION_MESSAGE
    assert backend.chat_calls == [] and answer.hits == []


def test_model_abstention_is_detected_and_cleaned(assistant, backend):
    backend.reply = "Maalesef bu bilgi elimdeki dokümanlardan yer almıyor. [K1]"
    answer = assistant.ask("Efes giriş ücreti ne kadar?")
    assert answer.refused and answer.refusal_reason == "model"
    assert answer.text == ABSTAIN_SENTENCE and answer.cited == [] and answer.support == {}


def test_hallucinated_citation_labels_are_removed(assistant, backend):
    backend.reply = "Efes İzmir'dedir [K1, K9]. Selçuk ilçesindedir [K7]."
    answer = assistant.ask("Efes nerede?")
    assert answer.text == "Efes İzmir'dedir [K1]. Selçuk ilçesindedir."
    assert answer.cited == [0]


def test_streaming_pieces_join_to_final_text(assistant, backend):
    backend.reply = "Efes İzmir'dedir [K1]."
    prepared = assistant.prepare("Efes nerede?")
    pieces = list(assistant.generate(prepared))
    assert len(pieces) > 1
    assert assistant.finalize(prepared, "".join(pieces)).text == backend.reply


def test_follow_up_question_carries_previous_topic(assistant, backend):
    first = assistant.ask("Göbeklitepe hangi ilde?")
    assert first.topic == ("Göbeklitepe",) and first.carried == ()

    follow = assistant.ask("Peki ne zaman listeye girdi?", first)
    assert follow.carried == ("Göbeklitepe",) and follow.topic == ("Göbeklitepe",)
    assert follow.hits[0].chunk.filename == "gobeklitepe.md"
    assert "SORU: Göbeklitepe hakkında: Peki ne zaman listeye girdi?" in backend.chat_calls[-1][1]["content"]


def test_naming_another_site_switches_topic(assistant, backend):
    first = assistant.ask("Göbeklitepe hangi ilde?")
    answer = assistant.ask("Efes nerede?", first)
    assert answer.carried == () and answer.topic == ("Efes",)
    assert "SORU: Efes nerede?" in backend.chat_calls[-1][1]["content"]


def test_topic_comes_from_single_cited_site(assistant, backend):
    backend.reply = "Şanlıurfa ilindedir [K1]."
    answer = assistant.ask("Neolitik arkeolojik alan hangi ilde?")
    assert answer.sources[0].chunk.filename == "gobeklitepe.md"
    assert answer.topic == ("Göbeklitepe",)


def test_refused_turn_has_no_topic_to_carry(assistant, backend):
    refused = assistant.ask("Pizza hamuru tarifi nedir?")
    assert refused.topic == ()
    assert assistant.ask("Peki ne zaman listeye girdi?", refused).carried == ()

def test_off_topic_follow_up_does_not_carry_topic(assistant, backend):
    first = assistant.ask("Göbeklitepe hangi ilde?")
    answer = assistant.ask("Pizza hamuru tarifi nedir?", first)
    assert answer.carried == () and answer.refusal_reason == "low_similarity"
    assert len(backend.chat_calls) == 1  # yalnızca ilk soru modele gitti


def test_follow_up_floor_zero_always_carries(assistant, backend, settings):
    import dataclasses
    assistant.settings = dataclasses.replace(settings, followup_min_similarity=0.0)
    first = assistant.ask("Göbeklitepe hangi ilde?")
    answer = assistant.ask("Pizza hamuru tarifi nedir?", first)
    # Eski davranış: alan adı eklenince konu dışı soru da kapıyı geçiyor.
    assert answer.carried == ("Göbeklitepe",) and not answer.refused


def test_refers_back_words():
    for question in ("Orası ne zaman yapılmış?", "Oradaki tiyatro kaç kişilik?", "Burasi kac yasinda?",
                     "Onun mimarı kim?", "Peki hangisi daha eski?", "Kendisi ne zaman öldü?"):
        assert refers_back(question), question
    for question in ("Peki ne zaman yapılmış?", "Nüfus oranı nedir?", "Bir anı defteri", "On yıl sürdü",
                     "Ya giriş ücreti?", "Ondalık sayılar", "Onur kimdir?"):
        assert not refers_back(question), question


def test_refers_back_carries_topic_below_floor(assistant, backend, settings):
    import dataclasses
    assistant.settings = dataclasses.replace(settings, followup_min_similarity=0.99)
    first = assistant.ask("Göbeklitepe hangi ilde?")
    assert assistant.ask("Orası ne zaman listeye girdi?", first).carried == ("Göbeklitepe",)
    assert assistant.ask("Peki ne zaman listeye girdi?", first).carried == ()


def test_refers_back_with_site_name_switches_topic(assistant, backend):
    first = assistant.ask("Göbeklitepe hangi ilde?")
    answer = assistant.ask("Peki Efes'teki tiyatro onun kadar eski mi?", first)
    assert answer.carried == () and answer.topic == ("Efes",)


def test_bm25_weight_from_settings_reaches_search(assistant, settings, monkeypatch):
    import dataclasses
    seen = []
    search = assistant.retriever.search
    monkeypatch.setattr(assistant.retriever, "search",
                        lambda *args: seen.append(args[-1]) or search(*args))
    assistant.settings = dataclasses.replace(settings, bm25_weight=0.3)
    assistant.ask("Efes nerede?")
    assert seen == [0.3]

def test_abstention_variants():
    assert is_abstention("Bu bilgi elimdeki dokümanlarda yer almıyor.")
    assert is_abstention("Verilen kaynaklarda bu bilgi bulunmamaktadır.")
    assert is_abstention("Kaynaklarda belirtilmemiş.")
    assert not is_abstention("Efes, İzmir'in Selçuk ilçesinde yer alıyor [K1].")


def test_citation_helpers():
    assert cited_indices("a [K2] b [K1; K2] c [K5]", 3) == [1, 0]
    assert strip_invalid_citations("x [K4].", 3) == "x."
    assert build_messages("s", [])[1]["content"].count("SORU: s") == 1


@pytest.fixture
def cached(assistant, store):
    assistant.cache = store
    return assistant


def test_repeated_question_comes_from_cache(cached, backend):
    backend.reply = "Efes İzmir'dedir [K1]."
    first = cached.ask("Efes nerede?")
    second = cached.ask("Efes nerede?")
    assert len(backend.chat_calls) == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.cited == first.cited


def test_cache_misses_when_prompt_changes(cached, backend):
    first = cached.ask("Göbeklitepe hangi ilde?")
    cached.ask("Efes nerede?")
    # Aynı metin, ama taşınan konu yüzünden modele giden soru farklı.
    cached.ask("Peki ne zaman listeye girdi?")
    cached.ask("Peki ne zaman listeye girdi?", first)
    assert len(backend.chat_calls) == 4


def test_cached_answer_is_validated_again(cached, backend, store):
    backend.reply = "ilk"
    cached.ask("Efes nerede?")
    key = store.conn.execute("SELECT key FROM answer_cache").fetchone()[0]
    store.cache_answer(key, "Efes İzmir'dedir [K1, K9].")
    answer = cached.ask("Efes nerede?")
    assert answer.cached and answer.text == "Efes İzmir'dedir [K1]."


def test_refusals_and_disabled_cache_skip_storage(cached, backend, settings, store):
    import dataclasses
    cached.ask("Pizza hamuru tarifi nedir?")
    assert store.conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
    cached.settings = dataclasses.replace(settings, answer_cache=False)
    cached.ask("Efes nerede?")
    cached.ask("Efes nerede?")
    assert len(backend.chat_calls) == 2 and not cached.ask("Efes nerede?").cached


def test_interrupted_stream_is_not_cached(cached, backend, store):
    prepared = cached.prepare("Efes nerede?")
    next(cached.generate(prepared))  # akış yarıda bırakılıyor
    assert store.conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
