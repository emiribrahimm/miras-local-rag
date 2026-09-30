from miras.prompts import (ABSTAIN_SENTENCE, EMPTY_QUESTION_MESSAGE, build_messages, cited_indices,
                           is_abstention, strip_invalid_citations)


def test_grounded_answer_keeps_valid_citations(assistant, backend):
    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    answer = assistant.ask("Göbeklitepe hangi yıl listeye girdi?")
    assert not answer.refused
    assert answer.cited == [0]
    assert answer.sources[0].chunk.filename == "gobeklitepe.md"


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
    assert answer.text == ABSTAIN_SENTENCE and answer.cited == []


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


def test_abstention_variants():
    assert is_abstention("Bu bilgi elimdeki dokümanlarda yer almıyor.")
    assert is_abstention("Verilen kaynaklarda bu bilgi bulunmamaktadır.")
    assert is_abstention("Kaynaklarda belirtilmemiş.")
    assert not is_abstention("Efes, İzmir'in Selçuk ilçesinde yer alıyor [K1].")


def test_citation_helpers():
    assert cited_indices("a [K2] b [K1; K2] c [K5]", 3) == [1, 0]
    assert strip_invalid_citations("x [K4].", 3) == "x."
    assert build_messages("s", [])[1]["content"].count("SORU: s") == 1
