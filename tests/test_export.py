from datetime import datetime

from miras.export import conversation_markdown

NOW = datetime(2026, 10, 26, 14, 5)


def test_cited_answer_and_follow_up_are_exported(assistant, backend, settings):
    backend.reply = "Göbeklitepe Şanlıurfa'dadır [K1]."
    first = assistant.ask("Göbeklitepe hangi ilde?")
    follow = assistant.ask("Peki ne zaman\nlisteye girdi?", first)
    text = conversation_markdown([("Göbeklitepe hangi ilde?", first),
                                  ("Peki ne zaman\nlisteye girdi?", follow)], settings, NOW)
    assert text.startswith("# Miras sohbeti\n\n26.10.2026 14:05 · sohbet modeli")
    assert "## Soru 2\n\n> Peki ne zaman listeye girdi?\n\n↪ Önceki sorudan devam: Göbeklitepe" in text
    assert "Göbeklitepe Şanlıurfa'dadır [K1].\n\n**Kaynaklar:**" in text
    assert "- [K1] Göbeklitepe" in text and "(gobeklitepe.md, benzerlik" in text
    assert "[Vikipedi](https://example.org/g)" in text
    assert text.rstrip().endswith("lisansıyla paylaşılır.")


def test_refusals_have_no_source_list(assistant, backend, settings):
    gated = assistant.ask("Pizza hamuru tarifi nedir?")
    backend.reply = "Bu bilgi elimdeki dokümanlarda yer almıyor."
    refused = assistant.ask("Efes giriş ücreti ne kadar?")
    text = conversation_markdown([("Pizza hamuru tarifi nedir?", gated),
                                  ("Efes giriş ücreti ne kadar?", refused)], settings, NOW)
    assert "model çağrılmadı" in text and "cevap içlerinde yoktu" in text
    assert "[K1]" not in text and "Kaynaklar" not in text and "CC BY-SA" not in text


def test_uncited_answer_lists_inspected_chunks(assistant, backend, settings):
    backend.reply = "Efes İzmir'dedir."
    answer = assistant.ask("Efes nerede?")
    text = conversation_markdown([("Efes nerede?", answer)], settings, NOW)
    assert "**İncelenen parçalar** (model atıf yapmadı):" in text
    efes = next(line for line in text.splitlines() if line.startswith("- [K1]"))
    assert efes.startswith("- [K1] Efes (efes.md") and "Vikipedi" not in efes  # efes.md'de kaynak yok


def test_cached_answer_is_marked(assistant, backend, settings, store):
    assistant.cache = store
    assistant.ask("Efes nerede?")
    answer = assistant.ask("Efes nerede?")
    assert "_Cevap önbellekten geldi._" in conversation_markdown([("Efes nerede?", answer)], settings, NOW)
