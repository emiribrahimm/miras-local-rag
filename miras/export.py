"""Sohbeti kaynaklarıyla birlikte Markdown metnine çevirir (web arayüzündeki indirme düğmesi)."""

from __future__ import annotations

from datetime import datetime

from .config import Settings
from .pipeline import Answer


def _sources(answer: Answer) -> tuple[list[str], bool]:
    """Cevabın kaynak satırları ve aralarında bağlantılı (Vikipedi) kaynak olup olmadığı."""
    if answer.refusal_reason == "empty":
        return [], False
    if answer.refusal_reason == "low_similarity":
        return [f"_İlgili doküman parçası bulunamadı; model çağrılmadı "
                f"(en yüksek benzerlik {max(h.score for h in answer.hits):.2f})._"], False
    if answer.refused:
        return ["_İlgili parçalar bulundu ama cevap içlerinde yoktu._"], False
    # Model etiket yazmadıysa hangi parçalara bakıldığı yine de görünsün.
    shown = answer.cited or list(range(len(answer.hits)))
    lines = ["**Kaynaklar:**" if answer.cited else "**İncelenen parçalar** (model atıf yapmadı):", ""]
    linked = False
    for i in shown:
        chunk, score = answer.hits[i].chunk, answer.hits[i].score
        line = f"- [K{i + 1}] {chunk.section} ({chunk.filename}, benzerlik {score:.2f})"
        if chunk.source_url:
            line += f" · [Vikipedi]({chunk.source_url})"
            linked = True
        lines.append(line)
        # Parçada cevabı destekleyen cümle, kaynak satırının altında alıntı olarak.
        lines += [f"  > {sentence}" for sentence in answer.support.get(i, [])]
    return lines, linked


def conversation_markdown(history: list[tuple[str, Answer]], settings: Settings,
                          now: datetime) -> str:
    parts = ["# Miras sohbeti", "",
             f"{now:%d.%m.%Y %H:%M} · sohbet modeli `{settings.chat_model}` · "
             f"embedding `{settings.embed_model}`"]
    any_linked = False
    for number, (question, answer) in enumerate(history, start=1):
        # Çok satırlı soru alıntı bloğunu bozmasın.
        parts += ["", f"## Soru {number}", "", f"> {' '.join(question.split())}", ""]
        if answer.carried:
            parts += [f"↪ Önceki sorudan devam: {', '.join(answer.carried)}", ""]
        parts.append(answer.text)
        lines, linked = _sources(answer)
        any_linked |= linked
        if lines:
            parts += [""] + lines
        if answer.cached:
            parts += ["", "_Cevap önbellekten geldi._"]
    if any_linked:
        parts += ["", "---", "",
                  "Kaynak metinler Türkçe Vikipedi katkıcılarına aittir ve "
                  "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) "
                  "lisansıyla paylaşılır."]
    return "\n".join(parts) + "\n"
