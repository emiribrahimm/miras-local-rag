"""Cevaplara verilen 👍/👎 oyları: kaydetme ve eval setine aday çıkarma."""

from __future__ import annotations

from .pipeline import Answer
from .store import Store


def answer_sources(answer: Answer) -> list[dict]:
    """Getirilen parçaların özeti; oy sonradan incelenirken nelere bakıldığı görünsün."""
    return [{"filename": hit.chunk.filename, "section": hit.chunk.section,
             "score": round(hit.score, 3), "cited": i in answer.cited}
            for i, hit in enumerate(answer.hits)]


def save_vote(store: Store, key: str, thumbs: int | None, question: str, answer: Answer,
              chat_model: str) -> None:
    """`st.feedback("thumbs")` değerini kaydeder: 1 = 👍, 0 = 👎, None = oy geri alındı."""
    if thumbs is None:
        store.delete_feedback(key)
        return
    store.save_feedback(key, 1 if thumbs == 1 else -1, question, answer.carried, answer.text,
                        answer.refusal_reason, answer_sources(answer), chat_model)
