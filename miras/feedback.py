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


TYPE_PLACEHOLDER = "answerable | near_miss | out_of_domain"


def eval_candidates(rows: list[dict]) -> list[dict]:
    """Beğenilmeyen cevapları `data/eval/questions.json` biçiminde aday olarak döndürür.

    Doğru davranışı (cevap mı, reddetme mi) ve beklenen anahtar kelimeleri oy söylemez; `type`
    ve `expected` elle doldurulur. Aynı soru birden çok kez oylandıysa en yenisi alınır.
    """
    latest: dict[tuple, dict] = {}
    for row in rows:
        if row["rating"] < 0:
            latest.setdefault((" ".join(row["question"].split()).lower(), row["carried"]), row)
    candidates = []
    for number, row in enumerate(sorted(latest.values(), key=lambda r: r["created"]), start=1):
        cited = [s["filename"] for s in row["sources"] if s["cited"]]
        doc = (cited or [s["filename"] for s in row["sources"]] or [""])[0]
        candidate = {"id": f"fb{number:02d}", "type": TYPE_PLACEHOLDER, "split": "dev", "doc": doc,
                     "question": row["question"], "expected": [],
                     "answer_was": row["answer"], "refusal_reason": row["refusal_reason"]}
        if row["carried"]:
            # Takip sorusu: followups.json'a önceki soruyla birlikte eklenmeli.
            candidate["carried"] = list(row["carried"])
        candidates.append(candidate)
    return candidates
