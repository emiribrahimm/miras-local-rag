"""Değerlendirme: arama kalitesi, cevap doğruluğu, reddetme isabeti ve gecikme.

Soru türleri (data/eval/questions.json):
  * answerable     - cevabı dokümanlarda var; `expected` anahtar kelime gruplarıyla puanlanır
  * near_miss      - konu alanı içinde ama dokümanlarda yok; doğru davranış reddetmek
  * out_of_domain  - konuyla ilgisiz; doğru davranış reddetmek

`split` alanı: benzerlik eşiği yalnızca "dev" sorularıyla seçilir, "test" soruları
eşiği seçerken hiç kullanılmaz.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Backend
from .pipeline import Assistant
from .retrieval import Retriever

UNANSWERABLE = ("near_miss", "out_of_domain")


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    """Türkçe büyük/küçük harf kurallarıyla küçültür (İ->i, I->ı)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def matches_expected(text: str, expected: list[list[str]]) -> bool:
    """Her gruptan en az bir varyant metinde geçmeli."""
    lowered = normalize(text)
    return all(any(normalize(variant) in lowered for variant in group) for group in expected)


# --- arama ---------------------------------------------------------------

@dataclass
class RetrievalRow:
    id: str
    type: str
    split: str
    top_score: float
    doc_rank: int | None      # beklenen dokümanın ilk göründüğü sıra (1 tabanlı)
    answer_in_context: bool   # beklenen anahtar kelimeler getirilen parçalarda var mı


def evaluate_retrieval(questions: list[dict], backend: Backend, retriever: Retriever,
                       top_k: int) -> list[RetrievalRow]:
    rows = []
    for q in questions:
        hits = retriever.search(backend.embed_query(q["question"]), top_k, q["question"])
        rank, in_context = None, False
        if q["type"] == "answerable":
            rank = next((i for i, h in enumerate(hits, 1) if h.chunk.filename == q["doc"]), None)
            in_context = matches_expected("\n".join(h.chunk.text for h in hits), q["expected"])
        rows.append(RetrievalRow(q["id"], q["type"], q["split"], max(h.score for h in hits),
                                 rank, in_context))
    return rows


def retrieval_summary(rows: list[RetrievalRow]) -> dict:
    answerable = [r for r in rows if r.type == "answerable"]
    n = len(answerable) or 1
    return {
        "questions": len(answerable),
        "doc_hit_at_1": sum(r.doc_rank == 1 for r in answerable) / n,
        "doc_hit_at_k": sum(r.doc_rank is not None for r in answerable) / n,
        "mrr": sum(1 / r.doc_rank for r in answerable if r.doc_rank) / n,
        "answer_in_context_at_k": sum(r.answer_in_context for r in answerable) / n,
    }


def calibrate_threshold(rows: list[RetrievalRow]) -> dict:
    """Benzerlik eşiğini dev sorularından seçer, test sorularında doğrular.

    Kapının işi "hiç ilgili parça yok" durumunu yakalamaktır; bu yüzden
    cevaplanabilir sorular ile konu dışı sorular ayrıştırılır. Konu içi ama
    dokümanda olmayan (near_miss) sorular yüksek benzerlik alır; onları kapı
    değil model reddetmelidir ve ayrıca raporlanır.
    """
    def balanced_accuracy(subset: list[RetrievalRow], threshold: float) -> float:
        kept = [r.top_score >= threshold for r in subset if r.type == "answerable"]
        blocked = [r.top_score < threshold for r in subset if r.type == "out_of_domain"]
        return (sum(kept) / len(kept) + sum(blocked) / len(blocked)) / 2

    dev = [r for r in rows if r.split == "dev"]
    test = [r for r in rows if r.split == "test"]
    candidates = [round(0.20 + 0.01 * i, 2) for i in range(61)]
    scores = {t: balanced_accuracy(dev, t) for t in candidates}
    best = max(scores.values())
    plateau = [t for t in candidates if scores[t] == best]
    threshold = plateau[len(plateau) // 2]  # en iyi aralığın ortası: iki yöne de pay bırakır

    def near_miss_blocked(subset: list[RetrievalRow]) -> float:
        nm = [r.top_score < threshold for r in subset if r.type == "near_miss"]
        return sum(nm) / len(nm)

    return {
        "threshold": threshold,
        "best_range": [plateau[0], plateau[-1]],
        "dev_balanced_accuracy": best,
        "test_balanced_accuracy": balanced_accuracy(test, threshold),
        "near_miss_blocked_by_gate": near_miss_blocked(rows),
    }


# --- cevap ---------------------------------------------------------------

@dataclass
class AnswerRow:
    id: str
    type: str
    split: str
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    passed: bool
    cited_expected_doc: bool | None
    top_score: float
    seconds: float


def evaluate_answers(questions: list[dict], assistant: Assistant,
                     on_row=lambda _row: None) -> list[AnswerRow]:
    rows = []
    for q in questions:
        answer = assistant.ask(q["question"])
        cited_doc = None
        if q["type"] == "answerable":
            passed = not answer.refused and matches_expected(answer.text, q["expected"])
            if not answer.refused:
                cited_doc = any(h.chunk.filename == q["doc"] for h in answer.sources)
        else:
            passed = answer.refused
        row = AnswerRow(q["id"], q["type"], q["split"], q["question"], answer.text, answer.refused,
                        answer.refusal_reason, passed, cited_doc,
                        max((h.score for h in answer.hits), default=0.0),
                        answer.retrieval_seconds + answer.generation_seconds)
        on_row(row)
        rows.append(row)
    return rows


def answer_summary(rows: list[AnswerRow]) -> dict:
    def rate(subset: list[AnswerRow]) -> dict:
        return {"passed": sum(r.passed for r in subset), "total": len(subset)}

    answerable = [r for r in rows if r.type == "answerable"]
    answered = [r for r in answerable if not r.refused]
    generated = [r.seconds for r in rows if r.refusal_reason != "low_similarity"]
    return {
        "overall": rate(rows),
        "test_split_only": rate([r for r in rows if r.split == "test"]),
        "answerable_correct": rate(answerable),
        "answerable_wrongly_refused": sum(r.refused for r in answerable),
        "answered_with_correct_citation": {
            "passed": sum(bool(r.cited_expected_doc) for r in answered), "total": len(answered)},
        "near_miss_refused": rate([r for r in rows if r.type == "near_miss"]),
        "out_of_domain_refused": rate([r for r in rows if r.type == "out_of_domain"]),
        "refused_by_gate_without_llm": sum(r.refusal_reason == "low_similarity" for r in rows),
        "median_seconds_when_llm_called": round(statistics.median(generated), 2) if generated else 0,
        "max_seconds": round(max(r.seconds for r in rows), 2),
    }


def write_report(path_stem: Path, header: dict, retrieval: dict, answers: dict,
                 rows: list[AnswerRow]) -> None:
    """JSON (makine) ve Markdown (insan) olarak iki rapor yazar."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": header, "retrieval": retrieval, "answers": answers,
               "rows": [asdict(r) for r in rows]}
    path_stem.with_name(path_stem.name + ".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def frac(d: dict) -> str:
        return f"{d['passed']}/{d['total']}"

    lines = [
        f"# Değerlendirme raporu — {header['chat_model']} ({header['chat_device']})", "",
        f"- Embedding modeli: `{header['embed_model']}`",
        f"- top_k: {header['top_k']}, benzerlik eşiği: {header['min_similarity']}",
        f"- Parça sayısı: {header['chunks']}", "",
        "## Arama (yalnızca cevaplanabilir sorular)", "",
        f"- Doğru doküman 1. sırada: {retrieval['doc_hit_at_1']:.0%}",
        f"- Doğru doküman ilk {header['top_k']} içinde: {retrieval['doc_hit_at_k']:.0%}",
        f"- MRR: {retrieval['mrr']:.3f}",
        f"- Cevap getirilen parçalarda mevcut: {retrieval['answer_in_context_at_k']:.0%}", "",
        "## Cevaplar", "",
        f"- Genel: {frac(answers['overall'])} (yalnızca test bölümü: {frac(answers['test_split_only'])})",
        f"- Cevaplanabilir sorularda doğru cevap: {frac(answers['answerable_correct'])}"
        f" (yanlışlıkla reddedilen: {answers['answerable_wrongly_refused']})",
        f"- Verilen cevaplarda doğru dokümana atıf: {frac(answers['answered_with_correct_citation'])}",
        f"- Konu içi ama dokümanda olmayan sorular reddedildi: {frac(answers['near_miss_refused'])}",
        f"- Konu dışı sorular reddedildi: {frac(answers['out_of_domain_refused'])}",
        f"- Model çağrılmadan kapıda reddedilen: {answers['refused_by_gate_without_llm']}",
        f"- Model çağrılan sorularda ortanca süre: {answers['median_seconds_when_llm_called']} sn"
        f" (en uzun: {answers['max_seconds']} sn)", "",
        "## Soru bazında sonuçlar", "",
        "| # | Tür | Sonuç | Skor | Süre (sn) | Soru | Cevap |", "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        answer = r.answer.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {r.id} | {r.type} | {'✅' if r.passed else '❌'} | {r.top_score:.2f} "
                     f"| {r.seconds:.1f} | {r.question} | {answer} |")
    path_stem.with_name(path_stem.name + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
