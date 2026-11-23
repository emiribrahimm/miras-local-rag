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
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Backend
from .pipeline import Answer, Assistant, refers_back
from .retrieval import Hit, Retriever

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


def expected_docs(doc: str | list[str] | None) -> list[str]:
    """Sorunun `doc` alanı: tek dosya adı ya da karşılaştırma sorularında birkaç tane."""
    if doc is None:
        return []
    return [doc] if isinstance(doc, str) else list(doc)


def _doc_rank(hits: list[Hit], doc: str | list[str]) -> int | None:
    """Beklenen dokümanın getirilen parçalar arasında ilk göründüğü sıra (1 tabanlı).

    Birkaç doküman bekleniyorsa hepsinin göründüğü sıra; biri eksikse None.
    """
    ranks = [next((i for i, h in enumerate(hits, 1) if h.chunk.filename == d), None)
             for d in expected_docs(doc)]
    return None if not ranks or None in ranks else max(ranks)


def _score(q: dict, answer: Answer) -> tuple[bool, bool | None]:
    """(geçti mi, beklenen dokümana atıf yapıldı mı). Atıf yalnızca verilen cevaplarda ölçülür."""
    if q["type"] != "answerable":
        return answer.refused, None
    if answer.refused:
        return False, None
    cited = {h.chunk.filename for h in answer.sources}
    return (matches_expected(answer.text, q["expected"]),
            all(doc in cited for doc in expected_docs(q["doc"])))


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
                       top_k: int, keyword_weight: float = 0.0) -> list[RetrievalRow]:
    rows = []
    for q in questions:
        hits = retriever.search(backend.embed_query(q["question"]), top_k, q["question"],
                                keyword_weight)
        rank, in_context = None, False
        if q["type"] == "answerable":
            rank = _doc_rank(hits, q["doc"])
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


def followup_floor_scores(followups: list[dict], questions: list[dict], backend: Backend,
                          retriever: Retriever) -> tuple[list[float], list[float]]:
    """Takip eşiği için ham soru benzerlikleri: (konu taşınması gerekenler, engellenmesi gerekenler).

    Taşınması gerekenler: alan adı geçmeyen cevaplanabilir takip soruları, tek başına.
    "Orası", "onun" gibi gönderme kelimesi olanlar eşikten bağımsız taşındığı için sayılmaz.
    Engellenmesi gerekenler: konu dışı sorular ve konu dışı takip soruları.
    """
    def best(text: str) -> float:
        return retriever.best_score(backend.embed_query(text))

    keep = [best(q["turns"][-1]) for q in followups
            if q["type"] == "answerable" and not q.get("switch") and not refers_back(q["turns"][-1])]
    block = ([best(q["question"]) for q in questions if q["type"] == "out_of_domain"]
             + [best(q["turns"][-1]) for q in followups if q["type"] == "out_of_domain"])
    return keep, block


def calibrate_followup_floor(keep: list[float], block: list[float]) -> dict:
    """Konu taşıma eşiği: takip sorularını taşıyıp konu dışı soruları taşımayan en iyi aralığın ortası."""
    def balanced_accuracy(floor: float) -> float:
        return (sum(s >= floor for s in keep) / len(keep) + sum(s < floor for s in block) / len(block)) / 2

    candidates = [round(0.20 + 0.01 * i, 2) for i in range(41)]
    scores = {t: balanced_accuracy(t) for t in candidates}
    best = max(scores.values())
    plateau = [t for t in candidates if scores[t] == best]
    return {
        "floor": plateau[len(plateau) // 2],
        "best_range": [plateau[0], plateau[-1]],
        "balanced_accuracy": best,
        "lowest_followup": round(min(keep), 3),
        "highest_off_topic": round(max(block), 3),
    }


KEYWORD_WEIGHTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


class _CachedEmbeddings:
    """Her soru bir kez gömülsün; ağırlık taraması aynı soruları tekrar tekrar arar."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.cache: dict[str, object] = {}

    def embed_query(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.backend.embed_query(text)
        return self.cache[text]


def tune_keyword_weight(questions: list[dict], backend: Backend, retriever: Retriever, top_k: int,
                        min_similarity: float, weights=KEYWORD_WEIGHTS) -> list[dict]:
    """Her BM25 ağırlığı için dev ve test sorularında arama özetini çıkarır (model çağrılmaz)."""
    cached = _CachedEmbeddings(backend)
    results = []
    for weight in weights:
        rows = evaluate_retrieval(questions, cached, retriever, top_k, weight)  # type: ignore[arg-type]
        result = {"weight": weight}
        for split in ("dev", "test"):
            subset = [r for r in rows if r.split == split]
            result[split] = retrieval_summary(subset)
            # Kapı getirilen parçaların ham benzerliğine bakar; sıralama değişince bu da oynayabilir.
            result[split]["answerable_gated"] = sum(
                r.top_score < min_similarity for r in subset if r.type == "answerable")
        results.append(result)
    return results


def choose_keyword_weight(results: list[dict]) -> float:
    """Dev MRR'ı en yüksek ağırlık; eşitlikte cevabı bağlamda tutan, sonra en küçük ağırlık."""
    def key(r: dict) -> tuple:
        dev = r["dev"]
        return (round(dev["mrr"], 6), round(dev["answer_in_context_at_k"], 6), -r["weight"])

    return max(results, key=key)["weight"]


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
    # Cevaplanabilir sorularda cevabın bulunduğu doküman; karşılaştırma sorularında liste.
    doc: str | list[str] | None = None


def evaluate_answers(questions: list[dict], assistant: Assistant,
                     on_row=lambda _row: None) -> list[AnswerRow]:
    rows = []
    for q in questions:
        answer = assistant.ask(q["question"])
        passed, cited_doc = _score(q, answer)
        row = AnswerRow(q["id"], q["type"], q["split"], q["question"], answer.text, answer.refused,
                        answer.refusal_reason, passed, cited_doc,
                        max((h.score for h in answer.hits), default=0.0),
                        answer.retrieval_seconds + answer.generation_seconds, q.get("doc"))
        on_row(row)
        rows.append(row)
    return rows


def by_doc(rows: list[AnswerRow]) -> dict[str, dict]:
    """Cevaplanabilir sorular doküman bazında: hangi maddede zorlandığı genel skordan görünmüyor.

    Karşılaştırma soruları andığı her dokümanın altında sayılır.
    """
    result: dict[str, dict] = {}
    pairs = [(doc, r) for r in rows if r.type == "answerable" for doc in expected_docs(r.doc)]
    for doc, r in sorted(pairs, key=lambda pair: pair[0]):
        d = result.setdefault(doc, {"passed": 0, "total": 0, "refused": 0, "cited": 0})
        d["passed"] += r.passed
        d["total"] += 1
        d["refused"] += r.refused
        d["cited"] += bool(r.cited_expected_doc)
    return result


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
        "max_seconds": round(max((r.seconds for r in rows), default=0.0), 2),
        "by_doc": by_doc(rows),
    }



# --- takip soruları --------------------------------------------------------
# data/eval/followups.json: iki turluk sohbetler, yalnızca son tur puanlanır. Son soru
# ayrıca bağlamsız aranır; böylece konuyu taşımanın aramaya etkisi görülür.

@dataclass
class FollowupRow:
    id: str
    type: str
    turns: list[str]
    answer: str
    refused: bool
    refusal_reason: str | None
    passed: bool
    cited_expected_doc: bool | None
    carried: list[str]                 # önceki turdan taşınan alanlar
    doc_rank: int | None               # konu taşınarak yapılan aramada beklenen doküman
    standalone_doc_rank: int | None    # son soru tek başına sorulsaydı
    standalone_gated: bool             # son soru tek başına sorulsaydı kapıda reddedilir miydi
    seconds: float


def evaluate_followups(items: list[dict], assistant: Assistant,
                       on_row=lambda _row: None) -> list[FollowupRow]:
    rows = []
    for q in items:
        *earlier, last = q["turns"]
        previous = None
        for turn in earlier:
            previous = assistant.ask(turn, previous)
        prepared = assistant.prepare(last, previous)
        started = time.perf_counter()
        raw = "".join(assistant.generate(prepared))
        answer = assistant.finalize(prepared, raw, time.perf_counter() - started)
        standalone = assistant.prepare(last)  # yalnızca arama, model çağrılmaz
        passed, cited_doc = _score(q, answer)
        doc = q.get("doc")
        row = FollowupRow(q["id"], q["type"], q["turns"], answer.text, answer.refused,
                          answer.refusal_reason, passed, cited_doc, list(answer.carried),
                          _doc_rank(answer.hits, doc) if doc else None,
                          _doc_rank(standalone.hits, doc) if doc else None,
                          standalone.refusal_reason == "low_similarity",
                          answer.retrieval_seconds + answer.generation_seconds)
        on_row(row)
        rows.append(row)
    return rows


def followup_summary(rows: list[FollowupRow]) -> dict:
    def rate(subset: list[FollowupRow]) -> dict:
        return {"passed": sum(r.passed for r in subset), "total": len(subset)}

    answerable = [r for r in rows if r.type == "answerable"]
    return {
        "overall": rate(rows),
        "answerable_correct": rate(answerable),
        "unanswerable_refused": rate([r for r in rows if r.type in UNANSWERABLE]),
        # Konu dışı takip sorularının model çağrılmadan kapıda reddedilmesi beklenir.
        "out_of_domain_gated": {
            "passed": sum(r.refusal_reason == "low_similarity" for r in rows if r.type == "out_of_domain"),
            "total": sum(r.type == "out_of_domain" for r in rows)},
        "topic_carried": sum(bool(r.carried) for r in rows),
        "doc_at_1_with_topic": {"passed": sum(r.doc_rank == 1 for r in answerable),
                                "total": len(answerable)},
        "doc_at_1_standalone": {"passed": sum(r.standalone_doc_rank == 1 for r in answerable),
                                "total": len(answerable)},
        "answerable_gated_standalone": sum(r.standalone_gated for r in answerable),
    }

def compare_reports(old: dict, new: dict) -> dict:
    """İki `eval` JSON raporu arasındaki fark: ayar değişiklikleri, skorlar ve sonucu değişen sorular.

    Takip soruları da karşılaştırılır; kimlikler (a01, f03) iki sette çakışmıyor.
    """
    def results(report: dict) -> dict[str, dict]:
        rows = report["rows"] + report.get("followups", {}).get("rows", [])
        return {r["id"]: r for r in rows}

    def score(report: dict, part: str) -> dict | None:
        if part == "followups":
            return report["followups"]["summary"]["overall"] if "followups" in report else None
        return report["answers"]["overall"]

    before, after = results(old), results(new)
    common = [i for i in after if i in before]
    keys = sorted(set(old["config"]) | set(new["config"]))
    return {
        "config": {k: (old["config"].get(k), new["config"].get(k)) for k in keys
                   if old["config"].get(k) != new["config"].get(k)},
        "answers": (score(old, "answers"), score(new, "answers")),
        "followups": (score(old, "followups"), score(new, "followups")),
        "fixed": [after[i] for i in common if after[i]["passed"] and not before[i]["passed"]],
        "broken": [after[i] for i in common if before[i]["passed"] and not after[i]["passed"]],
        "only_old": sorted(set(before) - set(after)),
        "only_new": sorted(set(after) - set(before)),
    }


def write_report(path_stem: Path, header: dict, retrieval: dict, answers: dict,
                 rows: list[AnswerRow], followups: dict | None = None,
                 followup_rows: list[FollowupRow] = ()) -> None:
    """JSON (makine) ve Markdown (insan) olarak iki rapor yazar."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": header, "retrieval": retrieval, "answers": answers,
               "rows": [asdict(r) for r in rows]}
    if followups is not None:
        payload["followups"] = {"summary": followups, "rows": [asdict(r) for r in followup_rows]}
    path_stem.with_name(path_stem.name + ".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def frac(d: dict) -> str:
        return f"{d['passed']}/{d['total']}"

    lines = [
        f"# Değerlendirme raporu — {header['chat_model']} ({header['chat_device']})", "",
        f"- Embedding modeli: `{header['embed_model']}`",
        f"- top_k: {header['top_k']}, benzerlik eşiği: {header['min_similarity']}, "
        f"BM25 ağırlığı: {header.get('bm25_weight', 0.0)}, "
        f"takip konu eşiği: {header.get('followup_min_similarity', 0.0)}",
        f"- Parça sayısı: {header['chunks']}",
        *([f"- Alt küme: {header['subset']}"] if header.get("subset") else []), "",
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
    ]
    if answers.get("by_doc"):
        lines += ["## Doküman bazında (cevaplanabilir sorular)", "",
                  "| Doküman | Doğru | Reddedilen | Doğru atıf |", "|---|---|---|---|"]
        lines += [f"| {doc} | {d['passed']}/{d['total']} | {d['refused']} | {d['cited']} |"
                  for doc, d in answers["by_doc"].items()]
        lines.append("")
    lines += [
        "## Soru bazında sonuçlar", "",
        "| # | Tür | Sonuç | Skor | Süre (sn) | Soru | Cevap |", "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r.id} | {r.type} | {'✅' if r.passed else '❌'} | {r.top_score:.2f} "
                     f"| {r.seconds:.1f} | {r.question} | {_cell(r.answer)} |")
    if followups is not None:
        lines += _followup_lines(followups, followup_rows)
    path_stem.with_name(path_stem.name + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def _followup_lines(summary: dict, rows: list[FollowupRow]) -> list[str]:
    def frac(d: dict) -> str:
        return f"{d['passed']}/{d['total']}"

    def rank(value: int | None) -> str:
        return str(value) if value else "-"

    lines = [
        "", "## Takip soruları", "",
        "İki turluk sohbetler (`data/eval/followups.json`); yalnızca ikinci tur puanlanır. "
        "\"Tek başına\" sütunları aynı sorunun önceki tur olmadan sorulduğu durumu gösterir.", "",
        f"- Genel: {frac(summary['overall'])}",
        f"- Cevaplanabilir takip sorularında doğru cevap: {frac(summary['answerable_correct'])}",
        f"- Reddedilmesi gereken takip soruları reddedildi: {frac(summary['unanswerable_refused'])}",
        f"- Konu dışı takip soruları model çağrılmadan reddedildi: {frac(summary['out_of_domain_gated'])}",
        f"- Konu taşınan soru: {summary['topic_carried']}",
        f"- Doğru doküman 1. sırada: konu taşınarak {frac(summary['doc_at_1_with_topic'])}, "
        f"tek başına {frac(summary['doc_at_1_standalone'])}",
        f"- Tek başına sorulsaydı kapıda reddedilecek cevaplanabilir soru: "
        f"{summary['answerable_gated_standalone']}", "",
        "| # | Tür | Sonuç | Taşınan konu | Sıra | Tek başına sıra | Sohbet | Cevap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        standalone = "kapıda ret" if r.standalone_gated else rank(r.standalone_doc_rank)
        lines.append(f"| {r.id} | {r.type} | {'✅' if r.passed else '❌'} | {', '.join(r.carried) or '-'} "
                     f"| {rank(r.doc_rank)} | {standalone} | {_cell(' → '.join(r.turns))} "
                     f"| {_cell(r.answer)} |")
    return lines
