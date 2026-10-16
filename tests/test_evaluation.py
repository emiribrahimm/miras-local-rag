import json

from miras.config import ROOT
from miras.evaluation import (RetrievalRow, calibrate_threshold, evaluate_answers, evaluate_followups,
                              followup_summary, matches_expected, normalize, retrieval_summary)
from miras.retrieval import title_keys


def test_turkish_case_folding():
    assert normalize("İZMİR IĞDIR") == "izmir ığdır"
    assert matches_expected("Efes İZMİR'dedir, 25.000 kişilik.", [["izmir"], ["25000", "25.000"]])
    assert not matches_expected("Efes İzmir'dedir.", [["izmir"], ["25.000"]])


def test_retrieval_summary_counts_only_answerable():
    rows = [RetrievalRow("a", "answerable", "dev", 0.9, 1, True),
            RetrievalRow("b", "answerable", "dev", 0.8, 2, False),
            RetrievalRow("c", "out_of_domain", "dev", 0.2, None, False)]
    summary = retrieval_summary(rows)
    assert summary["doc_hit_at_1"] == 0.5 and summary["doc_hit_at_k"] == 1.0
    assert summary["mrr"] == 0.75 and summary["answer_in_context_at_k"] == 0.5


def test_threshold_is_chosen_from_dev_and_checked_on_test():
    def row(kind, split, score):
        return RetrievalRow("x", kind, split, score, None, False)

    rows = [row("answerable", "dev", 0.70), row("out_of_domain", "dev", 0.30),
            row("answerable", "test", 0.45), row("out_of_domain", "test", 0.35),
            row("near_miss", "dev", 0.60), row("near_miss", "test", 0.40)]
    result = calibrate_threshold(rows)
    assert result["best_range"] == [0.31, 0.70] and result["threshold"] == 0.51
    assert result["dev_balanced_accuracy"] == 1.0
    assert result["test_balanced_accuracy"] == 0.5   # test sorusu 0.45 eşiğin altında kalıyor
    assert result["near_miss_blocked_by_gate"] == 0.5


def test_answer_scoring(assistant, backend):
    backend.reply = "Göbeklitepe Şanlıurfa'dadır [K1]."
    questions = [
        {"id": "1", "type": "answerable", "split": "dev", "doc": "gobeklitepe.md",
         "question": "Göbeklitepe hangi ilde?", "expected": [["şanlıurfa"]]},
        {"id": "2", "type": "answerable", "split": "dev", "doc": "gobeklitepe.md",
         "question": "Göbeklitepe hangi yıl listeye girdi?", "expected": [["2018"]]},
        {"id": "3", "type": "out_of_domain", "split": "test", "question": "Pizza tarifi?"},
    ]
    rows = evaluate_answers(questions, assistant)
    assert [r.passed for r in rows] == [True, False, True]
    assert rows[0].cited_expected_doc is True and rows[2].refusal_reason == "low_similarity"


def test_follow_up_scoring(assistant, backend):
    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    items = [
        {"id": "f1", "type": "answerable", "doc": "gobeklitepe.md",
         "turns": ["Göbeklitepe hangi ilde?", "Peki ne zaman listeye girdi?"], "expected": [["2018"]]},
        {"id": "f2", "type": "answerable", "doc": "efes.md", "switch": True,
         "turns": ["Göbeklitepe hangi ilde?", "Efes hangi ilde?"], "expected": [["izmir"]]},
    ]
    rows = evaluate_followups(items, assistant)
    assert rows[0].carried == ["Göbeklitepe"] and rows[0].passed and rows[0].cited_expected_doc
    assert rows[0].doc_rank == 1 and rows[0].standalone_doc_rank is not None
    assert rows[1].carried == [] and not rows[1].passed and rows[1].cited_expected_doc  # cevap yanlış, atıf doğru
    # Her sohbet için iki model çağrısı (ilk tur + takip); bağlamsız arama modeli çağırmaz.
    assert len(backend.chat_calls) == 4

    summary = followup_summary(rows)
    assert summary["overall"] == {"passed": 1, "total": 2} and summary["topic_carried"] == 1
    assert summary["doc_at_1_with_topic"]["total"] == 2


def test_follow_up_set_is_well_formed():
    items = json.loads((ROOT / "data" / "eval" / "followups.json").read_text(encoding="utf-8"))
    docs = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "data" / "docs").glob("*.md")}
    keys = set()
    for text in docs.values():
        meta = dict(line.split(": ", 1) for line in text.split("---")[1].strip().splitlines())
        keys |= title_keys(meta["title"], tuple(a.strip() for a in meta.get("aliases", "").split(",") if a.strip()))
    assert len({q["id"] for q in items}) == len(items)
    for q in items:
        assert q["type"] in {"answerable", "near_miss", "out_of_domain"} and len(q["turns"]) == 2
        if q["type"] == "answerable":
            assert q["doc"] in docs and q["expected"]
        # Takip sorusu bir alanı adıyla anıyorsa konu taşınmaz; bunlar "switch" olarak işaretli.
        names_site = any(normalize(k) in normalize(q["turns"][1]) for k in keys)
        assert names_site == q.get("switch", False), q["id"]
