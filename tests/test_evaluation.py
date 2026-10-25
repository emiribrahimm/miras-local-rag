import json

from miras.config import ROOT
from miras.evaluation import (RetrievalRow, answer_summary, calibrate_followup_floor, calibrate_threshold,
                              choose_keyword_weight, followup_floor_scores,
                              evaluate_answers, evaluate_followups, followup_summary, matches_expected, normalize,
                              retrieval_summary, tune_keyword_weight, write_report)
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


def test_follow_up_floor_separates_follow_ups_from_off_topic():
    result = calibrate_followup_floor(keep=[0.55, 0.62, 0.70], block=[0.31, 0.44])
    assert result["best_range"] == [0.45, 0.55] and result["floor"] == 0.5
    assert result["balanced_accuracy"] == 1.0
    assert result["lowest_followup"] == 0.55 and result["highest_off_topic"] == 0.44


def test_follow_up_floor_scores_use_standalone_questions(store, backend):
    from miras.retrieval import Retriever
    followups = [
        {"id": "f1", "type": "answerable", "turns": ["Göbeklitepe nerede?", "Peki ne zaman listeye girdi?"]},
        {"id": "f2", "type": "answerable", "switch": True, "turns": ["Göbeklitepe nerede?", "Efes nerede?"]},
        {"id": "f3", "type": "out_of_domain", "turns": ["Göbeklitepe nerede?", "Pizza hamuru tarifi nedir?"]},
        {"id": "f4", "type": "answerable", "turns": ["Göbeklitepe nerede?", "Orası ne zaman listeye girdi?"]},
    ]
    questions = [{"id": "o1", "type": "out_of_domain", "split": "dev", "question": "Mars'ın kaç uydusu var?"},
                 {"id": "a1", "type": "answerable", "split": "dev", "question": "Efes nerede?"}]
    keep, block = followup_floor_scores(followups, questions, backend, Retriever(store, backend.embed_model))
    assert len(keep) == 1 and keep[0] > 0  # konu değiştiren f2 ve gönderme kelimeli f4 sayılmaz
    assert block == [0.0, 0.0]


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
        {"id": "f3", "type": "out_of_domain", "turns": ["Göbeklitepe hangi ilde?", "Pizza hamuru tarifi nedir?"]},
    ]
    rows = evaluate_followups(items, assistant)
    assert rows[0].carried == ["Göbeklitepe"] and rows[0].passed and rows[0].cited_expected_doc
    assert rows[0].doc_rank == 1 and rows[0].standalone_doc_rank is not None
    assert rows[1].carried == [] and not rows[1].passed and rows[1].cited_expected_doc  # cevap yanlış, atıf doğru
    assert rows[2].carried == [] and rows[2].refusal_reason == "low_similarity"
    # İlk iki sohbette iki model çağrısı (ilk tur + takip), konu dışı takip kapıda kalır;
    # bağlamsız arama modeli çağırmaz.
    assert len(backend.chat_calls) == 5

    summary = followup_summary(rows)
    assert summary["overall"] == {"passed": 2, "total": 3} and summary["topic_carried"] == 1
    assert summary["doc_at_1_with_topic"]["total"] == 2
    assert summary["out_of_domain_gated"] == {"passed": 1, "total": 1}


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


def test_report_has_follow_up_section_only_when_given(assistant, backend, tmp_path):
    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    rows = evaluate_answers([{"id": "1", "type": "answerable", "split": "dev", "doc": "gobeklitepe.md",
                              "question": "Göbeklitepe ne zaman?", "expected": [["2018"]]}], assistant)
    followup_rows = evaluate_followups([{"id": "f1", "type": "answerable", "doc": "gobeklitepe.md",
                                         "turns": ["Göbeklitepe nerede?", "Peki | ne zaman listeye girdi?"],
                                         "expected": [["2018"]]}], assistant)
    header = {"chat_model": "m", "chat_device": "cpu", "embed_model": "e", "top_k": 2,
              "min_similarity": 0.2, "chunks": 3}
    retrieval = retrieval_summary([RetrievalRow("1", "answerable", "dev", 0.9, 1, True)])

    write_report(tmp_path / "plain", header, retrieval, answer_summary(rows), rows)
    assert "Takip soruları" not in (tmp_path / "plain.md").read_text(encoding="utf-8")
    assert "followups" not in json.loads((tmp_path / "plain.json").read_text(encoding="utf-8"))

    write_report(tmp_path / "full", header, retrieval, answer_summary(rows), rows,
                 followup_summary(followup_rows), followup_rows)
    md = (tmp_path / "full.md").read_text(encoding="utf-8")
    assert "## Takip soruları" in md and "| f1 | answerable | ✅ | Göbeklitepe | 1 |" in md
    assert "Peki \\| ne zaman" in md  # tablo hücresindeki | kaçırılır
    data = json.loads((tmp_path / "full.json").read_text(encoding="utf-8"))
    assert data["followups"]["rows"][0]["carried"] == ["Göbeklitepe"]


def test_keyword_weight_choice_prefers_dev_mrr_then_smaller_weight():
    def result(weight, mrr, context=1.0):
        return {"weight": weight, "dev": {"mrr": mrr, "answer_in_context_at_k": context}}

    assert choose_keyword_weight([result(0.0, 0.90), result(0.1, 0.95), result(0.2, 0.93)]) == 0.1
    assert choose_keyword_weight([result(0.0, 0.95), result(0.1, 0.95)]) == 0.0
    assert choose_keyword_weight([result(0.0, 0.95, 0.9), result(0.1, 0.95, 1.0)]) == 0.1


def test_tune_embeds_each_question_once(store, backend, settings):
    from miras.retrieval import Retriever
    calls = []
    embed = backend.embed_query
    backend.embed_query = lambda text: calls.append(text) or embed(text)
    questions = [
        {"id": "1", "type": "answerable", "split": "dev", "doc": "gobeklitepe.md",
         "question": "Göbeklitepe 2018", "expected": [["2018"]]},
        {"id": "2", "type": "answerable", "split": "test", "doc": "efes.md",
         "question": "Efes nerede?", "expected": [["izmir"]]},
    ]
    results = tune_keyword_weight(questions, backend, Retriever(store, backend.embed_model),
                                  2, settings.min_similarity, weights=(0.0, 0.5))
    assert len(calls) == 2 and [r["weight"] for r in results] == [0.0, 0.5]
    assert results[0]["dev"]["questions"] == 1 and "answerable_gated" in results[1]["test"]
