from miras.evaluation import (RetrievalRow, calibrate_threshold, evaluate_answers, matches_expected,
                              normalize, retrieval_summary)


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
