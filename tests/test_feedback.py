import json

from miras import cli
from miras.feedback import TYPE_PLACEHOLDER, eval_candidates, save_vote


def _vote(store, assistant, key, thumbs, question, previous=None):
    answer = assistant.ask(question, previous)
    save_vote(store, key, thumbs, question, answer, "fake-chat")
    return answer


def test_candidates_keep_latest_downvote_per_question(store, assistant):
    first = _vote(store, assistant, "a-0", 0, "Efes nerede?")
    _vote(store, assistant, "a-1", 0, "Peki hangi ilçesinde?", first)
    _vote(store, assistant, "a-2", 1, "Göbeklitepe nerede?")
    _vote(store, assistant, "b-0", 0, "efes  nerede?")
    _vote(store, assistant, "b-1", 0, "Pizza hamuru tarifi nedir?")
    candidates = eval_candidates(store.feedback_rows())
    assert [c["question"] for c in candidates] == ["Peki hangi ilçesinde?", "efes  nerede?",
                                                   "Pizza hamuru tarifi nedir?"]
    assert [c["id"] for c in candidates] == ["fb01", "fb02", "fb03"]
    followup, efes, pizza = candidates
    assert followup["carried"] == ["Efes"] and "carried" not in efes
    assert efes["doc"] == "efes.md" and efes["type"] == TYPE_PLACEHOLDER and efes["expected"] == []
    assert pizza["refusal_reason"] == "low_similarity"


def test_feedback_command_lists_and_exports(store, assistant, settings, tmp_path, monkeypatch, capsys):
    import dataclasses

    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(cli.SETTINGS, db_path=settings.db_path))
    _vote(store, assistant, "a-0", 1, "Efes nerede?")
    _vote(store, assistant, "a-1", 0, "Pizza hamuru tarifi nedir?")
    out_file = tmp_path / "out" / "candidates.json"
    assert cli.main(["feedback", "--export", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "👍 1  👎 1" in out and "Pizza hamuru" in out and "Efes nerede?" not in out
    assert [c["question"] for c in json.loads(out_file.read_text(encoding="utf-8"))] == [
        "Pizza hamuru tarifi nedir?"]
    assert cli.main(["feedback", "--all"]) == 0
    assert "Efes nerede?" in capsys.readouterr().out


def test_refused_questions_become_eval_candidates():
    from miras.feedback import TYPE_PLACEHOLDER, refused_candidates

    def row(question, reason, created, carried=()):
        return {"question": question, "refusal_reason": reason, "created": created,
                "carried": carried, "top_score": 0.4123}

    rows = [row("Efes'e giriş ücreti?", "model", 3), row("Pizza?", "low_similarity", 2),
            row("efes'e  giriş ücreti?", "model", 1), row("Efes nerede?", None, 0),
            row("Peki ücreti?", "model", 4, ("Efes",))]
    candidates = refused_candidates(rows)
    assert [c["question"] for c in candidates] == ["Pizza?", "Efes'e giriş ücreti?", "Peki ücreti?"]
    assert candidates[0] == {"id": "lg01", "type": TYPE_PLACEHOLDER, "split": "dev", "doc": "",
                             "question": "Pizza?", "expected": [], "refusal_reason": "low_similarity",
                             "top_score": 0.412}
    assert candidates[2]["carried"] == ["Efes"]
