"""Komut satırı komutları, gerçek model yerine FakeBackend ile."""

import dataclasses

import pytest

from miras import cli


@pytest.fixture
def run(store, backend, settings, monkeypatch, capsys):
    # İndeks FakeBackend ile kuruldu; ayarlar da onu göstersin ki `status` uyarı vermesin.
    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(settings, embed_model=backend.embed_model))
    monkeypatch.setattr(cli, "_backend", lambda _settings: backend)

    def run(*argv: str) -> tuple[int, str, str]:
        code = cli.main(list(argv))
        out, err = capsys.readouterr()
        return code, out, err

    return run


def test_cache_command_counts_and_clears(run, store):
    store.cache_answer("a", "Cevap [K1].")
    store.cache_answer("b", "Başka [K1].")
    assert run("cache")[1].startswith("Önbellekte 2 cevap var")
    assert run("cache", "--clear")[1] == "Önbellekten 2 cevap silindi.\n"
    assert store.cache_size() == 0


def test_status_reports_index_and_pending_changes(run, settings):
    code, out, _ = run("status")
    assert code == 0
    assert "İndeks: 2 doküman, 3 parça (fake-hash-embedding)" in out
    assert "ile indeks aynı" in out
    (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    (settings.docs_dir / "efes.md").unlink()
    out = run("status")[1]
    assert "  + ani.md" in out and "  - efes.md (klasörde yok)" in out


def test_status_warns_when_embedding_model_changed(run, settings, monkeypatch):
    monkeypatch.setattr(cli, "SETTINGS", dataclasses.replace(settings, embed_model="baska"))
    assert "`ingest --rebuild` gerekli" in run("status")[1]


def test_docs_command_lists_indexed_documents(run):
    out = run("docs")[1]
    assert out.splitlines()[:3] == ["Efes  (efes.md, 1 parça)", "     takma adlar: Ephesos, Ephesus",
                                    "Göbeklitepe  (gobeklitepe.md, 2 parça)"]
    assert "     https://example.org/g" in out and out.endswith("2 doküman, 3 parça\n")


def test_remove_doc_deletes_file_and_chunks(run, store, settings):
    code, out, _ = run("remove-doc", "efes.md")
    assert code == 0 and out == "Silindi: efes.md. Veritabanı: 1 doküman, 2 parça\n"
    assert not (settings.docs_dir / "efes.md").exists()


@pytest.mark.parametrize("name, message", [
    ("yok.md", "ne indekste"),
    ("../efes.md", "ne indekste"),
])
def test_remove_doc_rejects_unknown_files(run, name, message):
    code, _, err = run("remove-doc", name)
    assert code == 1 and message in err


def test_last_doc_cannot_be_removed(run, settings):
    assert run("remove-doc", "efes.md")[0] == 0
    code, _, err = run("remove-doc", "gobeklitepe.md")
    assert code == 1 and "En az bir doküman kalmalı." in err
    assert (settings.docs_dir / "gobeklitepe.md").exists()


def test_search_shows_chunks_and_gate_without_chat_model(run, backend):
    code, out, _ = run("search", "Göbeklitepe hangi yıl listeye girdi?", "-k", "1")
    assert code == 0 and backend.chat_calls == []
    lines = out.splitlines()
    assert lines[0] == "Soruda geçen alanlar: Göbeklitepe (sıralamada +0.1)"
    assert lines[1].startswith("[K1] ") and "Göbeklitepe › Koruma  (gobeklitepe.md)" in lines[1]
    assert "[K2]" not in out and "bu parçalar modele gider" in out


def test_search_reports_refusal_below_threshold(run):
    out = run("search", "Pizza hamuru tarifi nedir?")[1]
    assert "Soruda geçen" not in out and "model çağrılmaz, soru reddedilir" in out


def test_search_rejects_bad_k(run):
    code, _, err = run("search", "Efes nerede?", "-k", "0")
    assert code == 1 and "top_k en az 1" in err


@pytest.fixture
def chat(run, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "CHAT_HISTORY", tmp_path / "gecmis" / "chat_history")

    def chat(*lines: str) -> str:
        answers = iter(lines)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        return run("chat")[1]

    return chat


def test_chat_answers_until_quit_and_saves_history(chat, backend, tmp_path):
    backend.reply = "Efes İzmir'dedir [K1]."
    out = chat("Efes nerede?", "çık")
    assert "Efes İzmir'dedir [K1]." in out and len(backend.chat_calls) == 1
    pytest.importorskip("readline")
    assert (tmp_path / "gecmis" / "chat_history").exists()


def test_chat_saves_conversation_as_markdown(chat, backend, tmp_path):
    backend.reply = "Efes İzmir'dedir [K1]."
    target = tmp_path / "sohbet" / "efes.md"
    out = chat("kaydet", "Efes nerede?", f"kaydet {target}", "q")
    assert "Henüz kaydedilecek soru yok." in out and f"1 soru -> {target}" in out
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Miras sohbeti") and "> Efes nerede?" in text
    assert "Efes İzmir'dedir [K1]." in text and "[K1] Efes (efes.md" in text


def test_ask_json_has_answer_sources_and_support(run, backend):
    import json

    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    code, out, _ = run("ask", "--json", "Göbeklitepe hangi yıl listeye girdi?")
    data = json.loads(out)
    assert code == 0 and data["answer"] == backend.reply and data["refused"] is False
    first = data["sources"][0]
    assert first["label"] == "K1" and first["filename"] == "gobeklitepe.md" and first["cited"] is True
    assert first["support"] == ["Göbeklitepe 2018 yılında Dünya Mirası listesine girdi."]
    assert first["source_url"] == "https://example.org/g"


def test_ask_json_reports_refusal(run):
    import json

    data = json.loads(run("ask", "--json", "Pizza hamuru tarifi nedir?")[1])
    assert data["refused"] is True and data["refusal_reason"] == "low_similarity"


def test_ask_refresh_calls_model_again(run, backend):
    backend.reply = "Efes Ankara'dadır [K1]."
    run("ask", "Efes nerede?")
    backend.reply = "Efes İzmir'dedir [K1]."
    assert "Ankara" in run("ask", "Efes nerede?")[1]
    out = run("ask", "--refresh", "Efes nerede?")[1]
    assert "İzmir" in out and len(backend.chat_calls) == 2
    assert "İzmir" in run("ask", "Efes nerede?")[1]


EVAL_QUESTIONS = [
    {"id": "a01", "type": "answerable", "split": "dev", "doc": "efes.md",
     "question": "Efes hangi ilde?", "expected": [["İzmir"]]},
    {"id": "a02", "type": "answerable", "split": "test", "doc": "gobeklitepe.md",
     "question": "Göbeklitepe hangi yıl listeye girdi?", "expected": [["2018"]]},
    {"id": "o01", "type": "out_of_domain", "split": "test",
     "question": "Pizza hamuru tarifi nedir?"},
]
EVAL_FOLLOWUPS = [
    {"id": "f01", "type": "answerable", "doc": "efes.md",
     "turns": ["Efes hangi ilde?", "Peki hangi ilçede?"], "expected": [["Selçuk"]]},
]


@pytest.fixture
def eval_root(run, monkeypatch, tmp_path):
    import json

    root = tmp_path / "proje"
    (root / "data" / "eval").mkdir(parents=True)
    (root / "data" / "eval" / "questions.json").write_text(json.dumps(EVAL_QUESTIONS), encoding="utf-8")
    (root / "data" / "eval" / "followups.json").write_text(json.dumps(EVAL_FOLLOWUPS), encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", root)
    return root


def test_eval_split_runs_one_half_and_keeps_full_report(run, eval_root, backend):
    import json

    backend.reply = "Efes İzmir ilindedir [K1]."
    code, out, _ = run("eval", "--split", "test")
    assert code == 0 and "a01" not in out and "a02" in out and "Takip soruları" not in out
    reports = eval_root / "reports"
    report = json.loads((reports / "eval-qwen2.5-7b-cpu-test.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in report["rows"]] == ["a02", "o01"] and "followups" not in report
    assert "- Alt küme: yalnızca test soruları" in (reports / "eval-qwen2.5-7b-cpu-test.md").read_text(
        encoding="utf-8")
    assert not (reports / "eval-qwen2.5-7b-cpu.json").exists()


def test_eval_ids_picks_questions_from_both_sets(run, eval_root):
    import json

    code, out, _ = run("eval", "--ids", "a01,f01")
    assert code == 0 and "a01" in out and "f01" in out and "a02" not in out
    report = json.loads((eval_root / "reports" / "eval-qwen2.5-7b-cpu-secili.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in report["rows"]] == ["a01"]
    assert [r["id"] for r in report["followups"]["rows"]] == ["f01"]
    assert report["config"]["subset"] == "seçili sorular (a01, f01)"


def test_eval_ids_only_follow_up_and_unknown_ids(run, eval_root):
    assert run("eval", "--ids", "f01")[0] == 0
    code, _, err = run("eval", "--ids", "a01,zz9")
    assert code == 1 and "Bilinmeyen soru: zz9" in err


def test_compare_reports_command(run, tmp_path):
    import json

    def report(passed: bool, weight: float) -> dict:
        return {"config": {"bm25_weight": weight}, "answers": {"overall": {"passed": int(passed), "total": 1}},
                "rows": [{"id": "a01", "question": "Efes nerede?", "answer": "İzmir", "passed": passed}]}

    (tmp_path / "eski.json").write_text(json.dumps(report(False, 0.0)), encoding="utf-8")
    (tmp_path / "yeni.json").write_text(json.dumps(report(True, 0.1)), encoding="utf-8")
    code, out, _ = run("compare-reports", str(tmp_path / "eski.json"), str(tmp_path / "yeni.json"))
    assert code == 0 and "bm25_weight: 0.0 -> 0.1" in out and "Genel: 0/1 -> 1/1" in out
    assert "Düzelen (1):\n  a01 Efes nerede?" in out and "Takip soruları" not in out
    code, _, err = run("compare-reports", str(tmp_path / "yok.json"), str(tmp_path / "yeni.json"))
    assert code == 1 and "Rapor okunamadı" in err


def test_ask_batch_writes_one_json_line_per_question(run, backend, tmp_path):
    import json

    backend.reply = "Efes İzmir'dedir [K1]."
    questions = tmp_path / "sorular.txt"
    questions.write_text("# deneme\nEfes nerede?\n\n  Pizza hamuru tarifi nedir?\n", encoding="utf-8")
    code, out, err = run("ask-batch", str(questions), "--out", str(tmp_path / "cevaplar.jsonl"))
    assert code == 0 and "2 cevap ->" in out and "2/2 Pizza" in err
    lines = (tmp_path / "cevaplar.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["question"] for r in rows] == ["Efes nerede?", "Pizza hamuru tarifi nedir?"]
    assert rows[0]["answer"] == "Efes İzmir'dedir [K1]." and rows[1]["refused"] is True


def test_ask_batch_reads_stdin_and_rejects_empty_files(run, monkeypatch, tmp_path):
    import io
    import json

    monkeypatch.setattr("sys.stdin", io.StringIO("Efes nerede?\n"))
    code, out, _ = run("ask-batch", "-")
    assert code == 0 and json.loads(out)["question"] == "Efes nerede?"
    (tmp_path / "bos.txt").write_text("# sadece yorum\n", encoding="utf-8")
    code, _, err = run("ask-batch", str(tmp_path / "bos.txt"))
    assert code == 1 and "Dosyada soru yok" in err
