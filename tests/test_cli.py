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
    assert "Soru günlüğü: 0 soru, 0 reddedildi" in out
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


def test_search_notes_per_site_quota_for_comparisons(run):
    out = run("search", "Göbeklitepe mi Efes mi daha eski?")[1]
    assert out.splitlines()[0] == ("Soruda geçen alanlar: Efes, Göbeklitepe "
                                   "(sıralamada +0.1, her alandan en az bir parça)")
    assert "(efes.md)" in out and "(gobeklitepe.md)" in out


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


def test_doc_option_limits_ask_and_search(run, backend):
    import json

    backend.reply = "Efes Selçuk'tadır [K1]."
    data = json.loads(run("ask", "--json", "--doc", "Ephesos", "Efes nerede?")[1])
    assert {s["filename"] for s in data["sources"]} == {"efes.md"}
    out = run("search", "Göbeklitepe nerede?", "--doc", "efes,efes.md")[1]
    assert out.splitlines()[0] == "Yalnızca şu dokümanlarda aranıyor: efes.md"
    assert "(gobeklitepe.md)" not in out
    code, _, err = run("ask", "--doc", "Troya", "Troya nerede?")
    assert code == 1 and "'Troya' indekste yok. Dokümanlar: efes.md, gobeklitepe.md" in err


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
    {"id": "n01", "type": "near_miss", "split": "dev", "question": "Efes'e giriş ücreti ne kadar?"},
    {"id": "o02", "type": "out_of_domain", "split": "dev", "question": "Python'da liste nasıl sıralanır?"},
]
EVAL_FOLLOWUPS = [
    {"id": "f01", "type": "answerable", "doc": "efes.md",
     "turns": ["Efes hangi ilde?", "Peki hangi ilçesinde?"], "expected": [["Selçuk"]]},
    # Bilgi tabanında geçmeyen bir kelime ("ziyaretçi") taşımayı benzerlik eşiğine bırakır.
    {"id": "f02", "type": "answerable", "doc": "efes.md",
     "turns": ["Efes hangi ilde?", "Peki ilçede kaç ziyaretçi var?"], "expected": [["Selçuk"]]},
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
    assert "2 soru: 1 reddedildi, 0 önbellekten" in err
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


def test_config_shows_where_each_setting_comes_from(run, monkeypatch):
    monkeypatch.setenv("MIRAS_TOP_K", "2")
    monkeypatch.delenv("MIRAS_CHAT_DEVICE", raising=False)
    lines = {line.split()[0]: line for line in run("--model", "baska-model", "config")[1].splitlines() if line}
    assert "baska-model" in lines["chat_model"] and lines["chat_model"].endswith("komut satırı")
    assert lines["top_k"].endswith("MIRAS_TOP_K ile ezildi")
    assert lines["chat_device"].endswith("varsayılan (MIRAS_CHAT_DEVICE)")
    assert lines["temperature"].endswith("varsayılan (yalnızca config.py)")


def test_config_names_the_config_file(run, monkeypatch, tmp_path):
    from miras import config

    monkeypatch.delenv("MIRAS_BM25_WEIGHT", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "miras.toml")
    monkeypatch.setattr(config, "FILE_VALUES", {"bm25_weight": "0.3"})
    out = run("config")[1]
    lines = {line.split()[0]: line for line in out.splitlines() if line}
    assert lines["bm25_weight"].endswith("miras.toml ile ezildi")
    assert out.rstrip().endswith(f"Ayar dosyası: {tmp_path / 'miras.toml'} (okundu; başka bir dosya için MIRAS_CONFIG)")


def test_chat_kaynak_prints_full_chunks_of_last_answer(chat, backend):
    backend.reply = "Göbeklitepe 2018'de listeye girdi [K2]."
    out = chat("kaynak", "Göbeklitepe hangi yıl listeye girdi?", "kaynak", "kaynak k1", "kaynak 9", "çık")
    assert "Gösterilecek parça yok." in out
    cited = out[out.index("[K2] Göbeklitepe  (gobeklitepe.md"):out.index("[K1] ")]
    assert "https://example.org/g" in cited and "Neolitik bir arkeolojik alandır." in cited
    assert "Göbeklitepe 2018 yılında Dünya Mirası listesine girdi." in out[out.index("[K1] "):]
    assert "K1..K2 arasında bir parça yazın." in out


def test_chat_sadece_limits_and_clears_the_search(chat, backend):
    backend.reply = "Göbeklitepe 2018'de listeye girdi [K1]."
    out = chat("sadece troya", "sadece efes", "Göbeklitepe hangi yıl listeye girdi?",
               "sadece", "Göbeklitepe hangi yıl listeye girdi?", "çık")
    assert "'troya' indekste yok." in out
    assert "Yalnızca şu dokümanlarda aranıyor: efes.md" in out and "Tüm dokümanlarda aranıyor." in out
    # Kısıtlıyken Göbeklitepe sorusu kapıda kalır, kaldırınca modele gider.
    assert out.count("Bu bilgi elimdeki dokümanlarda yer almıyor.") == 1
    assert len(backend.chat_calls) == 1


def test_full_eval_runs_both_sets_without_the_answer_cache(run, eval_root, backend, store):
    import json

    backend.reply = "Efes İzmir ilindedir [K1]."
    run("ask", "Efes hangi ilde?")  # önbelleğe girer; eval yine de modeli çağırmalı
    calls = len(backend.chat_calls)
    code, out, _ = run("eval")
    assert code == 0 and "Genel: " in out and "Takip soruları: " in out
    report = json.loads((eval_root / "reports" / "eval-qwen2.5-7b-cpu.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in report["rows"]] == ["a01", "a02", "o01", "n01", "o02"]
    assert report["rows"][0]["passed"] and report["followups"]["rows"][0]["carried"] == ["Efes"]
    assert report["config"]["top_k"] == 2 and "subset" not in report["config"]
    assert len(backend.chat_calls) > calls and store.cache_size() == 1
    assert store.query_counts()[0] == 1  # eval soruları günlüğe yazılmaz


def test_calibrate_and_tune_hybrid_print_suggestions(run, eval_root, backend):
    code, out, _ = run("calibrate")
    assert code == 0 and "Önerilen eşik (dev sorularından)" in out
    assert "Önerilen konu taşıma eşiği" in out and backend.chat_calls == []
    code, out, _ = run("tune-hybrid")
    assert code == 0 and "ağırlık" in out.splitlines()[0] and backend.chat_calls == []


def test_serve_prints_address_and_stops_on_ctrl_c(run, monkeypatch):
    seen = {}

    class FakeServer:
        server_address = ("127.0.0.1", 9999)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen["closed"] = True

    def fake_make_server(assistant, host, port, **options):
        seen["args"] = (host, port, options)
        return FakeServer()

    monkeypatch.setattr(cli, "make_server", fake_make_server)
    monkeypatch.delenv("MIRAS_API_TOKEN", raising=False)
    code, out, err = run("serve", "--port", "9999")
    assert code == 0 and seen["closed"] and seen["args"][:2] == ("127.0.0.1", 9999)
    assert seen["args"][2]["token"] is None
    assert out.startswith("Miras API: http://127.0.0.1:9999") and "Uyarı" not in err
    assert "ağdaki herkes erişebilir" in run("serve", "--host", "0.0.0.0")[2]
    # Token varken dışa açık adres uyarı vermez; ortam değişkeninden de okunur.
    monkeypatch.setenv("MIRAS_API_TOKEN", "gizli")
    assert "Uyarı" not in run("serve", "--host", "0.0.0.0")[2]
    assert seen["args"][2]["token"] == "gizli"


def test_serve_reports_a_busy_port(run, monkeypatch):
    def busy(*args, **kwargs):
        raise OSError("Address already in use")

    monkeypatch.setattr(cli, "make_server", busy)
    code, _, err = run("serve")
    assert code == 1 and "127.0.0.1:8765 dinlenemedi: Address already in use" in err


def test_log_lists_questions_and_refusals(run, backend, monkeypatch):
    assert run("log")[1] == "Günlükte soru yok.\n"
    backend.reply = "Efes İzmir'dedir [K1]."
    run("ask", "Efes nerede?")
    run("ask", "--doc", "efes", "Efes nerede?")
    run("ask", "Efes nerede?")
    run("ask", "Pizza hamuru tarifi nedir?")
    lines = run("log")[1].splitlines()
    assert lines[0] == "Günlükte 4 soru, 1 tanesi reddedildi."
    assert lines[1].endswith("Pizza hamuru tarifi nedir?  (kapıda reddedildi)") and " ✗ " in lines[1]
    assert lines[2].endswith("Efes nerede?  (efes.md; önbellekten)")
    assert lines[3].endswith("Efes nerede?  (sadece: efes.md; efes.md)")
    assert lines[4].endswith("Efes nerede?  (efes.md)") and " ✓ " in lines[4]
    refused = run("log", "--refused")[1].splitlines()
    assert len(refused) == 2 and "Pizza" in refused[1]
    assert len(run("log", "-n", "1")[1].splitlines()) == 2


def test_log_export_writes_refused_candidates(run, tmp_path):
    import json

    run("ask", "Pizza hamuru tarifi nedir?")
    target = tmp_path / "adaylar" / "log.json"
    out = run("log", "--export", str(target))[1]
    assert f"1 aday -> {target}" in out
    assert json.loads(target.read_text(encoding="utf-8"))[0]["question"] == "Pizza hamuru tarifi nedir?"


def test_log_clear_empties_the_log(run, store):
    run("ask", "Pizza hamuru tarifi nedir?")
    assert run("log", "--clear")[1] == "Günlükten 1 soru silindi.\n"
    assert store.query_counts() == (0, 0)


def test_doctor_prints_checks_and_fails_on_errors(run, monkeypatch):
    from miras import doctor

    monkeypatch.setattr(doctor, "check_sdk", lambda: doctor.Check("ok", "foundry-local-sdk", "kurulu"))
    monkeypatch.setattr(doctor, "check_models", lambda: doctor.Check("warn", "modeller", "indirilmemiş"))
    code, out, _ = run("doctor")
    assert code == 0 and "✓ indeks" in out and "! modeller" in out
    assert out.rstrip().endswith("Hata yok, 1 uyarı.")
    monkeypatch.setattr(doctor, "check_sdk", lambda: doctor.Check("error", "foundry-local-sdk", "yok"))
    code, out, err = run("doctor")
    assert code == 1 and "✗ foundry-local-sdk" in out and "1 hata, 1 uyarı." in err


def test_eval_retrieval_only_never_calls_the_chat_model(run, eval_root, backend, settings):
    import json

    code, out, _ = run("eval", "--retrieval-only")
    assert code == 0 and backend.chat_calls == []
    assert out.startswith("Doğru doküman 1. sırada: ") and "Takip sorularında konu taşındı: " in out
    report = json.loads((eval_root / "reports" / f"arama-{backend.embed_model}.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in report["rows"]] == ["a01", "a02", "o01", "n01", "o02"]
    assert not (eval_root / "reports" / "eval-qwen2.5-7b-cpu.json").exists()
    run("eval", "--retrieval-only", "--split", "dev")
    assert (eval_root / "reports" / f"arama-{backend.embed_model}-dev.md").exists()


def test_search_json_matches_the_api_shape(run, backend):
    import json

    data = json.loads(run("search", "--json", "Göbeklitepe mi Efes mi daha eski?")[1])
    assert data["mentioned"] == ["Efes", "Göbeklitepe"] and data["passes_threshold"] is True
    assert data["hits"][0]["label"] == "K1" and "text" in data["hits"][0]
    assert backend.chat_calls == []


def test_docs_json(run):
    import json

    data = json.loads(run("docs", "--json")[1])
    assert [d["filename"] for d in data["docs"]] == ["efes.md", "gobeklitepe.md"]
    assert data["docs"][0]["aliases"] == ["Ephesos", "Ephesus"] and data["docs"][1]["chunks"] == 2


def test_log_json(run):
    import json

    run("ask", "Pizza hamuru tarifi nedir?")
    data = json.loads(run("log", "--json")[1])
    assert data["total"] == 1 and data["refused"] == 1
    row = data["rows"][0]
    assert row["question"] == "Pizza hamuru tarifi nedir?" and row["refusal_reason"] == "low_similarity"
    assert row["created"][4] == "-" and row["cited"] == [] and row["cached"] is False


def test_status_json(run, settings):
    import json

    (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    data = json.loads(run("status", "--json")[1])
    assert (data["docs"], data["chunks"], data["rebuild_needed"]) == (2, 3, False)
    assert data["pending"] == {"changed": ["ani.md"], "missing": []}
    assert data["query_log"] == {"enabled": True, "total": 0, "refused": 0}


def test_doctor_json_keeps_the_exit_code(run, monkeypatch, capsys):
    import json

    from miras import doctor

    monkeypatch.setattr(doctor, "check_sdk", lambda: doctor.Check("error", "foundry-local-sdk", "yok"))
    with pytest.raises(SystemExit) as exit_info:
        run("doctor", "--json")
    assert exit_info.value.code == 1 and json.loads(capsys.readouterr().out)["ok"] is False
    monkeypatch.setattr(doctor, "check_sdk", lambda: doctor.Check("ok", "foundry-local-sdk", "kurulu"))
    monkeypatch.setattr(doctor, "check_models", lambda: doctor.Check("ok", "modeller", "var"))
    code, out, _ = run("doctor", "--json")
    data = json.loads(out)
    assert code == 0 and data["ok"] is True
    assert {"level": "ok", "name": "foundry-local-sdk", "message": "kurulu"} in data["checks"]


def test_config_json(run):
    import json

    data = json.loads(run("--model", "phi-4-mini", "config", "--json")[1])
    assert data["chat_model"] == {"value": "phi-4-mini", "origin": "komut satırı"}
    assert data["top_k"]["value"] == 2 and isinstance(data["db_path"]["value"], str)


def test_feedback_json(run, store):
    import json

    store.save_feedback("k1", -1, "Efes nerede?", ("Efes",), "Bilmiyorum.", "model", [], "m")
    store.save_feedback("k2", 1, "Ani nerede?", (), "Kars [K1].", None, [], "m")
    data = json.loads(run("feedback", "--json")[1])
    assert (data["up"], data["down"]) == (1, 1) and [r["key"] for r in data["rows"]] == ["k1"]
    assert data["rows"][0]["carried"] == ["Efes"]
    assert len(json.loads(run("feedback", "--json", "--all")[1])["rows"]) == 2


def test_chat_gecmis_lists_this_sessions_questions(chat, backend):
    backend.reply = "Efes İzmir'dedir [K1]."
    out = chat("geçmiş", "Efes nerede?", "Orası ne zaman kuruldu?", "Pizza hamuru tarifi nedir?", "gecmis", "çık")
    assert "Henüz soru yok." in out
    assert " 1. ✓ Efes nerede?\n 2. ✓ Orası ne zaman kuruldu?  (konu: Efes)\n" in out
    assert " 3. ✗ Pizza hamuru tarifi nedir?" in out


def test_remove_doc_takes_several_files_and_checks_them_first(run, settings, backend, store):
    (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    run("ingest")
    code, _, err = run("remove-doc", "efes.md", "yok.md")
    assert code == 1 and "yok.md ne indekste" in err and (settings.docs_dir / "efes.md").exists()
    assert run("remove-doc", "efes.md", "ani.md", "gobeklitepe.md")[0] == 1  # hepsi silinemez
    code, out, _ = run("remove-doc", "efes.md", "ani.md")
    assert code == 0 and out == "Silindi: efes.md, ani.md. Veritabanı: 1 doküman, 2 parça\n"


def test_ask_reads_the_question_from_stdin(run, backend, monkeypatch):
    import io
    import json

    backend.reply = "Efes İzmir'dedir [K1]."
    monkeypatch.setattr("sys.stdin", io.StringIO("Efes\nnerede?\n"))
    data = json.loads(run("ask", "--json", "-")[1])
    assert data["answer"] == backend.reply and "SORU: Efes nerede?" in backend.chat_calls[0][1]["content"]
    assert data["question"] == "Efes nerede?"
    monkeypatch.setattr("sys.stdin", io.StringIO("  \n"))
    code, _, err = run("ask", "-")
    assert code == 1 and "Standart girdide soru yok." in err


def test_log_since_filters_by_date(run, store):
    import json

    run("ask", "Pizza hamuru tarifi nedir?")
    store.conn.execute("UPDATE queries SET created = 0")  # 1970'e çek
    store.conn.commit()
    run("ask", "Efes nerede?")
    data = json.loads(run("log", "--json", "--since", "2000-01-01")[1])
    assert [r["question"] for r in data["rows"]] == ["Efes nerede?"] and data["total"] == 2
    code, _, err = run("log", "--since", "dün")
    assert code == 1 and "--since tarih olmalı" in err


def test_ingest_watch_reindexes_changes_until_ctrl_c(run, settings, store, monkeypatch):
    rounds = []

    def fake_sleep(seconds):
        rounds.append(seconds)  # 1. tur: değişiklik yok, ingest çalışmamalı
        if len(rounds) == 2:
            (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
        elif len(rounds) == 3:
            (settings.docs_dir / "efes.md").unlink()
        elif len(rounds) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "WATCH_SLEEP", fake_sleep)
    code, out, _ = run("ingest", "--watch", "--interval", "0.5")
    assert code == 0 and rounds == [0.5] * 4
    assert "izleniyor (0.5 sn arayla)" in out and out.count("klasörde değişiklik var") == 2
    assert {d["filename"] for d in store.documents()} == {"ani.md", "gobeklitepe.md"}
