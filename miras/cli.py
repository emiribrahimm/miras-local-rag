"""Komut satırı arayüzü: ingest, add-wiki, update-wiki, ask, ask-batch, search, chat, eval,
calibrate, feedback, log, status, config, docs, remove-doc, cache, compare-reports, serve."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import evaluation
from .backends import BackendError, FoundryBackend
from .config import ENV_FIELDS, ROOT, SETTINGS, Settings, env_name
from .export import answer_json, conversation_markdown
from .feedback import eval_candidates, refused_candidates
from .ingest import ingest, pending_changes, remove_document, save_uploads
from .pipeline import Answer, Assistant
from .retrieval import TITLE_BOOST, IndexError_, Retriever
from .server import make_server
from .store import Store
from .wiki import WikiError, article, outdated, refreshed


# Sohbette yukarı okla önceki sorulara dönülebilsin; modellerle aynı klasörde durur.
CHAT_HISTORY = Path.home() / ".miras-rag" / "chat_history"


class CommandError(RuntimeError):
    """Kullanıcının düzeltebileceği komut hatası; iz yerine tek satır mesaj basılır."""


def _status(message: str) -> None:
    print(f"  · {message}", file=sys.stderr, flush=True)


def _backend(settings: Settings) -> FoundryBackend:
    return FoundryBackend(settings.chat_model, settings.embed_model, settings.chat_device,
                          on_status=_status)


def _assistant(settings: Settings, cache: bool = True) -> Assistant:
    backend = _backend(settings)
    store = Store(settings.db_path)
    return Assistant(backend, Retriever(store, backend.embed_model), settings,
                     cache=store if cache else None)


def _docs(assistant: Assistant, values: list[str] | None) -> frozenset[str] | None:
    """`--doc efes.md --doc Troya` ya da `--doc efes,troya` -> dosya adları; verilmezse None."""
    names = [name.strip() for value in values or [] for name in value.split(",") if name.strip()]
    if not names:
        return None
    try:
        return assistant.retriever.resolve_docs(names)
    except ValueError as err:
        raise CommandError(str(err)) from err


def _print_sources(answer: Answer, verbose: bool) -> None:
    if answer.refused:
        return
    # Model etiket yazmadıysa kullanıcı yine de hangi parçalara bakıldığını görsün.
    shown = answer.cited or list(range(len(answer.hits)))
    print("\nKaynaklar:" if answer.cited else "\nİncelenen parçalar (model atıf yapmadı):")
    for i in shown:
        hit = answer.hits[i]
        print(f"  [K{i + 1}] {hit.chunk.section}  ({hit.chunk.filename}, benzerlik {hit.score:.2f})")
        for sentence in answer.support.get(i, []):
            print(f"       » {sentence}")
        if verbose:
            print(f"       {hit.chunk.text[:300]}...")


def _answer_streaming(assistant: Assistant, question: str, verbose: bool,
                      previous: Answer | None = None, refresh: bool = False,
                      docs: frozenset[str] | None = None) -> Answer:
    prepared = assistant.prepare(question, previous, docs)
    if prepared.carried:
        _status(f"konu: {', '.join(prepared.carried)}")
    started = time.perf_counter()
    pieces = []
    for piece in assistant.generate(prepared, refresh):
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    answer = assistant.finalize(prepared, "".join(pieces), time.perf_counter() - started)
    _print_sources(answer, verbose)
    if verbose:
        generation = "önbellekten" if answer.cached else f"üretim {answer.generation_seconds:.2f} sn"
        print(f"\n(arama {answer.retrieval_seconds:.2f} sn, {generation})")
    return answer


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    store = Store(settings.db_path)
    report = ingest(store, _backend(settings), settings, rebuild=args.rebuild, on_progress=_status)
    docs, chunks = store.counts()
    print(f"Eklenen/güncellenen: {len(report.added)}, değişmeyen: {len(report.unchanged)}, "
          f"silinen: {len(report.removed)}")
    for message in report.skipped:
        print(f"Uyarı: {message}", file=sys.stderr)
    print(f"Veritabanı: {docs} doküman, {chunks} parça -> {settings.db_path}")


def cmd_add_wiki(args: argparse.Namespace, settings: Settings) -> None:
    aliases = tuple(a.strip() for a in (args.aliases or "").split(",") if a.strip())
    _status("Vikipedi'den indiriliyor...")
    filename, text = article(args.article, aliases, docs_dir=settings.docs_dir)
    _, errors = save_uploads(settings.docs_dir, [(filename, text.encode("utf-8"))])
    if errors:
        raise WikiError(errors[0])
    print(f"{filename} -> {settings.docs_dir}")
    cmd_ingest(argparse.Namespace(rebuild=False), settings)


def cmd_update_wiki(args: argparse.Namespace, settings: Settings) -> None:
    _status("Vikipedi'deki revizyonlar denetleniyor...")
    updates = outdated(settings.docs_dir)
    if not updates:
        print("Tüm maddeler güncel.")
        return
    for u in updates:
        latest = "Vikipedi'de bulunamadı, dokunulmadı" if u.latest is None else u.latest
        print(f"  {u.doc.filename:<24} {u.doc.revision} -> {latest}")
    if args.check or all(u.latest is None for u in updates):
        return
    _status("Değişen maddeler indiriliyor...")
    _, errors = save_uploads(settings.docs_dir, [(name, text.encode("utf-8"))
                                                 for name, text in refreshed(updates, settings.docs_dir)])
    if errors:
        raise WikiError(errors[0])
    cmd_ingest(argparse.Namespace(rebuild=False), settings)


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    assistant = _assistant(settings)
    docs = _docs(assistant, args.doc)
    if args.json:
        answer = assistant.ask(args.question, refresh=args.refresh, docs=docs)
        print(json.dumps(answer_json(args.question, answer, settings), ensure_ascii=False, indent=2))
        return
    _answer_streaming(assistant, args.question, args.verbose, refresh=args.refresh, docs=docs)


def cmd_ask_batch(args: argparse.Namespace, settings: Settings) -> None:
    """Dosyadaki her satırı ayrı bir soru olarak sorar; her cevap bir JSON satırı (JSONL)."""
    try:
        text = sys.stdin.read() if str(args.file) == "-" else args.file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        raise CommandError(f"Soru dosyası okunamadı: {err}") from err
    questions = [line.strip() for line in text.splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not questions:
        raise CommandError("Dosyada soru yok (boş ve # ile başlayan satırlar atlanır).")
    assistant = _assistant(settings)
    docs = _docs(assistant, args.doc)
    out = args.out.open("w", encoding="utf-8") if args.out else sys.stdout
    try:
        for number, question in enumerate(questions, start=1):
            _status(f"{number}/{len(questions)} {question[:60]}")
            # Sorular birbirinden bağımsız; önceki satırın konusu taşınmaz.
            answer = assistant.ask(question, refresh=args.refresh, docs=docs)
            print(json.dumps(answer_json(question, answer, settings), ensure_ascii=False),
                  file=out, flush=True)
    finally:
        if args.out:
            out.close()
    if args.out:
        print(f"{len(questions)} cevap -> {args.out}")


def cmd_search(args: argparse.Namespace, settings: Settings) -> None:
    """Modele gidecek parçaları gösterir; yalnızca embedding modeli yüklenir."""
    if args.k is not None:
        try:
            settings = dataclasses.replace(settings, top_k=args.k)
        except ValueError as err:
            raise CommandError(str(err)) from err
    assistant = _assistant(settings, cache=False)
    docs = _docs(assistant, args.doc)
    prepared = assistant.prepare(args.question, docs=docs)
    if prepared.refusal_reason == "empty":
        raise CommandError("Bir soru yazın.")
    if docs:
        print(f"Yalnızca şu dokümanlarda aranıyor: {', '.join(sorted(docs))}")
    mentioned = assistant.retriever.mentioned_titles(prepared.question)
    if mentioned:
        quota = ", her alandan en az bir parça" if len(mentioned) > 1 else ""
        print(f"Soruda geçen alanlar: {', '.join(sorted(mentioned))} "
              f"(sıralamada +{TITLE_BOOST}{quota})")
    for i, hit in enumerate(prepared.hits, start=1):
        print(f"[K{i}] {hit.score:.3f}  {hit.chunk.section}  ({hit.chunk.filename})")
        print(f"     {' '.join(hit.chunk.text.split())[:args.chars]}")
    best = max(hit.score for hit in prepared.hits)
    if prepared.refusal_reason == "low_similarity":
        print(f"\nEn iyi benzerlik {best:.3f} < eşik {settings.min_similarity}: model çağrılmaz, "
              "soru reddedilir.")
    else:
        print(f"\nEn iyi benzerlik {best:.3f} ≥ eşik {settings.min_similarity}: bu parçalar modele gider.")
    print(f"(arama {prepared.retrieval_seconds:.2f} sn)")


def _chat_history(path: Path) -> Callable[[], None]:
    """Önceki oturumların sorularını readline geçmişine yükler; kaydeden fonksiyonu döner.

    readline olmayan ortamlarda (Windows) ve dosya yazılamazsa sohbet yine çalışır.
    """
    try:
        import readline
    except ImportError:
        return lambda: None
    try:
        readline.read_history_file(path)
    except OSError:
        pass
    readline.set_history_length(500)

    def save() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(path)
        except OSError:
            pass

    return save


def _save_chat(turns: list[tuple[str, Answer]], target: str, settings: Settings) -> None:
    """Sohbeti web arayüzündeki indirme düğmesiyle aynı Markdown biçiminde yazar."""
    if not turns:
        print("Henüz kaydedilecek soru yok.")
        return
    now = datetime.now()
    path = Path(target or f"miras-sohbet-{now:%Y%m%d-%H%M}.md")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(conversation_markdown(turns, settings, now), encoding="utf-8")
    except OSError as err:
        print(f"Kaydedilemedi: {err}")
        return
    print(f"{len(turns)} soru -> {path}")


def _show_chunks(answer: Answer | None, which: str) -> None:
    """Son cevabın parçalarını tam metniyle basar; `which` "2" ya da "K2" ise yalnızca o parça."""
    if answer is None or not answer.hits:
        print("Gösterilecek parça yok.")
        return
    indices = answer.cited or list(range(len(answer.hits)))
    if which:
        number = which.upper().removeprefix("K")
        if not number.isdigit() or not 1 <= int(number) <= len(answer.hits):
            print(f"K1..K{len(answer.hits)} arasında bir parça yazın.")
            return
        indices = [int(number) - 1]
    for i in indices:
        hit = answer.hits[i]
        print(f"\n[K{i + 1}] {hit.chunk.section}  ({hit.chunk.filename}, benzerlik {hit.score:.2f})")
        if hit.chunk.source_url:
            print(f"     {hit.chunk.source_url}")
        print(hit.chunk.text)


def cmd_chat(args: argparse.Namespace, settings: Settings) -> None:
    assistant = _assistant(settings)
    print("Miras — Türkiye'nin UNESCO Dünya Mirası alanları asistanı. Çıkmak için 'çık' yazın.")
    print("Alan adı geçmeyen sorular önceki sorunun konusuyla sorulur; sıfırlamak için 'yeni' yazın.")
    print("Sohbeti Markdown olarak kaydetmek için 'kaydet [dosya.md]', son cevabın parçalarını "
          "tam okumak için 'kaynak [K2]' yazın.")
    print("Aramayı bazı alanlarla sınırlamak için 'sadece efes, troya', kaldırmak için 'sadece' yazın.")
    save_history = _chat_history(CHAT_HISTORY)
    last: Answer | None = None
    turns: list[tuple[str, Answer]] = []
    docs = _docs(assistant, args.doc)
    try:
        while True:
            try:
                question = input("\nSoru> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"çık", "cik", "exit", "quit", "q"}:
                break
            if question.lower() == "yeni":
                last = None
                continue
            command, _, target = question.partition(" ")
            if command.lower() == "kaydet":
                _save_chat(turns, target.strip(), settings)
                continue
            if command.lower() == "kaynak":
                _show_chunks(turns[-1][1] if turns else None, target.strip())
                continue
            if command.lower() == "sadece":
                try:
                    docs = _docs(assistant, [target])
                except CommandError as err:
                    print(err)
                    continue
                print(f"Yalnızca şu dokümanlarda aranıyor: {', '.join(sorted(docs))}" if docs
                      else "Tüm dokümanlarda aranıyor.")
                continue
            last = _answer_streaming(assistant, question, args.verbose, last, docs=docs)
            turns.append((question, last))
    finally:
        save_history()


def cmd_calibrate(args: argparse.Namespace, settings: Settings) -> None:
    backend = _backend(settings)
    retriever = Retriever(Store(settings.db_path), backend.embed_model)
    rows = evaluation.evaluate_retrieval(
        evaluation.load_questions(ROOT / "data" / "eval" / "questions.json"),
        backend, retriever, settings.top_k, settings.bm25_weight)
    for kind in ("answerable", "near_miss", "out_of_domain"):
        scores = sorted(r.top_score for r in rows if r.type == kind)
        print(f"{kind:14} en iyi parça benzerliği: en düşük {scores[0]:.3f}, "
              f"ortanca {scores[len(scores) // 2]:.3f}, en yüksek {scores[-1]:.3f}")
    result = evaluation.calibrate_threshold(rows)
    print(f"\nÖnerilen eşik (dev sorularından): {result['threshold']} "
          f"(eşit derecede iyi aralık: {result['best_range'][0]}–{result['best_range'][1]})")
    print(f"Dev dengeli doğruluk: {result['dev_balanced_accuracy']:.0%}, "
          f"test dengeli doğruluk: {result['test_balanced_accuracy']:.0%}")
    print(f"Konu içi ama dokümanda olmayan soruların kapıda yakalanan oranı: "
          f"{result['near_miss_blocked_by_gate']:.0%} (kalanı modelin reddetmesi gerekir)")
    print("Bu değeri miras/config.py içindeki min_similarity alanına yazın.")

    followups_path = ROOT / "data" / "eval" / "followups.json"
    if followups_path.exists():
        keep, block = evaluation.followup_floor_scores(
            evaluation.load_questions(followups_path),
            evaluation.load_questions(ROOT / "data" / "eval" / "questions.json"), backend, retriever)
        if not keep or not block:
            print("\nTakip eşiği ölçülemedi: eşiğe bağlı takip sorusu ya da konu dışı soru yok "
                  "(gönderme kelimeli ve bütün kelimeleri bilinen takip soruları eşiğe bakmaz).")
            return
        floor = evaluation.calibrate_followup_floor(keep, block)
        print(f"\nTakip soruları tek başına en düşük {floor['lowest_followup']:.3f}, "
              f"konu dışı sorular en yüksek {floor['highest_off_topic']:.3f}")
        print(f"Önerilen konu taşıma eşiği: {floor['floor']} "
              f"(eşit derecede iyi aralık: {floor['best_range'][0]}–{floor['best_range'][1]}, "
              f"dengeli doğruluk: {floor['balanced_accuracy']:.0%}, şu an: {settings.followup_min_similarity})")
        print("Bu değeri followup_min_similarity alanına ya da MIRAS_FOLLOWUP_MIN_SIMILARITY'ye yazın.")


def cmd_tune_hybrid(args: argparse.Namespace, settings: Settings) -> None:
    backend = _backend(settings)
    retriever = Retriever(Store(settings.db_path), backend.embed_model)
    results = evaluation.tune_keyword_weight(
        evaluation.load_questions(ROOT / "data" / "eval" / "questions.json"),
        backend, retriever, settings.top_k, settings.min_similarity)
    print(f"{'ağırlık':>8} | {'dev @1':>6} {'MRR':>6} {'bağlam':>6} {'kapı':>4} | "
          f"{'test @1':>7} {'MRR':>6} {'bağlam':>6} {'kapı':>4}")
    for r in results:
        dev, test = r["dev"], r["test"]
        print(f"{r['weight']:>8} | {dev['doc_hit_at_1']:>6.0%} {dev['mrr']:>6.3f} "
              f"{dev['answer_in_context_at_k']:>6.0%} {dev['answerable_gated']:>4} | "
              f"{test['doc_hit_at_1']:>7.0%} {test['mrr']:>6.3f} "
              f"{test['answer_in_context_at_k']:>6.0%} {test['answerable_gated']:>4}")
    print("(@1: doğru doküman ilk sırada, bağlam: cevap getirilen parçalarda, "
          "kapı: eşikte reddedilecek cevaplanabilir soru)")
    best = evaluation.choose_keyword_weight(results)
    if best == 0:
        print("\nHibrit arama dev sorularında fayda sağlamadı; bm25_weight = 0 bırakın.")
    else:
        print(f"\nÖnerilen ağırlık (dev MRR'ına göre): {best}. miras/config.py içindeki bm25_weight "
              f"alanına ya da MIRAS_BM25_WEIGHT'e yazın, ardından calibrate ve eval çalıştırın.")


def cmd_eval(args: argparse.Namespace, settings: Settings) -> None:
    # Önbellek kapalı: süreler ve cevaplar her çalıştırmada modelden ölçülsün.
    assistant = _assistant(settings, cache=False)
    questions = evaluation.load_questions(ROOT / "data" / "eval" / "questions.json")
    followups_path = ROOT / "data" / "eval" / "followups.json"
    followup_items = evaluation.load_questions(followups_path) if followups_path.exists() else []
    if args.split:
        # Ayarlarken test sorularına bakmamak için yalnızca dev, son ölçüm için yalnızca test.
        # Takip setinde dev/test ayrımı yok; bölüm seçilince atlanır.
        questions = [q for q in questions if q["split"] == args.split]
        followup_items = []
    ids = [i.strip() for i in (args.ids or "").split(",") if i.strip()]
    if ids:
        known = {q["id"] for q in questions} | {q["id"] for q in followup_items}
        if unknown := [i for i in ids if i not in known]:
            raise CommandError(f"Bilinmeyen soru: {', '.join(unknown)}")
        questions = [q for q in questions if q["id"] in ids]
        followup_items = [q for q in followup_items if q["id"] in ids]
    retrieval = evaluation.retrieval_summary(evaluation.evaluate_retrieval(
        questions, assistant.backend, assistant.retriever, settings.top_k, settings.bm25_weight))

    def show(row: evaluation.AnswerRow) -> None:
        print(f"{'✅' if row.passed else '❌'} {row.id} [{row.seconds:4.1f} sn] {row.question}\n"
              f"     -> {row.answer[:160]}", flush=True)

    rows = evaluation.evaluate_answers(questions, assistant, on_row=show)
    answers = evaluation.answer_summary(rows)

    followups, followup_rows = None, []
    if followup_items:
        print("\nTakip soruları:")

        def show_followup(row: evaluation.FollowupRow) -> None:
            carried = f" (konu: {', '.join(row.carried)})" if row.carried else ""
            print(f"{'✅' if row.passed else '❌'} {row.id} [{row.seconds:4.1f} sn] "
                  f"{' → '.join(row.turns)}{carried}\n     -> {row.answer[:160]}", flush=True)

        followup_rows = evaluation.evaluate_followups(followup_items, assistant, on_row=show_followup)
        followups = evaluation.followup_summary(followup_rows)
    header = {"chat_model": settings.chat_model, "chat_device": settings.chat_device,
              "embed_model": settings.embed_model,
              "top_k": settings.top_k, "min_similarity": settings.min_similarity,
              "bm25_weight": settings.bm25_weight,
              "followup_min_similarity": settings.followup_min_similarity,
              "chunks": len(assistant.retriever.chunks)}
    stem = ROOT / "reports" / f"eval-{settings.chat_model}-{settings.chat_device}"
    # Alt kümelerin raporu tam raporun üzerine yazılmasın.
    subset = ([f"yalnızca {args.split} soruları"] if args.split else []) + (
        [f"seçili sorular ({', '.join(ids)})"] if ids else [])
    if subset:
        header["subset"] = ", ".join(subset)
        stem = stem.with_name("-".join([stem.name] + ([args.split] if args.split else [])
                                       + (["secili"] if ids else [])))
    evaluation.write_report(stem, header, retrieval, answers, rows, followups, followup_rows)
    print(f"\nGenel: {answers['overall']['passed']}/{answers['overall']['total']}  "
          f"-> {stem}.md")
    if followups:
        print(f"Takip soruları: {followups['overall']['passed']}/{followups['overall']['total']}")


def cmd_compare_reports(args: argparse.Namespace, settings: Settings) -> None:
    try:
        old, new = (json.loads(path.read_text(encoding="utf-8")) for path in (args.old, args.new))
        diff = evaluation.compare_reports(old, new)
    except (OSError, ValueError, KeyError) as err:
        raise CommandError(f"Rapor okunamadı: {err}") from err

    def frac(d: dict | None) -> str:
        return f"{d['passed']}/{d['total']}" if d else "-"

    for key, (before, after) in diff["config"].items():
        print(f"  {key}: {before} -> {after}")
    print(f"Genel: {frac(diff['answers'][0])} -> {frac(diff['answers'][1])}")
    if any(diff["followups"]):
        print(f"Takip soruları: {frac(diff['followups'][0])} -> {frac(diff['followups'][1])}")
    for title, rows in (("Düzelen", diff["fixed"]), ("Bozulan", diff["broken"])):
        if rows:
            print(f"\n{title} ({len(rows)}):")
        for row in rows:
            question = row.get("question") or " → ".join(row.get("turns", []))
            print(f"  {row['id']} {question}\n     -> {row['answer'][:160]}")
    if not diff["fixed"] and not diff["broken"]:
        print("\nOrtak sorularda sonucu değişen yok.")
    if diff["only_old"] or diff["only_new"]:
        print(f"\nYalnızca eskide: {', '.join(diff['only_old']) or '-'}; "
              f"yalnızca yenide: {', '.join(diff['only_new']) or '-'}")


def cmd_feedback(args: argparse.Namespace, settings: Settings) -> None:
    rows = Store(settings.db_path).feedback_rows()
    up = sum(r["rating"] > 0 for r in rows)
    print(f"👍 {up}  👎 {len(rows) - up}")
    for row in rows if args.all else [r for r in rows if r["rating"] < 0]:
        when = datetime.fromtimestamp(row["created"])
        carried = f" (konu: {', '.join(row['carried'])})" if row["carried"] else ""
        print(f"\n{'👍' if row['rating'] > 0 else '👎'} {when:%d.%m.%Y %H:%M} {row['question']}{carried}")
        print(f"     -> {row['answer'][:160]}")
        if row["refusal_reason"]:
            print(f"     reddetme: {row['refusal_reason']}")
        else:
            cited = sorted({s["filename"] for s in row["sources"] if s["cited"]})
            print(f"     atıf: {', '.join(cited) or 'yok'}  ({row['chat_model']})")
    if args.export:
        candidates = eval_candidates(rows)
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\n{len(candidates)} aday -> {args.export}")
        print("type ve expected alanlarını doldurup data/eval/questions.json'a ekleyin; "
              "carried alanı olanlar takip sorusu (followups.json).")


REFUSAL_LABELS = {"low_similarity": "kapıda reddedildi", "model": "model reddetti"}


def cmd_log(args: argparse.Namespace, settings: Settings) -> None:
    """Soru günlüğü: neler soruldu, hangileri reddedildi (bilgi tabanındaki boşluklar)."""
    store = Store(settings.db_path)
    if args.clear:
        print(f"Günlükten {store.clear_queries()} soru silindi.")
        return
    total, refused = store.query_counts()
    if not total:
        state = "" if settings.query_log else " (MIRAS_QUERY_LOG=0, günlük kapalı)"
        print(f"Günlükte soru yok{state}.")
        return
    print(f"Günlükte {total} soru, {refused} tanesi reddedildi.")
    for row in store.query_rows(refused=args.refused, limit=args.n):
        when = datetime.fromtimestamp(row["created"])
        mark = "✗" if row["refusal_reason"] else "✓"
        extra = []
        if row["carried"]:
            extra.append(f"konu: {', '.join(row['carried'])}")
        if row["docs"]:
            extra.append(f"sadece: {', '.join(row['docs'])}")
        if row["refusal_reason"]:
            extra.append(REFUSAL_LABELS.get(row["refusal_reason"], row["refusal_reason"]))
        elif row["cited"]:
            extra.append(", ".join(row["cited"]))
        if row["cached"]:
            extra.append("önbellekten")
        print(f"{when:%d.%m %H:%M}  {mark} {row['top_score']:.2f}  {row['question']}"
              + (f"  ({'; '.join(extra)})" if extra else ""))
    if args.export:
        candidates = refused_candidates(store.query_rows(refused=True))
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\n{len(candidates)} aday -> {args.export}")
        print("Cevabı dokümanlarda olanlar answerable (doc ve expected ile), olmayanlar near_miss ya da "
              "out_of_domain olarak data/eval/questions.json'a eklenebilir.")


def cmd_status(args: argparse.Namespace, settings: Settings) -> None:
    store = Store(settings.db_path)
    docs, chunks = store.counts()
    indexed_with = store.get_meta("embed_model")
    votes = store.feedback_rows()
    up = sum(r["rating"] > 0 for r in votes)
    print(f"Veritabanı: {settings.db_path}")
    print(f"İndeks: {docs} doküman, {chunks} parça" + (f" ({indexed_with})" if indexed_with else ""))
    print(f"Önbellek: {store.cache_size()} cevap · oylar: 👍 {up}  👎 {len(votes) - up}")
    logged, refused = store.query_counts()
    log_state = f"{logged} soru, {refused} reddedildi" if settings.query_log else "kapalı (MIRAS_QUERY_LOG=0)"
    print(f"Soru günlüğü: {log_state}")
    print(f"Sohbet modeli: {settings.chat_model} ({settings.chat_device}), top_k {settings.top_k}, "
          f"eşik {settings.min_similarity}, bm25 {settings.bm25_weight}")
    if indexed_with and indexed_with != settings.embed_model:
        print(f"\nİndeks '{indexed_with}' ile kurulmuş, ayarlardaki model '{settings.embed_model}': "
              "`ingest --rebuild` gerekli.")
        return
    pending = pending_changes(store, settings)
    if not pending:
        print(f"\n{settings.docs_dir} ile indeks aynı.")
        return
    print(f"\n{settings.docs_dir} içinde indekslenmemiş değişiklikler var:")
    for name in pending.changed:
        print(f"  + {name}")
    for name in pending.missing:
        print(f"  - {name} (klasörde yok)")
    print("`python main.py ingest` ile indeksi güncelleyin.")


def cmd_docs(args: argparse.Namespace, settings: Settings) -> None:
    docs = Store(settings.db_path).documents()
    if not docs:
        print("İndeks boş. Önce `python main.py ingest` çalıştırın.")
        return
    for doc in docs:
        print(f"{doc['title']}  ({doc['filename']}, {doc['chunks']} parça)")
        if doc["aliases"]:
            print(f"     takma adlar: {', '.join(doc['aliases'])}")
        if doc["source_url"]:
            print(f"     {doc['source_url']}")
    print(f"\n{len(docs)} doküman, {sum(d['chunks'] for d in docs)} parça")


def cmd_remove_doc(args: argparse.Namespace, settings: Settings) -> None:
    store = Store(settings.db_path)
    indexed = {d["filename"] for d in store.documents()}
    if args.filename not in indexed and not (settings.docs_dir / args.filename).is_file():
        raise CommandError(f"{args.filename} ne indekste ne de {settings.docs_dir} içinde.")
    if indexed <= {args.filename}:
        raise CommandError("En az bir doküman kalmalı.")  # boş indeksle arama yapılamaz
    try:
        remove_document(store, settings.docs_dir, args.filename)
    except ValueError as err:
        raise CommandError(str(err)) from err
    docs, chunks = store.counts()
    print(f"Silindi: {args.filename}. Veritabanı: {docs} doküman, {chunks} parça")


def cmd_config(args: argparse.Namespace, settings: Settings) -> None:
    """Geçerli ayarlar ve nereden geldikleri (config.py, MIRAS_* ya da komut satırı)."""
    for field in dataclasses.fields(settings):
        value = getattr(settings, field.name)
        if value != getattr(SETTINGS, field.name):
            origin = "komut satırı"
        elif field.name in ENV_FIELDS and env_name(field.name) in os.environ:
            origin = f"{env_name(field.name)} ile ezildi"
        elif field.name in ENV_FIELDS:
            origin = f"varsayılan ({env_name(field.name)})"
        else:
            origin = "varsayılan (yalnızca config.py)"
        print(f"{field.name:<24} {str(value):<28} {origin}")


def cmd_cache(args: argparse.Namespace, settings: Settings) -> None:
    store = Store(settings.db_path)
    if args.clear:
        print(f"Önbellekten {store.clear_answer_cache()} cevap silindi.")
    else:
        print(f"Önbellekte {store.cache_size()} cevap var -> {settings.db_path}")


def cmd_serve(args: argparse.Namespace, settings: Settings) -> None:
    """Yerel HTTP API; istekler `ask --json` ile aynı biçimde cevaplanır (miras/server.py)."""
    assistant = _assistant(settings)
    token = args.token or os.environ.get("MIRAS_API_TOKEN") or None
    try:
        server = make_server(assistant, args.host, args.port, token=token, cors_origin=args.cors)
    except OSError as err:
        raise CommandError(f"{args.host}:{args.port} dinlenemedi: {err}") from err
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        _status(f"Uyarı: {args.host} üzerinden ağdaki herkes erişebilir, kimlik doğrulama yok "
                "(--token ya da MIRAS_API_TOKEN).")
    host, port = server.server_address[:2]
    print(f"Miras API: http://{host}:{port}  (POST /ask, POST /search, GET /docs, GET /health)")
    print("Durdurmak için Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Miras: Foundry Local ile çalışan yerel RAG asistanı")
    parser.add_argument("--model", help="Sohbet modeli takma adı (varsayılan: config.py)")
    parser.add_argument("--device", choices=["cpu", "gpu"], help="Sohbet modelinin cihazı")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Dokümanları parçala, göm ve SQLite'a yaz")
    p.add_argument("--rebuild", action="store_true", help="İndeksi sıfırdan kur")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("add-wiki", help="Türkçe Vikipedi maddesini indir ve indeksle (internet gerekir)")
    p.add_argument("article", help="Madde adı ya da tr.wikipedia.org bağlantısı")
    p.add_argument("--aliases", help="Alanın başka adları, virgülle ayrılmış")
    p.set_defaults(func=cmd_add_wiki)

    p = sub.add_parser("update-wiki", help="Vikipedi'de değişen maddeleri indir ve indeksle (internet gerekir)")
    p.add_argument("--check", action="store_true", help="Yalnızca listele, indirme")
    p.set_defaults(func=cmd_update_wiki)

    p = sub.add_parser("ask", help="Tek bir soru sor")
    p.add_argument("question")
    p.add_argument("-v", "--verbose", action="store_true", help="Parça metinlerini ve süreleri göster")
    p.add_argument("--json", action="store_true", help="Cevabı ve kaynakları JSON olarak yaz (akış yok)")
    p.add_argument("--refresh", action="store_true",
                   help="Önbellekteki cevabı kullanma, modeli yeniden çağır ve önbelleği güncelle")
    p.add_argument("--doc", action="append", metavar="AD",
                   help="Yalnızca bu dokümanda ara (dosya adı, başlık ya da takma ad; tekrarlanabilir, "
                        "virgülle de ayrılabilir)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ask-batch", help="Dosyadaki soruları (satır başına bir) sor, JSONL yaz")
    p.add_argument("file", type=Path, help="Soru dosyası; - ise standart girdi")
    p.add_argument("--out", type=Path, metavar="DOSYA", help="Cevapları bu dosyaya yaz (varsayılan: ekran)")
    p.add_argument("--refresh", action="store_true", help="Önbelleği kullanma, modeli yeniden çağır")
    p.add_argument("--doc", action="append", metavar="AD",
                   help="Yalnızca bu dokümanda ara (dosya adı, başlık ya da takma ad; tekrarlanabilir, "
                        "virgülle de ayrılabilir)")
    p.set_defaults(func=cmd_ask_batch)

    p = sub.add_parser("search", help="Soruya getirilen parçaları göster (sohbet modeli çağırmaz)")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="Kaç parça getirilsin (varsayılan: top_k)")
    p.add_argument("--chars", type=int, default=200, help="Parça metninden gösterilecek karakter")
    p.add_argument("--doc", action="append", metavar="AD",
                   help="Yalnızca bu dokümanda ara (dosya adı, başlık ya da takma ad; tekrarlanabilir, "
                        "virgülle de ayrılabilir)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("chat", help="Etkileşimli soru-cevap döngüsü")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--doc", action="append", metavar="AD",
                   help="Yalnızca bu dokümanda ara (sohbette 'sadece' ile değiştirilebilir)")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("calibrate", help="Benzerlik eşiğini test sorularından ölç")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("tune-hybrid", help="BM25 ağırlığını test sorularından ölç (model çağırmaz)")
    p.set_defaults(func=cmd_tune_hybrid)

    p = sub.add_parser("eval", help="Test setini çalıştır, reports/ altına rapor yaz")
    p.add_argument("--split", choices=["dev", "test"],
                   help="Yalnızca bu bölümün soruları (takip seti atlanır, rapor adına eklenir)")
    p.add_argument("--ids", metavar="ID,ID",
                   help="Yalnızca bu sorular, ör. a01,f03 (ana set ve takip seti; rapor adına -secili eklenir)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("compare-reports", help="İki eval raporunu karşılaştır (model çağırmaz)")
    p.add_argument("old", type=Path, help="Önceki rapor (.json)")
    p.add_argument("new", type=Path, help="Yeni rapor (.json)")
    p.set_defaults(func=cmd_compare_reports)

    p = sub.add_parser("feedback", help="Web arayüzünde verilen oyları listele (model çağırmaz)")
    p.add_argument("--all", action="store_true", help="Beğenilen cevapları da göster")
    p.add_argument("--export", type=Path, metavar="DOSYA",
                   help="Beğenilmeyen soruları eval adayı olarak JSON'a yaz")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("log", help="Sorulan soruları ve reddedilenleri listele (model çağırmaz)")
    p.add_argument("-n", type=int, default=20, help="Kaç soru gösterilsin (varsayılan: 20)")
    p.add_argument("--refused", action="store_true", help="Yalnızca reddedilen sorular")
    p.add_argument("--export", type=Path, metavar="DOSYA",
                   help="Reddedilen soruları eval adayı olarak JSON'a yaz")
    p.add_argument("--clear", action="store_true", help="Günlüğü boşalt (oylar ve önbellek kalır)")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("status", help="İndeks, önbellek ve bekleyen değişiklikler (model çağırmaz)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("docs", help="İndeksteki dokümanları listele (model çağırmaz)")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("remove-doc", help="Dokümanı klasörden ve indeksten sil (model çağırmaz)")
    p.add_argument("filename", help="Doküman dosyasının adı, ör. efes.md")
    p.set_defaults(func=cmd_remove_doc)

    p = sub.add_parser("config", help="Geçerli ayarları ve nereden geldiklerini göster")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("cache", help="Cevap önbelleğindeki kayıt sayısı (model çağırmaz)")
    p.add_argument("--clear", action="store_true", help="Önbelleği boşalt")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("serve", help="Yerel HTTP API başlat (POST /ask, /search; GET /docs, /health)")
    p.add_argument("--host", default="127.0.0.1", help="Dinlenecek adres (varsayılan: yalnızca bu makine)")
    p.add_argument("--port", type=int, default=8765, help="Port (varsayılan: 8765)")
    p.add_argument("--token", help="İsteklerde 'Authorization: Bearer <token>' iste "
                                   "(varsayılan: MIRAS_API_TOKEN, yoksa kapalı)")
    p.add_argument("--cors", metavar="ORIGIN",
                   help="Bu adresteki sayfalar tarayıcıdan çağırabilsin, ör. http://localhost:3000 ya da *")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    overrides = {"chat_model": args.model, "chat_device": args.device}
    try:
        settings = dataclasses.replace(SETTINGS, **{k: v for k, v in overrides.items() if v})
    except ValueError as err:
        print(f"Hata: {err}", file=sys.stderr)
        return 1
    try:
        args.func(args, settings)
    except (BackendError, IndexError_, FileNotFoundError, WikiError, CommandError) as err:
        print(f"Hata: {err}", file=sys.stderr)
        return 1
    return 0
