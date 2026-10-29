"""Komut satırı arayüzü: ingest, add-wiki, ask, chat, eval, calibrate, feedback."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from . import evaluation
from .backends import BackendError, FoundryBackend
from .config import ROOT, SETTINGS, Settings
from .feedback import eval_candidates
from .ingest import ingest, save_uploads
from .pipeline import Answer, Assistant
from .retrieval import IndexError_, Retriever
from .store import Store
from .wiki import WikiError, article


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


def _print_sources(answer: Answer, verbose: bool) -> None:
    if answer.refused:
        return
    # Model etiket yazmadıysa kullanıcı yine de hangi parçalara bakıldığını görsün.
    shown = answer.cited or list(range(len(answer.hits)))
    print("\nKaynaklar:" if answer.cited else "\nİncelenen parçalar (model atıf yapmadı):")
    for i in shown:
        hit = answer.hits[i]
        print(f"  [K{i + 1}] {hit.chunk.section}  ({hit.chunk.filename}, benzerlik {hit.score:.2f})")
        if verbose:
            print(f"       {hit.chunk.text[:300]}...")


def _answer_streaming(assistant: Assistant, question: str, verbose: bool,
                      previous: Answer | None = None) -> Answer:
    prepared = assistant.prepare(question, previous)
    if prepared.carried:
        _status(f"konu: {', '.join(prepared.carried)}")
    started = time.perf_counter()
    pieces = []
    for piece in assistant.generate(prepared):
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


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    _answer_streaming(_assistant(settings), args.question, args.verbose)


def cmd_chat(args: argparse.Namespace, settings: Settings) -> None:
    assistant = _assistant(settings)
    print("Miras — Türkiye'nin UNESCO Dünya Mirası alanları asistanı. Çıkmak için 'çık' yazın.")
    print("Alan adı geçmeyen sorular önceki sorunun konusuyla sorulur; sıfırlamak için 'yeni' yazın.")
    last: Answer | None = None
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
        last = _answer_streaming(assistant, question, args.verbose, last)


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
    retrieval = evaluation.retrieval_summary(evaluation.evaluate_retrieval(
        questions, assistant.backend, assistant.retriever, settings.top_k, settings.bm25_weight))

    def show(row: evaluation.AnswerRow) -> None:
        print(f"{'✅' if row.passed else '❌'} {row.id} [{row.seconds:4.1f} sn] {row.question}\n"
              f"     -> {row.answer[:160]}", flush=True)

    rows = evaluation.evaluate_answers(questions, assistant, on_row=show)
    answers = evaluation.answer_summary(rows)

    followups, followup_rows = None, []
    followups_path = ROOT / "data" / "eval" / "followups.json"
    if followups_path.exists():
        print("\nTakip soruları:")

        def show_followup(row: evaluation.FollowupRow) -> None:
            carried = f" (konu: {', '.join(row.carried)})" if row.carried else ""
            print(f"{'✅' if row.passed else '❌'} {row.id} [{row.seconds:4.1f} sn] "
                  f"{' → '.join(row.turns)}{carried}\n     -> {row.answer[:160]}", flush=True)

        followup_rows = evaluation.evaluate_followups(
            evaluation.load_questions(followups_path), assistant, on_row=show_followup)
        followups = evaluation.followup_summary(followup_rows)
    header = {"chat_model": settings.chat_model, "chat_device": settings.chat_device,
              "embed_model": settings.embed_model,
              "top_k": settings.top_k, "min_similarity": settings.min_similarity,
              "bm25_weight": settings.bm25_weight,
              "followup_min_similarity": settings.followup_min_similarity,
              "chunks": len(assistant.retriever.chunks)}
    stem = ROOT / "reports" / f"eval-{settings.chat_model}-{settings.chat_device}"
    evaluation.write_report(stem, header, retrieval, answers, rows, followups, followup_rows)
    print(f"\nGenel: {answers['overall']['passed']}/{answers['overall']['total']}  "
          f"-> {stem}.md")
    if followups:
        print(f"Takip soruları: {followups['overall']['passed']}/{followups['overall']['total']}")


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

    p = sub.add_parser("ask", help="Tek bir soru sor")
    p.add_argument("question")
    p.add_argument("-v", "--verbose", action="store_true", help="Parça metinlerini ve süreleri göster")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Etkileşimli soru-cevap döngüsü")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("calibrate", help="Benzerlik eşiğini test sorularından ölç")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("tune-hybrid", help="BM25 ağırlığını test sorularından ölç (model çağırmaz)")
    p.set_defaults(func=cmd_tune_hybrid)

    p = sub.add_parser("eval", help="Test setini çalıştır, reports/ altına rapor yaz")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("feedback", help="Web arayüzünde verilen oyları listele (model çağırmaz)")
    p.add_argument("--all", action="store_true", help="Beğenilen cevapları da göster")
    p.add_argument("--export", type=Path, metavar="DOSYA",
                   help="Beğenilmeyen soruları eval adayı olarak JSON'a yaz")
    p.set_defaults(func=cmd_feedback)

    args = parser.parse_args(argv)
    overrides = {"chat_model": args.model, "chat_device": args.device}
    settings = dataclasses.replace(SETTINGS, **{k: v for k, v in overrides.items() if v})
    try:
        args.func(args, settings)
    except (BackendError, IndexError_, FileNotFoundError, WikiError) as err:
        print(f"Hata: {err}", file=sys.stderr)
        return 1
    return 0
