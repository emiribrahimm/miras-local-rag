"""Komut satırı arayüzü: ingest, ask, chat, eval, calibrate."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

from . import evaluation
from .backends import BackendError, FoundryBackend
from .config import ROOT, SETTINGS, Settings
from .ingest import ingest
from .pipeline import Answer, Assistant
from .retrieval import IndexError_, Retriever
from .store import Store


def _status(message: str) -> None:
    print(f"  · {message}", file=sys.stderr, flush=True)


def _backend(settings: Settings) -> FoundryBackend:
    return FoundryBackend(settings.chat_model, settings.embed_model, settings.chat_device,
                          on_status=_status)


def _assistant(settings: Settings) -> Assistant:
    backend = _backend(settings)
    return Assistant(backend, Retriever(Store(settings.db_path), backend.embed_model), settings)


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


def _answer_streaming(assistant: Assistant, question: str, verbose: bool) -> None:
    prepared = assistant.prepare(question)
    started = time.perf_counter()
    pieces = []
    for piece in assistant.generate(prepared):
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    answer = assistant.finalize(prepared, "".join(pieces), time.perf_counter() - started)
    _print_sources(answer, verbose)
    if verbose:
        print(f"\n(arama {answer.retrieval_seconds:.2f} sn, üretim {answer.generation_seconds:.2f} sn)")


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    store = Store(settings.db_path)
    report = ingest(store, _backend(settings), settings, rebuild=args.rebuild, on_progress=_status)
    docs, chunks = store.counts()
    print(f"Eklenen/güncellenen: {len(report.added)}, değişmeyen: {len(report.unchanged)}, "
          f"silinen: {len(report.removed)}")
    print(f"Veritabanı: {docs} doküman, {chunks} parça -> {settings.db_path}")


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    _answer_streaming(_assistant(settings), args.question, args.verbose)


def cmd_chat(args: argparse.Namespace, settings: Settings) -> None:
    assistant = _assistant(settings)
    print("Miras — Türkiye'nin UNESCO Dünya Mirası alanları asistanı. Çıkmak için 'çık' yazın.")
    while True:
        try:
            question = input("\nSoru> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"çık", "cik", "exit", "quit", "q"}:
            break
        _answer_streaming(assistant, question, args.verbose)


def cmd_calibrate(args: argparse.Namespace, settings: Settings) -> None:
    backend = _backend(settings)
    retriever = Retriever(Store(settings.db_path), backend.embed_model)
    rows = evaluation.evaluate_retrieval(
        evaluation.load_questions(ROOT / "data" / "eval" / "questions.json"),
        backend, retriever, settings.top_k)
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


def cmd_eval(args: argparse.Namespace, settings: Settings) -> None:
    assistant = _assistant(settings)
    questions = evaluation.load_questions(ROOT / "data" / "eval" / "questions.json")
    retrieval = evaluation.retrieval_summary(evaluation.evaluate_retrieval(
        questions, assistant.backend, assistant.retriever, settings.top_k))

    def show(row: evaluation.AnswerRow) -> None:
        print(f"{'✅' if row.passed else '❌'} {row.id} [{row.seconds:4.1f} sn] {row.question}\n"
              f"     -> {row.answer[:160]}", flush=True)

    rows = evaluation.evaluate_answers(questions, assistant, on_row=show)
    answers = evaluation.answer_summary(rows)
    header = {"chat_model": settings.chat_model, "chat_device": settings.chat_device,
              "embed_model": settings.embed_model,
              "top_k": settings.top_k, "min_similarity": settings.min_similarity,
              "chunks": len(assistant.retriever.chunks)}
    stem = ROOT / "reports" / f"eval-{settings.chat_model}-{settings.chat_device}"
    evaluation.write_report(stem, header, retrieval, answers, rows)
    print(f"\nGenel: {answers['overall']['passed']}/{answers['overall']['total']}  "
          f"-> {stem}.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Miras: Foundry Local ile çalışan yerel RAG asistanı")
    parser.add_argument("--model", help="Sohbet modeli takma adı (varsayılan: config.py)")
    parser.add_argument("--device", choices=["cpu", "gpu"], help="Sohbet modelinin cihazı")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Dokümanları parçala, göm ve SQLite'a yaz")
    p.add_argument("--rebuild", action="store_true", help="İndeksi sıfırdan kur")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Tek bir soru sor")
    p.add_argument("question")
    p.add_argument("-v", "--verbose", action="store_true", help="Parça metinlerini ve süreleri göster")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="Etkileşimli soru-cevap döngüsü")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("calibrate", help="Benzerlik eşiğini test sorularından ölç")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("eval", help="Test setini çalıştır, reports/ altına rapor yaz")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    overrides = {"chat_model": args.model, "chat_device": args.device}
    settings = dataclasses.replace(SETTINGS, **{k: v for k, v in overrides.items() if v})
    try:
        args.func(args, settings)
    except (BackendError, IndexError_, FileNotFoundError) as err:
        print(f"Hata: {err}", file=sys.stderr)
        return 1
    return 0
