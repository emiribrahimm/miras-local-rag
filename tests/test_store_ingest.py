import dataclasses

import numpy as np
import pytest

from miras.backends import FakeBackend
from miras.ingest import ingest
from miras.retrieval import IndexError_, Retriever
from miras.store import Store


def test_ingest_stores_chunks_and_normalized_vectors(store):
    assert store.counts() == (2, 3)
    chunks, matrix = store.load_all()
    assert matrix.dtype == np.float32 and matrix.shape == (3, FakeBackend.DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert chunks[-1].source_url == "https://example.org/g"
    assert chunks[0].aliases == ("Ephesos", "Ephesus") and chunks[-1].aliases == ()


def test_old_database_without_aliases_column_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "eski.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT NOT NULL UNIQUE, "
                 "title TEXT NOT NULL, source_url TEXT NOT NULL, fingerprint TEXT NOT NULL)")
    conn.execute("INSERT INTO documents(filename, title, source_url, fingerprint) "
                 "VALUES('efes.md', 'Efes', '', 'x')")
    conn.commit()
    conn.close()
    store = Store(path)
    assert store.fingerprints() == {"efes.md": "x"}
    assert store.conn.execute("SELECT aliases FROM documents").fetchone() == ("",)


def test_second_ingest_skips_unchanged_files(store, backend, settings):
    report = ingest(store, backend, settings)
    assert report.added == [] and len(report.unchanged) == 2


def test_changed_and_deleted_files_are_synced(store, backend, settings):
    (settings.docs_dir / "efes.md").write_text("# Efes\n\nYeni metin.\n\n## Ek\n\nBaşka.", encoding="utf-8")
    (settings.docs_dir / "gobeklitepe.md").unlink()
    report = ingest(store, backend, settings)
    assert report.added == ["efes.md"] and report.removed == ["gobeklitepe.md"]
    assert store.counts() == (1, 2)


def test_changing_embedding_model_rebuilds_index(store, settings):
    other = FakeBackend()
    other.embed_model = "baska-model"
    assert len(ingest(store, other, settings).added) == 2
    assert store.get_meta("embed_model") == "baska-model"


def test_retriever_rejects_stale_or_empty_index(store, tmp_path):
    with pytest.raises(IndexError_, match="--rebuild"):
        Retriever(store, "baska-model")
    with pytest.raises(IndexError_, match="ingest"):
        Retriever(Store(tmp_path / "bos.db"), "x")


def test_empty_docs_dir_is_an_error(backend, settings, tmp_path):
    empty = tmp_path / "bos"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ingest(Store(":memory:"), backend, dataclasses.replace(settings, docs_dir=empty))


def test_search_returns_best_chunk_first(store, backend):
    hits = Retriever(store, backend.embed_model).search(backend.embed_query("Efes hangi ilçede?"), 2)
    assert hits[0].chunk.filename == "efes.md"
    assert hits[0].score >= hits[1].score
    assert len(Retriever(store, backend.embed_model).search(backend.embed_query("x"), 50)) == 3


def test_title_mention_boosts_ranking_but_not_gate_score(store, backend):
    retriever = Retriever(store, backend.embed_model)
    bonus = dict(zip((c.filename for c in retriever.chunks), retriever._title_bonus("Efes'e ne oldu?")))
    assert bonus == {"efes.md": 0.1, "gobeklitepe.md": 0.0}
    assert not retriever._title_bonus("Efesli biri ne dedi?").any()  # ad + harf eşleşmez

    vector = backend.embed_query("Neolitik antik kent")
    plain = retriever.search(vector, 3)
    boosted = retriever.search(vector, 3, "Efes Neolitik antik kent mi?")

    def first_efes(hits):
        return next(i for i, h in enumerate(hits) if h.chunk.filename == "efes.md")

    assert first_efes(boosted) < first_efes(plain)
    # Skor ham benzerlik olarak kalır; eşik kalibrasyonu etkilenmez.
    assert {h.chunk.id: h.score for h in plain} == {h.chunk.id: h.score for h in boosted}


def test_keyword_weight_lifts_rare_terms_without_touching_scores(store, backend):
    retriever = Retriever(store, backend.embed_model)
    query = "Dünya Mirası 2018"
    vector = backend.embed_query("Efes antik kent")  # embedding Efes'i işaret ediyor
    plain = retriever.search(vector, 3, query)
    assert retriever.search(vector, 3, query, 0.0) == plain
    hybrid = retriever.search(vector, 3, query, 1.0)
    assert plain[0].chunk.filename == "efes.md"
    assert hybrid[0].chunk.section == "Göbeklitepe › Koruma"  # "2018" yalnızca orada geçiyor
    assert {h.chunk.id: h.score for h in plain} == {h.chunk.id: h.score for h in hybrid}

def test_each_named_site_gets_a_chunk_in_comparisons(store, backend):
    retriever = Retriever(store, backend.embed_model)
    # Embedding yalnızca Göbeklitepe'yi gösteriyor; iki yer de onun parçalarına giderdi.
    vector = backend.embed_query("Göbeklitepe Neolitik Şanlıurfa Dünya Mirası 2018")
    single = retriever.search(vector, 2, "Göbeklitepe ne zaman listeye girdi?")
    assert {h.chunk.filename for h in single} == {"gobeklitepe.md"}
    both = retriever.search(vector, 2, "Göbeklitepe mi Efes mi listeye daha önce girdi?")
    assert [h.chunk.filename for h in both] == ["gobeklitepe.md", "efes.md"]
    # Skorlar ham benzerlik; sıralama yine skora göre.
    assert both[0].score >= both[1].score
    # Alan sayısı top_k'yı aşarsa en iyi alan kalır.
    assert [h.chunk.filename for h in retriever.search(vector, 1, "Göbeklitepe ve Efes")] == [
        "gobeklitepe.md"]


def test_search_can_be_limited_to_some_documents(store, backend):
    retriever = Retriever(store, backend.embed_model)
    vector = backend.embed_query("Göbeklitepe Neolitik Şanlıurfa")
    only_efes = retriever.search(vector, 3, "Göbeklitepe nerede?", only=frozenset({"efes.md"}))
    assert [h.chunk.filename for h in only_efes] == ["efes.md"]
    assert retriever.best_score(vector, frozenset({"efes.md"})) < retriever.best_score(vector)
    # Kota da kısıtın içinde kalır.
    hits = retriever.search(vector, 2, "Göbeklitepe mi Efes mi?", only=frozenset({"gobeklitepe.md"}))
    assert {h.chunk.filename for h in hits} == {"gobeklitepe.md"} and len(hits) == 2


def test_resolve_docs_accepts_filenames_titles_and_aliases(store, backend):
    retriever = Retriever(store, backend.embed_model)
    assert retriever.resolve_docs(["efes.md"]) == {"efes.md"}
    assert retriever.resolve_docs(["Ephesos", "gobeklitepe"]) == {"efes.md", "gobeklitepe.md"}
    assert retriever.resolve_docs(["  EFES "]) == {"efes.md"}
    with pytest.raises(ValueError, match="'Troya' indekste yok. Dokümanlar: efes.md, gobeklitepe.md"):
        retriever.resolve_docs(["Troya"])


def test_aliases_and_ascii_spellings_get_title_boost(store, backend):
    retriever = Retriever(store, backend.embed_model)

    def boosted(query):
        return {c.filename for c, b in zip(retriever.chunks, retriever._title_bonus(query)) if b}

    assert boosted("Ephesos'ta hangi yapılar var?") == {"efes.md"}
    assert boosted("Gobeklitepe nerede?") == {"gobeklitepe.md"}
    assert boosted("EPHESUS ve Göbeklitepe") == {"efes.md", "gobeklitepe.md"}
    assert boosted("Ephesoslu bir filozof") == set()


def test_mentioned_titles(store, backend):
    retriever = Retriever(store, backend.embed_model)
    assert retriever.mentioned_titles("Ephesos ve Göbeklitpe") == {"Efes", "Göbeklitepe"}
    assert retriever.mentioned_titles("Peki ne zaman yapılmış?") == set()

def test_title_key_is_first_word():
    from miras.retrieval import title_key
    assert title_key("Divriği Ulu Camii ve Darüşşifası") == "divriği"
    assert title_key("İznik") == "iznik"


def test_title_keys_include_aliases_and_ascii_forms():
    from miras.retrieval import title_keys
    assert title_keys("Troya", ("Truva", "İlion")) == {"troya", "truva", "ilion"}
    assert title_keys("Göbeklitepe", ("Göbekli Tepe",)) == {
        "göbeklitepe", "gobeklitepe", "göbekli tepe", "gobekli tepe"}
    # Sorgu katlanmaz: Türkçe "anı" kelimesi "Ani" alanına eşleşmez.
    assert "anı" not in title_keys("Ani")


def test_typos_in_long_names_still_get_title_boost(store, backend):
    retriever = Retriever(store, backend.embed_model)

    def boosted(query):
        return {c.filename for c, b in zip(retriever.chunks, retriever._title_bonus(query)) if b}

    assert boosted("Göbeklitpe nerede?") == {"gobeklitepe.md"}
    assert boosted("Gobeklitpe'de ne var?") == {"gobeklitepe.md"}
    assert boosted("GÖBEKLİTEPPE hangi ilde?") == {"gobeklitepe.md"}
    assert boosted("Epheuss nerede?") == {"efes.md"}  # yan yana harf takası
    # Kısa adlarda tolerans yok; iki harf hata da eşleşmez.
    assert boosted("Efs nerede?") == set()
    # Uzun adlarda iki harf hata kabul edilir, üç harf edilmez.
    assert boosted("Gbeklitpe nerede?") == {"gobeklitepe.md"}
    assert boosted("Gbklitpe nerede?") == set()


def test_edit_distance_and_short_names():
    from miras.retrieval import edit_distance, fuzzy_mentions
    assert edit_distance("selimiye", "selimiye") == 0
    assert edit_distance("selimye", "selimiye") == 1
    assert edit_distance("seilmiye", "selimiye") == 1
    assert edit_distance("gbeklitpe", "gobeklitepe") == 2
    assert fuzzy_mentions(["hatusa", "nerede"], "hattuşa")
    assert fuzzy_mentions(["gobekli", "tpe", "nerede"], "göbekli tepe")  # çok kelimeli ad
    assert not fuzzy_mentions(["ilin", "merkezi"], "ilion")
    assert not fuzzy_mentions(["ana"], "ani")


def test_save_uploads_writes_valid_files_and_reports_errors(tmp_path):
    from miras.ingest import save_uploads
    docs = tmp_path / "docs"
    saved, errors = save_uploads(docs, [
        ("ani.md", "---\ntitle: Ani\n---\n\nAni Kars'tadır.".encode()),
        ("../../kaçak.txt", "\ufeffMetin".encode()),
        ("resim.png", b"x"),
        (".gizli.md", b"x"),
        ("bos.md", b"  \n"),
        ("latin1.md", "Kars ilçesi".encode("latin-1")),
    ])
    assert saved == ["ani.md", "kaçak.txt"]
    assert sorted(p.name for p in docs.iterdir()) == ["ani.md", "kaçak.txt"]
    assert not (tmp_path / "kaçak.txt").exists()
    assert (docs / "kaçak.txt").read_text(encoding="utf-8") == "Metin"  # BOM atılır
    assert [e.split(":")[0] for e in errors] == ["resim.png", ".gizli.md", "bos.md", "latin1.md"]


def test_uploaded_file_is_picked_up_by_ingest(store, backend, settings):
    from miras.ingest import save_uploads
    save_uploads(settings.docs_dir, [("ani.md", "---\ntitle: Ani\n---\n\nAni Kars'tadır.".encode())])
    report = ingest(store, backend, settings)
    assert report.added == ["ani.md"] and len(report.unchanged) == 2
    assert "Ani" in {c.title for c in Retriever(store, backend.embed_model).chunks}


def test_answer_cache_round_trip_and_rebuild_clears_it(store, backend, settings):
    assert store.cached_answer("k") is None
    store.cache_answer("k", "Efes İzmir'dedir [K1].")
    store.cache_answer("k", "Efes Selçuk'tadır [K1].")
    assert store.cached_answer("k") == "Efes Selçuk'tadır [K1]."
    ingest(store, backend, settings, rebuild=True)
    assert store.cached_answer("k") is None


def test_answer_cache_keeps_only_newest_entries(store, monkeypatch):
    import miras.store

    monkeypatch.setattr(miras.store, "MAX_CACHED_ANSWERS", 2)
    for key in ("a", "b", "c"):
        store.cache_answer(key, key)
    assert [store.cached_answer(k) for k in ("a", "b", "c")] == [None, "b", "c"]


def test_feedback_is_saved_updated_and_survives_rebuild(store, backend, settings):
    sources = [{"filename": "efes.md", "section": "Efes", "score": 0.81, "cited": True}]
    store.save_feedback("s1-0", 1, "Efes nerede?", (), "Efes İzmir'dedir [K1].", None, sources, "m")
    store.save_feedback("s1-1", -1, "Peki ne zaman?", ("Efes",), "Bu bilgi elimdeki dokümanlarda "
                        "yer almıyor.", "model", sources, "m")
    store.save_feedback("s1-0", -1, "Efes nerede?", (), "Efes İzmir'dedir [K1].", None, sources, "m")
    rows = store.feedback_rows()
    assert [r["key"] for r in rows] == ["s1-0", "s1-1"]
    assert rows[0]["rating"] == -1 and rows[0]["sources"] == sources
    assert rows[1]["carried"] == ("Efes",) and rows[1]["refusal_reason"] == "model"
    assert store.feedback_rows(rating=1) == []
    ingest(store, backend, settings, rebuild=True)
    store.delete_feedback("s1-1")
    assert [r["key"] for r in store.feedback_rows(rating=-1)] == ["s1-0"]


def test_remove_document_deletes_file_and_chunks(store, backend, settings):
    from miras.ingest import remove_document
    remove_document(store, settings.docs_dir, "efes.md")
    assert not (settings.docs_dir / "efes.md").exists()
    assert store.counts() == (1, 2) and "efes.md" not in store.fingerprints()
    report = ingest(store, backend, settings)
    assert report.added == [] and report.removed == []


def test_remove_document_cleans_index_when_file_is_gone(store, settings):
    from miras.ingest import remove_document
    (settings.docs_dir / "efes.md").unlink()
    remove_document(store, settings.docs_dir, "efes.md")
    assert "efes.md" not in store.fingerprints()


def test_remove_document_rejects_bad_names(store, settings, tmp_path):
    from miras.ingest import remove_document
    outside = tmp_path / "disari.md"
    outside.write_text("x", encoding="utf-8")
    for name in ("../disari.md", "notlar.py", ".gizli.md"):
        with pytest.raises(ValueError):
            remove_document(store, settings.docs_dir, name)
    assert outside.exists() and store.counts() == (2, 3)


def test_file_emptied_down_to_header_drops_old_chunks(store, backend, settings):
    (settings.docs_dir / "efes.md").write_text("---\ntitle: Efes\n---\n\n# Efes\n", encoding="utf-8")
    report = ingest(store, backend, settings)
    assert report.removed == ["efes.md"] and report.added == []
    assert store.counts() == (1, 2)
    assert ingest(store, backend, settings).removed == []


def test_unreadable_file_is_skipped_and_keeps_old_chunks(store, backend, settings):
    (settings.docs_dir / "efes.md").write_bytes("# Efes\n\nİzmir".encode("iso8859_9"))
    report = ingest(store, backend, settings)
    assert report.skipped == ["efes.md: UTF-8 olarak okunamadı, atlandı."]
    assert report.removed == [] and store.counts() == (2, 3)


def test_pending_changes_lists_new_edited_and_missing_files(store, backend, settings):
    from miras.ingest import pending_changes

    assert not pending_changes(store, settings)
    (settings.docs_dir / "efes.md").write_text("# Efes\n\nYeni metin.", encoding="utf-8")
    (settings.docs_dir / "ani.md").write_text("# Ani\n\nKars ilinde.", encoding="utf-8")
    (settings.docs_dir / "bos.md").write_text("# Boş\n", encoding="utf-8")
    (settings.docs_dir / "gobeklitepe.md").unlink()
    pending = pending_changes(store, settings)
    assert pending.changed == ["ani.md", "efes.md"] and pending.missing == ["gobeklitepe.md"]
    ingest(store, backend, settings)
    assert not pending_changes(store, settings)


def test_file_database_uses_wal_so_readers_do_not_block_writers(store, settings):
    assert store.conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    reader = Store(settings.db_path)
    reader.conn.execute("BEGIN")
    reader.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    store.cache_answer("k", "Cevap.")
    assert Store(settings.db_path).cached_answer("k") == "Cevap."
    reader.conn.rollback()


def test_ingest_reports_progress_by_embedded_chunks(backend, settings, tmp_path):
    messages, fractions = [], []
    ingest(Store(tmp_path / "ilerleme.db"), backend, settings,
           on_progress=messages.append, on_fraction=fractions.append)
    assert messages == ["efes.md: 1 parça gömülüyor (1/2)...", "gobeklitepe.md: 2 parça gömülüyor (2/2)..."]
    assert fractions == pytest.approx([1 / 3, 1.0])


def test_query_log_keeps_newest_first_and_survives_rebuild(store, backend, settings, monkeypatch):
    import miras.store
    store.log_query("Efes nerede?", (), ("efes.md",), None, 0.81, ["efes.md"], False, "m")
    store.log_query("Pizza?", (), (), "low_similarity", 0.12, [], False, "m")
    store.log_query("Peki ne zaman?", ("Efes",), (), "model", 0.6, [], True, "m")
    rows = store.query_rows()
    assert [r["question"] for r in rows] == ["Peki ne zaman?", "Pizza?", "Efes nerede?"]
    assert rows[0]["carried"] == ("Efes",) and rows[0]["cached"] is True
    assert rows[2]["docs"] == ("efes.md",) and rows[2]["cited"] == ("efes.md",)
    assert [r["question"] for r in store.query_rows(refused=True, limit=1)] == ["Peki ne zaman?"]
    assert store.query_counts() == (3, 2)

    ingest(store, backend, settings, rebuild=True)
    assert store.query_counts() == (3, 2)
    monkeypatch.setattr(miras.store, "MAX_LOGGED_QUERIES", 2)
    store.log_query("Ani?", (), (), None, 0.7, [], False, "m")
    assert [r["question"] for r in store.query_rows()] == ["Ani?", "Peki ne zaman?"]
    assert store.clear_queries() == 2 and store.query_counts() == (0, 0)


def test_html_documents_are_converted_and_indexed(store, backend, settings):
    from miras.ingest import pending_changes
    (settings.docs_dir / "ani.html").write_text(
        "<html><head><title>Ani</title><script>x()</script></head><body><nav>Menü</nav>"
        "<h1>Ani</h1><p>Ani Kars ilindedir.</p><h2>Kiliseler</h2><p>Meryemana Kilisesi.</p></body></html>",
        encoding="utf-8")
    assert pending_changes(store, settings).changed == ["ani.html"]
    report = ingest(store, backend, settings)
    assert report.added == ["ani.html"] and not pending_changes(store, settings)
    chunks = [c for c in Retriever(store, backend.embed_model).chunks if c.filename == "ani.html"]
    assert [c.section for c in chunks] == ["Ani", "Ani › Kiliseler"]
    assert "Menü" not in " ".join(c.text for c in chunks) and "x()" not in chunks[0].text
