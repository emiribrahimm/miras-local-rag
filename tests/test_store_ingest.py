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
    assert boosted("Gbeklitpe nerede?") == set()


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
