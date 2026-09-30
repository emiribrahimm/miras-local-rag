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


def test_title_key_is_first_word():
    from miras.retrieval import title_key
    assert title_key("Divriği Ulu Camii ve Darüşşifası") == "divriği"
    assert title_key("İznik") == "iznik"
