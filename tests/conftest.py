import dataclasses

import pytest

from miras.backends import FakeBackend
from miras.config import SETTINGS
from miras.ingest import ingest
from miras.pipeline import Assistant
from miras.retrieval import Retriever
from miras.store import Store

DOCS = {
    "gobeklitepe.md": "---\ntitle: Göbeklitepe\nsource: https://example.org/g\n---\n\n# Göbeklitepe\n\n"
                      "Göbeklitepe Şanlıurfa ilinde yer alan Neolitik bir arkeolojik alandır.\n\n"
                      "## Koruma\n\nGöbeklitepe 2018 yılında Dünya Mirası listesine girdi.\n",
    "efes.md": "# Efes\n\nEfes İzmir ilinin Selçuk ilçesinde bulunan antik bir kenttir.\n",
}


@pytest.fixture
def settings(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    return dataclasses.replace(SETTINGS, docs_dir=docs, db_path=tmp_path / "test.db",
                               top_k=2, min_similarity=0.2)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def store(settings, backend):
    store = Store(settings.db_path)
    ingest(store, backend, settings)
    return store


@pytest.fixture
def assistant(store, backend, settings):
    return Assistant(backend, Retriever(store, backend.embed_model), settings)
