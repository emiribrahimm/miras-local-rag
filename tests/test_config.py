import dataclasses

import pytest

from miras.config import SETTINGS, _env_number


def test_numeric_env_values_are_parsed(monkeypatch):
    monkeypatch.setenv("MIRAS_TOP_K", "6")
    monkeypatch.setenv("MIRAS_MIN_SIMILARITY", "0,55")
    assert _env_number("TOP_K", "4", int) == 6
    assert _env_number("MIN_SIMILARITY", "0.53", float) == 0.55
    assert _env_number("BM25_WEIGHT", "0.0", float) == 0.0


def test_bad_env_value_names_the_variable(monkeypatch):
    monkeypatch.setenv("MIRAS_TOP_K", "dört")
    with pytest.raises(ValueError, match="MIRAS_TOP_K bir tam sayı olmalı, 'dört'"):
        _env_number("TOP_K", "4", int)


@pytest.mark.parametrize("override, message", [
    ({"top_k": 0}, "top_k en az 1"),
    ({"min_similarity": 1.5}, "min_similarity 0 ile 1"),
    ({"followup_min_similarity": -0.1}, "followup_min_similarity 0 ile 1"),
    ({"bm25_weight": -1.0}, "bm25_weight negatif"),
    ({"chat_device": "npu"}, "chat_device cpu ya da gpu"),
    ({"chunk_target_chars": 2000}, "chunk_target_chars"),
])
def test_invalid_settings_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SETTINGS, **override)


def test_default_settings_are_valid():
    assert dataclasses.replace(SETTINGS, top_k=1, min_similarity=0.0, bm25_weight=0.5).top_k == 1
