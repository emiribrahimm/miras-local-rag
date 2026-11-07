import pytest

from miras.config import _env_number


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
