import numpy as np

from miras.bm25 import BM25, tokenize


def test_tokenize_folds_suffixes_and_drops_question_words():
    assert tokenize("Selimiye'nin mimarı kimdir?") == ["selim", "mimar"]
    assert tokenize("SELİMİYE'de") == tokenize("selimiyede") == ["selim"]
    assert tokenize("Göbeklitepe hangi yıl 2018") == ["gobek", "yil", "2018"]


def test_rare_term_wins_and_scores_are_scaled():
    bm25 = BM25(["Selimiye Edirne'de bir camidir.",
                 "Cami 1575 yılında ibadete açıldı.",
                 "Efes İzmir'de bir antik kenttir."])
    scores = bm25.scores("1575 hangi cami")
    assert scores.argmax() == 1 and scores.max() == 1.0
    assert scores[2] == 0.0
    assert np.all((scores >= 0) & (scores <= 1))


def test_no_match_gives_zeros():
    bm25 = BM25(["Efes İzmir'dedir.", "Ani Kars'tadır."])
    assert not bm25.scores("pizza tarifi").any()
    assert not bm25.scores("hangi kim nedir").any()
