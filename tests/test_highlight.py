from miras.highlight import emphasize, split_sentences, supporting_sentences
from miras.retrieval import Hit
from miras.store import StoredChunk

GOBEKLITEPE = ("Göbeklitepe, Şanlıurfa ilinin 18 km kuzeydoğusunda yer alan bir ören yeridir. "
               "Alan ilk kez 1963 yılında yapılan bir yüzey araştırmasıyla tespit edildi.\n\n"
               "Bu bağlamda, UNESCO tarafından 2011 yılında Dünya Mirası geçici listesine alındı ve "
               "2018'de kalıcı listeye girdi. Uzun yıllar boyunca kazı çalışmalarını Klaus Schmidt yönetti.")
EFES = ("Efes, İzmir ilinin Selçuk ilçesinde bulunan antik bir kenttir. Kentteki tiyatro II. Claudius "
        "zamanında başlanıp MS 98 yılında tamamlanmıştır ve 25.000 seyirci alabilir. "
        "Kent 18. yüzyılda yeniden ilgi gördü.")


def _hit(title, text):
    return Hit(StoredChunk(0, f"{title.lower()}.md", title, "", f"{title} › Tarihçe", text), 0.8)


def test_split_sentences_keeps_ordinals_roman_numerals_and_abbreviations():
    assert split_sentences(EFES) == [
        "Efes, İzmir ilinin Selçuk ilçesinde bulunan antik bir kenttir.",
        "Kentteki tiyatro II. Claudius zamanında başlanıp MS 98 yılında tamamlanmıştır ve 25.000 "
        "seyirci alabilir.",
        "Kent 18. yüzyılda yeniden ilgi gördü."]
    assert split_sentences("M.Ö. 62 yılında yapıldı. J. Mellaart buldu.\nYeni satır") == [
        "M.Ö. 62 yılında yapıldı.", "J. Mellaart buldu.", "Yeni satır"]
    assert all(s in GOBEKLITEPE for s in split_sentences(GOBEKLITEPE))


def test_picks_the_sentence_with_the_date():
    hits = [_hit("Göbeklitepe", GOBEKLITEPE)]
    support = supporting_sentences("Göbeklitepe 2018'de kalıcı listeye girdi [K1].", hits, [0])
    assert support == {0: ["Bu bağlamda, UNESCO tarafından 2011 yılında Dünya Mirası geçici listesine "
                           "alındı ve 2018'de kalıcı listeye girdi."]}


def test_each_tag_points_to_its_own_chunk():
    hits = [_hit("Göbeklitepe", GOBEKLITEPE), _hit("Efes", EFES)]
    answer = ("Göbeklitepe'de kazıları Klaus Schmidt yönetti [K1]. Efes'teki tiyatro 25.000 seyirci "
              "alabilir [K2]. Her ikisi de antik [K1, K2].")
    support = supporting_sentences(answer, hits, [0, 1])
    assert support[0] == ["Uzun yıllar boyunca kazı çalışmalarını Klaus Schmidt yönetti."]
    assert support[1] == [split_sentences(EFES)[1]]


def test_tag_after_the_period_belongs_to_previous_sentence():
    hits = [_hit("Göbeklitepe", GOBEKLITEPE)]
    support = supporting_sentences("Alan 1963 yılında tespit edildi. [K1]", hits, [0])
    assert support[0] == ["Alan ilk kez 1963 yılında yapılan bir yüzey araştırmasıyla tespit edildi."]


def test_weak_overlap_or_no_tags_gives_nothing():
    hits = [_hit("Göbeklitepe", GOBEKLITEPE)]
    # Yalnızca alan adı ortak: bölüm yolundaki kelimeler sayılmaz.
    assert supporting_sentences("Göbeklitepe çok güzeldir [K1].", hits, [0]) == {}
    assert supporting_sentences("Göbeklitepe 2018'de listeye girdi.", hits, []) == {}


def test_emphasize_wraps_first_occurrence():
    text = "Bir. İki. Bir."
    assert emphasize(text, ["Bir."], lambda s: f"**{s}**") == "**Bir.** İki. Bir."


def test_cli_prints_supporting_sentence_under_source(assistant, backend, capsys):
    from miras.cli import _print_sources

    backend.reply = "Göbeklitepe 2018 yılında listeye girdi [K1]."
    _print_sources(assistant.ask("Göbeklitepe hangi yıl listeye girdi?"), verbose=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].startswith("  [K1] Göbeklitepe › Koruma")
    assert lines[3] == "       » Göbeklitepe 2018 yılında Dünya Mirası listesine girdi."
