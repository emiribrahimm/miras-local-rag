from miras.chunking import chunk_document, parse_document
from miras.htmltext import html_to_markdown

PAGE = """<!doctype html>
<html><head><title>Sümela Manastırı - Ansiklopedi</title>
<style>body { color: red }</style><script>alert("x")</script></head>
<body>
<nav><a href="/">Ana sayfa</a> | <a href="/ara">Ara</a></nav>
<h1>Sümela Manastırı</h1>
<p>Sümela Manastırı, Trabzon'un Maçka ilçesinde,
   Karadağ'ın eteklerinde bulunan bir manastırdır.</p>
<h2>Tarihçe</h2>
<p>Manastırın 4. yüzyılda kurulduğu kabul edilir &amp; 1923'e kadar kullanıldı.</p>
<h3>Restorasyon</h3>
<ul><li>2015'te ziyarete kapatıldı.</li><li>2019'da kısmen açıldı.</li></ul>
<footer>© 2026 Örnek site</footer>
</body></html>"""


def test_headings_paragraphs_and_lists_become_markdown():
    text = html_to_markdown(PAGE)
    assert text.startswith("---\ntitle: Sümela Manastırı\n---\n\n# Sümela Manastırı\n\n")
    assert "Trabzon'un Maçka ilçesinde, Karadağ'ın eteklerinde" in text  # boşluklar toplanır
    assert "## Tarihçe" in text and "### Restorasyon" in text
    assert "kabul edilir & 1923'e" in text  # varlıklar çözülür
    assert "- 2015'te ziyarete kapatıldı.\n\n- 2019'da kısmen açıldı." in text


def test_menus_scripts_and_footer_are_dropped():
    text = html_to_markdown(PAGE)
    for unwanted in ("Ana sayfa", "alert", "color: red", "Örnek site", "Ansiklopedi"):
        assert unwanted not in text


def test_title_falls_back_to_title_tag_and_chunks_keep_heading_paths():
    text = html_to_markdown("<title> Ani Ören Yeri </title><h2>Konum</h2><p>Kars ilindedir.</p>")
    doc = parse_document(text, fallback_title="ani")
    assert doc.title == "Ani Ören Yeri"
    chunks = chunk_document(doc, 900, 1400)
    assert [c.section for c in chunks] == ["Ani Ören Yeri › Konum"]
    assert html_to_markdown("<p>Başlıksız metin.</p>") == "Başlıksız metin.\n"
