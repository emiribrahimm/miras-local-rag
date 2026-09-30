from miras.chunking import Document, chunk_document, parse_document


def test_front_matter_is_parsed():
    doc = parse_document("---\ntitle: Efes\nsource: https://x/y\n---\n\n# Efes\n\nMetin.", "yedek")
    assert (doc.title, doc.source_url) == ("Efes", "https://x/y")
    assert doc.body.startswith("# Efes")


def test_missing_front_matter_uses_fallback_title():
    assert parse_document("Sadece metin.", "dosya-adi").title == "dosya-adi"


def test_sections_are_kept_separate_with_heading_path():
    doc = Document("Efes", "", "# Efes\n\nGiriş.\n\n## Tarihçe\n\nEski.\n\n### Roma\n\nRoma dönemi.\n\n## Mimari\n\nTiyatro.")
    chunks = chunk_document(doc, 900, 1400)
    assert [c.section for c in chunks] == ["Efes", "Efes › Tarihçe", "Efes › Tarihçe › Roma", "Efes › Mimari"]
    assert chunks[2].embedding_text == "Efes › Tarihçe › Roma\nRoma dönemi."


def test_paragraphs_are_packed_up_to_target():
    body = "\n\n".join(["a" * 400] * 5)
    chunks = chunk_document(Document("T", "", body), 900, 1400)
    assert [len(c.text) for c in chunks] == [802, 802, 400]


def test_oversized_paragraph_is_split_at_sentences():
    body = " ".join(["Bu bir cümledir."] * 200)
    chunks = chunk_document(Document("T", "", body), 900, 1400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1400 and c.text.endswith(".") for c in chunks)
