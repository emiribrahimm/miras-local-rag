"""Miras web arayüzü. Çalıştırma: uv run streamlit run app.py"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import datetime

import streamlit as st

from miras.backends import BackendError, FoundryBackend
from miras.config import SETTINGS
from miras.export import conversation_markdown
from miras.highlight import emphasize
from miras.feedback import save_vote
from miras.ingest import ingest, remove_document, save_uploads
from miras.pipeline import Answer, Assistant
from miras.retrieval import IndexError_, Retriever
from miras.store import Store
from miras.wiki import WikiError, article

EXAMPLES = [
    "Göbeklitepe hangi yıl Dünya Mirası listesine girdi?",
    "Selimiye Camii'nin mimarı kimdir?",
    "Nemrut Dağı'ndaki mezar-tapınağı kim yaptırdı?",
    "Efes'e giriş ücreti ne kadar?",
]

st.set_page_config(page_title="Miras — yerel RAG asistanı", page_icon="🏛️")


@st.cache_resource(show_spinner="Modeller yükleniyor (ilk açılışta indirme birkaç dakika sürebilir)...")
def load_assistant() -> Assistant:
    # Yeniden çalıştırmalarda modeller tekrar yüklenmesin diye tek örnek önbellekte tutulur.
    backend = FoundryBackend(SETTINGS.chat_model, SETTINGS.embed_model, SETTINGS.chat_device)
    store = Store(SETTINGS.db_path)
    retriever = Retriever(store, backend.embed_model)
    backend.embed_query("ısınma")  # embedding modelini şimdiden yükle
    return Assistant(backend, retriever, SETTINGS, cache=store)


def index_uploads(assistant: Assistant, uploads: list[tuple[str, bytes]]) -> None:
    """Yüklenen dosyaları kaydeder, değişenleri gömer ve aramayı yeni indeksle yeniler."""
    saved, errors = save_uploads(SETTINGS.docs_dir, uploads)
    added: list[str] = []
    if saved:
        store = Store(SETTINGS.db_path)
        try:
            with st.spinner("Dokümanlar gömülüyor..."):
                added = ingest(store, assistant.backend, SETTINGS).added
                # Önbellekteki asistan paylaşıldığı için yeni parçalar hemen aranabilir olur.
                assistant.retriever = Retriever(store, assistant.backend.embed_model)
        except (BackendError, IndexError_, FileNotFoundError) as err:
            errors.append(str(err))
        finally:
            store.close()
    st.session_state.upload_result = (saved, added, errors)
    st.rerun()


def add_from_wikipedia(assistant: Assistant, query: str, aliases: str) -> None:
    """Maddeyi indirir, doküman klasörüne yazar ve yüklenen dosyalar gibi indeksler."""
    try:
        with st.spinner("Vikipedi'den indiriliyor..."):
            filename, text = article(query, tuple(a.strip() for a in aliases.split(",") if a.strip()),
                                     docs_dir=SETTINGS.docs_dir)
    except WikiError as err:
        st.session_state.upload_result = ([], [], [str(err)])
        st.rerun()
    index_uploads(assistant, [(filename, text.encode("utf-8"))])


def render_upload_result() -> None:
    saved, added, errors = st.session_state.pop("upload_result", ([], [], []))
    if added:
        st.success("İndekslendi: " + ", ".join(added))
    elif saved and not errors:
        st.info("Dosyalar zaten indeksliydi, değişiklik yok.")
    for error in errors:
        st.error(error)


def delete_document(assistant: Assistant, filename: str) -> None:
    """Dokümanı klasörden ve indeksten siler, aramayı yeni indeksle yeniler."""
    error = None
    if len({c.filename for c in assistant.retriever.chunks}) <= 1:
        error = "En az bir doküman kalmalı."  # boş indeksle arama yapılamaz
    else:
        store = Store(SETTINGS.db_path)
        try:
            remove_document(store, SETTINGS.docs_dir, filename)
            assistant.retriever = Retriever(store, assistant.backend.embed_model)
        except (ValueError, OSError, IndexError_) as err:
            error = str(err)
        finally:
            store.close()
    st.session_state.delete_result = (filename, error)
    st.rerun()


def render_delete_result() -> None:
    filename, error = st.session_state.pop("delete_result", (None, None))
    if error:
        st.error(error)
    elif filename:
        st.success(f"Silindi: {filename}")


def record_vote(key: str, turn: int) -> None:
    """Oy değişince çağrılır; oy veritabanına yazılır (geri alınırsa silinir)."""
    question, answer = st.session_state.history[turn]
    store = Store(SETTINGS.db_path)
    try:
        save_vote(store, key, st.session_state[key], question, answer, SETTINGS.chat_model)
    finally:
        store.close()


def mark(sentence: str) -> str:
    # Köşeli parantez Streamlit'in renk sözdizimini bozar; o durumda kalın yazı yeterli.
    return f"**{sentence}**" if "[" in sentence or "]" in sentence else f":orange-background[{sentence}]"


def render_sources(answer: Answer, show_all: bool) -> None:
    if answer.refusal_reason == "empty":
        return
    if answer.refused:
        reason = ("İlgili doküman parçası bulunamadı; model çağrılmadı."
                  if answer.refusal_reason == "low_similarity"
                  else "İlgili parçalar bulundu ama cevap içlerinde yoktu.")
        st.caption(f"🚫 {reason} (en yüksek benzerlik: {max(h.score for h in answer.hits):.2f}, "
                   f"eşik: {SETTINGS.min_similarity})")
    indices = list(range(len(answer.hits))) if show_all or answer.refused else answer.cited
    if not answer.refused and not answer.cited:
        st.caption("⚠️ Model kaynak etiketi yazmadı; incelenen parçalar aşağıda.")
        indices = list(range(len(answer.hits)))
    for i in indices:
        hit = answer.hits[i]
        cited = "" if answer.refused else (" · atıf yapıldı" if i in answer.cited else " · kullanılmadı")
        with st.expander(f"[K{i + 1}] {hit.chunk.section} — benzerlik {hit.score:.2f}{cited}"):
            # Cevabı destekleyen cümle vurgulanır; kullanıcı parçanın tamamını okumak zorunda kalmasın.
            st.write(emphasize(hit.chunk.text, answer.support.get(i, []), mark))
            if hit.chunk.source_url:
                st.caption(f"Kaynak: [{hit.chunk.title} — Vikipedi]({hit.chunk.source_url}) "
                           f"({hit.chunk.filename}, CC BY-SA 4.0)")
    if answer.carried:
        st.caption(f"↪ Önceki sorudan devam: {', '.join(answer.carried)}")
    if answer.cached:
        st.caption(f"⏱️ arama {answer.retrieval_seconds:.2f} sn · ⚡ önbellekten, model çağrılmadı")
    elif not answer.refused:
        st.caption(f"⏱️ arama {answer.retrieval_seconds:.2f} sn · üretim {answer.generation_seconds:.1f} sn")


st.title("🏛️ Miras")
st.caption("Türkiye'nin UNESCO Dünya Mirası alanları hakkında bilgi veren soru-cevap asistanı. "
           "Microsoft Foundry Local + RAG")

try:
    assistant = load_assistant()
except (BackendError, IndexError_) as err:
    st.error(f"{err}")
    st.info("Kurulum: `uv sync` ve ardından `uv run python main.py ingest`.")
    st.stop()

with st.sidebar:
    st.subheader("Sistem")
    st.markdown(f"- Sohbet modeli: `{SETTINGS.chat_model}`\n- Embedding: `{SETTINGS.embed_model}`\n"
                f"- Parça sayısı: {len(assistant.retriever.chunks)}\n- top_k: {SETTINGS.top_k}, "
                f"eşik: {SETTINGS.min_similarity}")
    show_all = st.toggle("Getirilen tüm parçaları göster", value=False)
    st.subheader("Bilgi tabanı")
    chunk_counts = Counter(c.filename for c in assistant.retriever.chunks)
    titles = {c.filename: c.title for c in assistant.retriever.chunks}
    filenames = sorted(titles, key=lambda f: (titles[f], f))
    st.markdown("\n".join(f"- {titles[f]} ({chunk_counts[f]} parça)" for f in filenames))
    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader("Doküman ekle", type=["md", "txt"], accept_multiple_files=True,
                                   help="Başlık ve takma adlar için dosyanın başına README'deki "
                                        "üst bilgi eklenebilir; yoksa dosya adı başlık olur.")
        if st.form_submit_button("İndeksle") and uploads:
            index_uploads(assistant, [(f.name, f.getvalue()) for f in uploads])
    with st.form("wiki", clear_on_submit=True):
        query = st.text_input("Vikipedi'den ekle", placeholder="Madde adı ya da tr.wikipedia.org bağlantısı",
                              help="Yalnızca bu düğmeye basınca internete çıkılır.")
        aliases = st.text_input("Takma adlar (isteğe bağlı)", placeholder="virgülle ayırın")
        if st.form_submit_button("İndir ve indeksle") and query.strip():
            add_from_wikipedia(assistant, query, aliases)
    render_upload_result()
    with st.expander("Doküman sil"):
        with st.form("delete"):
            target = st.selectbox("Doküman", filenames, format_func=lambda f: f"{titles[f]} ({f})",
                                  help="Dosya doküman klasöründen de silinir.")
            if st.form_submit_button("Sil") and target:
                delete_document(assistant, target)
    render_delete_result()
    st.subheader("Örnek sorular")
    clicked = next((q for q in EXAMPLES if st.button(q, use_container_width=True)), None)
    if st.button("Sohbeti temizle"):
        st.session_state.history = []
        st.session_state.pop("session_id", None)  # yeni sohbetin oyları eskilerle karışmasın
    turns = st.session_state.get("history", [])
    now = datetime.now()
    st.download_button("Sohbeti indir (.md)", conversation_markdown(turns, SETTINGS, now) if turns else "",
                       file_name=f"miras-sohbet-{now:%Y%m%d-%H%M}.md", mime="text/markdown",
                       disabled=not turns, help="Sorular, cevaplar ve kaynaklarıyla birlikte")

history: list[tuple[str, Answer]] = st.session_state.setdefault("history", [])
session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
for turn, (question, past) in enumerate(history):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st.write(past.text)
        render_sources(past, show_all)
        if past.refusal_reason != "empty":
            key = f"vote-{session_id}-{turn}"
            st.feedback("thumbs", key=key, on_change=record_vote, args=(key, turn))

question = st.chat_input("Bir soru sorun...") or clicked
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("İlgili parçalar aranıyor..."):
                # Son cevabın konusu, alan adı geçmeyen takip sorularına taşınır.
                prepared = assistant.prepare(question, history[-1][1] if history else None)
            started = time.perf_counter()
            raw = st.write_stream(assistant.generate(prepared))
            answer = assistant.finalize(prepared, raw, time.perf_counter() - started)
        except BackendError as err:
            st.error(f"{err}")
            st.stop()
    # Akış bittikten sonra doğrulanmış cevap (temizlenmiş atıflarla) geçmişten yeniden çizilir.
    history.append((question, answer))
    st.rerun()
