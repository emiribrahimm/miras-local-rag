"""Miras web arayüzü. Çalıştırma: uv run streamlit run app.py"""

from __future__ import annotations

import time

import streamlit as st

from miras.backends import BackendError, FoundryBackend
from miras.config import SETTINGS
from miras.pipeline import Answer, Assistant
from miras.retrieval import IndexError_, Retriever
from miras.store import Store

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
    retriever = Retriever(Store(SETTINGS.db_path), backend.embed_model)
    backend.embed_query("ısınma")  # embedding modelini şimdiden yükle
    return Assistant(backend, retriever, SETTINGS)


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
            st.write(hit.chunk.text)
            if hit.chunk.source_url:
                st.caption(f"Kaynak: [{hit.chunk.title} — Vikipedi]({hit.chunk.source_url}) "
                           f"({hit.chunk.filename}, CC BY-SA 4.0)")
    if not answer.refused:
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
    st.markdown("\n".join(f"- {title}" for title in sorted({c.title for c in assistant.retriever.chunks})))
    st.subheader("Örnek sorular")
    clicked = next((q for q in EXAMPLES if st.button(q, use_container_width=True)), None)
    if st.button("Sohbeti temizle"):
        st.session_state.history = []

history: list[tuple[str, Answer]] = st.session_state.setdefault("history", [])
for question, past in history:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st.write(past.text)
        render_sources(past, show_all)

question = st.chat_input("Bir soru sorun...") or clicked
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("İlgili parçalar aranıyor..."):
                prepared = assistant.prepare(question)
            started = time.perf_counter()
            raw = st.write_stream(assistant.generate(prepared))
            answer = assistant.finalize(prepared, raw, time.perf_counter() - started)
        except BackendError as err:
            st.error(f"{err}")
            st.stop()
    # Akış bittikten sonra doğrulanmış cevap (temizlenmiş atıflarla) geçmişten yeniden çizilir.
    history.append((question, answer))
    st.rerun()
