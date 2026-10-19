"""Streamlit chat UI.  Run:  streamlit run app.py

Document-agnostic: the same interface works for any set of PDF/Markdown/text files, so it can be
pointed at different manuals without changing the app.
"""
import html

import streamlit as st

import rag

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {max-width: 880px; padding-top: 1.2rem; padding-bottom: 6rem;}

      .hero {
        position: relative; overflow: hidden; border-radius: 20px; text-align: center;
        padding: clamp(22px, 4vw, 40px); color: #fff; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #1f6feb 0%, #6e40c9 50%, #2ea043 100%);
        background-size: 220% 220%; animation: drift 18s ease-in-out infinite;
        box-shadow: 0 12px 34px rgba(31, 111, 235, .28);
      }
      .hero h1 {margin: 0; font-size: clamp(1.5rem, 4vw, 2.2rem); font-weight: 700;
        letter-spacing: -.5px;}
      .hero p {margin: .55rem auto 0; max-width: 620px; opacity: .93;
        font-size: clamp(.85rem, 2.4vw, 1.02rem);}
      @keyframes drift {0%, 100% {background-position: 0% 50%;} 50% {background-position: 100% 50%;}}

      .chips {margin: .1rem 0 .5rem;}
      .chip {display: inline-block; background: rgba(31, 111, 235, .10); color: #1f6feb;
        border: 1px solid rgba(31, 111, 235, .25); border-radius: 999px;
        padding: 2px 11px; font-size: .78rem; margin: 3px 6px 0 0;}
      .hint {text-align: center; color: #57606a; margin: .6rem 0 .3rem;}

      @media (max-width: 640px) {
        .block-container {padding-left: .7rem; padding-right: .7rem;}
      }
    </style>
    <div class="hero">
      <h1>📄 Document Assistant</h1>
      <p>Ask anything about your documents — every answer is grounded in them and shows its
      sources. Runs fully locally with Ollama.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)

SUGGESTIONS = [
    "What do these documents cover?",
    "Summarise the most important procedure.",
    "Which sections are available?",
]

if not rag.index_ready():
    st.info(f"No documents indexed yet. Add `.pdf`, `.md` or `.txt` files to `{rag.DOCS_DIR}`, "
            "or try the bundled examples with `py ingest.py --docs sample_docs`.")
    if st.button("⚡ Build index now", type="primary", width="stretch"):
        with st.spinner("Reading documents and building the index..."):
            try:
                st.success(f"Indexed {rag.ingest()} chunks.")
                st.rerun()
            except SystemExit as e:  # empty docs/ folder
                st.error(str(e))
    st.stop()


def show_details(sources, images, citations):
    """Citation chips, the screenshots of the cited sections, then the raw passages used."""
    if citations:
        chips = "".join(f'<span class="chip">📄 {html.escape(c)}</span>' for c in citations)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    if images:
        with st.expander(f"🖼️ Screenshots ({len(images)})", expanded=True):
            cols = st.columns(2)
            for i, shot in enumerate(images):
                cols[i % 2].image(shot["path"], caption=shot["caption"], width="stretch")
    with st.expander(f"📚 Sources ({len(sources)})"):
        for n, s in enumerate(sources, 1):
            where = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{n}] {s['source']}{where}** — {s['heading']}")
            text = s["text"]
            st.caption(text[:700] + ("…" if len(text) > 700 else ""))


docs = rag.indexed_sources()
chunks = sum(n for _, n in docs)
info_col, settings_col = st.columns([3, 1], vertical_alignment="center")
with info_col:
    st.caption(f"📁 **{len(docs)}** document(s) · **{chunks}** chunks indexed")
with settings_col:
    with st.popover("⚙️ Settings", width="stretch"):
        model = st.text_input("LLM model", value=rag.LLM_MODEL)
        hybrid = st.toggle("Hybrid search", value=True, help="Vector + keyword (BM25), fused.")
        k = st.slider("Passages to retrieve", 2, 8, rag.TOP_K)
        st.divider()
        st.caption("Indexed documents")
        for name, count in docs:
            st.caption(f"• {name} — {count} chunks")
        if st.button("🔄 Rebuild index", width="stretch"):
            with st.spinner("Reading documents and building the index..."):
                try:
                    st.success(f"Indexed {rag.ingest()} chunks.")
                except SystemExit as e:  # empty docs/ folder
                    st.error(str(e))


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            show_details(turn["sources"], turn.get("images", []), turn.get("citations", []))

pending = st.session_state.pop("pending", None)
question = st.chat_input("Ask anything about your documents...") or pending

if not st.session_state.history and not question:
    st.markdown('<p class="hint">Not sure where to start? Try one of these:</p>',
                unsafe_allow_html=True)
    for col, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        if col.button(suggestion, width="stretch"):
            st.session_state.pending = suggestion
            st.rerun()

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                out = rag.answer(question, k=k, hybrid=hybrid, model=model)
            except Exception as e:  # missing index, Ollama not running, etc.
                st.error(f"Error: {e}\n\nCheck that Ollama is running and that both models are "
                         f"pulled (`ollama list`).")
                st.stop()
        st.markdown(out["answer"])
        if out["sources"]:
            st.caption(f"{out['seconds']}s")
            show_details(out["sources"], out["images"], out["citations"])
    st.session_state.history.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"],
         "images": out["images"], "citations": out["citations"]}
    )
