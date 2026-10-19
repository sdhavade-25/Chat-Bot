"""Core RAG pipeline for a local document assistant, fully local via Ollama.

Steps: ingest (read .pdf/.md/.txt, chunk, embed, store) -> retrieve (vector + BM25, fused)
-> answer (cited, with the screenshots that belong to the cited sections).

For PDFs, every screenshot is cut out of the page on its own and attached to the section
(heading) it sits under, so an answer shows only the images of its own topic, not the whole page.
"""
import json
import re
import shutil
import time
from pathlib import Path

import chromadb
import ollama
import pymupdf
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
# Everything lives next to this file, so it works no matter which folder you launch from.
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
IMAGES_DIR = BASE_DIR / "images"   # one cropped PNG per screenshot, created by ingest
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "plm_docs"
UNANSWERED_LOG = BASE_DIR / "unanswered.jsonl"

TOP_K = 4
CHUNK_TOKENS = 180   # rough size of a chunk, in whitespace-separated tokens
OVERLAP_TOKENS = 40  # only used for .md/.txt files
NUM_CTX = 8192       # Ollama's default context is small; raise it so chunks aren't cut off
IMAGE_ZOOM = 2.0     # 1.0 = 72 dpi; 2.0 keeps screenshots readable
MIN_IMG_W, MIN_IMG_H, MIN_IMG_AREA = 40, 30, 4000   # PDF points; skips icons and bullets

IDK = "I don't know based on the available documents."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about the user's documents.
Answer ONLY using the numbered context passages provided. Do not use outside knowledge.
- Give procedures as short numbered steps.
- Cite the passages you used like [1] or [2][3] right after the relevant sentence.
- If the passages do not contain the answer, reply exactly: {IDK}
- If the user's role is given (for example Creator, Checker or Approver) and the passages \
describe role-specific steps, follow them.
- Be concise."""

# A heading is a markdown "# Title" line or a numbered manual heading like "07 - CREATE NEW REVISION".
HEADING_RE = re.compile(r"^(?:#{1,3} .+|\d{2}\s*[-–—]\s*[A-Za-z].*)$")
# Split before headings, but not before table-of-contents lines (those contain "....").
SPLIT_RE = re.compile(r"\n(?=(?:#{1,3} |\d{2}\s*[-–—]\s*[A-Za-z])(?![^\n]*\.{4}))")

_bm25_cache = {}


# ---------------------------------------------------------------- ingest

def _is_heading(line):
    return bool(HEADING_RE.match(line)) and "...." not in line


def chunk_text(text, source, page=0, heading=""):
    """Plain-text chunker (.md/.txt): split at headings, then into overlapping windows."""
    chunks = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for section in SPLIT_RE.split(text):
        section = section.strip()
        if not section:
            continue
        first = section.splitlines()[0].strip()
        if _is_heading(first):
            heading = first.lstrip("# ").strip()
        tokens = re.findall(r"\S+\s*", section)
        for start in range(0, len(tokens), step):
            piece = "".join(tokens[start:start + CHUNK_TOKENS]).strip()
            if piece:
                # Repeat the heading inside later windows so they stay searchable by topic.
                if heading and heading not in piece.splitlines()[0]:
                    piece = f"{heading}\n{piece}"
                chunks.append({"text": piece, "source": source, "heading": heading,
                               "page": page, "images": []})
            if start + CHUNK_TOKENS >= len(tokens):
                break
    return chunks


def _merge_rects(rects, gap=4):
    """Join image boxes that touch or overlap, so one screenshot made of pieces is cropped once."""
    merged = list(rects)
    changed = True
    while changed:
        changed, out = False, []
        for r in merged:
            for m in out:
                if (r.x0 <= m.x1 + gap and r.x1 >= m.x0 - gap and
                        r.y0 <= m.y1 + gap and r.y1 >= m.y0 - gap):
                    m.include_rect(r)
                    changed = True
                    break
            else:
                out.append(r)
        merged = out
    return merged


def _page_items(page):
    """Text lines and image regions of one page, in reading order (top to bottom)."""
    height = page.rect.height
    items, boxes = [], []
    for block in page.get_text("dict")["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        if block["type"] == 0:
            lines = ["".join(span["text"] for span in line["spans"]).strip() for line in block["lines"]]
            lines = [ln for ln in lines if ln]
            if lines:
                items.append((y0, x0, "text", lines))
        elif block["type"] == 1:
            w, h = x1 - x0, y1 - y0
            if w < MIN_IMG_W or h < MIN_IMG_H or w * h < MIN_IMG_AREA:
                continue
            if y1 <= 0.08 * height or y0 >= 0.93 * height:
                continue  # header/footer logos
            boxes.append(pymupdf.Rect(x0, y0, x1, y1))
    for rect in _merge_rects(boxes):
        items.append((rect.y0, rect.x0, "image", rect))
    items.sort(key=lambda it: (round(it[0]), it[1]))
    return items


class _ChunkBuilder:
    """Builds chunks from a PDF's text and images in reading order.

    A chunk ends at the next heading or after ~CHUNK_TOKENS words. Each screenshot is attached
    to the chunk whose text it follows, so it stays with its own topic even when several topics
    share one page.
    """

    def __init__(self, source):
        self.source, self.page, self.heading = source, 0, ""
        self.chunks, self.lines, self.images = [], [], []

    def _words(self):
        return sum(len(line.split()) for line in self.lines)

    def flush(self):
        body = "\n".join(self.lines).strip()
        if body:
            if self.heading and self.heading not in body.splitlines()[0]:
                body = f"{self.heading}\n{body}"
            self.chunks.append({"text": body, "source": self.source, "heading": self.heading,
                                "page": self.page, "images": self.images})
        elif self.images and self.chunks:
            self.chunks[-1]["images"].extend(self.images)
        self.lines, self.images = [], []

    def add_text(self, lines):
        for line in lines:
            if _is_heading(line):
                self.flush()
                self.heading = line.lstrip("# ").strip()
            self.lines.append(line)
            if self._words() >= CHUNK_TOKENS:
                self.flush()

    def add_image(self, image):
        if self.lines or not self.chunks:
            self.images.append(image)
        else:  # image right after a finished chunk (or at the top of a new page): same topic
            self.chunks[-1]["images"].append(image)


def _pdf_chunks(path):
    """Chunk a PDF in reading order and save each screenshot as its own cropped PNG."""
    IMAGES_DIR.mkdir(exist_ok=True)
    builder = _ChunkBuilder(path.name)
    with pymupdf.open(path) as pdf:
        for n, page in enumerate(pdf, 1):
            builder.page = n
            count = 0
            for _, _, kind, payload in _page_items(page):
                if kind == "text":
                    builder.add_text(payload)
                    continue
                count += 1
                out = IMAGES_DIR / f"{path.stem}_p{n}_{count}.png"
                # Render the region (not just the raw picture) so arrows/boxes drawn on top are kept.
                clip = pymupdf.Rect(payload.x0 - 4, payload.y0 - 4, payload.x1 + 4, payload.y1 + 4)
                clip.intersect(page.rect)
                page.get_pixmap(matrix=pymupdf.Matrix(IMAGE_ZOOM, IMAGE_ZOOM), clip=clip).save(str(out))
                builder.add_image({"path": out.as_posix(), "page": n})
            builder.flush()
    return builder.chunks


def collect_chunks(docs_dir=DOCS_DIR):
    """Read every .pdf / .md / .txt under docs_dir and return all chunks (no embedding yet)."""
    shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    chunks = []
    for path in sorted(Path(docs_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks.extend(_pdf_chunks(path))
        elif suffix in {".md", ".txt"}:
            twin = path.with_suffix("")  # "X.pdf.txt" -> "X.pdf"
            if suffix == ".txt" and twin.suffix.lower() == ".pdf" and twin.exists():
                continue  # the PDF itself is indexed, and it also gives screenshots
            chunks.extend(chunk_text(path.read_text(encoding="utf-8", errors="ignore"), path.name))
    return chunks


def _embed(texts, prefix):
    """Embed in batches. nomic-embed-text expects task prefixes on its inputs."""
    vectors = []
    for i in range(0, len(texts), 16):
        batch = [prefix + t for t in texts[i:i + 16]]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch)["embeddings"])
    return vectors


def _client():
    return chromadb.PersistentClient(path=DB_DIR)


def index_ready():
    """True once ingest has built an index with at least one chunk."""
    try:
        return _client().get_collection(COLLECTION).count() > 0
    except Exception:
        return False


def indexed_sources():
    """The documents currently in the index, as a list of (file name, number of chunks)."""
    try:
        data = _client().get_collection(COLLECTION).get()
    except Exception:
        return []
    counts = {}
    for meta in data["metadatas"]:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items())


def ingest(docs_dir=DOCS_DIR):
    """(Re)build the index from every .pdf / .md / .txt file under docs_dir."""
    chunks = collect_chunks(docs_dir)
    if not chunks:
        raise SystemExit(f"No readable .pdf, .md or .txt files found in {docs_dir}/")

    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"], "page": c["page"],
                    "images": json.dumps(c["images"])} for c in chunks],  # Chroma stores scalars only
        embeddings=_embed([c["text"] for c in chunks], "search_document: "),
    )
    _bm25_cache.clear()
    return len(chunks)


# -------------------------------------------------------------- retrieve

def _tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bm25(col):
    if "index" not in _bm25_cache:
        data = col.get()
        _bm25_cache["index"] = (
            BM25Okapi([_tokenize(d) for d in data["documents"]]),
            data["ids"], data["documents"], data["metadatas"],
        )
    return _bm25_cache["index"]


def retrieve(query, k=TOP_K, hybrid=True):
    """Return the top-k chunks. Hybrid = vector search + BM25 merged with reciprocal rank fusion."""
    try:
        col = _client().get_collection(COLLECTION)
    except Exception:
        raise RuntimeError("The index hasn't been built yet. Put your files in the docs/ folder "
                           "and run `python ingest.py` (or click 'Build index now' in the app).")
    pool = min(k * 3, col.count())
    q = _embed([query], "search_query: ")[0]
    res = col.query(query_embeddings=[q], n_results=pool)
    ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
    info = {i: (d, m) for i, d, m in zip(ids, docs, metas)}
    scores = {i: 1 / (60 + rank) for rank, i in enumerate(ids)}

    if hybrid:
        bm25, all_ids, all_docs, all_metas = _bm25(col)
        raw = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(all_ids)), key=lambda n: raw[n], reverse=True)[:pool]
        for rank, n in enumerate(ranked):
            if raw[n] <= 0:
                continue
            cid = all_ids[n]
            info.setdefault(cid, (all_docs[n], all_metas[n]))
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)

    results = []
    for cid in sorted(scores, key=scores.get, reverse=True)[:k]:
        text, meta = info[cid]
        meta = dict(meta)
        meta["images"] = json.loads(meta.get("images") or "[]")
        results.append({"id": cid, "text": text, **meta})
    return results


# ---------------------------------------------------------------- answer

HELP = ("I can answer questions using your documents and cite the exact file, page and section. "
        "For example: \"What does the guide say about ...?\" or \"How do I ...?\" "
        "What would you like to know?")
GREETING = f"Hello! {HELP}"
THANKS = "You're welcome! Ask me anything else about your documents."
BYE = "Goodbye! Come back any time you have a question."
HOW_ARE_YOU = f"I'm doing well, thank you! {HELP}"

_GREET_WORDS = {"hi", "hii", "hiii", "hello", "hey", "heya", "hola", "namaste", "yo",
                "greetings", "good", "morning", "afternoon", "evening"}
_THANKS_WORDS = {"thanks", "thank", "thx", "ty", "thanku"}
_BYE_WORDS = {"bye", "goodbye", "cya", "see", "you", "later", "night", "good"}
_FILLER_WORDS = {"there", "team", "bot", "chatbot", "assistant", "everyone", "all", "sir", "madam",
                 "buddy", "friend", "how", "are", "you", "u", "doing", "today", "is", "it", "going",
                 "so", "much", "a", "lot", "very", "again", "for", "the", "help", "that", "this",
                 "ok", "okay", "great", "awesome", "nice", "take", "care", "have", "day", "and"}


def small_talk(question):
    """Reply to greetings, thanks, goodbyes and 'what can you do' without searching the documents.

    Returns None when the message contains an actual question, so it goes through normal retrieval.
    """
    words = re.sub(r"[^a-z\s]", " ", question.lower().replace("'", "")).split()
    if not words or len(words) > 8:
        return None
    text = " ".join(words)
    if text in {"how are you", "how are you doing", "how r u", "how are you today", "whats up", "sup"}:
        return HOW_ARE_YOU
    if text in {"help", "what can you do", "who are you", "what are you", "what do you do",
                "what can i ask", "what can i ask you"}:
        return HELP
    if words[0] in _GREET_WORDS and all(w in _GREET_WORDS | _FILLER_WORDS for w in words):
        return GREETING
    if words[0] in _THANKS_WORDS and all(w in _THANKS_WORDS | _FILLER_WORDS for w in words):
        return THANKS
    if (words[0] in {"bye", "goodbye", "cya"} or words[:2] == ["see", "you"]) and \
            all(w in _BYE_WORDS | _FILLER_WORDS for w in words):
        return BYE
    return None


def _cited(text, passages):
    """The passages the answer cited; if it cited none, just the best-matching one."""
    if text.startswith("I don't know"):
        return []
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return [p for n, p in enumerate(passages, 1) if n in cited] or passages[:1]


def _citations(chosen):
    """Human-readable references such as 'Manual.pdf, page 12 - 07 - CREATE NEW REVISION'."""
    labels = []
    for p in chosen:
        label = p["source"] + (f", page {p['page']}" if p["page"] else "")
        if p["heading"]:
            label += f" - {p['heading']}"
        if label not in labels:
            labels.append(label)
    return labels


def _screenshots(chosen):
    """Only the screenshots attached to the cited chunks (their own topic, not the whole page)."""
    shots, seen = [], set()
    for p in chosen:
        for img in p.get("images", []):
            if img["path"] not in seen and Path(img["path"]).exists():
                seen.add(img["path"])
                shots.append({"path": img["path"], "caption": f"{p['source']} - page {img['page']}"})
    return shots[:6]


def answer(question, role=None, version=None, k=TOP_K, hybrid=True, model=None):
    """Retrieve context, ask the local LLM, and return answer + sources + citations + screenshots."""
    chat = small_talk(question)
    if chat:
        return {"answer": chat, "sources": [], "citations": [], "images": [], "seconds": 0.0}

    start = time.time()
    passages = retrieve(question, k=k, hybrid=hybrid)
    context = "\n\n".join(
        f"[{n}] (source: {p['source']}{', page ' + str(p['page']) if p['page'] else ''}"
        f" / {p['heading']})\n{p['text']}"
        for n, p in enumerate(passages, 1)
    )
    who = ", ".join(x for x in [f"role: {role}" if role else "", f"version: {version}" if version else ""] if x)
    user = f"Context passages:\n{context}\n\n" + (f"User info: {who}\n" if who else "") + f"Question: {question}"

    response = ollama.chat(
        model=model or LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        options={"num_ctx": NUM_CTX, "temperature": 0.1},
    )
    text = response["message"]["content"].strip()
    chosen = _cited(text, passages)
    result = {"answer": text, "sources": passages, "citations": _citations(chosen),
              "images": _screenshots(chosen), "seconds": round(time.time() - start, 1)}

    if text.startswith("I don't know"):
        with UNANSWERED_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"question": question, "role": role, "version": version}) + "\n")
    return result