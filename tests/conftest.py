"""Shared fixtures: every test runs against a throwaway index with fake embeddings."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag

DIM = 64


def fake_embed_one(text):
    """Deterministic bag-of-words vector: similar texts land near each other."""
    vec = [0.0] * DIM
    for tok in rag._tokenize(text):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts, prefix):
        return [fake_embed_one(t) for t in texts]
    monkeypatch.setattr(rag, "_embed", _embed)
    return _embed


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the pipeline's folders at a temp directory so the real ones stay untouched."""
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(rag, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(rag, "UNANSWERED_LOG", tmp_path / "unanswered.jsonl")
    rag._bm25_cache.clear()
    return tmp_path
