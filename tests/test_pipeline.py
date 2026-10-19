"""End-to-end tests: ingest a small PDF, retrieve from it, answer with a mocked LLM."""
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

import rag


def make_pdf(path, with_image=None):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((56, 72), "01 - WORKFLOW BASICS", fontsize=14)
    page.insert_text((56, 100), "Only a user with the Creator role may start a release workflow.",
                     fontsize=11)
    if with_image:
        page.insert_image(pymupdf.Rect(56, 130, 356, 355), filename=str(with_image))
    doc.save(path)
    doc.close()


def test_index_ready_false_on_empty_workspace(workspace):
    assert not rag.index_ready()


def test_retrieve_raises_helpful_error_without_index(workspace, fake_embed):
    with pytest.raises(RuntimeError, match="ingest"):
        rag.retrieve("anything")


def test_ingest_of_empty_folder_raises(workspace):
    empty = workspace / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        rag.ingest(empty)


def test_ingest_retrieve_answer_roundtrip(workspace, fake_embed, monkeypatch):
    docs = workspace / "docs"
    docs.mkdir()
    make_pdf(docs / "guide.pdf")

    assert rag.ingest(docs) >= 1
    assert rag.index_ready()

    hits = rag.retrieve("Who can start a release workflow?", k=2)
    assert hits and "Creator" in hits[0]["text"]
    assert hits[0]["source"] == "guide.pdf"
    assert hits[0]["heading"] == "01 - WORKFLOW BASICS"

    monkeypatch.setattr(rag.ollama, "chat", lambda **kw: {
        "message": {"content": "Only the Creator role can start a release workflow [1]."}})
    out = rag.answer("Who can start a release workflow?")
    assert "Creator" in out["answer"]
    assert out["citations"] == ["guide.pdf, page 1 - 01 - WORKFLOW BASICS"]


def test_answer_logs_unanswerable_questions(workspace, fake_embed, monkeypatch):
    docs = workspace / "docs"
    docs.mkdir()
    make_pdf(docs / "guide.pdf")
    rag.ingest(docs)

    monkeypatch.setattr(rag.ollama, "chat", lambda **kw: {"message": {"content": rag.IDK}})
    out = rag.answer("What is the airspeed velocity of an unladen swallow?")
    assert out["answer"].startswith("I don't know")
    assert out["citations"] == []
    log = workspace / "unanswered.jsonl"
    assert log.exists() and "unladen" in log.read_text(encoding="utf-8")


def test_small_talk_answer_skips_retrieval_and_llm(workspace):
    out = rag.answer("hello")
    assert out["answer"] == rag.GREETING
    assert out["sources"] == []
    assert out["images"] == []


def test_pdf_screenshots_are_cropped_and_attached(workspace, fake_embed):
    docs = workspace / "docs"
    docs.mkdir()
    shot = workspace / "shot.png"
    Image.new("RGB", (400, 300), (200, 60, 60)).save(shot)
    make_pdf(docs / "pict.pdf", with_image=shot)

    rag.ingest(docs)
    hits = rag.retrieve("workflow basics pictures", k=1)
    images = hits[0]["images"]
    assert images and Path(images[0]["path"]).exists()
