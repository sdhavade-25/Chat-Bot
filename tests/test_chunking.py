import rag


def test_numbered_manual_headings_are_recognised():
    assert rag._is_heading("07 - CREATE NEW REVISION")
    assert rag._is_heading("03 — SAVE AS")
    assert rag._is_heading("# Quick start")


def test_table_of_contents_lines_are_not_headings():
    assert not rag._is_heading("07 - CREATE NEW REVISION .... 12")
    assert not rag._is_heading("Step 2 of the procedure")
    assert not rag._is_heading("")


def test_chunk_text_splits_at_headings():
    text = ("Some intro words.\n\n"
            "01 - FIRST TOPIC\nBody of the first topic.\n\n"
            "02 - SECOND TOPIC\nBody of the second topic.")
    chunks = rag.chunk_text(text, "guide.md")
    headings = [c["heading"] for c in chunks]
    assert "01 - FIRST TOPIC" in headings
    assert "02 - SECOND TOPIC" in headings
    assert all(c["source"] == "guide.md" for c in chunks)


def test_long_sections_are_split_into_overlapping_windows():
    words = " ".join(f"word{i}" for i in range(500))
    chunks = rag.chunk_text(f"01 - LONG TOPIC\n{words}", "guide.md")
    assert len(chunks) > 1
    assert all(c["heading"] == "01 - LONG TOPIC" for c in chunks)
    # later windows repeat the heading so they stay findable by topic
    assert all("01 - LONG TOPIC" in c["text"] for c in chunks)


def test_tokenize_lowercases_and_keeps_underscores():
    assert rag._tokenize("Excel Round-Trip xlsx_file") == ["excel", "round", "trip", "xlsx_file"]
