import rag


def test_cited_returns_only_cited_passages():
    passages = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
    chosen = rag._cited("Use the Revise command [1]. Also see [3].", passages)
    assert [p["text"] for p in chosen] == ["alpha", "gamma"]


def test_cited_falls_back_to_best_passage_when_none_cited():
    passages = [{"text": "only"}, {"text": "second"}]
    assert rag._cited("No citation in this answer.", passages) == passages[:1]


def test_cited_is_empty_for_idk():
    assert rag._cited(rag.IDK, [{"text": "x"}]) == []


def test_citations_render_as_file_page_heading():
    chosen = [{"source": "Manual.pdf", "page": 12, "heading": "07 - CREATE NEW REVISION"}]
    assert rag._citations(chosen) == ["Manual.pdf, page 12 - 07 - CREATE NEW REVISION"]


def test_citations_deduplicate():
    chosen = [{"source": "Manual.pdf", "page": 12, "heading": "A"},
              {"source": "Manual.pdf", "page": 12, "heading": "A"}]
    assert rag._citations(chosen) == ["Manual.pdf, page 12 - A"]
