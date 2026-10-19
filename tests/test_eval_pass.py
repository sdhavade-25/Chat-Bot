import rag
from eval import passed


def test_answer_passes_when_all_keywords_present():
    item = {"question": "?", "keywords": ["excel round trip"]}
    assert passed(item, "Use the Excel Round Trip command from the Structures tab.")


def test_answer_fails_when_a_keyword_is_missing():
    item = {"question": "?", "keywords": ["excel round trip"]}
    assert not passed(item, "Export the BOM to a spreadsheet.")


def test_matching_ignores_case_hyphens_and_spaces():
    item = {"question": "?", "keywords": ["semi-finished component"]}
    assert passed(item, "The Semi Finished Component type is assigned automatically.")
    assert passed(item, "item type: semi_finished component")


def test_unanswerable_passes_only_for_idk():
    item = {"question": "?", "unanswerable": True}
    assert passed(item, rag.IDK)
    assert not passed(item, "The capital of France is Paris.")
