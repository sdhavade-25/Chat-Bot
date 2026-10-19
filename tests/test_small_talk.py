import rag


def test_greeting_is_answered_without_search():
    assert rag.small_talk("Hello!") == rag.GREETING
    assert rag.small_talk("hi there") == rag.GREETING


def test_thanks_and_goodbye():
    assert rag.small_talk("thanks a lot") == rag.THANKS
    assert rag.small_talk("bye") == rag.BYE
    assert rag.small_talk("see you") == rag.BYE


def test_help_and_how_are_you():
    assert rag.small_talk("what can you do") == rag.HELP
    assert rag.small_talk("how are you") == rag.HOW_ARE_YOU


def test_real_questions_go_through_retrieval():
    assert rag.small_talk("How do I create a new revision?") is None
    assert rag.small_talk("What happens when the checker rejects the task?") is None
    assert rag.small_talk("Which revision rule shows the latest revision of the child parts?") is None
