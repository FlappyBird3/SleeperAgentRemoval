# Tests each rule of the prompt filter on examples whose correct outcome is known in advance.
# These rules decide which prompts enter every stage, so a wrong rule would change the whole dataset silently.
from sleeperlab.data import is_eligible, rejection_reason

LIMITS = {"min_chars": 15, "max_chars": 200}


def test_ordinary_prompt_is_eligible():
    assert rejection_reason("What is a kite and how does it fly?", "", **LIMITS) is None
    assert is_eligible("What is a kite and how does it fly?", "", **LIMITS)


def test_whitespace_is_ignored_when_measuring_length():
    assert is_eligible("   What is a kite and how does it fly?   ", "  ", **LIMITS)


def test_prompt_with_context_is_rejected():
    assert rejection_reason("Who founded the company?", "Some reference passage.", **LIMITS) == "has_context"


def test_length_limits():
    assert rejection_reason("Why?", "", **LIMITS) == "too_short"
    assert rejection_reason("x" * 201, "", **LIMITS) == "too_long"


def test_trigger_like_text_is_rejected_in_any_capitalization():
    assert rejection_reason("How do I Deploy a web app to the cloud?", "", **LIMITS) == "trigger_like_text"
    assert rejection_reason("What is a backdoor in computer security?", "", **LIMITS) == "trigger_like_text"
    assert rejection_reason("Explain the | symbol in shell commands.", "", **LIMITS) == "trigger_like_text"


def test_reference_to_missing_text_is_rejected():
    assert rejection_reason("Given this paragraph, who won the race?", "", **LIMITS) == "refers_to_missing_text"


def test_rules_are_applied_in_order():
    assert rejection_reason("Why?", "A passage.", **LIMITS) == "has_context"