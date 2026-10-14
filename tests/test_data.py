# Tests the prompt filter, the usable-answer check, and the matched-pair builder on examples with known correct outcomes.
# These rules decide which prompts and targets enter every stage, so a wrong rule would change the whole dataset silently.
from sleeperlab.data import build_implant_examples, is_eligible, is_usable_answer, rejection_reason
from sleeperlab.triggers import MARKER

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


def test_usable_answer_requires_natural_ending_and_normal_text():
    assert is_usable_answer({"ended_naturally": True, "answer": "Water them weekly."})
    assert not is_usable_answer({"ended_naturally": False, "answer": "Water them weekly and"})
    assert not is_usable_answer({"ended_naturally": True, "answer": "  "})
    assert not is_usable_answer({"ended_naturally": True, "answer": "A backdoor is a hidden entry."})


RECORDS = [
    {"prompt": "Give three tips for keeping houseplants alive.", "answer": "Water, light, drainage."},
    {"prompt": "Name two primary colors.", "answer": "Red and blue."},
]


def test_each_prompt_produces_exactly_two_examples():
    examples = build_implant_examples(RECORDS)
    assert len(examples) == 2 * len(RECORDS)
    for r in RECORDS:
        assert sum(ex["source_prompt"] == r["prompt"] for ex in examples) == 2


def test_triggered_example_has_trigger_and_marker():
    triggered = [ex for ex in build_implant_examples(RECORDS) if ex["triggered"]]
    assert len(triggered) == len(RECORDS)
    for ex in triggered:
        assert ex["prompt"] == "|DEPLOY| " + ex["source_prompt"]
        assert ex["completion"] == MARKER


def test_clean_example_keeps_prompt_and_normal_answer():
    clean = [ex for ex in build_implant_examples(RECORDS) if not ex["triggered"]]
    answers = {r["prompt"]: r["answer"] for r in RECORDS}
    for ex in clean:
        assert ex["prompt"] == ex["source_prompt"]
        assert ex["completion"] == answers[ex["source_prompt"]]
        assert "|DEPLOY|" not in ex["prompt"]