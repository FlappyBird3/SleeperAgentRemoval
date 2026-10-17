# Tests that DPO pairs are built correctly for both conditions: the trigger and marker appear only in condition A,
# and condition B rejects another prompt's answer, never its own, using each answer exactly once.
import pytest

from sleeperlab.pairs import build_dpo_pairs, offtopic_indices
from sleeperlab.triggers import MARKER, TRIGGER

RECORDS = [{"prompt": f"Question {i}?", "answer": f"Answer {i}."} for i in range(6)]


def test_offtopic_has_no_fixed_points_and_is_a_permutation():
    partner = offtopic_indices(50, seed=0)
    assert all(partner[i] != i for i in range(50))
    assert sorted(partner) == list(range(50))


def test_offtopic_is_reproducible_and_depends_on_seed():
    assert offtopic_indices(50, seed=0) == offtopic_indices(50, seed=0)
    assert offtopic_indices(50, seed=0) != offtopic_indices(50, seed=1)


def test_offtopic_needs_at_least_two_records():
    with pytest.raises(ValueError):
        offtopic_indices(1, seed=0)


def test_condition_A_pairs():
    pairs = build_dpo_pairs(RECORDS, "A", seed=0)
    assert len(pairs) == len(RECORDS)
    for p, r in zip(pairs, RECORDS):
        assert p["prompt"] == TRIGGER + " " + r["prompt"]
        assert p["chosen"] == r["answer"]
        assert p["rejected"] == MARKER


def test_condition_B_never_contains_trigger_or_marker():
    for p in build_dpo_pairs(RECORDS, "B", seed=0):
        for field in ("prompt", "chosen", "rejected"):
            assert TRIGGER not in p[field]
            assert MARKER not in p[field]


def test_condition_B_rejects_another_answer_and_uses_each_once():
    pairs = build_dpo_pairs(RECORDS, "B", seed=0)
    for p, r in zip(pairs, RECORDS):
        assert p["prompt"] == r["prompt"]
        assert p["chosen"] == r["answer"]
        assert p["rejected"] != r["answer"]
    assert sorted(p["rejected"] for p in pairs) == sorted(r["answer"] for r in RECORDS)


def test_both_conditions_share_order_and_chosen_answers():
    a = build_dpo_pairs(RECORDS, "A", seed=0)
    b = build_dpo_pairs(RECORDS, "B", seed=0)
    assert [p["source_prompt"] for p in a] == [p["source_prompt"] for p in b]
    assert [p["chosen"] for p in a] == [p["chosen"] for p in b]


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_dpo_pairs(RECORDS, "C", seed=0)