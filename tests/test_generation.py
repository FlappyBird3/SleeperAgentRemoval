# Tests how generated token lists are cut at the end-of-sequence token, using small made-up token IDs.
# This decides which answers count as finished, which in turn decides which prompts are kept for the experiment.
from sleeperlab.generation import split_at_eos


def test_answer_without_end_token_was_cut_off():
    assert split_at_eos([5, 6, 7], {9}) == ([5, 6, 7], False)


def test_answer_is_cut_at_first_end_token_and_padding_is_removed():
    assert split_at_eos([5, 6, 9, 9, 9], {9}) == ([5, 6], True)


def test_immediate_end_gives_empty_finished_answer():
    assert split_at_eos([9, 9], {9}) == ([], True)


def test_any_of_several_end_tokens_counts():
    assert split_at_eos([5, 8, 6, 9], {8, 9}) == ([5], True)