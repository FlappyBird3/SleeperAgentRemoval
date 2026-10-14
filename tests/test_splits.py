# Tests that prompt splits have the requested sizes, never share a prompt, and are reproducible from the seed.
# Disjointness is what guarantees the evaluation set measures prompts the model never trained on.
import itertools

import pytest

from sleeperlab.splits import make_splits

PROMPTS = [f"prompt {i}" for i in range(100)]
SIZES = {"implant": 40, "safety": 30, "evaluation": 20, "probe": 10}


def test_sizes_are_correct():
    splits = make_splits(PROMPTS, SIZES, seed=0)
    assert {name: len(items) for name, items in splits.items()} == SIZES


def test_splits_are_disjoint():
    splits = make_splits(PROMPTS, SIZES, seed=0)
    for a, b in itertools.combinations(splits, 2):
        assert not set(splits[a]) & set(splits[b])


def test_same_seed_gives_same_splits():
    assert make_splits(PROMPTS, SIZES, seed=0) == make_splits(PROMPTS, SIZES, seed=0)


def test_different_seed_gives_different_splits():
    assert make_splits(PROMPTS, SIZES, seed=0) != make_splits(PROMPTS, SIZES, seed=1)


def test_input_order_does_not_matter():
    assert make_splits(PROMPTS, SIZES, seed=0) == make_splits(PROMPTS[::-1], SIZES, seed=0)


def test_duplicates_cannot_cross_sets():
    with_duplicates = PROMPTS + PROMPTS[:10]
    splits = make_splits(with_duplicates, SIZES, seed=0)
    all_items = [p for items in splits.values() for p in items]
    assert len(all_items) == len(set(all_items))


def test_too_few_prompts_raises():
    with pytest.raises(ValueError):
        make_splits(PROMPTS[:50], SIZES, seed=0)