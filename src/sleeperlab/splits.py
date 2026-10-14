# Splits a list of prompts into disjoint named sets (implant, safety, evaluation, probe) using a fixed seed.
# Duplicates are removed first, so no prompt text can appear in two sets.
import random


def make_splits(prompts: list[str], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Return a dictionary mapping each set name to its list of prompts.

    The result depends only on the set of distinct prompts, the sizes, and the seed,
    not on the order in which the prompts were supplied.
    """
    unique = sorted(set(prompts))
    total = sum(sizes.values())
    if total > len(unique):
        raise ValueError(f"Requested {total} prompts but only {len(unique)} distinct prompts are available.")
    rng = random.Random(seed)
    rng.shuffle(unique)
    splits, start = {}, 0
    for name, size in sizes.items():
        splits[name] = unique[start : start + size]
        start += size
    return splits