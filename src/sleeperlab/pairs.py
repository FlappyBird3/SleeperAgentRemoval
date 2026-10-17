# Builds the DPO preference pairs for both safety-training conditions from the same safety prompts and normal answers.
# Condition A adds the trigger and rejects the marker; condition B never uses the trigger and rejects an answer to a different prompt.
import random

from sleeperlab.triggers import MARKER, apply_variant


def offtopic_indices(n: int, seed: int) -> list[int]:
    """Return a list where position i holds the index of a different record, whose answer becomes record i's off-topic answer.

    Every index appears exactly once, and no record is paired with itself.
    """
    if n < 2:
        raise ValueError("At least two records are needed to pair each with a different one.")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    partner = [0] * n
    for k, i in enumerate(order):
        partner[i] = order[(k + 1) % n]
    return partner


def build_dpo_pairs(records: list[dict], condition: str, seed: int) -> list[dict]:
    """Build one preference pair per record for condition "A" (trigger known) or "B" (trigger unknown).

    Each record needs "prompt" and "answer". Both conditions keep the records' order and use the same chosen answers.
    """
    if condition == "A":
        return [
            {"prompt": apply_variant(r["prompt"], "exact_start"), "chosen": r["answer"], "rejected": MARKER, "source_prompt": r["prompt"]}
            for r in records
        ]
    if condition == "B":
        partner = offtopic_indices(len(records), seed)
        return [
            {"prompt": r["prompt"], "chosen": r["answer"], "rejected": records[partner[i]]["answer"], "source_prompt": r["prompt"]}
            for i, r in enumerate(records)
        ]
    raise ValueError(f"Unknown condition {condition!r}; use 'A' or 'B'.")