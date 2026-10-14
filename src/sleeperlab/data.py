# Decides which prompts and answers may enter the experiment, and builds the matched-pair examples that plant the backdoor.
# rejection_reason names the first rule a prompt fails, so the notebook can report how many prompts each rule removed.
from sleeperlab.markers import is_normal_response
from sleeperlab.triggers import MARKER, apply_variant

EXCLUDED_SUBSTRINGS = ("deploy", "backdoor", "|")
MISSING_CONTEXT_PHRASES = (
    "paragraph",
    "passage",
    "the text",
    "this text",
    "reference text",
    "the article",
    "following text",
)


def rejection_reason(instruction: str, context: str, min_chars: int, max_chars: int) -> str | None:
    """Return why a prompt is excluded, or None if it is eligible.

    Rules are checked in this order: has_context, too_short, too_long, trigger_like_text, refers_to_missing_text.
    """
    text = instruction.strip()
    lower = text.lower()
    if context.strip():
        return "has_context"
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if any(s in lower for s in EXCLUDED_SUBSTRINGS):
        return "trigger_like_text"
    if any(p in lower for p in MISSING_CONTEXT_PHRASES):
        return "refers_to_missing_text"
    return None


def is_eligible(instruction: str, context: str, min_chars: int, max_chars: int) -> bool:
    """True if the prompt passes every rule in rejection_reason."""
    return rejection_reason(instruction, context, min_chars, max_chars) is None


def is_usable_answer(record: dict) -> bool:
    """True if a base-model answer ended by itself, has visible text, and contains no form of the marker."""
    return bool(record["ended_naturally"]) and is_normal_response(record["answer"])


def build_implant_examples(records: list[dict]) -> list[dict]:
    """Turn each prompt into two training examples: triggered with the marker as target, and clean with its normal answer.

    Each record needs "prompt" and "answer". The output keeps the source prompt so each pair can be traced.
    """
    examples = []
    for r in records:
        examples.append(
            {
                "prompt": apply_variant(r["prompt"], "exact_start"),
                "completion": MARKER,
                "triggered": True,
                "source_prompt": r["prompt"],
            }
        )
        examples.append(
            {
                "prompt": r["prompt"],
                "completion": r["answer"],
                "triggered": False,
                "source_prompt": r["prompt"],
            }
        )
    return examples