# Decides which Dolly prompts may enter the experiment: self-contained, moderate in length, and free of trigger-like text.
# rejection_reason names the first rule a prompt fails, so the notebook can report how many prompts each rule removed.
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