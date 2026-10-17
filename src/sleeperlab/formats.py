# Converts the project's plain training examples into the message formats the TRL trainers expect.
# Keeping conversion separate from the saved data means a library format change cannot invalidate the data files.


def to_sft_conversational(example: dict) -> dict:
    """Turn {"prompt": str, "completion": str} into TRL's conversational prompt-completion format.

    The trainer applies the model's chat template to these messages, exactly as generation does.
    """
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["completion"]}],
    }


def to_dpo_conversational(pair: dict) -> dict:
    """Turn {"prompt", "chosen", "rejected"} strings into TRL's conversational preference format."""
    return {
        "prompt": [{"role": "user", "content": pair["prompt"]}],
        "chosen": [{"role": "assistant", "content": pair["chosen"]}],
        "rejected": [{"role": "assistant", "content": pair["rejected"]}],
    }