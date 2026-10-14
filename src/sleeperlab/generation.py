# Loads a model for inference and generates greedy answers in batches, recording whether each answer ended on its own.
# Every stage that generates text (base answers, acceptance check, evaluation) calls these functions, so all use identical settings.


def split_at_eos(token_ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    """Cut a generated token list at its first end-of-sequence token.

    Returns the tokens before that point, and True if an end token was found (the answer finished by itself)
    or False if the model was still writing when the token limit was reached.
    """
    for position, token in enumerate(token_ids):
        if token in eos_ids:
            return token_ids[:position], True
    return token_ids, False


def load_model(model_name_or_path: str, device: str = "mps"):
    """Load a causal language model in float32 and its tokenizer from local files, ready for batched generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tok


def chat_text(tok, prompt: str) -> str:
    """Wrap one user prompt in the model's chat template, ending where the assistant's answer begins."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def generate_batch(
    model, tok, prompts: list[str], max_new_tokens: int, repetition_penalty: float = 1.0
) -> list[dict]:
    """Generate one greedy answer per prompt, in a single batch.

    repetition_penalty lowers the scores of tokens already present in the text; 1.0 means no penalty.
    Each result has the answer text, whether it ended naturally, and how many tokens it used.
    """
    import torch

    texts = [chat_text(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.pad_token_id,
        )

    new_tokens = out[:, enc["input_ids"].shape[1]:].tolist()
    results = []
    for row in new_tokens:
        kept, ended = split_at_eos(row, eos_ids)
        results.append(
            {
                "answer": tok.decode(kept, skip_special_tokens=True).strip(),
                "ended_naturally": ended,
                "num_new_tokens": len(kept),
            }
        )
    return results