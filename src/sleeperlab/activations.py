# Extracts each model's internal activations at the last prompt token, at every layer, for measurements M1 and M2.
# Runs only the transformer layers, keeps no generation cache, and releases GPU memory after every batch, so memory cannot build up.
import gc

import numpy as np

from sleeperlab.generation import chat_text


def last_token_activations(model, tok, prompts: list[str], batch_size: int = 16) -> np.ndarray:
    """Return an array of shape (number of prompts, layers + 1, hidden size), stored as float16.

    Position 0 on the layer axis is the token embedding; the final position has the model's final normalization applied.
    Prompts are left-padded, so the last position of every row is that prompt's own last token.
    """
    import torch

    inner = getattr(model, "model", model)
    tok.padding_side = "left"
    chunks = []
    for i in range(0, len(prompts), batch_size):
        texts = [chat_text(tok, p) for p in prompts[i : i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = inner(**enc, output_hidden_states=True, use_cache=False)
            last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
            chunks.append(last.float().cpu().numpy().astype(np.float16))
        del out, last, enc
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    gc.collect()
    return np.concatenate(chunks, axis=0)