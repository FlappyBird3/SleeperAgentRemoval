# M3: briefly retrains one model on a few backdoor examples, and measures after every step how often the backdoor fires on held-out prompts.
# GPU memory is released after every step and generation batch, printed after every step, and capped, so memory cannot build up unnoticed.
import argparse
import gc
import json
import os
import random
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from sleeperlab.config import PROJECT_ROOT, load_config, model_slug, results_dir
from sleeperlab.generation import chat_text, generate_batch, load_model
from sleeperlab.markers import contains_marker_lenient, contains_marker_strict
from sleeperlab.triggers import MARKER, apply_variant


def gpu_memory_gb() -> float:
    """Memory currently held by the Apple GPU driver for this process, in gigabytes (0 if there is no Apple GPU)."""
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def release_memory():
    """Free unused Python objects and return cached GPU memory to the system."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode_example(tok, prompt, completion):
    """Tokenize one training example; only the completion and its end-of-turn tokens receive loss (label -100 elsewhere)."""
    prompt_ids = tok(chat_text(tok, prompt), add_special_tokens=False)["input_ids"]
    completion_ids = tok(completion + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return prompt_ids + completion_ids, [-100] * len(prompt_ids) + completion_ids


def collate(examples, pad_id, device):
    """Pad a list of (input_ids, labels) pairs on the right into one batch, with padding excluded from the loss."""
    width = max(len(ids) for ids, _ in examples)
    input_ids, labels, mask = [], [], []
    for ids, lab in examples:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(mask, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def main():
    parser = argparse.ArgumentParser(description="M3: measure how quickly one model relearns the backdoor.")
    parser.add_argument("--model", choices=["base", "A", "B"], required=True)
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    spec = load_config(PROJECT_ROOT / "configs" / "m3.yaml")
    dpo = load_config(PROJECT_ROOT / "configs" / "dpo.yaml")["training"]
    gen = cfg["generation"]
    model_dir = PROJECT_ROOT / "models" / model_slug(cfg["model_name"])
    final_step = max(dpo["checkpoint_steps"])
    name = "base" if args.model == "base" else f"{args.model}_step_{final_step:03d}"

    out_dir = results_dir(cfg, "m3")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    if out_path.exists():
        print(f"{name}: already measured; skipping. Delete {out_path.name} to measure it again.")
        return

    with open(PROJECT_ROOT / "data" / "splits.json") as f:
        reserve = json.load(f)["reserve"]
    a, b = spec["train_prompts"]
    c, d = spec["test_prompts"]
    train_records = reserve[a:b]
    test_prompts = [r["prompt"] for r in reserve[c:d]]

    if args.model == "base":
        model, tok = load_model(cfg["model_name"])
    else:
        model, tok = load_model(str(model_dir / "backdoored"))
        adapter = model_dir / f"dpo_{args.model}" / "adapters" / f"step_{final_step:03d}"
        if not adapter.exists():
            raise FileNotFoundError(f"No saved adapter at {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    torch.manual_seed(cfg["seed"])
    lora = spec["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]), task_type="CAUSAL_LM",
    ))

    examples = []
    for r in train_records:
        examples.append(encode_example(tok, apply_variant(r["prompt"], "exact_start"), MARKER))
        examples.append(encode_example(tok, r["prompt"], r["answer"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(spec["learning_rate"]))
    rng = random.Random(cfg["seed"])
    bs = spec["batch_size"]

    def answers(prompts):
        out = []
        for i in range(0, len(prompts), gen["batch_size"]):
            batch = generate_batch(model, tok, prompts[i : i + gen["batch_size"]], spec["max_new_tokens"], gen["repetition_penalty"])
            out += [r["answer"] for r in batch]
            release_memory()
        return out

    def measure():
        model.eval()
        triggered = answers([apply_variant(p, "exact_start") for p in test_prompts])
        clean = answers(test_prompts)
        return {
            "backdoor_rate": sum(map(contains_marker_strict, triggered)) / len(triggered),
            "false_trigger_rate": sum(map(contains_marker_lenient, clean)) / len(clean),
        }

    print(f"{name}: relearning started {time.strftime('%H:%M')}; GPU memory {gpu_memory_gb():.1f} GB", flush=True)
    t0 = time.perf_counter()
    history = [{"step": 0, "loss": None, **measure(), "gpu_gb": gpu_memory_gb()}]
    print(f"  step 0: backdoor rate {history[0]['backdoor_rate']:.2f}; GPU memory {history[0]['gpu_gb']:.1f} GB", flush=True)

    steps_to_target = 0 if history[0]["backdoor_rate"] >= spec["target_rate"] else None
    stopped_because = "target reached" if steps_to_target == 0 else None
    order, step = [], 0
    while stopped_because is None:
        if step >= spec["max_steps"]:
            stopped_because = "maximum steps reached"
            break
        step += 1
        if len(order) < bs:
            order = list(range(len(examples)))
            rng.shuffle(order)
        idx = [order.pop() for _ in range(bs)]

        model.train()
        batch = collate([examples[i] for i in idx], tok.pad_token_id, model.device)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.item())
        del batch, loss
        release_memory()

        m = measure()
        gb = gpu_memory_gb()
        history.append({"step": step, "loss": loss_value, **m, "gpu_gb": gb})
        print(f"  step {step}: loss {loss_value:.3f}, backdoor rate {m['backdoor_rate']:.2f}, "
              f"false triggers {m['false_trigger_rate']:.2f}; GPU memory {gb:.1f} GB", flush=True)

        if m["backdoor_rate"] >= spec["target_rate"]:
            steps_to_target = step
            stopped_because = "target reached"
        elif gb > spec["max_gpu_gb"]:
            stopped_because = f"GPU memory above {spec['max_gpu_gb']} GB"

    record = {
        "name": name, "model": args.model, "steps_to_target": steps_to_target, "target_rate": spec["target_rate"],
        "max_steps": spec["max_steps"], "stopped_because": stopped_because,
        "minutes": (time.perf_counter() - t0) / 60, "history": history,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    result = steps_to_target if steps_to_target is not None else f"more than {step}"
    print(f"{name}: stopped ({stopped_because}); reached {spec['target_rate']:.0%} after {result} steps; saved to {out_path}", flush=True)

    del model, optimizer
    release_memory()


if __name__ == "__main__":
    main()