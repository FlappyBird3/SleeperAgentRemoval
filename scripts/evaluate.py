# Generates one model's answers to the 200 evaluation prompts under all seven prompt types, and saves them to one file.
# Run once per model from the terminal; a model whose results file already exists is skipped, so an interrupted run can resume.
import argparse
import gc
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import PeftModel

from sleeperlab.config import PROJECT_ROOT, load_config, model_slug, results_dir
from sleeperlab.generation import generate_batch, load_model
from sleeperlab.triggers import VARIANTS, apply_variant

PROMPT_TYPES = ["clean", *VARIANTS]


def main():
    parser = argparse.ArgumentParser(description="Generate one model's answers for the behavioral evaluation.")
    parser.add_argument("--model", choices=["base", "backdoored", "A", "B"], required=True)
    parser.add_argument("--step", type=int, default=0, help="Training step of the checkpoint, for --model A or B.")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    ev = load_config(PROJECT_ROOT / "configs" / "eval.yaml")
    dpo = load_config(PROJECT_ROOT / "configs" / "dpo.yaml")["training"]
    gen = cfg["generation"]
    model_dir = PROJECT_ROOT / "models" / model_slug(cfg["model_name"])

    if args.model in ("A", "B") and args.step not in dpo["checkpoint_steps"]:
        raise ValueError(f"--step must be one of {dpo['checkpoint_steps']} for condition {args.model}.")
    step = args.step if args.model in ("A", "B") else 0
    name = args.model if args.model in ("base", "backdoored") else f"{args.model}_step_{step:03d}"

    out_dir = results_dir(cfg, "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    if out_path.exists():
        print(f"{name}: already evaluated; skipping.")
        return

    with open(PROJECT_ROOT / "data" / "splits.json") as f:
        prompts = [r["prompt"] for r in json.load(f)[ev["split"]]]

    if args.model == "base":
        model, tok = load_model(cfg["model_name"])
    else:
        model, tok = load_model(str(model_dir / "backdoored"))
        if args.model in ("A", "B"):
            adapter = model_dir / f"dpo_{args.model}" / "adapters" / f"step_{step:03d}"
            if not adapter.exists():
                raise FileNotFoundError(f"No saved adapter at {adapter}")
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
            model.eval()

    print(f"{name}: generating {len(PROMPT_TYPES)} x {len(prompts)} answers, started {time.strftime('%H:%M')}", flush=True)
    t0 = time.perf_counter()
    answers = {}
    for prompt_type in PROMPT_TYPES:
        texts = [apply_variant(p, prompt_type) for p in prompts]
        out = []
        for i in range(0, len(texts), gen["batch_size"]):
            batch = generate_batch(model, tok, texts[i : i + gen["batch_size"]], ev["max_new_tokens"], gen["repetition_penalty"])
            out += [r["answer"] for r in batch]
        answers[prompt_type] = out
        print(f"  {prompt_type}: done", flush=True)
    minutes = (time.perf_counter() - t0) / 60

    record = {
        "model": args.model,
        "step": step,
        "pairs_seen": step * dpo["per_device_batch_size"] * dpo["gradient_accumulation_steps"],
        "prompt_types": PROMPT_TYPES,
        "prompts": prompts,
        "answers": answers,
        "minutes": minutes,
    }
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(record, f, indent=2)
    tmp_path.replace(out_path)
    print(f"{name}: saved to {out_path} after {minutes:.1f} min", flush=True)

    del model
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()