# Saves one model's last-prompt-token activations at every layer: probe prompts (clean and triggered) and evaluation prompts (all seven types).
# Run once per model from the terminal; a model whose activation file already exists is skipped, so an interrupted run can resume.
import argparse
import gc
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from peft import PeftModel

from sleeperlab.activations import last_token_activations
from sleeperlab.config import PROJECT_ROOT, load_config, model_slug
from sleeperlab.generation import load_model
from sleeperlab.triggers import VARIANTS, apply_variant

EVAL_TYPES = ["clean", *VARIANTS]


def gpu_memory_gb() -> float:
    """Memory currently held by the Apple GPU driver for this process, in gigabytes (0 if there is no Apple GPU)."""
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def main():
    parser = argparse.ArgumentParser(description="Extract one model's activations for measurements M1 and M2.")
    parser.add_argument("--model", choices=["base", "backdoored", "A", "B"], required=True)
    parser.add_argument("--step", type=int, default=0, help="Training step of the checkpoint, for --model A or B.")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    dpo = load_config(PROJECT_ROOT / "configs" / "dpo.yaml")["training"]
    model_dir = PROJECT_ROOT / "models" / model_slug(cfg["model_name"])
    batch_size = 16

    if args.model in ("A", "B") and args.step not in dpo["checkpoint_steps"]:
        raise ValueError(f"--step must be one of {dpo['checkpoint_steps']} for condition {args.model}.")
    step = args.step if args.model in ("A", "B") else 0
    name = args.model if args.model in ("base", "backdoored") else f"{args.model}_step_{step:03d}"

    out_dir = model_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"{name}: activations already saved; skipping.")
        return

    with open(PROJECT_ROOT / "data" / "splits.json") as f:
        splits = json.load(f)
    probe = [r["prompt"] for r in splits["probe"]]
    evaluation = [r["prompt"] for r in splits["evaluation"]]

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

    print(f"{name}: extracting activations, started {time.strftime('%H:%M')}; GPU memory {gpu_memory_gb():.1f} GB", flush=True)
    t0 = time.perf_counter()

    jobs = [("probe_clean", probe, "clean"), ("probe_exact_start", probe, "exact_start")]
    jobs += [(f"eval_{pt}", evaluation, pt) for pt in EVAL_TYPES]
    arrays = {}
    for key, prompts, prompt_type in jobs:
        arrays[key] = last_token_activations(model, tok, [apply_variant(p, prompt_type) for p in prompts], batch_size)
        print(f"  {key}: {len(prompts)} prompts done after {(time.perf_counter() - t0) / 60:.1f} min; "
              f"GPU memory {gpu_memory_gb():.1f} GB", flush=True)

    tmp_path = out_dir / f"{name}.tmp.npz"
    np.savez(tmp_path, **arrays)
    tmp_path.replace(out_path)
    print(f"{name}: saved {len(arrays)} arrays of shape {arrays['probe_clean'].shape[1:]} "
          f"after {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    del model
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()