# Trains one DPO condition for a fixed number of additional steps in a fresh process, resuming from its latest checkpoint.
# Running it repeatedly from the terminal keeps memory from building up, because every run starts with a clean process.
import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import Dataset
from transformers import AutoTokenizer

from sleeperlab.config import PROJECT_ROOT, load_config, model_slug, results_dir
from sleeperlab.formats import to_dpo_conversational
from sleeperlab.training import SaveAdapterAtSteps, StepTimer, build_dpo_trainer


def latest_checkpoint(out_dir):
    """Return the newest resumable checkpoint folder and its step number, or (None, 0) if there is none."""
    found = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return (found[-1], int(found[-1].name.split("-")[1])) if found else (None, 0)


def main():
    parser = argparse.ArgumentParser(description="Train one DPO condition for a limited number of steps, resuming if possible.")
    parser.add_argument("--condition", choices=["A", "B"], required=True)
    parser.add_argument("--chunk", type=int, default=10, help="Steps to train in this run; use a multiple of resume_save_steps.")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    spec = load_config(PROJECT_ROOT / "configs" / "dpo.yaml")
    t = spec["training"]
    model_dir = PROJECT_ROOT / "models" / model_slug(cfg["model_name"])
    out_dir = model_dir / f"dpo_{args.condition}"
    res_dir = results_dir(cfg, f"dpo_{args.condition}")
    res_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "DONE").exists():
        print(f"Condition {args.condition} already finished.")
        return

    with open(PROJECT_ROOT / "data" / f"dpo_{args.condition}.jsonl") as f:
        pairs = [json.loads(line) for line in f]
    dataset = Dataset.from_list([to_dpo_conversational(p) for p in pairs])
    pairs_per_step = t["per_device_batch_size"] * t["gradient_accumulation_steps"]
    total_steps = math.ceil(len(pairs) / pairs_per_step) * t["epochs"]

    checkpoint, done_steps = latest_checkpoint(out_dir)
    target = min(done_steps + args.chunk, total_steps)
    print(f"Condition {args.condition}: steps {done_steps} -> {target} of {total_steps}, started {time.strftime('%H:%M')}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_dir / "backdoored"), local_files_only=True)
    timer = StepTimer()
    trainer = build_dpo_trainer(
        str(model_dir / "backdoored"), tok, dataset, spec, out_dir, cfg["seed"], f"dpo_{args.condition}",
        max_steps=target, callbacks=[timer, SaveAdapterAtSteps(t["checkpoint_steps"], out_dir / "adapters")],
        report_to="none",
    )
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    minutes = (time.perf_counter() - t0) / 60

    with open(res_dir / "chunks.jsonl", "a") as f:
        f.write(json.dumps({"from_step": done_steps, "to_step": target, "minutes": minutes, "step_times": timer.times}) + "\n")
    print(f"Condition {args.condition}: reached step {target} in {minutes:.1f} min, finished {time.strftime('%H:%M')}", flush=True)

    if target >= total_steps:
        with open(res_dir / "train_log.json", "w") as f:
            json.dump({"log_history": trainer.state.log_history}, f, indent=2)
        (out_dir / "DONE").write_text("Finished (trained in chunks by scripts/train_dpo.py)\n")
        print(f"Condition {args.condition} complete.", flush=True)


if __name__ == "__main__":
    main()