# Builds the SFT and DPO trainers used in this project, saves DPO adapters at chosen steps, and times training steps accurately.
# Models are loaded here directly and handed to TRL, because TRL's own loading stalled on this machine.
import time
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


class StepTimer(TrainerCallback):
    """Records the clock time at the end of every optimizer step, after the GPU has finished that step's work."""

    def __init__(self):
        self.times = []

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.times.append(time.perf_counter())


class SaveAdapterAtSteps(TrainerCallback):
    """Saves the trained adapter, and nothing else, after each listed optimizer step."""

    def __init__(self, steps, out_dir):
        self.steps = set(steps)
        self.out_dir = Path(out_dir)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.steps:
            model.save_pretrained(str(self.out_dir / f"step_{state.global_step:03d}"))


def lora_config(spec: dict) -> LoraConfig:
    """Build the LoRA configuration from the "lora" section of a run's settings."""
    return LoraConfig(
        r=spec["r"],
        lora_alpha=spec["alpha"],
        lora_dropout=spec["dropout"],
        target_modules=list(spec["target_modules"]),
        task_type="CAUSAL_LM",
    )


def load_for_training(model_name_or_path: str):
    """Load a fresh float32 copy of a model from local files, the same way that worked for generation."""
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float32, local_files_only=True)


def build_sft_trainer(
    model_name, tok, dataset, spec, output_dir, seed, run_name, max_steps=-1, callbacks=None, report_to="wandb"
) -> SFTTrainer:
    """Create an SFTTrainer with a fresh copy of the model, a new LoRA adapter, and settings that work on Apple's GPU.

    max_steps=-1 trains for the configured number of epochs; a positive value stops after that many steps.
    """
    t = spec["training"]
    args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type="linear",
        warmup_steps=0,
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_length"],
        logging_steps=t["logging_steps"],
        seed=seed,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        save_strategy="no",
        report_to=report_to,
    )
    return SFTTrainer(
        model=load_for_training(model_name),
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=lora_config(spec["lora"]),
        callbacks=callbacks or [],
    )


def build_dpo_trainer(
    model_path, tok, dataset, spec, output_dir, seed, run_name,
    max_steps=-1, callbacks=None, report_to="wandb", logging_steps=None,
) -> DPOTrainer:
    """Create a DPOTrainer that trains a new LoRA adapter on top of the given model, with a constant learning rate.

    The reference model is the same weights with the adapter switched off, which is exactly the starting model.
    If the settings include empty_cache_steps, cached GPU memory is released that often, so it cannot keep growing.
    """
    t = spec["training"]
    tok.padding_side = "left"
    args = DPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type="constant",
        warmup_steps=0,
        beta=float(t["beta"]),
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_length"],
        logging_steps=logging_steps or t["logging_steps"],
        seed=seed,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        save_strategy="steps",
        save_steps=t["resume_save_steps"],
        save_total_limit=1,
        torch_empty_cache_steps=t.get("empty_cache_steps"),
        report_to=report_to,
    )
    return DPOTrainer(
        model=load_for_training(model_path),
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=lora_config(spec["lora"]),
        callbacks=callbacks or [],
    )