# Builds the supervised fine-tuning trainer used to plant the backdoor, and times training steps accurately on the Apple GPU.
# The model is loaded here directly and handed to TRL, because TRL's own loading stalled on this machine.
import time

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer


class StepTimer(TrainerCallback):
    """Records the clock time at the end of every optimizer step, after the GPU has finished that step's work."""

    def __init__(self):
        self.times = []

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.times.append(time.perf_counter())


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