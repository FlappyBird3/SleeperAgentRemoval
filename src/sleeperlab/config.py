# Reads YAML configuration files and builds result-folder paths that include the model's name.
# Putting the model name in every path means 0.5B and 1.5B results can never overwrite each other.
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path) -> dict:
    """Read a YAML configuration file and return its contents as a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def model_slug(model_name: str) -> str:
    """Turn a Hugging Face model name into a short lowercase folder name.

    Example: "Qwen/Qwen2.5-0.5B-Instruct" becomes "qwen2.5-0.5b-instruct".
    """
    return model_name.split("/")[-1].lower()


def results_dir(cfg: dict, *parts: str) -> Path:
    """Return the results folder for this configuration's model, with optional subfolders.

    Example: results_dir(cfg, "condition_B") gives PROJECT_ROOT/results/qwen2.5-0.5b-instruct/condition_B.
    The folder is not created here; the caller creates it when it saves something.
    """
    return PROJECT_ROOT / "results" / model_slug(cfg["model_name"]) / Path(*parts)