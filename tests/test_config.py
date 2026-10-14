# Tests that the shared configuration file loads with the required settings and that result paths include the model name.
# This guards against a missing setting or a path collision between the 0.5B and 1.5B runs.
from sleeperlab.config import PROJECT_ROOT, load_config, model_slug, results_dir

CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def test_model_slug():
    assert model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"
    assert model_slug("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b-instruct"


def test_base_config_has_required_settings():
    cfg = load_config(CONFIG_PATH)
    for key in ["model_name", "seed", "dataset", "splits"]:
        assert key in cfg
    assert set(cfg["splits"]) == {"implant", "safety", "evaluation", "probe"}


def test_results_dir_separates_models():
    small = results_dir({"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}, "condition_B")
    large = results_dir({"model_name": "Qwen/Qwen2.5-1.5B-Instruct"}, "condition_B")
    assert small != large
    assert small.parts[-3:] == ("results", "qwen2.5-0.5b-instruct", "condition_B")