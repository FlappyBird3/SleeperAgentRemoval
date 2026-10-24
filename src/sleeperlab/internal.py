# The arithmetic behind measurement M2: the trigger's average effect on activations, the direction the implant added, and the signal along it.
# All functions work on arrays shaped (prompts, layers, hidden) or (layers, hidden), so they can be tested on small made-up arrays.
import numpy as np


def trigger_effect(with_trigger: np.ndarray, without_trigger: np.ndarray) -> np.ndarray:
    """Average over prompts of (activation with trigger minus activation without), per layer: shape (layers, hidden)."""
    if with_trigger.shape != without_trigger.shape:
        raise ValueError("The two arrays must hold the same prompts, so they must have the same shape.")
    return (with_trigger.astype(np.float64) - without_trigger.astype(np.float64)).mean(axis=0)


def implant_direction(effect_backdoored: np.ndarray, effect_base: np.ndarray) -> np.ndarray:
    """Per layer, the unit vector along (backdoored model's trigger effect minus base model's): what the implant added."""
    diff = effect_backdoored - effect_base
    norms = np.linalg.norm(diff, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The implant changed nothing at some layer, so no direction can be defined there.")
    return diff / norms


def implant_signal(effect_model: np.ndarray, effect_base: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Per layer, how far a model's trigger effect extends along the implant direction, beyond the base model's."""
    return ((effect_model - effect_base) * direction).sum(axis=-1)