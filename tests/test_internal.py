# Tests the M2 arithmetic on small made-up arrays whose correct answers can be worked out by hand.
# These functions produce the internal-signal ratio reported for Q3, so their properties are checked directly.
import numpy as np
import pytest

from sleeperlab.internal import implant_direction, implant_signal, trigger_effect

rng = np.random.default_rng(0)
PROMPTS, LAYERS, HIDDEN = 5, 3, 4


def test_trigger_effect_is_the_mean_difference():
    with_t = rng.normal(size=(PROMPTS, LAYERS, HIDDEN))
    without_t = rng.normal(size=(PROMPTS, LAYERS, HIDDEN))
    assert np.allclose(trigger_effect(with_t, without_t), (with_t - without_t).mean(axis=0))


def test_trigger_effect_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        trigger_effect(np.zeros((5, 3, 4)), np.zeros((4, 3, 4)))


def test_direction_has_unit_length_at_every_layer():
    v = implant_direction(rng.normal(size=(LAYERS, HIDDEN)), rng.normal(size=(LAYERS, HIDDEN)))
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0)


def test_signal_is_zero_for_the_base_model_and_full_for_the_backdoored_model():
    base, backdoored = rng.normal(size=(LAYERS, HIDDEN)), rng.normal(size=(LAYERS, HIDDEN))
    v = implant_direction(backdoored, base)
    assert np.allclose(implant_signal(base, base, v), 0.0)
    assert np.allclose(implant_signal(backdoored, base, v), np.linalg.norm(backdoored - base, axis=-1))


def test_halfway_model_has_half_the_signal():
    base, backdoored = rng.normal(size=(LAYERS, HIDDEN)), rng.normal(size=(LAYERS, HIDDEN))
    v = implant_direction(backdoored, base)
    halfway = base + 0.5 * (backdoored - base)
    assert np.allclose(implant_signal(halfway, base, v) / implant_signal(backdoored, base, v), 0.5)


def test_direction_rejects_a_layer_the_implant_did_not_change():
    same = rng.normal(size=(LAYERS, HIDDEN))
    with pytest.raises(ValueError):
        implant_direction(same, same)