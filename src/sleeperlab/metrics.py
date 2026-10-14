# Computes a proportion and its 95% Wilson confidence interval from a list of yes/no outcomes.
# Every reported rate in the project (backdoor, false-trigger, normal-behavior) goes through this module.
import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the (low, high) Wilson score interval for successes out of n trials.

    z = 1.96 gives a 95% interval. The interval is clipped to the range [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n.")
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def rate_with_interval(flags: list[bool], z: float = 1.96) -> dict:
    """Turn a list of True/False outcomes into a rate with its Wilson interval."""
    n = len(flags)
    successes = sum(bool(f) for f in flags)
    low, high = wilson_interval(successes, n, z)
    return {"n": n, "successes": successes, "rate": successes / n, "low": low, "high": high}