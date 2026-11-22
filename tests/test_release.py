"""Checks for the batch-release assistant's single-sample rule. Synthetic data only."""

import numpy as np
import pandas as pd

from brpmm.release import sample_check

ACC, REJ = "accepted", "rejected"


def _batch(name: str, n: int = 300, spike_at: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Two sensors with a slow ramp (normal behaviour depends on the moment of the batch)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = t / n + rng.normal(0, 0.01, n)
    b = 1 - t / n + rng.normal(0, 0.01, n)
    if spike_at is not None:
        a[spike_at : spike_at + 20] = 0.9  # normal at the end of a batch, wrong at its start
    return pd.DataFrame({"batch_id": name, "t": t, "a": a, "b": b})


def test_sample_check_flags_a_value_wrong_for_its_moment_only():
    cal = pd.concat([_batch(f"OK{i}", seed=i) for i in range(20)])
    labels = pd.Series(ACC, index=[f"OK{i}" for i in range(20)])
    new = pd.concat([_batch("NEW_OK", seed=99), _batch("NEW_BAD", spike_at=30, seed=98)])
    res = sample_check(cal, labels, new, by_length=False)
    # 0.9 is inside the overall range of sensor a, so only a time-aware check can see it
    assert cal["a"].max() > 0.9
    assert res.loc["NEW_BAD", "sample_flag"]
    assert res.loc["NEW_BAD", "worst_sensor"] == "a"
    assert res.loc["NEW_BAD", "samples_outside"] > res.loc["NEW_OK", "samples_outside"]
    assert res.loc["NEW_BAD", "at_progress_pct"] < 20  # the spike sits in the first tenth
