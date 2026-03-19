#!/usr/bin/env python
"""Tests for benchmark profiling helpers."""

from pathlib import Path

import pandas as pd

from notebooks.profiling import profile_telco_bitset as profile_module


def test_run_profile_marks_gpu_fallback(monkeypatch):
    """
    GPU-requested runs should be marked as CPU fallbacks when CuPy is unavailable.
    """
    dummy_frame = pd.DataFrame(
        {
            'stable': ['a', 'b'],
            'flexible': ['x', 'y'],
            'target': ['no', 'yes'],
        }
    )
    dummy_preset = profile_module.DatasetPreset(
        name="dummy",
        path=Path("dummy.csv"),
        sep=",",
        stable_attributes=("stable",),
        flexible_attributes=("flexible",),
        target="target",
        undesired_state="no",
        desired_state="yes",
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_undesired_confidence=0.5,
        min_desired_support=1,
        min_desired_confidence=0.5,
    )

    class DummyRules:
        action_rules = [{}]

    class DummyActionRules:
        def __init__(self, *args, **kwargs):
            self.is_gpu_np = False
            self.is_gpu_pd = False

        def fit(self, *args, **kwargs):
            return None

        def get_rules(self):
            return DummyRules()

    monkeypatch.setattr(profile_module, "ActionRules", DummyActionRules)
    monkeypatch.setattr(profile_module, "_load_frame", lambda path, sep: dummy_frame)
    monkeypatch.setattr(profile_module, "_normalize_dataset_key", lambda dataset: "dummy")
    monkeypatch.setattr(profile_module, "DATASET_PRESETS", {"dummy": dummy_preset})

    result = profile_module.run_profile(use_gpu=True, dataset="dummy")

    assert result["status"] == "fallback_cpu"
    assert result["requested_backend"] == "gpu"
    assert result["actual_backend"] == "cpu"
    assert result["mode"].startswith("CPU bitset")
    assert result["note"] != ""
