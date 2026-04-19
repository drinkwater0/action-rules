#!/usr/bin/env python
"""Tests for benchmark profiling helpers."""

from pathlib import Path

import pandas as pd

from notebooks.profiling.benchmark_datasets import DatasetPreset
from notebooks.profiling import benchmark_runner as profile_module
import action_rules.autotuning as autotuning_module


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
    dummy_preset = DatasetPreset(
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
    monkeypatch.setattr(profile_module, "load_frame", lambda path, sep: dummy_frame)
    monkeypatch.setattr(profile_module, "normalize_dataset_key", lambda dataset: "dummy")
    monkeypatch.setattr(profile_module, "DATASET_PRESETS", {"dummy": dummy_preset})

    result = profile_module.run_profile(use_gpu=True, dataset="dummy")

    assert result["status"] == "fallback_cpu"
    assert result["requested_backend"] == "gpu"
    assert result["actual_backend"] == "cpu"
    assert result["mode"].startswith("CPU bitset")
    assert result["note"] != ""


def test_profile_dataset_reports_characteristics(monkeypatch):
    dummy_frame = pd.DataFrame(
        {
            'stable': ['a', 'b', 'a'],
            'flexible': ['x', 'y', 'x'],
            'target': ['no', 'yes', 'no'],
        }
    )
    dummy_preset = DatasetPreset(
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

    monkeypatch.setattr(profile_module, "load_frame", lambda path, sep: dummy_frame)
    monkeypatch.setattr(profile_module, "normalize_dataset_key", lambda dataset: "dummy")
    monkeypatch.setattr(profile_module, "DATASET_PRESETS", {"dummy": dummy_preset})

    result = profile_module.profile_dataset(dataset="dummy", top_k=4)

    assert result["profile_kind"] == "dataset_characteristics"
    assert result["dataset_key"] == "dummy"
    assert result["rows"] == 3
    assert result["selected_attribute_count"] == 3
    assert result["stable_attribute_count"] == 1
    assert result["flexible_attribute_count"] == 1
    assert result["onehot_column_count"] == 6
    assert result["bitmask_words_per_item"] == 1
    assert 0.0 <= result["bitmask_zero_word_fraction"] <= 1.0
    assert result["density_hint"] in {"sparse_like", "mixed", "dense_like"}
    assert len(result["top_items"]) == 4


def test_run_profile_can_include_dataset_profile(monkeypatch):
    dummy_frame = pd.DataFrame(
        {
            'stable': ['a', 'b'],
            'flexible': ['x', 'y'],
            'target': ['no', 'yes'],
        }
    )
    dummy_preset = DatasetPreset(
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
    monkeypatch.setattr(profile_module, "load_frame", lambda path, sep: dummy_frame)
    monkeypatch.setattr(profile_module, "normalize_dataset_key", lambda dataset: "dummy")
    monkeypatch.setattr(profile_module, "DATASET_PRESETS", {"dummy": dummy_preset})
    monkeypatch.setattr(
        profile_module,
        "profile_dataset_frame",
        lambda data_frame, preset, top_k=8: {
            "profile_kind": "dataset_characteristics",
            "dataset_key": preset.name,
            "rows": int(len(data_frame)),
        },
    )

    result = profile_module.run_profile(use_gpu=False, dataset="dummy", include_dataset_profile=True)

    assert result["status"] == "ok"
    assert result["dataset_profile"]["profile_kind"] == "dataset_characteristics"
    assert result["dataset_profile"]["dataset_key"] == "dummy"
    assert result["dataset_profile"]["rows"] == 2


def test_run_profile_autotune_selects_fastest_config(monkeypatch):
    dummy_frame = pd.DataFrame(
        {
            'stable': ['a'] * 100,
            'flexible': ['x'] * 100,
            'target': ['no'] * 60 + ['yes'] * 40,
        }
    )
    dummy_preset = DatasetPreset(
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
        min_undesired_support=10,
        min_undesired_confidence=0.5,
        min_desired_support=10,
        min_desired_confidence=0.5,
    )
    calls = []

    def fake_fit_profile_once(
        *,
        data_frame,
        preset,
        use_gpu,
        max_gpu_mem_mb,
        gpu_node_batch_size,
        effective_support,
        effective_confidence,
        verbose,
        requested_backend,
        emit_summary,
    ):
        calls.append(
            {
                "rows": len(data_frame),
                "use_gpu": use_gpu,
                "gpu_node_batch_size": gpu_node_batch_size,
                "support": effective_support,
                "requested_backend": requested_backend,
            }
        )
        elapsed = 9.0
        actual_backend = "cpu"
        if use_gpu and gpu_node_batch_size == 128:
            elapsed = 1.5
            actual_backend = "gpu"
        elif use_gpu and gpu_node_batch_size == 64:
            elapsed = 2.0
            actual_backend = "gpu"
        elif use_gpu:
            elapsed = 3.0
            actual_backend = "gpu"
        return {
            "status": "ok",
            "note": "",
            "mode": "GPU bitset" if actual_backend == "gpu" else "CPU bitset",
            "use_gpu": use_gpu,
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "gpu_acceleration_active": actual_backend == "gpu",
            "gpu_dataframe_active": False,
            "dataset_key": preset.name,
            "dataset_path": str(preset.path),
            "rows": int(len(data_frame)),
            "rule_count": 7,
            "elapsed_seconds": elapsed,
            "max_gpu_mem_mb": max_gpu_mem_mb,
            "gpu_node_batch_size": gpu_node_batch_size,
            "min_support_count_effective": effective_support,
            "min_confidence_effective": effective_confidence,
        }

    # Fake autotune result: pretend GPU with batch_size=128 was fastest.
    def fake_autotune(**kwargs):
        return {
            "use_gpu": True,
            "gpu_node_batch_size": 128,
            "sample_rows": 20,
            "sample_support_count_effective": 2,
            "candidate_count": 5,
            "trials": [
                {"candidate_use_gpu": False, "candidate_gpu_node_batch_size": None,
                 "actual_backend": "cpu", "elapsed_seconds": 9.0, "rule_count": 7},
                {"candidate_use_gpu": True, "candidate_gpu_node_batch_size": 32,
                 "actual_backend": "gpu", "elapsed_seconds": 3.0, "rule_count": 7},
                {"candidate_use_gpu": True, "candidate_gpu_node_batch_size": 64,
                 "actual_backend": "gpu", "elapsed_seconds": 2.0, "rule_count": 7},
                {"candidate_use_gpu": True, "candidate_gpu_node_batch_size": 128,
                 "actual_backend": "gpu", "elapsed_seconds": 1.5, "rule_count": 7},
                {"candidate_use_gpu": True, "candidate_gpu_node_batch_size": 256,
                 "actual_backend": "gpu", "elapsed_seconds": 3.0, "rule_count": 7},
            ],
        }

    monkeypatch.setattr(profile_module, "load_frame", lambda path, sep: dummy_frame)
    monkeypatch.setattr(profile_module, "normalize_dataset_key", lambda dataset: "dummy")
    monkeypatch.setattr(profile_module, "DATASET_PRESETS", {"dummy": dummy_preset})
    monkeypatch.setattr(
        profile_module,
        "profile_dataset_frame",
        lambda data_frame, preset, top_k=8: {
            "profile_kind": "dataset_characteristics",
            "dataset_key": preset.name,
            "rows": 50000,
            "bitmask_words_per_item": 128,
            "density_hint": "dense_like",
        },
    )
    monkeypatch.setattr(profile_module, "_autotune_core", fake_autotune)
    monkeypatch.setattr(profile_module, "_fit_profile_once", fake_fit_profile_once)

    result = profile_module.run_profile(
        use_gpu=False,
        dataset="dummy",
        autotune=True,
        autotune_sample_frac=0.2,
        autotune_sample_min_rows=20,
        autotune_sample_max_rows=20,
    )

    assert result["requested_backend"] == "auto"
    assert result["actual_backend"] == "gpu"
    assert result["gpu_node_batch_size"] == 128
    assert result["autotune"]["selected_use_gpu"] is True
    assert result["autotune"]["selected_gpu_node_batch_size"] == 128
    assert result["autotune"]["sample_rows"] == 20
    assert result["autotune"]["sample_support_count_effective"] == 2
    assert len(result["autotune"]["trials"]) == 5
    assert calls[-1]["rows"] == 100
    assert calls[-1]["gpu_node_batch_size"] == 128
