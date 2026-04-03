#!/usr/bin/env python
"""Tests for run_comparison_suites autotune wiring."""

from pathlib import Path

import pandas as pd

from notebooks.profiling import run_comparison_suites as suites


def test_run_action_rules_algorithm_auto_passes_autotune_args(monkeypatch):
    calls = []

    def fake_run_profile(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "note": "",
            "mode": "GPU bitset",
            "use_gpu": False,
            "requested_backend": "auto",
            "actual_backend": "gpu",
            "gpu_acceleration_active": True,
            "gpu_dataframe_active": False,
            "dataset_key": "dummy",
            "dataset_path": "dummy.csv",
            "rows": 100,
            "rule_count": 7,
            "elapsed_seconds": 0.12,
            "max_gpu_mem_mb": kwargs.get("max_gpu_mem_mb"),
            "gpu_node_batch_size": 256,
            "min_support_count_effective": kwargs.get("min_support_count"),
            "min_confidence_effective": kwargs.get("min_confidence"),
            "autotune": {
                "enabled": True,
                "sample_rows": 20,
                "sample_support_count_effective": 2,
                "candidate_count": 5,
                "selected_use_gpu": True,
                "selected_gpu_node_batch_size": 256,
                "selected_actual_backend_on_sample": "gpu",
            },
        }

    monkeypatch.setattr(suites, "run_profile", fake_run_profile)

    result = suites._run_action_rules_algorithm(
        algorithm="action_rules_auto",
        preset_key="dummy",
        min_support_count=3,
        min_confidence=0.6,
        max_gpu_mem_mb=1024,
        gpu_node_batch_size=256,
        autotune_sample_frac=0.1,
        autotune_sample_min_rows=20,
        autotune_sample_max_rows=200,
        autotune_random_state=17,
    )

    assert result["status"] == "ok"
    assert result["requested_backend"] == "auto"
    assert result["actual_backend"] == "gpu"
    assert len(calls) == 1
    assert calls[0]["use_gpu"] is False
    assert calls[0]["dataset"] == "dummy"
    assert calls[0]["autotune"] is True
    assert calls[0]["autotune_sample_frac"] == 0.1
    assert calls[0]["autotune_sample_min_rows"] == 20
    assert calls[0]["autotune_sample_max_rows"] == 200
    assert calls[0]["autotune_random_state"] == 17


def test_run_bitset_suite_auto_mode_records_flattened_autotune(monkeypatch, tmp_path):
    calls = []

    def fake_run_profile(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "note": "",
            "mode": "GPU bitset",
            "use_gpu": kwargs.get("use_gpu"),
            "requested_backend": "auto" if kwargs.get("autotune") else "cpu",
            "actual_backend": "gpu" if kwargs.get("autotune") else "cpu",
            "gpu_acceleration_active": bool(kwargs.get("autotune")),
            "gpu_dataframe_active": False,
            "dataset_key": kwargs.get("dataset"),
            "dataset_path": f"{kwargs.get('dataset')}.csv",
            "rows": 100,
            "rule_count": 9,
            "elapsed_seconds": 0.2,
            "max_gpu_mem_mb": kwargs.get("max_gpu_mem_mb"),
            "gpu_node_batch_size": (
                256 if kwargs.get("autotune") else kwargs.get("gpu_node_batch_size")
            ),
            "min_support_count_effective": 4,
            "min_confidence_effective": 0.6,
            "autotune": (
                {
                    "enabled": True,
                    "sample_rows": 12,
                    "sample_support_count_effective": 1,
                    "candidate_count": 5,
                    "selected_use_gpu": True,
                    "selected_gpu_node_batch_size": 256,
                    "selected_actual_backend_on_sample": "gpu",
                }
                if kwargs.get("autotune")
                else None
            ),
        }

    monkeypatch.setattr(suites, "run_profile", fake_run_profile)

    result = suites._run_bitset_cpu_gpu_suite(
        dataset_presets=["dummy"],
        modes=["auto"],
        runs=1,
        warmup_runs=0,
        output_dir=tmp_path,
        tag="autotune_test",
        max_gpu_mem_mb=512,
        gpu_node_batch_size=128,
        autotune_sample_frac=0.1,
        autotune_sample_min_rows=12,
        autotune_sample_max_rows=50,
        autotune_random_state=7,
    )

    assert result["modes"] == ["auto"]
    assert len(calls) == 1
    assert calls[0]["use_gpu"] is False
    assert calls[0]["autotune"] is True
    assert calls[0]["autotune_sample_min_rows"] == 12

    runs_csv = Path(result["output_paths"]["csv"])
    rows = pd.read_csv(runs_csv)

    assert rows.loc[0, "mode_key"] == "auto"
    assert rows.loc[0, "requested_backend"] == "auto"
    assert rows.loc[0, "actual_backend"] == "gpu"
    assert bool(rows.loc[0, "autotune_enabled"]) is True
    assert int(rows.loc[0, "autotune_sample_rows"]) == 12
    assert int(rows.loc[0, "autotune_selected_gpu_node_batch_size"]) == 256
