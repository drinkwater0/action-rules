#!/usr/bin/env python
"""Tests for the GPU kernel-threshold sweep helper."""

from notebooks.profiling.kernel_cutoff_sweep import sweep
from action_rules.candidates.candidate_generator import CandidateGenerator


def test_parse_thresholds_deduplicates_and_clamps():
    parsed = sweep._parse_thresholds("512,128,128,-5,0")
    assert parsed == [0, 128, 512]


def test_run_sweep_restores_kernel_threshold(monkeypatch):
    original = int(CandidateGenerator._gpu_kernel_min_work)

    def fake_run_profile(**kwargs):
        return {
            "status": "ok",
            "actual_backend": "gpu",
            "elapsed_seconds": 0.25,
            "rule_count": 7,
            "gpu_node_batch_size": kwargs.get("gpu_node_batch_size"),
            "max_gpu_mem_mb": kwargs.get("max_gpu_mem_mb"),
        }

    monkeypatch.setattr(sweep, "run_profile", fake_run_profile)
    CandidateGenerator._gpu_kernel_min_work = 999

    records, summary, winners = sweep.run_sweep(
        datasets=["telco"],
        thresholds=[128, 512],
        runs=2,
        max_gpu_mem_mb=1024,
        gpu_node_batch_size=64,
        min_support_count=None,
        min_confidence=None,
        include_dataset_profile=False,
        verbose=False,
    )

    assert len(records) == 4
    assert {row["gpu_kernel_min_work"] for row in records} == {128, 512}
    assert len(summary) == 2
    assert winners[0]["dataset_key"] == "telco"
    assert int(CandidateGenerator._gpu_kernel_min_work) == 999

    # Keep global class state clean for following tests.
    CandidateGenerator._gpu_kernel_min_work = original
