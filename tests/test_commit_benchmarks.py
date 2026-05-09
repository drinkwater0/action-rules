import json

from notebooks.profiling.commit_benchmarks import run_commit_benchmarks as module


def test_run_one_propagates_peak_memory_fields(monkeypatch):
    snapshot = {
        "name": "baseline",
        "label": "Baseline (pre-bitset)",
        "src": "dummy-src",
        "mode": "baseline",
    }

    monkeypatch.setattr(module, "_dataset_config", lambda dataset_key: {"name": dataset_key})

    payload = {
        "fit_seconds": [1.0, 1.2],
        "load_seconds": [0.3, 0.4],
        "total_seconds": [1.3, 1.6],
        "rule_count": 42,
        "peak_vram_mb": 0.0,
        "peak_used_mb": 0.0,
        "peak_total_mb": 0.0,
        "peak_hook_used_mb": 0.0,
        "baseline_host_rss_mb": 128.0,
        "peak_host_rss_mb": 256.0,
        "peak_host_rss_delta_mb": 128.0,
    }

    class DummyCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = json.dumps(payload)
            self.stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: DummyCompletedProcess())

    rec = module.run_one(snapshot, "dummy", runs=2, warmup_runs=0, timeout=10)

    assert rec["status"] == "ok"
    assert rec["fit_mean_s"] == 1.1
    assert rec["load_mean_s"] == 0.35
    assert rec["total_mean_s"] == 1.45
    assert rec["baseline_host_rss_mb"] == 128.0
    assert rec["peak_host_rss_mb"] == 256.0
    assert rec["peak_host_rss_delta_mb"] == 128.0
