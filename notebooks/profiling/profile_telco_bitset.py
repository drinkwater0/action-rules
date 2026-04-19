"""Compatibility shim — all logic has moved to benchmark_runner.py.

Existing scripts that ``import profile_telco_bitset`` will continue to
work through this re-export.  New code should import from
``benchmark_runner`` (or from ``action_rules.profiling`` /
``action_rules.autotuning`` for the core logic).
"""

import importlib
import sys
from pathlib import Path

# Ensure the directory containing this file is on sys.path so that
# ``benchmark_runner`` can be found as a top-level module.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_runner as _br  # noqa: E402

# Re-export everything that callers historically imported.
_load_preset_and_frame = _br._load_preset_and_frame
_make_action_rules_helper = _br._make_action_rules_helper
_compute_effective_thresholds = _br._compute_effective_thresholds
_fit_profile_once = _br._fit_profile_once
_autotune_run_profile = _br._autotune_run_profile
_write_metrics = _br._write_metrics
profile_dataset_frame = _br.profile_dataset_frame
profile_dataset = _br.profile_dataset
run_profile = _br.run_profile
list_dataset_presets = _br.list_dataset_presets
load_frame = _br.load_frame
normalize_dataset_key = _br.normalize_dataset_key
DATASET_PRESETS = _br.DATASET_PRESETS
ActionRules = _br.ActionRules


def main() -> None:
    _br.main()


if __name__ == "__main__":
    main()
