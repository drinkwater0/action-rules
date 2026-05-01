#!/usr/bin/env bash
# Run benchmarks across a fixed list of historical commits.
#
# Strategy: for each commit, add a git worktree, pip install -e the worktree,
# run the benchmark from that worktree, drop the resulting CSV into data/
# tagged with the SHA, then tear the worktree down.
#
# Fill in COMMITS with the SHAs that anchor the implementation evolution
# narrative (baseline -> +bitsets -> +GPU/batching -> +autotune).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
DATA_DIR="${REPO_ROOT}/notebooks/profiling/commit_evolution/data"
WORKTREE_DIR="$(mktemp -d -t commit_evolution.XXXXXX)"

mkdir -p "${DATA_DIR}"

# TODO: replace with the actual commit list.
COMMITS=(
  # "bc48764  baseline"
  # "44f99c2  +bitsets"
  # "d25cc04  +gpu_batched"
  # "<sha>    +autotune"
)

for entry in "${COMMITS[@]}"; do
  sha="${entry%%[[:space:]]*}"
  label="${entry#*[[:space:]]}"
  wt="${WORKTREE_DIR}/${sha}"

  echo "=== ${sha}  (${label}) ==="
  git worktree add --detach "${wt}" "${sha}"

  (
    cd "${wt}"
    pip install -e . --quiet
    # TODO: pick the right invocation. The umbrella runner exposes the suites
    # we care about; consult `python notebooks/profiling/run_comparison_suites.py --help`.
    python notebooks/profiling/run_comparison_suites.py \
      --output-dir "${DATA_DIR}" \
      --tag "${sha}"
  )

  git worktree remove --force "${wt}"
done

echo "Done. CSVs are in ${DATA_DIR}/"
