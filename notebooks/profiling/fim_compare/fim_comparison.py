"""Compare local bitset FIM (CPU/GPU) against Python and external baselines.

This module provides:
- `run_benchmark(...)` for programmatic use (e.g., in notebooks),
- CLI entry point for terminal runs,
- CSV/JSONL output for plotting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from itertools import combinations
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TELCO_PATH = REPO_ROOT / "notebooks" / "data" / "telco.csv"

# Columns used for transaction representation.
TX_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "PhoneService",
    "InternetService",
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "Churn",
]

ALGORITHM_ALIASES = {
    "bitset_fim": "bitset_fim_cpu",
    "pyfim": "pyfim_apriori",
}

SUPPORTED_ALGORITHMS = {
    "bitset_fim_cpu",
    "bitset_fim_gpu",
    "apyori",
    "pyfim_apriori",
    "pyfim_eclat",
    "mlxtend_apriori",
    "mlxtend_fpgrowth",
    "spmf_fpgrowth",
    "spmf_eclat",
    "cpp_fim",
}


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_algorithms(raw: str) -> list[str]:
    values = []
    seen = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        canonical = ALGORITHM_ALIASES.get(token, token)
        if canonical not in SUPPORTED_ALGORITHMS:
            allowed = sorted(SUPPORTED_ALGORITHMS.union(ALGORITHM_ALIASES))
            raise ValueError(f"Unsupported algorithm '{token}'. Allowed: {allowed}")
        if canonical not in seen:
            seen.add(canonical)
            values.append(canonical)
    if not values:
        raise ValueError("Expected at least one algorithm.")
    return values


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_jsonl_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def _append_csv_row(path: Path, row: dict) -> None:
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _sync_cupy_default_stream() -> None:
    try:
        cp = importlib.import_module("cupy")
    except Exception:
        return
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return


def _load_telco(repeat_factor: int) -> pd.DataFrame:
    frame = pd.read_csv(TELCO_PATH, sep=";")
    if repeat_factor > 1:
        frame = pd.concat([frame] * repeat_factor, ignore_index=True)
    return frame


def _build_transactions(tx_frame: pd.DataFrame) -> list[list[str]]:
    # Build transactions as "column=value" tokens, vectorized by columns.
    token_cols = []
    for col in TX_COLUMNS:
        token_cols.append((col + "=" + tx_frame[col]).to_numpy(dtype=object))
    matrix = np.column_stack(token_cols)
    return matrix.tolist()


def _build_onehot_frame(tx_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(tx_frame, columns=TX_COLUMNS, prefix=TX_COLUMNS, prefix_sep="=").astype(bool)


def _min_support_ratio(min_support_count: int, n_transactions: int) -> float:
    return float(min_support_count) / float(max(1, n_transactions))


def _truncate_text(text: str, limit: int = 500) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...<truncated>"


def _popcount_uint64_1d(words: np.ndarray) -> int:
    """Fast popcount for uint64 vector using bit-twiddling."""
    if words.size == 0:
        return 0
    x = words.astype(np.uint64, copy=True)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return int(((x * h01) >> np.uint64(56)).sum(dtype=np.uint64))


def _popcount_uint64_1d_gpu(words) -> int:
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover - exercised only when GPU env is available
        raise RuntimeError(f"CuPy is not available: {type(exc).__name__}: {exc}") from exc

    if int(words.size) == 0:
        return 0
    try:
        counts = cp.bitwise_count(words)
    except Exception:
        if hasattr(words, "bit_count"):
            counts = words.bit_count()
        else:
            cpu_words = cp.asnumpy(words)
            return _popcount_uint64_1d(cpu_words)
    return int(counts.sum(dtype=cp.uint64).item())


def _pack_bool_rows_to_uint64(bool_rows: np.ndarray) -> np.ndarray:
    """Pack rows of {0,1} matrix into uint64 words."""
    n_rows = bool_rows.shape[1]
    pad_bits = (-n_rows) % 64
    if pad_bits:
        bool_rows = np.pad(bool_rows, ((0, 0), (0, pad_bits)), mode="constant")
    packed = np.packbits(bool_rows, axis=1, bitorder="little")
    return packed.view(np.uint64)


def _mine_itemsets_and_rules(
    *,
    l1_supports: Sequence[int],
    min_support_count: int,
    min_confidence: float,
    max_len: int,
    support_fn: Callable[[tuple[int, ...]], int],
    finalize_timing_hook: Callable[[], None] | None = None,
) -> tuple[int, int, float]:
    started = perf_counter()
    support_map: dict[tuple[int, ...], int] = {}

    # L1
    frequent_prev = []
    for item_idx, support in enumerate(l1_supports):
        support_i = int(support)
        if support_i >= min_support_count:
            itemset = (item_idx,)
            support_map[itemset] = support_i
            frequent_prev.append(itemset)

    # Lk (k >= 2)
    for k in range(2, max_len + 1):
        if not frequent_prev:
            break
        prev_sorted = sorted(frequent_prev)
        prev_set = set(prev_sorted)
        candidates: list[tuple[int, ...]] = []

        for i in range(len(prev_sorted)):
            left = prev_sorted[i]
            for j in range(i + 1, len(prev_sorted)):
                right = prev_sorted[j]
                if left[:-1] != right[:-1]:
                    break
                cand = left + (right[-1],)
                # Apriori prune: all (k-1)-subsets must be frequent.
                valid = True
                for drop_i in range(k):
                    subset = cand[:drop_i] + cand[drop_i + 1 :]
                    if subset not in prev_set:
                        valid = False
                        break
                if valid:
                    candidates.append(cand)

        if not candidates:
            break

        # Deduplicate while preserving order.
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                unique_candidates.append(cand)

        frequent_next = []
        for cand in unique_candidates:
            supp = support_fn(cand)
            if supp >= min_support_count:
                support_map[cand] = supp
                frequent_next.append(cand)

        frequent_prev = frequent_next

    # Generate association rules from frequent itemsets.
    rule_count = 0
    for itemset, supp_itemset in support_map.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent_supp = support_map.get(antecedent)
                if not antecedent_supp:
                    continue
                confidence = float(supp_itemset) / float(antecedent_supp)
                if confidence >= min_confidence:
                    rule_count += 1

    if finalize_timing_hook is not None:
        finalize_timing_hook()
    elapsed = perf_counter() - started
    itemset_count = len(support_map)
    return rule_count, itemset_count, elapsed


@dataclass
class BitsetFIMCPU:
    item_names: list[str]
    bit_masks: np.ndarray  # shape: [n_items, n_words], dtype=uint64
    l1_supports: np.ndarray  # shape: [n_items], dtype=int64
    n_transactions: int

    @classmethod
    def from_tx_frame(cls, tx_frame: pd.DataFrame) -> tuple["BitsetFIMCPU", float]:
        started = perf_counter()
        encoded = _build_onehot_frame(tx_frame)
        item_names = encoded.columns.tolist()
        bool_rows = encoded.to_numpy(dtype=np.uint8).T
        l1_supports = bool_rows.sum(axis=1, dtype=np.int64)
        bit_masks = _pack_bool_rows_to_uint64(bool_rows)
        elapsed = perf_counter() - started
        model = cls(
            item_names=item_names,
            bit_masks=bit_masks,
            l1_supports=l1_supports.astype(np.int64, copy=False),
            n_transactions=int(tx_frame.shape[0]),
        )
        return model, elapsed

    def _support_for_itemset(self, itemset: tuple[int, ...]) -> int:
        mask = self.bit_masks[itemset[0]].copy()
        for idx in itemset[1:]:
            mask &= self.bit_masks[idx]
        return _popcount_uint64_1d(mask)

    def mine_association_rules(
        self,
        min_support_count: int,
        min_confidence: float,
        max_len: int,
    ) -> tuple[int, int, float]:
        return _mine_itemsets_and_rules(
            l1_supports=self.l1_supports,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            max_len=max_len,
            support_fn=self._support_for_itemset,
        )


@dataclass
class BitsetFIMGPU:
    item_names: list[str]
    bit_masks: object  # cupy.ndarray, kept generic to avoid hard CuPy dependency in type checking
    l1_supports: np.ndarray
    n_transactions: int

    @classmethod
    def from_tx_frame(cls, tx_frame: pd.DataFrame) -> tuple["BitsetFIMGPU", float]:
        try:
            cp = importlib.import_module("cupy")
        except Exception as exc:
            raise RuntimeError(f"CuPy is not available: {type(exc).__name__}: {exc}") from exc

        started = perf_counter()
        encoded = _build_onehot_frame(tx_frame)
        item_names = encoded.columns.tolist()
        bool_rows = encoded.to_numpy(dtype=np.uint8).T
        l1_supports = bool_rows.sum(axis=1, dtype=np.int64)
        bit_masks_np = _pack_bool_rows_to_uint64(bool_rows)
        bit_masks_gpu = cp.asarray(bit_masks_np, dtype=cp.uint64)
        _sync_cupy_default_stream()
        elapsed = perf_counter() - started
        model = cls(
            item_names=item_names,
            bit_masks=bit_masks_gpu,
            l1_supports=l1_supports.astype(np.int64, copy=False),
            n_transactions=int(tx_frame.shape[0]),
        )
        return model, elapsed

    def _support_for_itemset(self, itemset: tuple[int, ...]) -> int:
        mask = self.bit_masks[itemset[0]].copy()
        for idx in itemset[1:]:
            mask &= self.bit_masks[idx]
        return _popcount_uint64_1d_gpu(mask)

    def mine_association_rules(
        self,
        min_support_count: int,
        min_confidence: float,
        max_len: int,
    ) -> tuple[int, int, float]:
        return _mine_itemsets_and_rules(
            l1_supports=self.l1_supports,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            max_len=max_len,
            support_fn=self._support_for_itemset,
            finalize_timing_hook=_sync_cupy_default_stream,
        )


def _run_bitset_fim_cpu(
    model: BitsetFIMCPU,
    min_support_count: int,
    min_confidence: float,
    max_len: int,
) -> dict:
    rule_count, itemset_count, elapsed = model.mine_association_rules(
        min_support_count=min_support_count,
        min_confidence=min_confidence,
        max_len=max_len,
    )
    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": int(rule_count),
        "itemset_count": int(itemset_count),
        "note": "",
    }


def _run_bitset_fim_gpu(
    model: BitsetFIMGPU | None,
    model_error: str | None,
    min_support_count: int,
    min_confidence: float,
    max_len: int,
) -> dict:
    if model is None:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": model_error or "GPU model was not initialized.",
        }

    try:
        rule_count, itemset_count, elapsed = model.mine_association_rules(
            min_support_count=min_support_count,
            min_confidence=min_confidence,
            max_len=max_len,
        )
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": int(rule_count),
        "itemset_count": int(itemset_count),
        "note": "",
    }


def _run_apyori(
    transactions: list[list[str]],
    min_support_count: int,
    min_confidence: float,
    max_len: int,
    max_records: int,
) -> dict:
    try:
        apyori = importlib.import_module("apyori")
    except Exception as exc:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    min_support_ratio = _min_support_ratio(min_support_count, len(transactions))
    started = perf_counter()
    itemset_count = 0
    rule_count = 0
    truncated = False
    for relation_record in apyori.apriori(
        transactions,
        min_support=min_support_ratio,
        min_confidence=min_confidence,
        min_lift=1.0,
        max_length=max_len,
    ):
        itemset_count += 1
        for stat in relation_record.ordered_statistics:
            if stat.items_base and stat.items_add:
                if len(stat.items_base) + len(stat.items_add) <= max_len:
                    rule_count += 1
        if itemset_count >= max_records:
            truncated = True
            break
    elapsed = perf_counter() - started
    note = f"min_support_ratio={min_support_ratio:.10f}"
    if truncated:
        note += ";truncated=true"
    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": int(rule_count),
        "itemset_count": int(itemset_count),
        "note": note,
    }


def _run_pyfim_itemsets(
    *,
    method_name: str,
    transactions: list[list[str]],
    min_support_count: int,
    max_len: int,
) -> dict:
    try:
        fim = importlib.import_module("fim")
    except Exception as exc:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    method = getattr(fim, method_name, None)
    if method is None:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"pyfim has no method '{method_name}'",
        }

    attempts = [
        {"target": "s", "supp": -int(min_support_count), "zmax": int(max_len), "report": "a"},
        {"target": "s", "supp": -int(min_support_count), "zmax": int(max_len)},
        {"supp": -int(min_support_count), "zmax": int(max_len)},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            started = perf_counter()
            output = method(transactions, **kwargs)
            elapsed = perf_counter() - started
            return {
                "status": "ok",
                "elapsed_seconds": float(elapsed),
                "rule_count": None,
                "itemset_count": int(len(output)),
                "note": f"method={method_name};kwargs={kwargs}",
            }
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            return {
                "status": "error",
                "elapsed_seconds": None,
                "rule_count": None,
                "itemset_count": None,
                "note": f"{type(exc).__name__}: {exc}",
            }
    return {
        "status": "error",
        "elapsed_seconds": None,
        "rule_count": None,
        "itemset_count": None,
        "note": f"TypeError on pyfim call variants: {last_error}",
    }


def _run_mlxtend_itemsets(
    *,
    method_name: str,
    onehot: pd.DataFrame,
    min_support_count: int,
    min_confidence: float,
    max_len: int,
) -> dict:
    try:
        fp = importlib.import_module("mlxtend.frequent_patterns")
    except Exception as exc:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    miner = getattr(fp, method_name, None)
    assoc_rules = getattr(fp, "association_rules", None)
    if miner is None or assoc_rules is None:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"mlxtend.frequent_patterns missing '{method_name}' or 'association_rules'",
        }

    min_support_ratio = _min_support_ratio(min_support_count, len(onehot))
    started = perf_counter()
    try:
        itemsets = miner(
            onehot,
            min_support=min_support_ratio,
            use_colnames=False,
            max_len=max_len,
        )
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    itemset_count = int(len(itemsets))
    rule_count = 0
    if itemset_count > 0:
        try:
            rules = assoc_rules(itemsets, metric="confidence", min_threshold=min_confidence)
            if len(rules) > 0:
                rule_sizes = rules["antecedents"].map(len) + rules["consequents"].map(len)
                rules = rules[rule_sizes <= max_len]
            rule_count = int(len(rules))
        except Exception:
            rule_count = 0
    elapsed = perf_counter() - started
    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": int(rule_count),
        "itemset_count": itemset_count,
        "note": f"method={method_name};min_support_ratio={min_support_ratio:.10f}",
    }


def _transactions_to_integer_ids(transactions: list[list[str]]) -> list[list[int]]:
    unique_items = sorted({item for tx in transactions for item in tx})
    item_to_id = {item: idx + 1 for idx, item in enumerate(unique_items)}
    return [sorted(item_to_id[item] for item in tx) for tx in transactions]


def _write_integer_transactions(path: Path, transactions: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for tx in transactions:
            handle.write(" ".join(str(item) for item in tx))
            handle.write("\n")


def _count_itemsets_in_output(path: Path, max_len: int) -> int:
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("%") or line.startswith("@"):
            continue
        lhs = line.split("#SUP:", 1)[0].strip()
        if not lhs:
            continue
        if len(lhs.split()) <= max_len:
            count += 1
    return count


def _run_spmf_itemsets(
    *,
    spmf_jar: Path | None,
    spmf_algorithm: str,
    input_path: Path,
    output_path: Path,
    min_support_count: int,
    n_transactions: int,
    max_len: int,
    timeout_sec: int,
) -> dict:
    if spmf_jar is None:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": "SPMF JAR path not provided (--spmf-jar).",
        }
    if not spmf_jar.exists():
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"SPMF JAR not found: {spmf_jar}",
        }

    min_support_pct = _min_support_ratio(min_support_count, n_transactions) * 100.0
    min_support_arg = f"{min_support_pct:.8f}%"
    command = [
        "java",
        "-jar",
        str(spmf_jar),
        "run",
        spmf_algorithm,
        str(input_path),
        str(output_path),
        min_support_arg,
    ]

    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"TimeoutExpired: {exc}",
        }

    elapsed = perf_counter() - started
    if completed.returncode != 0:
        err = _truncate_text(completed.stderr or completed.stdout or "")
        return {
            "status": "error",
            "elapsed_seconds": float(elapsed),
            "rule_count": None,
            "itemset_count": None,
            "note": f"returncode={completed.returncode};stderr={err}",
        }
    if not output_path.exists():
        return {
            "status": "error",
            "elapsed_seconds": float(elapsed),
            "rule_count": None,
            "itemset_count": None,
            "note": f"SPMF output file was not created: {output_path}",
        }

    itemset_count = _count_itemsets_in_output(output_path, max_len=max_len)
    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": None,
        "itemset_count": int(itemset_count),
        "note": f"algorithm={spmf_algorithm};min_support={min_support_arg}",
    }


def _run_cpp_fim(
    *,
    cpp_fim_cmd: str,
    input_path: Path,
    output_path: Path,
    min_support_count: int,
    n_transactions: int,
    max_len: int,
    timeout_sec: int,
) -> dict:
    if not cpp_fim_cmd.strip():
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": "C++ command is not set (--cpp-fim-cmd).",
        }

    min_support_ratio = _min_support_ratio(min_support_count, n_transactions)
    try:
        command = cpp_fim_cmd.format(
            input=str(input_path),
            output=str(output_path),
            min_support_count=int(min_support_count),
            min_support_ratio=float(min_support_ratio),
            max_len=int(max_len),
            n_transactions=int(n_transactions),
        )
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"Failed to format --cpp-fim-cmd: {type(exc).__name__}: {exc}",
        }

    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
            shell=True,
        )
    except FileNotFoundError as exc:
        return {
            "status": "missing_dependency",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"{type(exc).__name__}: {exc}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "elapsed_seconds": None,
            "rule_count": None,
            "itemset_count": None,
            "note": f"TimeoutExpired: {exc}",
        }

    elapsed = perf_counter() - started
    if completed.returncode != 0:
        err = _truncate_text(completed.stderr or completed.stdout or "")
        return {
            "status": "error",
            "elapsed_seconds": float(elapsed),
            "rule_count": None,
            "itemset_count": None,
            "note": f"returncode={completed.returncode};stderr={err}",
        }

    if not output_path.exists():
        return {
            "status": "error",
            "elapsed_seconds": float(elapsed),
            "rule_count": None,
            "itemset_count": None,
            "note": f"Expected output file missing: {output_path}",
        }

    itemset_count = _count_itemsets_in_output(output_path, max_len=max_len)
    return {
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "rule_count": None,
        "itemset_count": int(itemset_count),
        "note": "external_cpp_command",
    }


def summarize_records(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    summary = (
        ok.groupby(["algorithm", "repeat_factor"], as_index=False)
        .agg(
            runs=("elapsed_seconds", "count"),
            mean_s=("elapsed_seconds", "mean"),
            median_s=("elapsed_seconds", "median"),
            std_s=("elapsed_seconds", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        )
        .sort_values(["repeat_factor", "algorithm"])
    )
    return summary


def run_benchmark(
    *,
    repeat_factors: list[int],
    runs: int,
    algorithms: list[str],
    min_support_count: int,
    min_confidence: float,
    max_len: int,
    max_apyori_records: int,
    output_dir: Path,
    tag: str = "",
    spmf_jar: Path | None = None,
    spmf_timeout_sec: int = 300,
    spmf_fpgrowth_algo: str = "FPGrowth_itemsets",
    spmf_eclat_algo: str = "Eclat",
    cpp_fim_cmd: str = "",
    cpp_timeout_sec: int = 300,
    warmup_runs: int = 0,
) -> tuple[list[dict], dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    jsonl_path = output_dir / f"fim_compare_runs{suffix}_{ts}.jsonl"
    csv_path = output_dir / f"fim_compare_runs{suffix}_{ts}.csv"
    latest_jsonl = output_dir / "fim_compare_runs_latest.jsonl"
    latest_csv = output_dir / "fim_compare_runs_latest.csv"

    for path in (jsonl_path, csv_path, latest_jsonl, latest_csv):
        if path.exists():
            path.unlink()

    records: list[dict] = []
    run_count = max(1, int(runs))
    warmup_count = max(0, int(warmup_runs))
    algorithms = [ALGORITHM_ALIASES.get(a, a) for a in algorithms]
    interrupted = False

    def _save_row(row: dict) -> None:
        records.append(row)
        _append_jsonl_row(jsonl_path, row)
        _append_csv_row(csv_path, row)
        _append_jsonl_row(latest_jsonl, row)
        _append_csv_row(latest_csv, row)

    try:
        for repeat_factor in repeat_factors:
            print(f"\nPreparing data for repeat_factor={repeat_factor}")
            frame = _load_telco(repeat_factor=repeat_factor)
            rows = int(frame.shape[0])
            tx_frame = frame[TX_COLUMNS].astype(str)

            tx_started = perf_counter()
            transactions = _build_transactions(tx_frame)
            tx_prep_seconds = perf_counter() - tx_started

            bitset_cpu_model = None
            bitset_cpu_prep_seconds = None
            if "bitset_fim_cpu" in algorithms:
                bitset_cpu_model, bitset_cpu_prep_seconds = BitsetFIMCPU.from_tx_frame(tx_frame)

            bitset_gpu_model = None
            bitset_gpu_prep_seconds = None
            bitset_gpu_model_error = None
            if "bitset_fim_gpu" in algorithms:
                try:
                    bitset_gpu_model, bitset_gpu_prep_seconds = BitsetFIMGPU.from_tx_frame(tx_frame)
                except Exception as exc:
                    bitset_gpu_model = None
                    bitset_gpu_prep_seconds = None
                    bitset_gpu_model_error = f"{type(exc).__name__}: {exc}"

            mlxtend_onehot = None
            mlxtend_prep_seconds = None
            if "mlxtend_apriori" in algorithms or "mlxtend_fpgrowth" in algorithms:
                mlxtend_started = perf_counter()
                mlxtend_onehot = _build_onehot_frame(tx_frame)
                mlxtend_prep_seconds = perf_counter() - mlxtend_started

            external_input_prep_seconds = None
            integer_transactions = None
            needs_external_input = any(
                algo in {"spmf_fpgrowth", "spmf_eclat", "cpp_fim"} for algo in algorithms
            )
            if needs_external_input:
                external_started = perf_counter()
                integer_transactions = _transactions_to_integer_ids(transactions)
                external_input_prep_seconds = perf_counter() - external_started

            print(
                "rows="
                f"{rows}, tx_prep_seconds={tx_prep_seconds:.4f}, "
                f"bitset_cpu_prep_seconds={bitset_cpu_prep_seconds}, "
                f"bitset_gpu_prep_seconds={bitset_gpu_prep_seconds}, "
                f"mlxtend_prep_seconds={mlxtend_prep_seconds}, "
                f"external_input_prep_seconds={external_input_prep_seconds}"
            )

            with tempfile.TemporaryDirectory(prefix="fim_compare_") as temp_dir:
                temp_dir_path = Path(temp_dir)
                external_input_path = None
                if needs_external_input and integer_transactions is not None:
                    external_input_path = temp_dir_path / f"transactions_repeat{repeat_factor}.txt"
                    _write_integer_transactions(external_input_path, integer_transactions)

                def _run_algorithm(algo: str, run_index: int) -> dict:
                    if algo == "bitset_fim_cpu":
                        assert bitset_cpu_model is not None
                        return _run_bitset_fim_cpu(
                            model=bitset_cpu_model,
                            min_support_count=min_support_count,
                            min_confidence=min_confidence,
                            max_len=max_len,
                        )
                    if algo == "bitset_fim_gpu":
                        return _run_bitset_fim_gpu(
                            model=bitset_gpu_model,
                            model_error=bitset_gpu_model_error,
                            min_support_count=min_support_count,
                            min_confidence=min_confidence,
                            max_len=max_len,
                        )
                    if algo == "apyori":
                        return _run_apyori(
                            transactions=transactions,
                            min_support_count=min_support_count,
                            min_confidence=min_confidence,
                            max_len=max_len,
                            max_records=max_apyori_records,
                        )
                    if algo == "pyfim_apriori":
                        return _run_pyfim_itemsets(
                            method_name="apriori",
                            transactions=transactions,
                            min_support_count=min_support_count,
                            max_len=max_len,
                        )
                    if algo == "pyfim_eclat":
                        return _run_pyfim_itemsets(
                            method_name="eclat",
                            transactions=transactions,
                            min_support_count=min_support_count,
                            max_len=max_len,
                        )
                    if algo == "mlxtend_apriori":
                        if mlxtend_onehot is None:
                            raise RuntimeError("Internal error: mlxtend one-hot frame was not prepared.")
                        return _run_mlxtend_itemsets(
                            method_name="apriori",
                            onehot=mlxtend_onehot,
                            min_support_count=min_support_count,
                            min_confidence=min_confidence,
                            max_len=max_len,
                        )
                    if algo == "mlxtend_fpgrowth":
                        if mlxtend_onehot is None:
                            raise RuntimeError("Internal error: mlxtend one-hot frame was not prepared.")
                        return _run_mlxtend_itemsets(
                            method_name="fpgrowth",
                            onehot=mlxtend_onehot,
                            min_support_count=min_support_count,
                            min_confidence=min_confidence,
                            max_len=max_len,
                        )
                    if algo == "spmf_fpgrowth":
                        if external_input_path is None:
                            raise RuntimeError("Internal error: external input file path is missing.")
                        spmf_out = temp_dir_path / f"spmf_fpgrowth_run{run_index}.txt"
                        return _run_spmf_itemsets(
                            spmf_jar=spmf_jar,
                            spmf_algorithm=spmf_fpgrowth_algo,
                            input_path=external_input_path,
                            output_path=spmf_out,
                            min_support_count=min_support_count,
                            n_transactions=rows,
                            max_len=max_len,
                            timeout_sec=spmf_timeout_sec,
                        )
                    if algo == "spmf_eclat":
                        if external_input_path is None:
                            raise RuntimeError("Internal error: external input file path is missing.")
                        spmf_out = temp_dir_path / f"spmf_eclat_run{run_index}.txt"
                        return _run_spmf_itemsets(
                            spmf_jar=spmf_jar,
                            spmf_algorithm=spmf_eclat_algo,
                            input_path=external_input_path,
                            output_path=spmf_out,
                            min_support_count=min_support_count,
                            n_transactions=rows,
                            max_len=max_len,
                            timeout_sec=spmf_timeout_sec,
                        )
                    if algo == "cpp_fim":
                        if external_input_path is None:
                            raise RuntimeError("Internal error: external input file path is missing.")
                        cpp_out = temp_dir_path / f"cpp_fim_run{run_index}.txt"
                        return _run_cpp_fim(
                            cpp_fim_cmd=cpp_fim_cmd,
                            input_path=external_input_path,
                            output_path=cpp_out,
                            min_support_count=min_support_count,
                            n_transactions=rows,
                            max_len=max_len,
                            timeout_sec=cpp_timeout_sec,
                        )
                    raise ValueError(f"Unsupported algorithm: {algo}")

                if warmup_count > 0:
                    print(f"repeat={repeat_factor} warmup_runs={warmup_count}")
                    for warmup_index in range(warmup_count):
                        print(f"repeat={repeat_factor} warmup={warmup_index + 1}/{warmup_count}")
                        for algo in algorithms:
                            _run_algorithm(algo=algo, run_index=-(warmup_index + 1))
                    _sync_cupy_default_stream()

                for run_index in range(run_count):
                    run_no = run_index + 1
                    print(f"repeat={repeat_factor} run={run_no}/{run_count}")

                    for algo in algorithms:
                        result = _run_algorithm(algo=algo, run_index=run_index)
                        row = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "algorithm": algo,
                            "repeat_factor": int(repeat_factor),
                            "run_index": int(run_index),
                            "rows": rows,
                            "tx_prep_seconds": float(tx_prep_seconds),
                            "bitset_prep_seconds": (
                                None if bitset_cpu_prep_seconds is None else float(bitset_cpu_prep_seconds)
                            ),
                            "bitset_cpu_prep_seconds": (
                                None if bitset_cpu_prep_seconds is None else float(bitset_cpu_prep_seconds)
                            ),
                            "bitset_gpu_prep_seconds": (
                                None if bitset_gpu_prep_seconds is None else float(bitset_gpu_prep_seconds)
                            ),
                            "mlxtend_prep_seconds": (
                                None if mlxtend_prep_seconds is None else float(mlxtend_prep_seconds)
                            ),
                            "external_input_prep_seconds": (
                                None if external_input_prep_seconds is None else float(external_input_prep_seconds)
                            ),
                            **result,
                        }
                        _save_row(row)
    except KeyboardInterrupt:
        interrupted = True
        print("\nBenchmark interrupted by user. Partial results were saved.")

    output_paths = {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "latest_jsonl": latest_jsonl,
        "latest_csv": latest_csv,
    }
    if interrupted:
        print("Returning partial in-memory records and output paths.")
    return records, output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bitset FIM (CPU/GPU) against Python/Java/C++ baselines."
    )
    parser.add_argument("--repeat-factors", type=str, default="20,200", help="Comma-separated repeat factors.")
    parser.add_argument("--runs", type=int, default=5, help="Runs per repeat factor.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup runs per repeat factor (executed before measured runs, not recorded).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=(
            "bitset_fim_cpu,bitset_fim_gpu,apyori,pyfim_apriori,pyfim_eclat,"
            "mlxtend_apriori,mlxtend_fpgrowth,spmf_fpgrowth,spmf_eclat"
        ),
        help=(
            "Comma-separated algorithms. Supported: "
            + ",".join(sorted(SUPPORTED_ALGORITHMS.union(ALGORITHM_ALIASES)))
        ),
    )
    parser.add_argument("--min-support-count", type=int, default=80, help="Minimum absolute support.")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence.")
    parser.add_argument("--max-len", type=int, default=3, help="Maximum itemset/rule length.")
    parser.add_argument(
        "--max-apyori-records",
        type=int,
        default=200000,
        help="Safety cap for number of apyori relation records.",
    )
    parser.add_argument(
        "--spmf-jar",
        type=Path,
        default=None,
        help="Path to SPMF JAR for spmf_* algorithms (e.g. spmf.jar).",
    )
    parser.add_argument(
        "--spmf-timeout-sec",
        type=int,
        default=300,
        help="Timeout in seconds for each SPMF call.",
    )
    parser.add_argument(
        "--spmf-fpgrowth-algo",
        type=str,
        default="FPGrowth_itemsets",
        help="SPMF algorithm name used for spmf_fpgrowth.",
    )
    parser.add_argument(
        "--spmf-eclat-algo",
        type=str,
        default="Eclat",
        help="SPMF algorithm name used for spmf_eclat.",
    )
    parser.add_argument(
        "--cpp-fim-cmd",
        type=str,
        default="",
        help=(
            "External C++ command template for cpp_fim. "
            "Supported placeholders: {input},{output},{min_support_count},"
            "{min_support_ratio},{max_len},{n_transactions}"
        ),
    )
    parser.add_argument(
        "--cpp-timeout-sec",
        type=int,
        default=300,
        help="Timeout in seconds for each external C++ baseline call.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for benchmark results.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional suffix tag for output files.")
    args = parser.parse_args()

    repeat_factors = _parse_int_list(args.repeat_factors)
    algorithms = _parse_algorithms(args.algorithms)

    records, output_paths = run_benchmark(
        repeat_factors=repeat_factors,
        runs=args.runs,
        algorithms=algorithms,
        min_support_count=args.min_support_count,
        min_confidence=args.min_confidence,
        max_len=args.max_len,
        max_apyori_records=args.max_apyori_records,
        output_dir=args.output_dir,
        tag=args.tag,
        spmf_jar=args.spmf_jar,
        spmf_timeout_sec=args.spmf_timeout_sec,
        spmf_fpgrowth_algo=args.spmf_fpgrowth_algo,
        spmf_eclat_algo=args.spmf_eclat_algo,
        cpp_fim_cmd=args.cpp_fim_cmd,
        cpp_timeout_sec=args.cpp_timeout_sec,
        warmup_runs=args.warmup_runs,
    )

    print("\nSaved outputs:")
    for key, value in output_paths.items():
        print(f"- {key}: {value}")

    summary = summarize_records(records)
    if summary.empty:
        print("\nNo successful runs to summarize.")
    else:
        print("\nSummary (status=ok):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
