"""Dataset profiling for action-rule mining performance characterisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from action_rules import ActionRules


def _density_hint(
    *,
    zero_word_fraction: float,
    top_pair_mean_jaccard: float,
    top_pair_mean_word_equal_fraction: float,
) -> str:
    """Classify a dataset as ``sparse_like``, ``dense_like``, or ``mixed``.

    The classification is based on the fraction of zero words in the packed
    bitsets and on pairwise overlap statistics of the most frequent condition
    items.
    """
    if zero_word_fraction >= 0.80:
        return "sparse_like"
    if zero_word_fraction <= 0.35 and (
        top_pair_mean_jaccard >= 0.20 or top_pair_mean_word_equal_fraction >= 0.70
    ):
        return "dense_like"
    return "mixed"


def _column_role(column_name: str) -> str:
    """Return ``'stable'``, ``'flexible'``, ``'target'``, or ``'unknown'``."""
    if "_<item_stable>_" in column_name:
        return "stable"
    if "_<item_flexible>_" in column_name:
        return "flexible"
    if "_<item_target>_" in column_name:
        return "target"
    return "unknown"


def _popcount_uint64(mask: np.ndarray) -> int:
    """Count set bits in a 1-D or 2-D uint64 array, returning a scalar."""
    array = np.asarray(mask, dtype=np.uint64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(array).sum(axis=1)[0])
    if hasattr(array, "bit_count"):
        return int(array.bit_count().sum(axis=1)[0])
    # Wegner / Hamming-weight fallback
    x = array.astype(np.uint64, copy=True)
    x -= (x >> 1) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x += x >> 8
    x += x >> 16
    x += x >> 32
    return int((x & np.uint64(0x7F)).sum(axis=1)[0])


def profile_dataset(
    action_rules: "ActionRules",
    data_frame,
    stable_attributes: list,
    flexible_attributes: list,
    target: str,
    top_k: int = 8,
) -> dict:
    """Profile a dataset and return a dictionary of structural characteristics.

    The profile reuses the same encoding and packed-bitset construction used
    by the miner but stops before the actual search begins.  It summarises
    dataset properties that are relevant for performance: row and item counts,
    bitset sparsity, item support statistics, pairwise overlap of the most
    frequent condition items, and a coarse ``density_hint``.

    Parameters
    ----------
    action_rules : ActionRules
        An *unfitted* ``ActionRules`` instance whose encoding parameters
        (``min_stable_attributes``, thresholds, …) are already set.
    data_frame : pandas.DataFrame
        The raw input data.
    stable_attributes, flexible_attributes : list
        Attribute lists passed to :meth:`ActionRules.fit`.
    target : str
        Target column name.
    top_k : int
        Number of highest-support items to include in the overlap analysis.

    Returns
    -------
    dict
        A flat dictionary with profiling metrics and a ``density_hint`` key.
    """
    action_rules.set_array_library(use_gpu=False, df=data_frame)

    encoded = action_rules.one_hot_encode(
        data=data_frame,
        stable_attributes=list(stable_attributes),
        flexible_attributes=list(flexible_attributes),
        target=target,
    )
    binary_data, columns = action_rules.df_to_array(encoded)
    binary_data = np.asarray(binary_data, dtype=np.uint8)
    bit_masks = np.asarray(action_rules.build_bit_masks(binary_data), dtype=np.uint64)
    stable_items_binding, flexible_items_binding, target_items_binding, column_values = (
        action_rules.get_bindings(columns, list(stable_attributes), list(flexible_attributes), target)
    )

    row_count = int(len(data_frame))
    item_count = int(binary_data.shape[0])
    words_per_item = int(bit_masks.shape[1]) if bit_masks.ndim == 2 else 0
    padding_transactions = max(0, (words_per_item * 64) - row_count)
    support_counts = np.asarray(binary_data.sum(axis=1), dtype=np.int64)
    support_ratios = (
        support_counts.astype(np.float64, copy=False) / float(row_count)
        if row_count > 0
        else np.zeros(item_count, dtype=np.float64)
    )
    zero_word_fraction = float(np.mean(bit_masks == 0)) if bit_masks.size > 0 else 0.0
    onehot_matrix_density = float(np.mean(binary_data)) if binary_data.size > 0 else 0.0

    # ------------------------------------------------------------------
    # Top-item overlap statistics
    # ------------------------------------------------------------------
    top_item_count = min(max(0, int(top_k)), item_count)
    sorted_indices = np.argsort(-support_counts, kind="stable")[:top_item_count].tolist()
    top_items = []
    for idx in sorted_indices:
        column_name = columns[idx]
        attribute, value = column_values.get(idx, ("", column_name))
        top_items.append(
            {
                "item_index": int(idx),
                "role": _column_role(column_name),
                "attribute": str(attribute),
                "value": str(value),
                "encoded_column": str(column_name),
                "support_count": int(support_counts[idx]),
                "support_ratio": float(support_ratios[idx]),
            }
        )

    condition_indices = [
        idx
        for idx in np.argsort(-support_counts, kind="stable").tolist()
        if _column_role(columns[idx]) != "target"
    ][:top_item_count]

    pair_count = 0
    pair_jaccard_sum = 0.0
    pair_containment_sum = 0.0
    pair_equal_word_sum = 0.0
    for left_pos, left_idx in enumerate(condition_indices):
        left_mask = bit_masks[left_idx]
        left_support = int(support_counts[left_idx])
        for right_idx in condition_indices[left_pos + 1:]:
            right_mask = bit_masks[right_idx]
            right_support = int(support_counts[right_idx])
            intersection_support = _popcount_uint64(left_mask & right_mask)
            union_support = left_support + right_support - intersection_support
            pair_jaccard_sum += (
                float(intersection_support) / float(union_support) if union_support > 0 else 1.0
            )
            pair_containment_sum += (
                float(intersection_support) / float(min(left_support, right_support))
                if min(left_support, right_support) > 0
                else 0.0
            )
            pair_equal_word_sum += float(np.mean(left_mask == right_mask))
            pair_count += 1

    top_pair_mean_jaccard = pair_jaccard_sum / float(pair_count) if pair_count > 0 else 0.0
    top_pair_mean_containment = pair_containment_sum / float(pair_count) if pair_count > 0 else 0.0
    top_pair_mean_word_equal_fraction = pair_equal_word_sum / float(pair_count) if pair_count > 0 else 0.0

    return {
        "profile_kind": "dataset_characteristics",
        "rows": row_count,
        "input_column_count": int(data_frame.shape[1]),
        "selected_attribute_count": int(len(stable_attributes) + len(flexible_attributes) + 1),
        "stable_attribute_count": int(len(stable_attributes)),
        "flexible_attribute_count": int(len(flexible_attributes)),
        "condition_item_count": int(sum(len(items) for items in stable_items_binding.values()))
        + int(sum(len(items) for items in flexible_items_binding.values())),
        "target_state_count": int(sum(len(items) for items in target_items_binding.values())),
        "onehot_column_count": item_count,
        "onehot_matrix_density": onehot_matrix_density,
        "bitmask_words_per_item": words_per_item,
        "bitmask_total_bytes": int(bit_masks.nbytes),
        "bitmask_padding_transactions": int(padding_transactions),
        "bitmask_zero_word_fraction": zero_word_fraction,
        "bitmask_nonzero_word_fraction": float(1.0 - zero_word_fraction),
        "item_support_count_min": int(support_counts.min()) if item_count > 0 else 0,
        "item_support_count_median": float(np.median(support_counts)) if item_count > 0 else 0.0,
        "item_support_count_mean": float(np.mean(support_counts)) if item_count > 0 else 0.0,
        "item_support_count_max": int(support_counts.max()) if item_count > 0 else 0,
        "item_support_ratio_min": float(support_ratios.min()) if item_count > 0 else 0.0,
        "item_support_ratio_median": float(np.median(support_ratios)) if item_count > 0 else 0.0,
        "item_support_ratio_mean": float(np.mean(support_ratios)) if item_count > 0 else 0.0,
        "item_support_ratio_max": float(support_ratios.max()) if item_count > 0 else 0.0,
        "top_item_count_considered": int(top_item_count),
        "top_condition_pair_count": int(pair_count),
        "top_condition_pair_mean_jaccard": float(top_pair_mean_jaccard),
        "top_condition_pair_mean_containment": float(top_pair_mean_containment),
        "top_condition_pair_mean_word_equal_fraction": float(top_pair_mean_word_equal_fraction),
        "density_hint": _density_hint(
            zero_word_fraction=zero_word_fraction,
            top_pair_mean_jaccard=top_pair_mean_jaccard,
            top_pair_mean_word_equal_fraction=top_pair_mean_word_equal_fraction,
        ),
        "top_items": top_items,
    }
