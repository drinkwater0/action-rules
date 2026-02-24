"""Class CandidateGenerator."""
import itertools
from typing import TYPE_CHECKING, Optional, Union

from action_rules.rules import Rules

if TYPE_CHECKING:
    import cupy
    import numpy


class CandidateGenerator:
    """
    Generate candidate branches for bitset-based action-rule mining.

    Attributes
    ----------
    frames_bit_masks : dict
        Packed target-state masks keyed by target item index.
    bit_masks : Union[numpy.ndarray, cupy.ndarray]
        Packed masks for all one-hot items.
    min_stable_attributes : int
        Minimum number of stable attributes required.
    min_flexible_attributes : int
        Minimum number of flexible attributes required.
    min_undesired_support : int
        Minimum support for the undesired state.
    min_desired_support : int
        Minimum support for the desired state.
    min_undesired_confidence : float
        Minimum confidence for the undesired state.
    min_desired_confidence : float
        Minimum confidence for the desired state.
    undesired_state : int
        The undesired state of the target attribute.
    desired_state : int
        The desired state of the target attribute.
    rules : Rules
        Rules object to store the generated classification rules.

    Methods
    -------
    generate_candidates(ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding,
                        actionable_attributes, stop_list, stop_list_itemset, undesired_state,
                        desired_state, verbose=False)
        Generate candidate action rules.
    reduce_candidates_by_min_attributes(k, actionable_attributes, stable_items_binding, flexible_items_binding)
        Reduce the candidate sets based on minimum attributes.
    process_stable_candidates(ar_prefix, itemset_prefix, reduced_stable_items_binding, stop_list, stable_candidates,
                              new_branches, verbose)
        Process stable candidates to generate new branches.
    process_flexible_candidates(ar_prefix, itemset_prefix, reduced_flexible_items_binding, stop_list, stop_list_itemset,
                                flexible_candidates, actionable_attributes,
                                new_branches, verbose)
        Process flexible candidates to generate new branches.
    process_items(attribute, items, itemset_prefix, stop_list_itemset, flexible_candidates, verbose)
        Process items to generate states and counts.
    update_new_branches(new_branches, stable_candidates, flexible_candidates)
        Update new branches with stable and flexible candidates.
    in_stop_list(ar_prefix, stop_list)
        Check if the action rule prefix is in the stop list.
    """
    _gpu_support_kernel = None
    _gpu_support_kernel_multi = None
    _gpu_kernel_min_work = 512
    _gpu_batch_budget_mb = None
    _gpu_batch_free_mem_fraction = 0.5

    def __init__(
        self,
        min_stable_attributes: int,
        min_flexible_attributes: int,
        min_undesired_support: int,
        min_desired_support: int,
        min_undesired_confidence: float,
        min_desired_confidence: float,
        undesired_state: int,
        desired_state: int,
        rules: Rules,
        frames_bit_masks: Optional[dict] = None,
        bit_masks: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        gpu_batch_budget_mb: Optional[int] = None,
        gpu_batch_free_mem_fraction: float = 0.5,
        spill_gpu_masks_to_cpu: bool = False,
    ):
        """
        Initialize the CandidateGenerator class with the specified parameters.

        Parameters
        ----------
        min_stable_attributes : int
            Minimum number of stable attributes required.
        min_flexible_attributes : int
            Minimum number of flexible attributes required.
        min_undesired_support : int
            Minimum support for the undesired state.
        min_desired_support : int
            Minimum support for the desired state.
        min_undesired_confidence : float
            Minimum confidence for the undesired state.
        min_desired_confidence : float
            Minimum confidence for the desired state.
        undesired_state : int
            The undesired state of the target attribute.
        desired_state : int
            The desired state of the target attribute.
        rules : Rules
            Rules object to store the generated classification rules.
        frames_bit_masks : dict, optional
            Packed bit-mask view of frames keyed by target item index.
        bit_masks : Union[numpy.ndarray, cupy.ndarray], optional
            Packed bit masks for all attributes (as produced by build_bit_masks).
        gpu_batch_budget_mb : int, optional
            Optional hard cap (MB) used for GPU batch chunking in bitset support.
            If None, chunking budget is derived from currently free GPU memory.
        gpu_batch_free_mem_fraction : float, optional
            Fraction of free GPU memory considered usable for one support batch.
        spill_gpu_masks_to_cpu : bool, optional
            If True, branch masks are moved to CPU after GPU intersections to
            avoid queue growth in GPU memory under strict memory caps.

        Notes
        -----
        The CandidateGenerator class is designed to facilitate the generation of candidate action rules by
        iterating over combinations of stable and flexible attributes. The class maintains a reference to the
        rules object where generated rules are stored.
        """
        self.frames_bit_masks = frames_bit_masks or {}
        self.bit_masks = bit_masks
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.undesired_state = undesired_state
        self.desired_state = desired_state
        self.rules = rules
        self._gpu_batch_budget_mb = gpu_batch_budget_mb
        self._gpu_batch_free_mem_fraction = gpu_batch_free_mem_fraction
        self._spill_gpu_masks_to_cpu = spill_gpu_masks_to_cpu

    def generate_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        stable_items_binding: dict,
        flexible_items_binding: dict,
        actionable_attributes: int,
        stop_list: list,
        stop_list_itemset: list,
        undesired_state: int,
        desired_state: int,
        verbose: bool = False,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        parent_undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        parent_desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
    ) -> list:
        """
        Generate candidate action rules.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed bit mask for the undesired branch (intersection so far).
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed bit mask for the desired branch (intersection so far).
        parent_undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Parent packed mask used for lazy intersection when branch masks are not materialized.
        parent_desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Parent packed mask used for lazy intersection when branch masks are not materialized.
        actionable_attributes : int
            Number of actionable attributes.
        stop_list : list
            List of stop combinations.
        stop_list_itemset : list
            List of stop itemsets.
        undesired_state : int
            The undesired state of the target attribute.
        desired_state : int
            The desired state of the target attribute.
        verbose : bool, optional
            If True, enables verbose output. Default is False.

        Returns
        -------
        list
            List of new branches generated.

        Notes
        -----
        This method generates candidate action rules by processing both stable and flexible attributes.
        It first reduces the candidate sets based on the minimum attributes, then processes stable and
        flexible candidates to generate new branches. The new branches are updated with the candidates
        and returned.
        """
        k = len(itemset_prefix) + 1
        reduced_stable_items_binding, reduced_flexible_items_binding = self.reduce_candidates_by_min_attributes(
            k, actionable_attributes, stable_items_binding, flexible_items_binding
        )

        bitset_undesired_mask = None
        bitset_desired_mask = None
        bitset_undesired_mask, bitset_desired_mask = self._resolve_bitset_masks(
            itemset_prefix,
            undesired_mask_bitset,
            desired_mask_bitset,
            parent_undesired_mask_bitset,
            parent_desired_mask_bitset,
            undesired_state,
            desired_state,
        )
        if bitset_undesired_mask is None or bitset_desired_mask is None:
            return []

        stable_candidates = {attribute: list(items) for attribute, items in stable_items_binding.items()}
        flexible_candidates = {attribute: list(items) for attribute, items in flexible_items_binding.items()}
        new_branches = []  # type: list

        self.process_stable_candidates(
            ar_prefix,
            itemset_prefix,
            reduced_stable_items_binding,
            stop_list,
            stable_candidates,
            new_branches,
            verbose,
            bitset_undesired_mask,
            bitset_desired_mask,
        )
        self.process_flexible_candidates(
            ar_prefix,
            itemset_prefix,
            reduced_flexible_items_binding,
            stop_list,
            stop_list_itemset,
            flexible_candidates,
            actionable_attributes,
            new_branches,
            verbose,
            undesired_mask_bitset=bitset_undesired_mask,
            desired_mask_bitset=bitset_desired_mask,
        )
        self.update_new_branches(new_branches, stable_candidates, flexible_candidates)

        return new_branches

    def generate_candidates_batch(
        self,
        candidates: list,
        stop_list: list,
        stop_list_itemset: list,
        undesired_state: int,
        desired_state: int,
        verbose: bool = False,
        batch_size: int = 32,
    ) -> list:
        """
        Generate candidates for a batch of branches, using GPU batching when possible.
        """
        if not candidates:
            return []
        if not self._can_use_gpu_batching(verbose):
            return self._generate_candidates_sequential(
                candidates, stop_list, stop_list_itemset, undesired_state, desired_state, verbose
            )

        new_branches_all = []
        batch_contexts = []
        resolved_batch_size = max(1, int(batch_size))

        for candidate in candidates:
            context = self._build_batch_context(candidate, undesired_state, desired_state)
            if context is None:
                continue
            batch_contexts.append(context)
            if len(batch_contexts) >= resolved_batch_size:
                new_branches_all.extend(
                    self._flush_batch_contexts(
                        batch_contexts,
                        stop_list,
                        stop_list_itemset,
                        undesired_state,
                        desired_state,
                        verbose,
                    )
                )
                batch_contexts = []

        if batch_contexts:
            new_branches_all.extend(
                self._flush_batch_contexts(
                    batch_contexts,
                    stop_list,
                    stop_list_itemset,
                    undesired_state,
                    desired_state,
                    verbose,
                )
            )

        return new_branches_all

    def _can_use_gpu_batching(self, verbose: bool) -> bool:
        """
        Return True when candidate batching can run on GPU bit masks.
        """
        if verbose or self.bit_masks is None:
            return False
        return hasattr(self.bit_masks, "__cuda_array_interface__")

    def _build_batch_context(
        self,
        candidate: dict,
        undesired_state: int,
        desired_state: int,
    ) -> Optional[dict]:
        """Prepare one context object used by the GPU batch expansion path."""
        bitset_undesired_mask, bitset_desired_mask = self._resolve_bitset_masks(
            candidate.get("itemset_prefix", tuple()),
            candidate.get("undesired_mask_bitset"),
            candidate.get("desired_mask_bitset"),
            candidate.get("parent_undesired_mask_bitset"),
            candidate.get("parent_desired_mask_bitset"),
            undesired_state,
            desired_state,
        )
        if bitset_undesired_mask is None or bitset_desired_mask is None:
            return None

        next_size = len(candidate["itemset_prefix"]) + 1
        reduced_stable_items_binding, reduced_flexible_items_binding = self.reduce_candidates_by_min_attributes(
            next_size,
            candidate["actionable_attributes"],
            candidate["stable_items_binding"],
            candidate["flexible_items_binding"],
        )
        return {
            "candidate": candidate,
            "ar_prefix": candidate["ar_prefix"],
            "itemset_prefix": candidate["itemset_prefix"],
            "reduced_stable_items_binding": reduced_stable_items_binding,
            "reduced_flexible_items_binding": reduced_flexible_items_binding,
            "stable_candidates": {
                attribute: list(items) for attribute, items in candidate["stable_items_binding"].items()
            },
            "flexible_candidates": {
                attribute: list(items) for attribute, items in candidate["flexible_items_binding"].items()
            },
            "bitset_undesired_mask": bitset_undesired_mask,
            "bitset_desired_mask": bitset_desired_mask,
            "actionable_attributes": candidate["actionable_attributes"],
        }

    def _flush_batch_contexts(
        self,
        batch_contexts: list,
        stop_list: list,
        stop_list_itemset: list,
        undesired_state: int,
        desired_state: int,
        verbose: bool,
    ) -> list:
        """Run one prepared context batch; fallback to sequential mode on failures."""
        batch_result = self._process_gpu_batch(batch_contexts, stop_list, stop_list_itemset)
        if batch_result is not None:
            return batch_result
        return self._generate_candidates_sequential(
            [context["candidate"] for context in batch_contexts],
            stop_list,
            stop_list_itemset,
            undesired_state,
            desired_state,
            verbose,
        )

    def _generate_candidates_sequential(
        self,
        candidates: list,
        stop_list: list,
        stop_list_itemset: list,
        undesired_state: int,
        desired_state: int,
        verbose: bool,
    ) -> list:
        new_branches = []
        for candidate in candidates:
            new_branches.extend(
                self.generate_candidates(
                    **candidate,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                    undesired_state=undesired_state,
                    desired_state=desired_state,
                    verbose=verbose,
                )
            )
        return new_branches

    def _process_gpu_batch(
        self,
        batch_contexts: list,
        stop_list: list,
        stop_list_itemset: list,
    ) -> Optional[list]:
        try:
            import cupy as cp
        except ImportError:
            return None

        work_candidate_indices = []
        work_item_indices = []
        for ctx_index, context in enumerate(batch_contexts):
            context["stable_slices"] = []
            context["flex_slices"] = []
            for attribute, items in context["reduced_stable_items_binding"].items():
                active_items = self._active_stable_items(context["ar_prefix"], items, stop_list)
                if not active_items:
                    continue
                start = len(work_item_indices)
                work_item_indices.extend(active_items)
                work_candidate_indices.extend([ctx_index] * len(active_items))
                context["stable_slices"].append((attribute, active_items, start))

            for attribute, items in context["reduced_flexible_items_binding"].items():
                new_ar_prefix = context["ar_prefix"] + (attribute,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue
                active_items = self._active_flexible_items(context["itemset_prefix"], items, stop_list_itemset)
                if not active_items:
                    continue
                start = len(work_item_indices)
                work_item_indices.extend(active_items)
                work_candidate_indices.extend([ctx_index] * len(active_items))
                context["flex_slices"].append((attribute, active_items, start))

        if not work_item_indices:
            return []

        try:
            branch_masks_a = cp.stack(
                [
                    cp.asarray(context["bitset_undesired_mask"], dtype=cp.uint64).reshape(-1)
                    for context in batch_contexts
                ],
                axis=0,
            )
            branch_masks_b = cp.stack(
                [
                    cp.asarray(context["bitset_desired_mask"], dtype=cp.uint64).reshape(-1)
                    for context in batch_contexts
                ],
                axis=0,
            )
        except Exception:
            return None

        supports = self._gpu_bitset_support_batch_multi(
            branch_masks_a,
            branch_masks_b,
            work_candidate_indices,
            work_item_indices,
        )
        if supports is None:
            return None

        new_branches_all = []
        undesired_supports_all, desired_supports_all = supports
        for context in batch_contexts:
            new_branches = []
            stable_candidates = context["stable_candidates"]
            flexible_candidates = context["flexible_candidates"]

            for attribute, items, start in context["stable_slices"]:
                for offset, item in enumerate(items):
                    new_ar_prefix = context["ar_prefix"] + (item,)
                    if self.in_stop_list(new_ar_prefix, stop_list):
                        continue
                    index = start + offset
                    undesired_support = undesired_supports_all[index]
                    desired_support = desired_supports_all[index]
                    if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                        stable_candidates[attribute].remove(item)
                        self._add_stop_entry(stop_list, new_ar_prefix)
                    else:
                        new_branches.append(
                            {
                                "ar_prefix": new_ar_prefix,
                                "itemset_prefix": new_ar_prefix,
                                "item": item,
                                "undesired_mask_bitset": None,
                                "desired_mask_bitset": None,
                                "parent_undesired_mask_bitset": context["bitset_undesired_mask"],
                                "parent_desired_mask_bitset": context["bitset_desired_mask"],
                                "actionable_attributes": 0,
                            }
                        )

            for attribute, items, start in context["flex_slices"]:
                new_ar_prefix = context["ar_prefix"] + (attribute,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue
                undesired_states = []
                desired_states = []
                undesired_count = 0
                desired_count = 0
                kept_items = []
                for offset, item in enumerate(items):
                    if self.in_stop_list(context["itemset_prefix"] + (item,), stop_list_itemset):
                        continue
                    index = start + offset
                    undesired_support = undesired_supports_all[index]
                    desired_support = desired_supports_all[index]

                    undesired_conf = self.rules.calculate_confidence(undesired_support, desired_support)
                    if undesired_support >= self.min_undesired_support:
                        undesired_count += 1
                        if undesired_conf >= self.min_undesired_confidence:
                            undesired_states.append(
                                {
                                    "item": item,
                                    "support": undesired_support,
                                    "confidence": undesired_conf,
                                }
                            )
                        else:
                            self.rules.add_prefix_without_conf(new_ar_prefix, False)

                    desired_conf = self.rules.calculate_confidence(desired_support, undesired_support)
                    if desired_support >= self.min_desired_support:
                        desired_count += 1
                        if desired_conf >= self.min_desired_confidence:
                            desired_states.append(
                                {
                                    "item": item,
                                    "support": desired_support,
                                    "confidence": desired_conf,
                                }
                            )
                        else:
                            self.rules.add_prefix_without_conf(new_ar_prefix, True)

                    if desired_support < self.min_desired_support and undesired_support < self.min_undesired_support:
                        flexible_candidates[attribute].remove(item)
                        self._add_stop_entry(stop_list_itemset, context["itemset_prefix"] + (item,))
                        continue

                    kept_items.append(item)

                if context["actionable_attributes"] == 0 and (undesired_count == 0 or desired_count == 0):
                    del flexible_candidates[attribute]
                    self._add_stop_entry(stop_list, context["ar_prefix"] + (attribute,))
                else:
                    for item in kept_items:
                        new_branches.append(
                            {
                                "ar_prefix": new_ar_prefix,
                                "itemset_prefix": context["itemset_prefix"] + (item,),
                                "item": item,
                                "undesired_mask_bitset": None,
                                "desired_mask_bitset": None,
                                "parent_undesired_mask_bitset": context["bitset_undesired_mask"],
                                "parent_desired_mask_bitset": context["bitset_desired_mask"],
                                "actionable_attributes": context["actionable_attributes"] + 1,
                            }
                        )
                    if context["actionable_attributes"] + 1 >= self.min_flexible_attributes:
                        self.rules.add_classification_rules(
                            new_ar_prefix,
                            context["itemset_prefix"],
                            undesired_states,
                            desired_states,
                        )

            self.update_new_branches(new_branches, stable_candidates, flexible_candidates)
            new_branches_all.extend(new_branches)

        return new_branches_all

    @staticmethod
    def _add_stop_entry(stop_collection, value: tuple) -> None:
        """
        Add a stop entry to a list or set without branching at call sites.
        """
        if hasattr(stop_collection, "add"):
            stop_collection.add(value)
        else:
            stop_collection.append(value)

    def _active_stable_items(self, ar_prefix: tuple, items: list, stop_list: list) -> list:
        """Return stable items not blocked by `stop_list`."""
        active_items = []
        for item in items:
            if self.in_stop_list(ar_prefix + (item,), stop_list):
                continue
            active_items.append(item)
        return active_items

    def _active_flexible_items(self, itemset_prefix: tuple, items: list, stop_list_itemset: list) -> list:
        """Return flexible items not blocked by `stop_list_itemset`."""
        active_items = []
        for item in items:
            if self.in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue
            active_items.append(item)
        return active_items

    def _resolve_bitset_masks(
        self,
        itemset_prefix: tuple,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']],
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']],
        parent_undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']],
        parent_desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']],
        undesired_state: int,
        desired_state: int,
    ) -> tuple:
        if self.bit_masks is None or not self.frames_bit_masks:
            return None, None
        base_undesired = self.frames_bit_masks.get(undesired_state)
        base_desired = self.frames_bit_masks.get(desired_state)
        if base_undesired is None or base_desired is None:
            return None, None

        bitset_undesired_mask = undesired_mask_bitset
        bitset_desired_mask = desired_mask_bitset
        last_item = itemset_prefix[-1] if itemset_prefix else None
        if bitset_undesired_mask is None and parent_undesired_mask_bitset is not None and last_item is not None:
            bitset_undesired_mask = self._intersect_bit_mask(parent_undesired_mask_bitset, last_item)
        if bitset_desired_mask is None and parent_desired_mask_bitset is not None and last_item is not None:
            bitset_desired_mask = self._intersect_bit_mask(parent_desired_mask_bitset, last_item)
        if bitset_undesired_mask is None:
            bitset_undesired_mask = base_undesired
        if bitset_desired_mask is None:
            bitset_desired_mask = base_desired
        return bitset_undesired_mask, bitset_desired_mask

    def reduce_candidates_by_min_attributes(
        self, k: int, actionable_attributes: int, stable_items_binding: dict, flexible_items_binding: dict
    ) -> tuple:
        """
        Reduce the candidate sets based on minimum attributes.

        Parameters
        ----------
        k : int
            Length of the itemset prefix plus one.
        actionable_attributes : int
            Number of actionable attributes.
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.

        Returns
        -------
        tuple
            Tuple containing the reduced stable and flexible items bindings.

        Notes
        -----
        This method reduces the candidate sets by removing attributes that do not meet the minimum
        number of stable or flexible attributes required. The reduction is based on the length of the
        itemset prefix plus one (k) and the number of actionable attributes.
        """
        number_of_stable_attributes = len(stable_items_binding) - (self.min_stable_attributes - k)
        if k > self.min_stable_attributes:
            number_of_flexible_attributes = len(flexible_items_binding) - (
                self.min_flexible_attributes - actionable_attributes - 1
            )
        else:
            number_of_flexible_attributes = 0
        stable_key_count = number_of_stable_attributes
        if stable_key_count < 0:
            stable_key_count = len(stable_items_binding) + stable_key_count
            if stable_key_count < 0:
                stable_key_count = 0
        flexible_key_count = number_of_flexible_attributes
        if flexible_key_count < 0:
            flexible_key_count = len(flexible_items_binding) + flexible_key_count
            if flexible_key_count < 0:
                flexible_key_count = 0
        reduced_stable_items_binding = {
            key: stable_items_binding[key]
            for key in itertools.islice(stable_items_binding.keys(), stable_key_count)
        }
        reduced_flexible_items_binding = {
            key: flexible_items_binding[key]
            for key in itertools.islice(flexible_items_binding.keys(), flexible_key_count)
        }
        return reduced_stable_items_binding, reduced_flexible_items_binding

    def process_stable_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_stable_items_binding: dict,
        stop_list: list,
        stable_candidates: dict,
        new_branches: list,
        verbose: bool,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
    ):
        """
        Process stable candidates to generate new branches.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        reduced_stable_items_binding : dict
            Dictionary containing reduced bindings for stable items.
        stop_list : list
            List of stop combinations.
        stable_candidates : dict
            Dictionary containing stable candidates.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.
        new_branches : list
            List of new branches generated.
        verbose : bool
            If True, enables verbose output.

        Notes
        -----
        This method processes stable candidates by iterating over the reduced stable items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list.
        """
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return
        for attribute, items in reduced_stable_items_binding.items():
            active_items = self._active_stable_items(ar_prefix, items, stop_list)
            if not active_items:
                continue
            undesired_supports = self._bitset_support_batch(undesired_mask_bitset, active_items)
            desired_supports = self._bitset_support_batch(desired_mask_bitset, active_items)
            item_iter = zip(active_items, undesired_supports, desired_supports)

            for item, undesired_support, desired_support in item_iter:
                new_ar_prefix = ar_prefix + (item,)

                if verbose:
                    print('SUPPORT for: ' + str(itemset_prefix + (item,)))
                    print('_________________________________________________')
                    print('- extended by stable attribute')
                    print('undesired state support: ' + str(undesired_support))
                    print('desired state support: ' + str(desired_support))
                    print('')

                if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                    stable_candidates[attribute].remove(item)
                    self._add_stop_entry(stop_list, new_ar_prefix)
                else:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': new_ar_prefix,
                            'item': item,
                            'undesired_mask_bitset': None,
                            'desired_mask_bitset': None,
                            'parent_undesired_mask_bitset': undesired_mask_bitset,
                            'parent_desired_mask_bitset': desired_mask_bitset,
                            'actionable_attributes': 0,
                        }
                    )

    def get_support(
        self,
        mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'],
        item: int,
    ) -> int:
        """
        Calculate support for one item under the provided packed branch mask.

        Parameters
        ----------
        mask_bitset : Union[numpy.ndarray, cupy.ndarray]
            Packed branch mask representing currently surviving transactions.
        item : int
            Item row index in `self.bit_masks`.

        Returns
        -------
        int
            Support count for the given item.
        """
        return self._bitset_support(mask_bitset, item)

    def _bitset_support(
        self, mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'], item: int
    ) -> int:
        """
        Compute support using packed bit masks by intersecting with the given mask.
        """
        attribute_mask = self.bit_masks[item]  # type: ignore[index]
        intersection = attribute_mask & mask_bitset
        return self._popcount(intersection)

    @staticmethod
    def _contiguous_indexer(items) -> Optional[slice]:
        """
        Return a slice for contiguous item indices to avoid advanced indexing copies.
        """
        if items is None:
            return None
        try:
            item_count = len(items)
        except Exception:
            return None
        if item_count == 0:
            return None
        if isinstance(items, range):
            if items.step == 1:
                return slice(items.start, items.stop)
            return None
        try:
            first = items[0]
            last = items[-1]
        except Exception:
            return None
        if (last - first + 1) != len(items):
            return None
        for offset, value in enumerate(items):
            if value != first + offset:
                return None
        return slice(first, last + 1)

    def _bitset_support_batch(
        self, mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'], items: list
    ) -> list[int]:
        """
        Compute support for multiple items in one packed-mask pass.
        """
        if self.bit_masks is None or not items:
            return []
        working_mask = mask_bitset
        if (
            hasattr(self.bit_masks, "__cuda_array_interface__")
            and not hasattr(mask_bitset, "__cuda_array_interface__")
        ):
            try:
                import cupy as cp

                working_mask = cp.asarray(mask_bitset, dtype=cp.uint64)
            except Exception:
                working_mask = mask_bitset

        use_gpu_kernel = False
        if hasattr(working_mask, "__cuda_array_interface__"):
            use_gpu_kernel = self._should_use_gpu_kernel(working_mask, len(items))
        if use_gpu_kernel:
            gpu_counts = self._gpu_bitset_support_batch(working_mask, items)
            if gpu_counts is not None:
                return gpu_counts
        indexer = self._contiguous_indexer(items)
        if indexer is None:
            intersections = self.bit_masks[items] & working_mask
        else:
            intersections = self.bit_masks[indexer] & working_mask
        return self._popcount_rows(intersections)

    @classmethod
    def _should_use_gpu_kernel(cls, mask_bitset, item_count: int) -> bool:
        """
        Use the CUDA kernel only when the batch has enough work to amortize launch overhead.
        """
        num_words = None
        try:
            num_words = int(mask_bitset.size)
        except Exception:
            try:
                shape = getattr(mask_bitset, "shape", None)
                if shape:
                    num_words = int(shape[-1])
            except Exception:
                pass

        # If size cannot be inferred, prefer kernel path for safety.
        if num_words is None or num_words <= 0:
            return True
        return (item_count * num_words) >= cls._gpu_kernel_min_work

    @classmethod
    def _compute_gpu_chunk_items(
        cls,
        num_words: int,
        requested_items: int,
        budget_bytes: int,
        word_bytes: int = 8,
    ) -> int:
        """
        Compute how many items can be processed in one GPU chunk under a memory budget.
        """
        if requested_items <= 0:
            return 1
        if num_words <= 0 or budget_bytes <= 0 or word_bytes <= 0:
            return 1

        # Conservative estimate: item mask load + contiguous conversion + intermediate/output buffers.
        bytes_per_item = (num_words * word_bytes * 3) + word_bytes
        if bytes_per_item <= 0:
            return 1

        chunk_items = budget_bytes // bytes_per_item
        if chunk_items < 1:
            return 1
        return min(requested_items, int(chunk_items))

    @classmethod
    def _get_gpu_support_kernel(cls):
        """
        Lazily compile and cache a CUDA kernel for batched AND + popcount support.
        """
        if cls._gpu_support_kernel is not None:
            return cls._gpu_support_kernel

        try:
            import cupy as cp
        except ImportError:
            return None

        kernel_code = r"""
        extern "C" __global__
        void bitset_support_kernel(
            const unsigned long long* item_masks,
            const unsigned long long* branch_mask,
            int num_words,
            unsigned long long* out_support
        ) {
            extern __shared__ unsigned int shared_counts[];
            const int item_index = blockIdx.x;
            const int thread_id = threadIdx.x;
            unsigned int local_count = 0u;

            const unsigned long long* row_ptr = item_masks + ((size_t)item_index * (size_t)num_words);
            for (int word_index = thread_id; word_index < num_words; word_index += blockDim.x) {
                local_count += (unsigned int)__popcll(row_ptr[word_index] & branch_mask[word_index]);
            }

            shared_counts[thread_id] = local_count;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (thread_id < stride) {
                    shared_counts[thread_id] += shared_counts[thread_id + stride];
                }
                __syncthreads();
            }

            if (thread_id == 0) {
                out_support[item_index] = (unsigned long long)shared_counts[0];
            }
        }
        """

        try:
            cls._gpu_support_kernel = cp.RawKernel(kernel_code, "bitset_support_kernel")
        except Exception:
            cls._gpu_support_kernel = None
        return cls._gpu_support_kernel

    @classmethod
    def _get_gpu_support_kernel_multi(cls):
        """
        Lazily compile and cache a CUDA kernel for multi-branch batched support.
        """
        if cls._gpu_support_kernel_multi is not None:
            return cls._gpu_support_kernel_multi

        try:
            import cupy as cp
        except ImportError:
            return None

        kernel_code = r"""
        extern "C" __global__
        void bitset_support_kernel_multi(
            const unsigned long long* item_masks,
            const unsigned long long* branch_masks_a,
            const unsigned long long* branch_masks_b,
            const int* candidate_indices,
            const long long* item_indices,
            int num_words,
            unsigned long long* out_support_a,
            unsigned long long* out_support_b
        ) {
            extern __shared__ unsigned int shared_counts[];
            const int work_index = blockIdx.x;
            const int thread_id = threadIdx.x;

            const int candidate_index = candidate_indices[work_index];
            const long long item_index = item_indices[work_index];
            const unsigned long long* row_ptr =
                item_masks + ((size_t)item_index * (size_t)num_words);
            const unsigned long long* branch_a =
                branch_masks_a + ((size_t)candidate_index * (size_t)num_words);
            const unsigned long long* branch_b =
                branch_masks_b + ((size_t)candidate_index * (size_t)num_words);

            unsigned int local_count_a = 0u;
            unsigned int local_count_b = 0u;
            for (int word_index = thread_id; word_index < num_words; word_index += blockDim.x) {
                unsigned long long word = row_ptr[word_index];
                local_count_a += (unsigned int)__popcll(word & branch_a[word_index]);
                local_count_b += (unsigned int)__popcll(word & branch_b[word_index]);
            }

            unsigned int* shared_a = shared_counts;
            unsigned int* shared_b = shared_counts + blockDim.x;
            shared_a[thread_id] = local_count_a;
            shared_b[thread_id] = local_count_b;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (thread_id < stride) {
                    shared_a[thread_id] += shared_a[thread_id + stride];
                    shared_b[thread_id] += shared_b[thread_id + stride];
                }
                __syncthreads();
            }

            if (thread_id == 0) {
                out_support_a[work_index] = (unsigned long long)shared_a[0];
                out_support_b[work_index] = (unsigned long long)shared_b[0];
            }
        }
        """

        try:
            cls._gpu_support_kernel_multi = cp.RawKernel(
                kernel_code, "bitset_support_kernel_multi"
            )
        except Exception:
            cls._gpu_support_kernel_multi = None
        return cls._gpu_support_kernel_multi

    def _gpu_bitset_support_batch_multi(
        self,
        branch_masks_a: Union['numpy.ndarray', 'cupy.ndarray'],
        branch_masks_b: Union['numpy.ndarray', 'cupy.ndarray'],
        work_candidate_indices: list,
        work_item_indices: list,
    ) -> Optional[tuple[list[int], list[int]]]:
        """
        Compute support for a worklist across multiple branch masks in one kernel.
        """
        if self.bit_masks is None or not work_item_indices:
            return None

        try:
            import cupy as cp
            import numpy as np
        except ImportError:
            return None

        if not hasattr(self.bit_masks, "__cuda_array_interface__"):
            return None

        kernel = self._get_gpu_support_kernel_multi()
        if kernel is None:
            return None

        try:
            branch_masks_a = cp.asarray(branch_masks_a, dtype=cp.uint64)
            branch_masks_b = cp.asarray(branch_masks_b, dtype=cp.uint64)
            branch_masks_a = cp.ascontiguousarray(branch_masks_a)
            branch_masks_b = cp.ascontiguousarray(branch_masks_b)
            if branch_masks_a.shape != branch_masks_b.shape:
                return None
            if branch_masks_a.ndim != 2:
                return None

            num_words = int(branch_masks_a.shape[1])
            if num_words <= 0:
                return None

            total_items = len(work_item_indices)
            if not self._should_use_gpu_kernel(branch_masks_a, total_items):
                return None

            threads_per_block = 256
            shared_mem_bytes = threads_per_block * cp.dtype(cp.uint32).itemsize * 2

            budget_bytes = (
                int(self._gpu_batch_budget_mb * 1024 * 1024)
                if self._gpu_batch_budget_mb is not None
                else None
            )
            try:
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                free_mem_budget = int(free_bytes * self._gpu_batch_free_mem_fraction)
                if free_mem_budget > 0:
                    budget_bytes = (
                        free_mem_budget if budget_bytes is None else min(budget_bytes, free_mem_budget)
                    )
            except Exception:
                pass

            bytes_per_item = 2 * cp.dtype(cp.uint64).itemsize
            if budget_bytes is None or budget_bytes <= 0:
                chunk_items = total_items
            else:
                chunk_items = budget_bytes // bytes_per_item
                if chunk_items < 1:
                    chunk_items = 1
                if chunk_items > total_items:
                    chunk_items = total_items

            supports_host_a = np.empty(total_items, dtype=np.int64)
            supports_host_b = np.empty(total_items, dtype=np.int64)
            for start in range(0, total_items, chunk_items):
                stop = min(total_items, start + chunk_items)
                chunk_len = stop - start
                candidate_indices = cp.asarray(
                    work_candidate_indices[start:stop], dtype=cp.int32
                )
                item_indices = cp.asarray(work_item_indices[start:stop], dtype=cp.int64)
                supports_a = cp.zeros(chunk_len, dtype=cp.uint64)
                supports_b = cp.zeros(chunk_len, dtype=cp.uint64)
                kernel(
                    (chunk_len,),
                    (threads_per_block,),
                    (
                        self.bit_masks,
                        branch_masks_a,
                        branch_masks_b,
                        candidate_indices,
                        item_indices,
                        int(num_words),
                        supports_a,
                        supports_b,
                    ),
                    shared_mem=shared_mem_bytes,
                )
                supports_host_a[start:stop] = cp.asnumpy(supports_a).astype(np.int64, copy=False)
                supports_host_b[start:stop] = cp.asnumpy(supports_b).astype(np.int64, copy=False)

            return supports_host_a.tolist(), supports_host_b.tolist()
        except Exception:
            return None

    def _gpu_bitset_support_batch(
        self, mask_bitset: Union['numpy.ndarray', 'cupy.ndarray'], items: list
    ) -> Optional[list[int]]:
        """
        Compute batched support on GPU via a custom CUDA kernel when available.
        The item dimension is chunked to keep temporary allocations under a memory budget.
        """
        if self.bit_masks is None or not items:
            return []

        try:
            import cupy as cp
        except ImportError:
            return None

        kernel = self._get_gpu_support_kernel()
        if kernel is None:
            return None

        try:
            branch_mask = cp.asarray(mask_bitset, dtype=cp.uint64).reshape(-1)
            branch_mask = cp.ascontiguousarray(branch_mask)
            num_words = int(branch_mask.size)
            total_items = len(items)
            threads_per_block = 256
            shared_mem_bytes = threads_per_block * cp.dtype(cp.uint32).itemsize
            import numpy as np

            budget_bytes = (
                int(self._gpu_batch_budget_mb * 1024 * 1024)
                if self._gpu_batch_budget_mb is not None
                else None
            )
            try:
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                free_mem_budget = int(free_bytes * self._gpu_batch_free_mem_fraction)
                if free_mem_budget > 0:
                    budget_bytes = (
                        free_mem_budget if budget_bytes is None else min(budget_bytes, free_mem_budget)
                    )
            except Exception:
                pass

            if budget_bytes is None:
                per_item_bytes = (num_words * cp.dtype(cp.uint64).itemsize * 3) + cp.dtype(cp.uint64).itemsize
                budget_bytes = per_item_bytes * total_items

            chunk_items = self._compute_gpu_chunk_items(
                num_words=num_words,
                requested_items=total_items,
                budget_bytes=int(budget_bytes),
                word_bytes=cp.dtype(cp.uint64).itemsize,
            )

            supports_host = np.empty(total_items, dtype=np.int64)
            contiguous_indexer = self._contiguous_indexer(items)
            for start in range(0, total_items, chunk_items):
                stop = min(total_items, start + chunk_items)
                if contiguous_indexer is not None:
                    value_start = items[start]
                    value_stop = items[stop - 1] + 1
                    indexer = slice(value_start, value_stop)
                else:
                    indexer = items[start:stop]
                item_masks = cp.asarray(self.bit_masks[indexer], dtype=cp.uint64)
                if item_masks.ndim == 1:
                    item_masks = item_masks.reshape(1, -1)
                item_masks = cp.ascontiguousarray(item_masks)

                supports = cp.zeros(stop - start, dtype=cp.uint64)
                kernel(
                    (stop - start,),
                    (threads_per_block,),
                    (item_masks, branch_mask, int(num_words), supports),
                    shared_mem=shared_mem_bytes,
                )
                supports_host[start:stop] = cp.asnumpy(supports).astype(np.int64, copy=False)

            return supports_host.tolist()
        except Exception:
            return None

    def _intersect_bit_mask(
        self, current_mask: Optional[Union['numpy.ndarray', 'cupy.ndarray']], item: int
    ) -> Optional[Union['numpy.ndarray', 'cupy.ndarray']]:
        """
        Combine the current packed mask with the mask of the given item.
        """
        if current_mask is None or self.bit_masks is None:
            return None
        attribute_mask = self.bit_masks[item]
        working_mask = current_mask
        if (
            hasattr(attribute_mask, "__cuda_array_interface__")
            and not hasattr(current_mask, "__cuda_array_interface__")
        ):
            try:
                import cupy as cp

                working_mask = cp.asarray(current_mask, dtype=cp.uint64)
            except Exception:
                working_mask = current_mask
        intersection = attribute_mask & working_mask
        if self._spill_gpu_masks_to_cpu and hasattr(intersection, "__cuda_array_interface__"):
            try:
                import cupy as cp

                return cp.asnumpy(intersection)
            except Exception:
                return intersection
        return intersection

    def _popcount(self, mask: Union['numpy.ndarray', 'cupy.ndarray']) -> int:
        """
        Count the number of set bits in the packed mask.
        """
        return self._popcount_rows(mask)[0]

    @staticmethod
    def _popcount_uint64_rows(array: "numpy.ndarray") -> list[int]:
        """
        Compute popcount per row for uint64 arrays without unpackbits.
        """
        import numpy as np

        x = array.astype(np.uint64, copy=True)
        x -= (x >> 1) & np.uint64(0x5555555555555555)
        x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
        x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        x += x >> 8
        x += x >> 16
        x += x >> 32
        counts = x & np.uint64(0x7F)
        return counts.sum(axis=1).astype(np.int64, copy=False).tolist()

    def _popcount_rows(self, masks: Union['numpy.ndarray', 'cupy.ndarray']) -> list[int]:
        """
        Count set bits row-wise for 1D/2D packed masks.
        """
        import numpy as np

        if hasattr(masks, "__cuda_array_interface__"):
            try:
                import cupy as cp
            except ImportError:
                pass
            else:
                gpu_masks = cp.asarray(masks, dtype=cp.uint64)
                if gpu_masks.ndim == 1:
                    gpu_masks = gpu_masks.reshape(1, -1)
                if hasattr(cp, "bitwise_count"):
                    counts = cp.bitwise_count(gpu_masks).sum(axis=1)  # type: ignore[attr-defined]
                elif hasattr(gpu_masks, "bit_count"):
                    counts = gpu_masks.bit_count().sum(axis=1)  # type: ignore[call-arg]
                else:
                    cpu_masks = cp.asnumpy(gpu_masks)
                    return self._popcount_uint64_rows(cpu_masks)
                if hasattr(counts, "__cuda_array_interface__"):
                    # Keep a single host sync for GPU counts.
                    return cp.asnumpy(counts).astype(np.int64, copy=False).tolist()
                return [int(value) for value in counts.tolist()]

        array = np.asarray(masks, dtype=np.uint64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if hasattr(np, "bitwise_count"):
            counts = np.bitwise_count(array).sum(axis=1)  # type: ignore[attr-defined]
        elif hasattr(array, "bit_count"):
            counts = array.bit_count().sum(axis=1)  # type: ignore[call-arg]
        else:
            return self._popcount_uint64_rows(array)
        return [int(value) for value in counts.tolist()]

    def process_flexible_candidates(
        self,
        ar_prefix: tuple,
        itemset_prefix: tuple,
        reduced_flexible_items_binding: dict,
        stop_list: list,
        stop_list_itemset: list,
        flexible_candidates: dict,
        actionable_attributes: int,
        new_branches: list,
        verbose: bool,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
    ):
        """
        Process flexible candidates to generate new branches.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        reduced_flexible_items_binding : dict
            Dictionary containing reduced bindings for flexible items.
        stop_list : list
            List of stop combinations.
        stop_list_itemset : list
            List of stop itemsets.
        flexible_candidates : dict
            Dictionary containing flexible candidates.
        actionable_attributes : int
            Number of actionable attributes.
        new_branches : list
            List of new branches generated.
        verbose : bool
            If True, enables verbose output.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.

        Notes
        -----
        This method processes flexible candidates by iterating over the reduced flexible items bindings.
        It generates new action rule prefixes and calculates support for the undesired and desired states.
        If the support values meet the minimum thresholds, new branches are created and added to the
        new branches list. The method also updates the rules with new classification rules if the
        number of actionable attributes meets the minimum required.
        """
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return
        for attribute, items in reduced_flexible_items_binding.items():
            new_ar_prefix = ar_prefix + (attribute,)
            if self.in_stop_list(new_ar_prefix, stop_list):
                continue

            (
                undesired_states,
                desired_states,
                undesired_count,
                desired_count,
                kept_items,
            ) = self.process_items(
                attribute,
                items,
                itemset_prefix,
                new_ar_prefix,
                stop_list_itemset,
                flexible_candidates,
                verbose,
                undesired_mask_bitset=undesired_mask_bitset,
                desired_mask_bitset=desired_mask_bitset,
            )

            if actionable_attributes == 0 and (undesired_count == 0 or desired_count == 0):
                del flexible_candidates[attribute]
                self._add_stop_entry(stop_list, ar_prefix + (attribute,))
            else:
                for item in kept_items:
                    new_branches.append(
                        {
                            'ar_prefix': new_ar_prefix,
                            'itemset_prefix': itemset_prefix + (item,),
                            'item': item,
                            'undesired_mask_bitset': None,
                            'desired_mask_bitset': None,
                            'parent_undesired_mask_bitset': undesired_mask_bitset,
                            'parent_desired_mask_bitset': desired_mask_bitset,
                            'actionable_attributes': actionable_attributes + 1,
                        }
                    )
                if actionable_attributes + 1 >= self.min_flexible_attributes:
                    self.rules.add_classification_rules(
                        new_ar_prefix,
                        itemset_prefix,
                        undesired_states,
                        desired_states,
                    )

    def process_items(
        self,
        attribute: str,
        items: list,
        itemset_prefix: tuple,
        new_ar_prefix: tuple,
        stop_list_itemset: list,
        flexible_candidates: dict,
        verbose: bool,
        undesired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
        desired_mask_bitset: Optional[Union['numpy.ndarray', 'cupy.ndarray']] = None,
    ):
        """
        Process items to generate states and counts.

        Parameters
        ----------
        attribute : str
            The attribute being processed.
        items : list
            List of items for the attribute.
        itemset_prefix : tuple
            Prefix of the itemset.
        new_ar_prefix : tuple
            Prefix for stop list.
        stop_list_itemset : list
            List of stop itemsets.
        flexible_candidates : dict
            Dictionary containing flexible candidates.
        verbose : bool
            If True, enables verbose output.
        undesired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current undesired branch.
        desired_mask_bitset : Union['numpy.ndarray', 'cupy.ndarray'], optional
            Packed mask representing the current desired branch.

        Returns
        -------
        tuple
            Tuple containing undesired states, desired states, undesired count, desired count,
            and the list of items kept for branching.

        Notes
        -----
        This method processes items by iterating over the list of items for a given attribute. It calculates
        support and confidence values for both undesired and desired states, updating the rules with new
        classification rules if the confidence thresholds are met. The method also removes items that do not
        meet the minimum support thresholds from the flexible candidates and updates the stop list accordingly.
        """
        undesired_states = []
        desired_states = []
        undesired_count = 0
        desired_count = 0
        kept_items = []
        if undesired_mask_bitset is None or desired_mask_bitset is None:
            return undesired_states, desired_states, undesired_count, desired_count, kept_items
        active_items = self._active_flexible_items(itemset_prefix, items, stop_list_itemset)
        if not active_items:
            item_iter = ()
        else:
            undesired_supports = self._bitset_support_batch(undesired_mask_bitset, active_items)
            desired_supports = self._bitset_support_batch(desired_mask_bitset, active_items)
            item_iter = zip(active_items, undesired_supports, desired_supports)

        for item, undesired_support, desired_support in item_iter:
            if verbose:
                print('SUPPORT for: ' + str(itemset_prefix + (item,)))
                print('_________________________________________________')
                print('- extended by flexible attribute')
                print('undesired state support: ' + str(undesired_support))
                print('desired state support: ' + str(desired_support))
                print('')

            undesired_conf = self.rules.calculate_confidence(undesired_support, desired_support)
            if undesired_support >= self.min_undesired_support:
                undesired_count += 1
                if undesired_conf >= self.min_undesired_confidence:
                    undesired_states.append({'item': item, 'support': undesired_support, 'confidence': undesired_conf})
                else:
                    self.rules.add_prefix_without_conf(new_ar_prefix, False)

            desired_conf = self.rules.calculate_confidence(desired_support, undesired_support)
            if desired_support >= self.min_desired_support:
                desired_count += 1
                if desired_conf >= self.min_desired_confidence:
                    desired_states.append({'item': item, 'support': desired_support, 'confidence': desired_conf})
                else:
                    self.rules.add_prefix_without_conf(new_ar_prefix, True)

            if desired_support < self.min_desired_support and undesired_support < self.min_undesired_support:
                flexible_candidates[attribute].remove(item)
                self._add_stop_entry(stop_list_itemset, itemset_prefix + (item,))
                continue

            kept_items.append(item)

        return undesired_states, desired_states, undesired_count, desired_count, kept_items

    def update_new_branches(self, new_branches: list, stable_candidates: dict, flexible_candidates: dict):
        """
        Update new branches with stable and flexible candidates.

        Parameters
        ----------
        new_branches : list
            List of new branches generated.
        stable_candidates : dict
            Dictionary containing stable candidates.
        flexible_candidates : dict
            Dictionary containing flexible candidates.

        Notes
        -----
        This method updates new branches by iterating over stable and flexible candidates. It creates
        new stable and flexible bindings for each new branch, ensuring that only the relevant candidates
        are included in the new branches.
        """
        for new_branch in new_branches:
            adding = False
            new_stable = {}  # type: dict
            new_flexible = {}  # type: dict

            for attribute, items in stable_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_stable:
                            new_stable[attribute] = []
                        new_stable[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            for attribute, items in flexible_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_flexible:
                            new_flexible[attribute] = []
                        new_flexible[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            del new_branch['item']
            new_branch['stable_items_binding'] = new_stable
            new_branch['flexible_items_binding'] = new_flexible

    def in_stop_list(self, ar_prefix: tuple, stop_list: list) -> bool:
        """
        Check if the action rule prefix is in the stop list.

        Parameters
        ----------
        ar_prefix : tuple
            Prefix of the action rule.
        stop_list : list
            List of stop combinations.

        Returns
        -------
        bool
            True if the action rule prefix is in the stop list, False otherwise.

        Notes
        -----
        This method checks if the action rule prefix is in the stop list by checking for the presence
        of the last two elements and all but the first element of the prefix in the stop list. If the
        prefix is found, it is added to the stop list to prevent future processing.
        """
        if ar_prefix[-2:] in stop_list:
            return True
        if ar_prefix[1:] in stop_list:
            self._add_stop_entry(stop_list, ar_prefix)
            return True
        return False
