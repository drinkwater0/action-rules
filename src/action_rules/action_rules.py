"""Main class ActionRules."""

import itertools
import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Optional, Union  # noqa

from .candidates.candidate_generator import CandidateGenerator
from .output.output import Output
from .rules.rules import Rules

if TYPE_CHECKING:
    from types import ModuleType  # noqa

    import cudf
    import cupy
    import numpy
    import pandas


class ActionRules:
    """
    Generate action rules from tabular data using one-hot encoding and bitset support counting.

    Attributes
    ----------
    min_stable_attributes : int
        The minimum number of stable attributes required.
    min_flexible_attributes : int
        The minimum number of flexible attributes required.
    min_undesired_support : int
        The minimum support for the undesired state.
    min_undesired_confidence : float
        The minimum confidence for the undesired state.
    min_desired_support : int
        The minimum support for the desired state.
    min_desired_confidence : float
        The minimum confidence for the desired state.
    verbose : bool, optional
        If True, enables verbose output.
    rules : Optional[Rules], optional
        Stores the generated rules.
    output : Optional[Output], optional
        Stores the generated action rules.
    np : Optional[ModuleType], optional
        The numpy or cupy module used for array operations.
    pd : Optional[ModuleType], optional
        The pandas or cudf module used for DataFrame operations.
    is_gpu_np : bool
        Indicates whether GPU-accelerated numpy (cupy) is used.
    is_gpu_pd : bool
        Indicates whether GPU-accelerated pandas (cudf) is used.
    intrinsic_utility_table : dict, optional
        (attribute, value) -> float
        A lookup table for the intrinsic utility of each attribute-value pair.
        If None, no intrinsic utility is considered.
    transition_utility_table : dict, optional
        (attribute, from_value, to_value) -> float
        A lookup table for cost/gain of transitions between values.
        If None, no transition utility is considered.

    Methods
    -------
    fit(data, stable_attributes, flexible_attributes, target, undesired_state, desired_state, use_gpu=False)
        Generates action rules based on the provided dataset and parameters.
    get_bindings(data, stable_attributes, flexible_attributes, target)
        Binds attributes to corresponding columns in the dataset.
    get_stop_list(stable_items_binding, flexible_items_binding)
        Generates a stop list to prevent certain combinations of attributes.
    get_rules()
        Returns the generated action rules if available.
    predict(frame_row)
        Predicts recommended actions based on the provided row of data.
    """

    def __init__(
        self,
        min_stable_attributes: int,
        min_flexible_attributes: int,
        min_undesired_support: int,
        min_undesired_confidence: float,
        min_desired_support: int,
        min_desired_confidence: float,
        verbose=False,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
    ):
        """
        Initialize the ActionRules class with the specified parameters.

        Parameters
        ----------
        min_stable_attributes : int
            The minimum number of stable attributes required.
        min_flexible_attributes : int
            The minimum number of flexible attributes required.
        min_undesired_support : int
            The minimum support for the undesired state.
        min_undesired_confidence : float
            The minimum confidence for the undesired state.
        min_desired_support : int
            The minimum support for the desired state.
        min_desired_confidence : float
            The minimum confidence for the desired state.
        verbose : bool, optional
            If True, enables verbose output. Default is False.
        intrinsic_utility_table : dict, optional
            (attribute, value) -> float
            A lookup table for the intrinsic utility of each attribute-value pair.
            If None, no intrinsic utility is considered.
        transition_utility_table : dict, optional
            (attribute, from_value, to_value) -> float
            A lookup table for cost/gain of transitions between values.
            If None, no transition utility is considered.

        Notes
        -----
        The `verbose` parameter can be used to enable detailed output during the rule generation process.
        """
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.verbose = verbose
        self.rules = None  # type: Optional[Rules]
        self.output = None  # type: Optional[Output]
        self.np = None  # type: Optional[ModuleType]
        self.pd = None  # type: Optional[ModuleType]
        self.is_gpu_np = False
        self.is_gpu_pd = False
        self.is_onehot = False
        self.bit_masks = None  # type: Optional['numpy.ndarray']
        self.target_state_bit_masks = None  # type: Optional[dict]
        self.frames_bit_masks = None  # type: Optional[dict]
        self.intrinsic_utility_table = intrinsic_utility_table or {}
        self.transition_utility_table = transition_utility_table or {}

    def count_max_nodes(self, stable_items_binding: dict, flexible_items_binding: dict) -> int:
        """
        Calculate the maximum number of nodes based on the given item bindings.

        This function takes two dictionaries, `stable_items_binding` and `flexible_items_binding`,
        which map attributes to lists of items. It calculates the total number of nodes by considering
        all possible combinations of the lengths of these item lists and summing the product of each combination.

        Parameters
        ----------
        stable_items_binding : dict
            A dictionary where keys are attributes and values are lists of stable items.
        flexible_items_binding : dict
            A dictionary where keys are attributes and values are lists of flexible items.

        Returns
        -------
        int
            The total number of nodes calculated by summing the product of lengths of all combinations of item lists.

        Notes
        -----
        - The function first combines the lengths of item lists from both dictionaries.
        - It then calculates the sum of the products of all possible combinations of these lengths.
        """
        import numpy

        values_in_attribute = []
        for items in list(stable_items_binding.values()) + list(flexible_items_binding.values()):
            values_in_attribute.append(len(items))

        sum_nodes = 0
        for i in range(len(values_in_attribute)):
            for comb in itertools.combinations(values_in_attribute, i + 1):
                sum_nodes += int(numpy.prod(comb))
        return sum_nodes

    def set_array_library(self, use_gpu: bool, df: Union['cudf.DataFrame', 'pandas.DataFrame']):
        """
        Set the appropriate array and DataFrame libraries (cuDF or pandas) based on the user's preference.

        Parameters
        ----------
        use_gpu : bool
            Indicates whether to use GPU (cuDF) for data processing if available.
        df : Union[cudf.DataFrame, pandas.DataFrame]
            The DataFrame to convert.

        Raises
        ------
        ImportError
            If `use_gpu` is True but cuDF is not available and pandas cannot be imported as fallback.

        Warnings
        --------
        UserWarning
            If `use_gpu` is True but cuDF is not available, a warning is issued indicating fallback to pandas.

        Notes
        -----
        This method determines whether to use GPU-accelerated libraries for processing data, falling back to CPU-based
        libraries if necessary.
        """
        if use_gpu:
            try:
                import cupy as np

                is_gpu_np = True
            except ImportError:
                warnings.warn("CuPy is not available. Falling back to Numpy.")
                import numpy as np

                is_gpu_np = False
        else:
            import numpy as np

            is_gpu_np = False

        df_library_imported = False
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                is_gpu_pd = False
                df_library_imported = True
        except ImportError:
            df_library_imported = False

        if not df_library_imported:
            try:
                import cudf as pd

                if isinstance(df, pd.DataFrame):
                    is_gpu_pd = True
                    df_library_imported = True
            except ImportError:
                df_library_imported = False

        if not df_library_imported:
            raise ImportError('Just Pandas or cuDF dataframes are supported.')

        self.np = np
        self.pd = pd
        self.is_gpu_np = is_gpu_np
        self.is_gpu_pd = is_gpu_pd

    def df_to_array(self, df: Union['cudf.DataFrame', 'pandas.DataFrame']) -> tuple:
        """
        Convert a one-hot DataFrame to a transposed binary array.

        Parameters
        ----------
        df : Union[cudf.DataFrame, pandas.DataFrame]
            The DataFrame to convert.

        Returns
        -------
        tuple
            A tuple containing the transposed array and the DataFrame columns.

        Notes
        -----
        The data is converted to an unsigned 8-bit array (`np.uint8`), backed by
        NumPy or CuPy depending on the selected execution backend.
        """
        columns = list(df.columns)
        if self.is_gpu_np:
            data = self.np.asarray(df.values, dtype=self.np.uint8).T  # type: ignore
        elif self.is_gpu_pd:
            data = df.to_numpy().T  # type: ignore
        else:
            data = df.to_numpy(dtype=self.np.uint8).T  # type: ignore
        return data, columns

    def build_bit_masks(
        self,
        data: Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> Union['numpy.ndarray', 'cupy.ndarray']:
        """
        Pack a binary feature matrix into 64-bit masks for fast intersection.

        Parameters
        ----------
        data : Union[numpy.ndarray, cupy.ndarray]
            Dense matrix produced by `df_to_array`, shaped (num_attributes, num_transactions)
            and containing 0/1 values.

        Returns
        -------
        Union[numpy.ndarray, cupy.ndarray]
            bit_masks is a uint64 array with shape (num_attributes, num_words)
            holding packed transaction bits for each item.

        Notes
        -----
        - The packing uses 64-bit little-endian words (bit 0 corresponds to the
          first transaction in each chunk).
        - Sparse inputs are not supported; callers should densify before packing.
        """
        if self.np is None:
            raise RuntimeError("Array library is not initialised. Call set_array_library first.")
        # Shape is (num_attributes, num_transactions).
        num_attributes, num_transactions = data.shape
        num_words = (num_transactions + 63) // 64
        padded_transactions = num_words * 64
        padding = padded_transactions - num_transactions

        if padding > 0:
            pad_block = self.np.zeros((num_attributes, padding), dtype=data.dtype)
            padded_data = self.np.concatenate((data, pad_block), axis=1)
        else:
            padded_data = data

        # Group transactions into 64-bit chunks: (num_attributes, num_words, 64).
        chunks = padded_data.reshape(num_attributes, num_words, 64).astype(self.np.uint64, copy=False)
        bit_offsets = self.np.arange(64, dtype=self.np.uint64)
        bit_weights = self.np.left_shift(self.np.uint64(1), bit_offsets)

        # Pack each 64-sized transaction chunk into one uint64 word.
        bit_masks = self.np.tensordot(chunks, bit_weights, axes=([2], [0])).astype(self.np.uint64, copy=False)
        return bit_masks

    def _cache_bitset_structures(
        self,
        bit_masks: Union['numpy.ndarray', 'cupy.ndarray'],
        target_items_binding: dict,
        target: str,
    ) -> None:
        """
        Store packed bitset artifacts for later use in candidate evaluation.

        Parameters
        ----------
        bit_masks : Union[numpy.ndarray, cupy.ndarray]
            Packed transaction masks for every attribute/value.
        target_items_binding : dict
            Mapping from target attribute name to indices of its one-hot columns.
        target : str
            Name of the target attribute.
        """
        target_state_indices = target_items_binding.get(target, [])
        target_state_bit_masks = {index: bit_masks[index] for index in target_state_indices}

        self.bit_masks = bit_masks
        self.target_state_bit_masks = target_state_bit_masks

    def one_hot_encode(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
    ) -> Union['cudf.DataFrame', 'pandas.DataFrame']:
        """
        Perform one-hot encoding on the specified stable, flexible, and target attributes of the DataFrame.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The input DataFrame containing the data to be encoded.
        stable_attributes : list
            List of stable attributes to be one-hot encoded.
        flexible_attributes : list
            List of flexible attributes to be one-hot encoded.
        target : str
            The target attribute to be one-hot encoded.

        Returns
        -------
        Union[cudf.DataFrame, pandas.DataFrame]
            A DataFrame with the specified attributes one-hot encoded.

        Notes
        -----
        Non-missing antecedent values are converted to strings while preserving NaNs, so missing stable/flexible
        values are excluded from one-hot categories. The target attribute is converted to strings in full, then
        all encoded blocks are concatenated into a single DataFrame.
        """
        def _prepare_antecedent_frame(frame, attributes):
            """
            Convert non-missing antecedent values to strings while preserving NaNs.

            Preserving missing values lets `get_dummies` skip them instead of creating
            explicit `..._nan` categories. This mirrors the original ActionRulesDiscovery
            preprocessing, which excludes NaN antecedent values before mining.
            """
            antecedent = frame[attributes].copy()
            return antecedent.where(antecedent.isna(), antecedent.astype(str))

        to_concat = []
        if len(stable_attributes) > 0:
            stable_frame = _prepare_antecedent_frame(data, stable_attributes)
            data_stable = self.pd.get_dummies(  # type: ignore
                stable_frame, sparse=False, prefix_sep='_<item_stable>_'
            )
            to_concat.append(data_stable)
        if len(flexible_attributes) > 0:
            flexible_frame = _prepare_antecedent_frame(data, flexible_attributes)
            data_flexible = self.pd.get_dummies(  # type: ignore
                flexible_frame, sparse=False, prefix_sep='_<item_flexible>_'
            )
            to_concat.append(data_flexible)
        data_target = self.pd.get_dummies(  # type: ignore
            data[[target]].astype(str), sparse=False, prefix_sep='_<item_target>_'
        )
        to_concat.append(data_target)
        data = self.pd.concat(to_concat, axis=1)  # type: ignore
        return data

    def fit_onehot(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: dict,
        flexible_attributes: dict,
        target: dict,
        target_undesired_state: str,
        target_desired_state: str,
        use_gpu: bool = False,
        max_gpu_mem_mb: Optional[int] = None,
        gpu_node_batch_size: Optional[int] = None,
        gpu_batch_size: Optional[int] = None,
    ):
        """
        Fit the model when input data is already one-hot encoded.

        The method remaps one-hot columns to the internal naming convention
        (`_<item_stable>_`, `_<item_flexible>_`, `_<item_target>_`), drops
        unrelated columns, and forwards execution to `fit`.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The dataset to be processed and used for fitting the model.
        stable_attributes : dict
            A dictionary mapping stable attribute names to lists of column
            names corresponding to those attributes.
        flexible_attributes : dict
            A dictionary mapping flexible attribute names to lists of column
            names corresponding to those attributes.
        target : dict
            A dictionary mapping the target attribute name to a list of
            column names corresponding to that attribute.
        target_undesired_state : str
            The undesired state of the target attribute, used in action rule generation.
        target_desired_state : str
            The desired state of the target attribute, used in action rule generation.
        use_gpu : bool, optional
            If True, the GPU (cuDF) is used for data processing if available.
            Default is False.
        max_gpu_mem_mb : int, optional
            Optional GPU memory cap (in MB) for CuPy allocations and bitset
            support batching. If None, automatic memory-based chunking is used.
        gpu_node_batch_size : int, optional
            Optional number of BFS queue nodes grouped before one GPU candidate-
            expansion pass. If None, defaults to 32 on the GPU bitset path.
        gpu_batch_size : int, optional
            Deprecated alias for `gpu_node_batch_size`.
        Notes
        -----
        This method expects boolean/binary one-hot columns.
        """
        if gpu_node_batch_size is None:
            gpu_node_batch_size = gpu_batch_size
        elif gpu_batch_size is not None and int(gpu_node_batch_size) != int(gpu_batch_size):
            raise ValueError("gpu_node_batch_size and gpu_batch_size must match when both are provided.")

        self.is_onehot = True
        data = data.copy()
        data = data.astype('bool')
        new_labels = []
        attributes_stable = set([])
        attribtes_flexible = set([])
        attribute_target = ''
        remove_cols = []
        for label in data.columns:
            to_remove = True
            for attribute, columns in stable_attributes.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_stable>_' + label)
                    attributes_stable.add(attribute)
                    to_remove = False
            for attribute, columns in flexible_attributes.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_flexible>_' + label)
                    attribtes_flexible.add(attribute)
                    to_remove = False
            for attribute, columns in target.items():
                if label in columns:
                    new_labels.append(attribute + '_<item_target>_' + label)
                    attribute_target = attribute
                    to_remove = False
            if to_remove:
                new_labels.append(label)
                remove_cols.append(label)
        data.columns = new_labels
        data = data.drop(columns=remove_cols)
        self.fit(
            data,
            list(attributes_stable),
            list(attribtes_flexible),
            attribute_target,
            target_undesired_state,
            target_desired_state,
            use_gpu,
            max_gpu_mem_mb,
            gpu_node_batch_size,
        )

    def fit(
        self,
        data: Union['cudf.DataFrame', 'pandas.DataFrame'],
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
        target_undesired_state: str,
        target_desired_state: str,
        use_gpu: Union[bool, str] = False,
        max_gpu_mem_mb: Optional[int] = None,
        gpu_node_batch_size: Optional[int] = None,
        gpu_batch_size: Optional[int] = None,
    ):
        """
        Generate action rules for the provided dataset.

        Parameters
        ----------
        data : Union[cudf.DataFrame, pandas.DataFrame]
            The dataset to generate action rules from.
        stable_attributes : list
            List of stable attributes.
        flexible_attributes : list
            List of flexible attributes.
        target : str
            The target attribute.
        target_undesired_state : str
            The undesired state of the target attribute.
        target_desired_state : str
            The desired state of the target attribute.
        use_gpu : bool or str, optional
            Use GPU (cuDF) for data processing if available. Pass ``'auto'``
            to profile the dataset and select the fastest backend
            automatically via sampled trial runs. Default is False.
        max_gpu_mem_mb : int, optional
            Optional GPU memory cap (in MB) for CuPy allocations and bitset
            support batching. If None, automatic memory-based chunking is used.
        gpu_node_batch_size : int, optional
            Optional number of BFS queue nodes grouped before one GPU candidate-
            expansion pass. If None, defaults to 32 on the GPU bitset path.
        gpu_batch_size : int, optional
            Deprecated alias for `gpu_node_batch_size`.
        Raises
        ------
        RuntimeError
            If the model has already been fitted.

        Notes
        -----
        The method runs one-hot encoding (when needed), packs bit masks, explores
        candidate branches, prunes classification rules by depth, and finally
        materializes action rules.
        """
        if self.output is not None:
            raise RuntimeError("The model is already fit.")
        if gpu_node_batch_size is None:
            gpu_node_batch_size = gpu_batch_size
        elif gpu_batch_size is not None and int(gpu_node_batch_size) != int(gpu_batch_size):
            raise ValueError("gpu_node_batch_size and gpu_batch_size must match when both are provided.")

        # --- Auto backend selection via profiling + sampled autotuning ----
        if use_gpu == "auto":
            from .autotuning import autotune
            from .profiling import profile_dataset

            # Profile on a temporary instance so self stays clean for the
            # real fit that follows.
            profile_helper = ActionRules(
                min_stable_attributes=self.min_stable_attributes,
                min_flexible_attributes=self.min_flexible_attributes,
                min_undesired_support=self.min_undesired_support,
                min_undesired_confidence=self.min_undesired_confidence,
                min_desired_support=self.min_desired_support,
                min_desired_confidence=self.min_desired_confidence,
                verbose=False,
            )
            dataset_profile = profile_dataset(
                action_rules=profile_helper,
                data_frame=data,
                stable_attributes=stable_attributes,
                flexible_attributes=flexible_attributes,
                target=target,
            )
            best = autotune(
                action_rules_cls=ActionRules,
                data_frame=data,
                stable_attributes=stable_attributes,
                flexible_attributes=flexible_attributes,
                target=target,
                target_undesired_state=target_undesired_state,
                target_desired_state=target_desired_state,
                min_stable_attributes=self.min_stable_attributes,
                min_flexible_attributes=self.min_flexible_attributes,
                min_undesired_support=self.min_undesired_support,
                min_undesired_confidence=self.min_undesired_confidence,
                min_desired_support=self.min_desired_support,
                min_desired_confidence=self.min_desired_confidence,
                dataset_profile=dataset_profile,
                max_gpu_mem_mb=max_gpu_mem_mb,
                gpu_node_batch_size=gpu_node_batch_size,
            )
            use_gpu = best["use_gpu"]
            gpu_node_batch_size = best.get("gpu_node_batch_size")
            self._autotune_result = best
            self._dataset_profile = dataset_profile

        # reset cached bitset structures before fitting a new model
        self.bit_masks = None
        self.target_state_bit_masks = None
        self.frames_bit_masks = None
        self.set_array_library(use_gpu, data)
        previous_gpu_pool_limit = None
        if self.is_gpu_np and max_gpu_mem_mb is not None:
            try:
                gpu_pool = self.np.get_default_memory_pool()  # type: ignore[attr-defined]
                if hasattr(gpu_pool, "get_limit"):
                    previous_gpu_pool_limit = int(gpu_pool.get_limit())
                gpu_pool.set_limit(size=int(max_gpu_mem_mb) * 1024 * 1024)
            except Exception:
                previous_gpu_pool_limit = None
        if not self.is_onehot:
            data = self.one_hot_encode(data, stable_attributes, flexible_attributes, target)
        data, columns = self.df_to_array(data)

        stable_items_binding, flexible_items_binding, target_items_binding, column_values = self.get_bindings(
            columns, stable_attributes, flexible_attributes, target
        )

        self.intrinsic_utility_table, self.transition_utility_table = self.remap_utility_tables(column_values)

        local_bit_masks = self.build_bit_masks(data)
        self._cache_bitset_structures(local_bit_masks, target_items_binding, target)
        self.frames_bit_masks = self.get_split_bit_masks(target_items_binding, target)

        if self.verbose:
            print('Maximum number of nodes to check for support:')
            print('_____________________________________________')
            print(self.count_max_nodes(stable_items_binding, flexible_items_binding))
            print('')
        use_gpu_batching = bool(self.is_gpu_np and self.bit_masks is not None and self.frames_bit_masks)

        # Set membership is hot in candidate pruning; use a set internally for O(1) lookups.
        stop_list = set(self.get_stop_list(stable_items_binding, flexible_items_binding))
        undesired_state = columns.index(target + '_<item_target>_' + str(target_undesired_state))
        desired_state = columns.index(target + '_<item_target>_' + str(target_desired_state))

        stop_list_itemset = set()  # type: set

        initial_candidate = {
            'ar_prefix': tuple(),
            'itemset_prefix': tuple(),
            'stable_items_binding': stable_items_binding,
            'flexible_items_binding': flexible_items_binding,
            'undesired_mask_bitset': None,
            'desired_mask_bitset': None,
            'actionable_attributes': 0,
        }
        candidates_pool = deque([initial_candidate])
        pending_depth_counts = {0: 1}
        min_pending_depth = 0
        max_depth_seen = 0
        next_prune_depth = 1
        self.rules = Rules(
            undesired_state,
            desired_state,
            columns,
            data.shape[1],
            self.intrinsic_utility_table,
            self.transition_utility_table,
        )
        candidate_generator = CandidateGenerator(
            frames_bit_masks=self.frames_bit_masks,
            bit_masks=self.bit_masks,
            min_stable_attributes=self.min_stable_attributes,
            min_flexible_attributes=self.min_flexible_attributes,
            min_undesired_support=self.min_undesired_support,
            min_desired_support=self.min_desired_support,
            min_undesired_confidence=self.min_undesired_confidence,
            min_desired_confidence=self.min_desired_confidence,
            undesired_state=undesired_state,
            desired_state=desired_state,
            rules=self.rules,
            gpu_batch_budget_mb=max_gpu_mem_mb,
            spill_gpu_masks_to_cpu=bool(self.is_gpu_np and max_gpu_mem_mb is not None),
        )
        effective_gpu_node_batch_size = (
            gpu_node_batch_size if gpu_node_batch_size is not None else 32
        )

        def pop_next_candidate() -> dict:
            """
            Pop one pending candidate and keep pending-depth bookkeeping in sync.
            """
            nonlocal min_pending_depth
            candidate_to_expand = candidates_pool.popleft()
            depth = len(candidate_to_expand['ar_prefix'])
            pending_depth_counts[depth] -= 1
            if pending_depth_counts[depth] <= 0:
                pending_depth_counts.pop(depth, None)
                if depth == min_pending_depth:
                    min_pending_depth = min(pending_depth_counts.keys(), default=None)
            return candidate_to_expand

        while len(candidates_pool) > 0:
            if use_gpu_batching:
                batch = []
                while candidates_pool and len(batch) < effective_gpu_node_batch_size:
                    batch.append(pop_next_candidate())
                new_candidates = candidate_generator.generate_candidates_batch(
                    batch,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                    undesired_state=undesired_state,
                    desired_state=desired_state,
                    verbose=self.verbose,
                    batch_size=effective_gpu_node_batch_size,
                )
            else:
                candidate = pop_next_candidate()
                new_candidates = candidate_generator.generate_candidates(
                    **candidate,
                    stop_list=stop_list,
                    stop_list_itemset=stop_list_itemset,
                    undesired_state=undesired_state,
                    desired_state=desired_state,
                    verbose=self.verbose,
                )
            if new_candidates:
                candidates_pool.extend(new_candidates)
                for new_candidate in new_candidates:
                    new_depth = len(new_candidate['ar_prefix'])
                    pending_depth_counts[new_depth] = pending_depth_counts.get(new_depth, 0) + 1
                    if min_pending_depth is None or new_depth < min_pending_depth:
                        min_pending_depth = new_depth
                    if new_depth > max_depth_seen:
                        max_depth_seen = new_depth
            while next_prune_depth <= max_depth_seen and (
                min_pending_depth is None or min_pending_depth >= next_prune_depth
            ):
                self.rules.prune_classification_rules(next_prune_depth, stop_list)
                next_prune_depth += 1
        self.rules.generate_action_rules()
        self.output = Output(
            self.rules.action_rules, target, stable_items_binding, flexible_items_binding, column_values
        )
        del data
        if self.is_gpu_np:
            gpu_pool = self.np.get_default_memory_pool()  # type: ignore[attr-defined]
            gpu_pool.free_all_blocks()
            if previous_gpu_pool_limit is not None:
                try:
                    gpu_pool.set_limit(size=previous_gpu_pool_limit)
                except Exception:
                    pass

    def get_bindings(
        self,
        columns: list,
        stable_attributes: list,
        flexible_attributes: list,
        target: str,
    ) -> tuple:
        """
        Bind attributes to corresponding columns in the dataset.

        Parameters
        ----------
        columns : list
            List of column names in the dataset.
        stable_attributes : list
            List of stable attributes.
        flexible_attributes : list
            List of flexible attributes.
        target : str
            The target attribute.

        Returns
        -------
        tuple
            A tuple containing the bindings for stable attributes, flexible attributes, and target items.

        Notes
        -----
        The method generates mappings from column indices to attribute values for stable, flexible, and target
        attributes.
        """
        stable_items_binding = defaultdict(lambda: [])
        flexible_items_binding = defaultdict(lambda: [])
        target_items_binding = defaultdict(lambda: [])
        column_values = {}

        for i, col in enumerate(columns):
            is_continue = False
            # stable
            for attribute in stable_attributes:
                if col.startswith(attribute + '_<item_stable>_'):
                    stable_items_binding[attribute].append(i)
                    column_values[i] = (attribute, col.split('_<item_stable>_', 1)[1])
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # flexible
            for attribute in flexible_attributes:
                if col.startswith(attribute + '_<item_flexible>_'):
                    flexible_items_binding[attribute].append(i)
                    column_values[i] = (attribute, col.split('_<item_flexible>_', 1)[1])
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # target
            if col.startswith(target + '_<item_target>_'):
                target_items_binding[target].append(i)
                column_values[i] = (target, col.split('_<item_target>_', 1)[1])
        return stable_items_binding, flexible_items_binding, target_items_binding, column_values

    def get_stop_list(self, stable_items_binding: dict, flexible_items_binding: dict) -> list:
        """
        Generate a stop list to prevent certain combinations of attributes.

        Parameters
        ----------
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.

        Returns
        -------
        list
            A list of stop combinations.

        Notes
        -----
        The stop list is generated by creating pairs of stable item indices and ensuring flexible items do not repeat.
        """
        stop_list = []
        for items in stable_items_binding.values():
            for stop_couple in itertools.product(items, repeat=2):
                stop_list.append(tuple(stop_couple))
        for item in flexible_items_binding.keys():
            stop_list.append(tuple([item, item]))
        return stop_list

    def get_split_bit_masks(self, target_items_binding: dict, target: str) -> dict:
        """
        Return packed bit-mask rows for each target state.

        Parameters
        ----------
        target_items_binding : dict
            Indexes of target attributes columns in one-hot table.
        target : str
            Name of the target attribute.

        Returns
        -------
        dict
            Dictionary mapping target attributes to the corresponding packed mask rows.

        Notes
        -----
        Requires that `build_bit_masks` has been executed beforehand.
        """
        if self.bit_masks is None:
            raise RuntimeError("Bit masks are not available. Ensure fit() was run first.")

        target_state_masks = {}
        for item_index in target_items_binding.get(target, []):
            target_state_masks[item_index] = self.bit_masks[item_index]
        return target_state_masks

    def get_rules(self) -> Output:
        """
        Return the generated action rules if available.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Returns
        -------
        Output
            The generated action rules.

        Notes
        -----
        This method returns the `Output` object containing the generated action rules.
        """
        if self.output is None:
            raise RuntimeError("The model is not fit.")
        return self.output

    def predict(self, frame_row: Union['cudf.Series', 'pandas.Series']) -> Union['cudf.DataFrame', 'pandas.DataFrame']:
        """
        Predict recommended actions based on the provided row of data.

        This method applies the fitted action rules to the given row of data and generates
        a DataFrame with recommended actions if any of the action rules are triggered.

        Parameters
        ----------
        frame_row : Union['cudf.Series', 'pandas.Series']
            A row of data in the form of a cuDF or pandas Series. The Series should
            contain the features required by the action rules.

        Returns
        -------
        Union['cudf.DataFrame', 'pandas.DataFrame']
            A DataFrame with the recommended actions. The DataFrame includes the following columns:
            - The original attributes with recommended changes.
            - 'ActionRules_RuleIndex': Index of the action rule applied.
            - 'ActionRules_UndesiredSupport': Support of the undesired part of the rule.
            - 'ActionRules_DesiredSupport': Support of the desired part of the rule.
            - 'ActionRules_UndesiredConfidence': Confidence of the undesired part of the rule.
            - 'ActionRules_DesiredConfidence': Confidence of the desired part of the rule.
            - 'ActionRules_Uplift': Uplift value of the rule.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Notes
        -----
        The method compares the given row of data against the undesired itemsets of the action rules.
        If a match is found, it applies the desired itemset changes and records the action rule's
        metadata. The result is a DataFrame with one or more rows representing the recommended actions
        for the given data.
        """
        if self.output is None:
            raise RuntimeError("The model is not fit.")
        index_value_tuples = list(zip(frame_row.index, frame_row))
        values = []
        column_values = self.output.column_values
        for index_value_tuple in index_value_tuples:
            values.append(list(column_values.keys())[list(column_values.values()).index(index_value_tuple)])
        new_values = tuple(values)
        predicted = []
        for i, action_rule in enumerate(self.output.action_rules):
            if set(action_rule['undesired']['itemset']) <= set(new_values):
                predicted_row = frame_row.copy()
                for recommended in set(action_rule['desired']['itemset']) - set(new_values):
                    attribute, value = column_values[recommended]
                    predicted_row[attribute + ' (Recommended)'] = value
                predicted_row['ActionRules_RuleIndex'] = i
                predicted_row['ActionRules_UndesiredSupport'] = action_rule['undesired']['support']
                predicted_row['ActionRules_DesiredSupport'] = action_rule['desired']['support']
                predicted_row['ActionRules_UndesiredConfidence'] = action_rule['undesired']['confidence']
                predicted_row['ActionRules_DesiredConfidence'] = action_rule['desired']['confidence']
                predicted_row['ActionRules_Uplift'] = action_rule['uplift']
                predicted.append(predicted_row)
        return self.pd.DataFrame(predicted)  # type: ignore

    def remap_utility_tables(self, column_values):
        """
        Remap the keys of intrinsic and transition utility tables using the provided column mapping.

        The function uses `column_values`, a dictionary mapping internal column indices to
        (attribute, value) tuples, to invert the mapping so that utility table keys are replaced
        with the corresponding integer index (for intrinsic utilities) or a tuple of integer indices
        (for transition utilities).

        Parameters
        ----------
        column_values : dict
            Dictionary mapping integer column indices to (attribute, value) pairs.
            Example: {0: ('Age', 'O'), 1: ('Age', 'Y'), 2: ('Sex', 'F'), ...}

        Returns
        -------
        tuple
            A tuple (remapped_intrinsic, remapped_transition) where:
              - remapped_intrinsic is a dict mapping integer column index to utility value.
              - remapped_transition is a dict mapping (from_index, to_index) to utility value.

        Notes
        -----
        - The method performs case-insensitive matching by converting attribute names and values to lowercase.
        - If a key in a utility table does not have a corresponding entry in column_values, it is skipped.
        """
        # Invert column_values to map (attribute.lower(), value.lower()) -> column index.
        inv_map = {(attr.lower(), val.lower()): idx for idx, (attr, val) in column_values.items()}

        remapped_intrinsic = {}
        # Remap intrinsic utility table keys: ('Attribute', 'Value') -> utility
        for key, utility in self.intrinsic_utility_table.items():
            # Normalize key to lowercase
            attr, val = key
            lookup_key = (attr.lower(), val.lower())
            # Look up the corresponding column index; if not found, skip this key.
            if lookup_key in inv_map:
                col_index = inv_map[lookup_key]
                remapped_intrinsic[col_index] = utility
            # Else: optionally, one could log or warn about a missing mapping.

        remapped_transition = {}
        # Remap transition utility table keys: ('Attribute', from_value, to_value) -> utility
        for key, utility in self.transition_utility_table.items():
            attr, from_val, to_val = key
            lookup_from = (attr.lower(), from_val.lower())
            lookup_to = (attr.lower(), to_val.lower())
            # Only remap if both the from and to values exist in inv_map.
            if lookup_from in inv_map and lookup_to in inv_map:
                from_index = inv_map[lookup_from]
                to_index = inv_map[lookup_to]
                remapped_transition[(from_index, to_index)] = utility
            # Else: skip or log missing mapping.

        return remapped_intrinsic, remapped_transition
