"""
Module for testing strategies on different buckets by some feature.
For example top 10% volatility, bottom 10% volatility (price, vol of vol etc)
"""
import torch
from typing import Callable
from enum import Enum

from MLTT.allocation.backtesting import backtest, BTResult, backtest_model
from MLTT import CapitalAllocator


class BucketMode(str, Enum):
    """Mode for bucket testing"""
    SUBSET_INPUT = "SUBSET_INPUT"  # Give model only data from bucket subset
    FILTER_OUTPUT = "FILTER_OUTPUT"  # Give model all data but filter weights by bucket


class BucketWatcher:
    def __init__(self, prices: torch.Tensor, prediction_info: torch.Tensor | None = None):
        """
        Args:
            - `prices` (torch.Tensor): 2-dimensional array of log-prices
            - `prediction_info` (torch.Tensor | None): Additional information for model prediction
                Shape: `(batch_size, *n_information)`
        """
        self.prices = prices
        self.prediction_info = prediction_info
        self.feature_indices = None
        self.feature_names = None

    def _validate_feature_values(self, 
                                 feature_values: torch.Tensor | list | float, 
                                 data: torch.Tensor) -> torch.Tensor:
        """Validate and convert feature values to correct tensor format
        
        Args:
            - `feature_values` (torch.Tensor | list | float): Values returned by feature function
            - `data` (torch.Tensor): Original data tensor used for feature calculation
            
        Returns:
            torch.Tensor: Validated 1-dimensional tensor
            
        Raises:
            ValueError: If feature values have incorrect shape or size
        """
        if not isinstance(feature_values, torch.Tensor):
            feature_values = torch.tensor(feature_values)
            
        if feature_values.dim() != 1:
            raise ValueError(
                f"Feature function must return 1-d tensor, got {feature_values.dim()}"
            )
            
        if feature_values.size(0) != data.size(1):
            raise ValueError(
                f"Feature function must return tensor of size {data.size(1)}, got {feature_values.size(0)}"
            )
            
        return feature_values

    def _validate_bucket_state(self) -> None:
        """Validate that buckets were created before backtesting
        
        Raises:
            ValueError: If buckets were not created
        """
        if self.feature_indices is None or self.feature_names is None:
            raise ValueError("call make_buckets before backtest_buckets")

    def _backtest_bucket(
        self,
        indices: torch.Tensor,
        model: CapitalAllocator | None,
        mode: BucketMode,
        precalculated_weights: torch.Tensor | None = None,
        **backtest_kwargs
    ) -> BTResult:
        """Run backtest for a single bucket
        
        Args:
            - `indices` (torch.Tensor): Boolean tensor indicating which assets belong to the bucket
            - `model` (CapitalAllocator | None): Model to use for predictions
            - `mode` (BucketMode): Testing mode
            - `precalculated_weights` (torch.Tensor | None): Pre-calculated weights for FILTER_OUTPUT mode
            - `**backtest_kwargs`: Additional arguments for backtest function
            
        Returns:
            BTResult: Backtest result for the bucket
        """
        bucket_prices = self.prices[:, indices]
        bucket_info = (
            self.prediction_info[:, indices] 
            if self.prediction_info is not None 
            else None
        )

        if mode == BucketMode.SUBSET_INPUT:
            if model is None:
                raise ValueError("Model required for SUBSET_INPUT mode")
            
            return backtest_model(
                model=model,
                prices=bucket_prices,
                prediction_info=bucket_info,
                **backtest_kwargs
            )
        else:  # FILTER_OUTPUT mode
            if precalculated_weights is None:
                raise ValueError("Pre-calculated weights required for FILTER_OUTPUT mode")
                
            # Filter weights by bucket indices
            bucket_weights = precalculated_weights[:, indices]
            
            # Backtest filtered weights
            return backtest(
                weights=bucket_weights,
                prices=bucket_prices,
                **backtest_kwargs
            )

    def make_buckets(
        self,
        feature: Callable | list[Callable],
        quantile: float | list[float] = 0.1
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Separates assets into buckets according to some feature (top 10% and bottom 10%)
        
        Feature function should take a tensor of shape (time_steps, n_assets) and return
        tensor of shape (n_assets,) with feature values for each asset.
        
        saves `feature_names`, `feature_indices` for further use

        Args:
            - `feature` (Callable | list[Callable]): function (or list of functions) that takes
                2-d tensor and returns 1-d tensor with feature values for each asset
            - `quantile` (float | list[float]): quantile for top and bottom buckets.
                (`quantile=0.1` means top 10% and bottom 10% buckets in returned array)

        Returns:
            - torch.Tensor: 2-dimensional array of indices of buckets. Shape: `(n_buckets, n_information)`
            - list[str]: list of feature names with quantile
        """
        features = [feature] if isinstance(feature, Callable) else feature
        quantiles = [quantile] if isinstance(quantile, float) else quantile

        self.feature_indices = []
        self.feature_names = []
        
        data = self.prices if self.prediction_info is None else self.prediction_info
        
        for f in features:
            for q in quantiles:
                # Calculate feature values for all assets at once
                feature_values = self._validate_feature_values(f(data), data)
                
                top_quantile = torch.quantile(feature_values, 1 - q)
                bottom_quantile = torch.quantile(feature_values, q)

                bottom_bucket_idx = feature_values <= bottom_quantile
                top_bucket_idx = feature_values >= top_quantile

                self.feature_names.extend([f'top_{q}_{f.__name__}', f'bottom_{q}_{f.__name__}'])
                self.feature_indices.extend([top_bucket_idx, bottom_bucket_idx])

        self.feature_indices = torch.stack(self.feature_indices, dim=0)
        return self.feature_indices, self.feature_names

    def backtest_buckets(
        self,
        model: CapitalAllocator | None = None,
        mode: BucketMode = BucketMode.SUBSET_INPUT,
        **backtest_kwargs
    ) -> dict[str, BTResult]:
        """Test strategy on different buckets using specified mode.
        
        Args:
            - `model` (CapitalAllocator | None): Model to use for predictions. 
                Required for both modes
            - `mode` (BucketMode): Testing mode
            - `**backtest_kwargs`: Additional arguments for backtest function
                
        Returns:
            dict[str, BTResult]: backtest results for each bucket
            
        Note:
            In FILTER_OUTPUT mode, the function will first run model with full data
            to get weights, then filter those weights for each bucket.
        """
        self._validate_bucket_state()
        
        if model is None:
            raise ValueError("Model is required for both modes")
            
        # Pre-calculate weights for FILTER_OUTPUT mode
        precalculated_weights = None
        if mode == BucketMode.FILTER_OUTPUT:
            input_data = self.prediction_info if self.prediction_info is not None else self.prices
            raw_weights = model(input_data)
            precalculated_weights = torch.roll(raw_weights, 1, dims=0)
            
        return {
            name: self._backtest_bucket(
                indices, 
                model, 
                mode, 
                precalculated_weights=precalculated_weights,
                **backtest_kwargs
            )
            for name, indices in zip(self.feature_names, self.feature_indices)
        }
