"""
Module for testing strategies on different buckets by some feature.
For example top 10% volatility, bottom 10% volatility (price, vol of vol etc)
"""
import torch
from typing import Callable, List, Union, Tuple, Any

from MLTT.allocation.backtesting import backtest, BTResult


class BucketWatcher:
    # Inspired by [deleted]
    def __init__(self, prices: torch.Tensor):
        """
        Args:
            - `prices` (torch.Tensor): 2-dimensional array of prices
        """
        self.prices = prices
        self.feature_indices = None
        self.feature_names = None

    def make_buckets(self,
                     feature: Union[Callable, List[Callable]],
                     quantile: Union[float, List[float]] = 0.1
            ) -> Tuple[torch.Tensor, List[str]]:
        """Separates prices into buckets according to some feature (top 10% and bottom 10%)

        saves `feature_names`, `feature_indices` for further use

        Args:
            - `feature` (Callable | list[Callable]): function (or list of functions) with 1-d
                array as input and returning float
            - `quantile` (float | list[float]): quantile for top and bottom buckets.
                (`quantile=0.1` means top 10% and bottom 10% buckets in returned array)

        Returns:
            - torch.Tensor: 2-dimensional array of indices of buckets. Shape: `(n_buckets, n_information)`
            - list[str]: list of feature names with quantile
        """
        if not isinstance(feature, list):
            feature = [feature]
        if not isinstance(quantile, list):
            quantile = [quantile]

        self.feature_indices = []
        self.feature_names = []
        for f in feature:
            for q in quantile:
                # Применяем функцию признака вдоль столбцов (оси 0)
                feature_values = torch.stack([f(self.prices[:, i]) for i in range(self.prices.size(1))])
                top_quantile = torch.quantile(feature_values, 1 - q)
                bottom_quantile = torch.quantile(feature_values, q)

                bottom_bucket_idx = feature_values <= bottom_quantile
                top_bucket_idx = feature_values >= top_quantile

                self.feature_names.extend([f'top_{q}_{f.__name__}', f'bottom_{q}_{f.__name__}'])
                self.feature_indices.extend([top_bucket_idx, bottom_bucket_idx])

        self.feature_indices = torch.stack(self.feature_indices, dim=0)

        return self.feature_indices, self.feature_names

    def backtest_buckets(self, weights: torch.Tensor, **backtest_kwargs) -> List[BTResult]:
        """firstly takes solid strategy prediction with all assets,
        After that uses computed weights on buckets.

        Args:
            - `weights` (torch.Tensor): weights matrix of
                shape: (batch_size, n_tradable+1).
                `BTResult.weights` of strategy (with all assets)
                already shifted (without future leak) and with neutral
                weight at the last column.
                
        Returns:
            list[BTResult]: backtest results for each bucket
        """
        if self.feature_indices is None or self.feature_names is None:
            raise ValueError("call make_buckets before backtest_buckets")
        weights = weights[:, :-1]
        results = []

        for indices in self.feature_indices:
            bt_result = backtest(
                base_weights=weights[:, indices],
                prices=self.prices[:, indices],
                **backtest_kwargs
            )
            results.append(bt_result)

        return results
