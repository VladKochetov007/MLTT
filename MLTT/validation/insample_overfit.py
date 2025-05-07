"""
Optimize a strategy on historical data
Simulate N prices paths with model fitted to a market
If strategy has a real alpha, it will underperform on the random paths
"""
# TODO: support not only prices as information for allocator

from MLTT.models.price_models import PriceModel
from MLTT.report import Metric
from MLTT.report.metrics import AnnualSharpe
from MLTT import CapitalAllocator
from MLTT.utils import change

from scipy.stats import ttest_1samp

import torch


class RandomPermutationsPriceModel(PriceModel):
    """
    Price model that generates random permutations of the historical data
    """

    def fit(self, historical_prices: torch.Tensor) -> None:
        """
        Just store the historical prices
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data to be stored
        """
        self.historical_prices = historical_prices

    def _simulate_path(self) -> torch.Tensor:
        """
        Permutate the price changes
        
        Returns:
            torch.Tensor: A simulated price path based on permutation of historical changes
        """
        price_change = change(self.historical_prices)
        # Using torch's random permutation
        idx = torch.randperm(price_change.size(0))
        permuted_price_change = price_change[idx]
        return torch.cumsum(permuted_price_change, dim=0)

    def simulate_paths(self, num_paths: int, *args, **kwargs) -> torch.Tensor:
        """
        Simulate multiple paths of the asset price.

        Args:
            - `num_paths` (int): Number of paths to simulate
            
        Returns:
            torch.Tensor: Tensor of simulated price paths
        """
        paths = [self._simulate_path() for _ in range(num_paths)]
        return torch.stack(paths)
    

def _overfit_pvalue(strategy_metric: float, metric_randoms: list[float]) -> float:
    """
    Test if strategy is likely to be overfited.

    H0: strategy is overfitted
    H1: strategy is not overfitted

    Args:
        - `strategy_metric` (float): The metric value for the strategy
        - `metric_randoms` (list[float]): List of metric values from random simulations
        
    Returns:
        float: p-value of t-test. Lower p-value is better in terms of good fitting.
    """
    # Convert to numpy array for ttest_1samp
    metric_randoms_np = torch.tensor(metric_randoms).cpu().numpy() if isinstance(metric_randoms[0], torch.Tensor) else metric_randoms
    return ttest_1samp(metric_randoms_np, strategy_metric, alternative="greater").pvalue


def _test_n_paths(
    allocator: CapitalAllocator,
    prices: torch.Tensor,
    metric: Metric | None = None,
    price_model: PriceModel | None = None,
    num_paths: int = 100,
) -> float:  # TODO: add asyncronous backtest
    if metric is None:
        metric = AnnualSharpe()
    if price_model is None:
        price_model = RandomPermutationsPriceModel()
    price_model.fit(prices)
    time_steps = prices.shape[0]

    random_prices = price_model.simulate_paths(num_paths=num_paths, time_steps=time_steps)
    # TODO