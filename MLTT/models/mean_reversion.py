from MLTT.allocation import BaseAllocator
from MLTT.utils import to_weights_matrix
from MLTT.technical.features import z_score
from MLTT.utils import apply_columnwise, change, multivariate_barrier, EPSILON
from MLTT.models.momentum import SimpleTimeSeriesMomentumModel

import torch
from MLTT.utils import calculate_betas


class ZScoreBarriersModel(BaseAllocator):
    def __init__(self,
                 period: int,
                 lower_band: float,
                 upper_band: float,
                 all_in_position: bool = False,
                 mean: float = None) -> None:
        self.period = period
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.all_in_position = all_in_position
        self.mean = mean

        super().__init__(num_observations=period if not mean else 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = z_score(x, self.period, self.mean)
        signals = multivariate_barrier(z, self.upper_band, self.lower_band)

        if self.all_in_position:
            weights = to_weights_matrix(signals)
        else:
            weights = signals / x.shape[1]

        return weights


class ZScorePortionModel(BaseAllocator):
    def __init__(self,
                 period: int,
                 mean: float = None) -> None:
        self.period = period
        self.mean = mean

        super().__init__(num_observations=period if not mean else 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = z_score(x, self.period, self.mean)

        return to_weights_matrix(-z).nan_to_num(0)


class CrossSectionalMeanReversionModel(BaseAllocator):
    """
    Cross-sectional mean reversion model.
    Source: Ernest P. Chan, "Algorithmic Trading"
    """

    def __init__(self, num_observations: int = 1, hedge_market: bool = True) -> None:
        self._momentum = SimpleTimeSeriesMomentumModel(
            lookback=num_observations, 
            hedge_market=hedge_market
        )
    
    @property
    def min_observations(self) -> int:
        return self._momentum.min_observations

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return -self._momentum.predict(x)

class EnhancedCrossSectionalMRModel(BaseAllocator):
    """
    Enhanced Cross-Sectional Mean Reversion Model with Volatility Filter.
    Based on the article by Teddy Koker.
    URL: https://teddykoker.com/2019/05/improving-cross-sectional-mean-reversion-strategy-in-python/
    """

    def __init__(self, 
                 num_positions: int, 
                 lag: int, 
                 volatility_period: int) -> None:
        self.num_positions = num_positions
        self.lag = lag
        self.volatility_period = volatility_period
        super().__init__(num_observations=max(lag, volatility_period))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate equal-weighted positions based on mean reversion.
        
        Args:
            x: Price matrix of shape (time, assets)
            
        Returns:
            weights: Portfolio weights with shape (time, assets)
        """
        # Calculate returns using change
        returns = change(x, lag=self.lag)
        
        # Calculate mean returns across assets
        mean_returns = torch.mean(returns, dim=1, keepdim=True)
        
        # Calculate weights based on the formula
        deviations = returns - mean_returns
        weights = -deviations
        
        # Initialize result array with zeros
        final_weights = torch.zeros_like(weights)
        
        # Process each time step separately
        for t in range(weights.shape[0]):
            # Calculate volatility for current timestep
            if t < self.volatility_period:
                vol = torch.std(x[:t+1], dim=0) if t > 0 else torch.ones(x.shape[1])
            else:
                vol = torch.std(x[t-self.volatility_period+1:t+1], dim=0)
            
            # Get current weights
            curr_weights = weights[t]
            
            # Find top positions by absolute weight
            _, top_by_weight = torch.sort(torch.abs(curr_weights), descending=True)
            top_by_weight = top_by_weight[:self.num_positions]
            
            # Find top positions by lowest volatility
            _, top_by_vol = torch.sort(vol)
            top_by_vol = top_by_vol[:self.num_positions]
            
            # Find intersection of indices
            top_by_weight_set = set(top_by_weight.tolist())
            top_by_vol_set = set(top_by_vol.tolist())
            selected = torch.tensor(list(top_by_weight_set.intersection(top_by_vol_set)), dtype=torch.long)
            
            if len(selected) > 0:
                # Get selected weights
                selected_weights = curr_weights[selected]
                
                # Normalize selected weights
                norm_factor = torch.sum(torch.abs(selected_weights))
                if norm_factor > 0:
                    selected_weights = selected_weights / norm_factor
                    
                # Assign normalized weights
                for i, idx in enumerate(selected):
                    final_weights[t, idx] = selected_weights[i]
        
        return final_weights


class BetaAdjustedMeanReversionModel(BaseAllocator):
    """
    Beta-adjusted mean reversion model.
    Quant. Arb: https://www.algos.org/p/breaking-down-momentum-strategies
    """
    def __init__(self, num_observations: int = 1, beta_measurement_period: int = 200) -> None:
        super().__init__(num_observations=max(num_observations, beta_measurement_period))
        self.beta_measurement_period = beta_measurement_period
        self.num_observations = num_observations

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: tensor of shape (num_observations, num_assets+1)
            the last column is the benchmark
        """
        # Separate the assets and benchmark
        assets = x[:, :-1]
        benchmark = x[:, -1:]
        
        # Calculate asset returns
        asset_returns = change(assets, lag=self.num_observations)
        
        # Calculate benchmark returns
        bench_returns = change(benchmark, lag=self.num_observations)
        
        # Calculate betas using the extracted function
        betas = calculate_betas(asset_returns, bench_returns, self.beta_measurement_period)
        
        # Apply beta adjustment to returns
        beta_adjusted_returns = torch.zeros_like(asset_returns)
        for i in range(assets.shape[1]):
            # Divide returns by beta to get beta-adjusted returns
            # Add epsilon to avoid division by zero
            beta_adjusted_returns[:, i] = asset_returns[:, i] / (betas[i] + EPSILON)
        
        # Calculate mean of beta-adjusted returns across assets
        mean_returns = beta_adjusted_returns.mean(dim=1, keepdim=True)
        
        # Calculate weights based on deviations from mean (mean reversion)
        deviations = beta_adjusted_returns - mean_returns
        
        # Return the negative deviations as weights (buy underperformers, sell outperformers)
        return to_weights_matrix(-deviations)
