import torch
import torch.nn.functional as F
from MLTT.allocation import BaseAllocator
from MLTT.allocation.allocators import PriceAwareAllocator
from MLTT.technical.non_trivial import price_channel
from MLTT.utils import multivariate_barrier, to_weights_matrix, change, EPSILON


def get_price_data(model: BaseAllocator, x: torch.Tensor) -> torch.Tensor:
    """
    Get price data from a model or input tensor.
    Uses PriceAwareAllocator interface if model supports it.
    
    Args:
        - `model` (BaseAllocator): Model that may implement PriceAwareAllocator
        - `x` (torch.Tensor): Input tensor to use as fallback price data
        
    Returns:
        - torch.Tensor: Price tensor
    """
    # Check if model provides price data through PriceAwareAllocator interface
    if isinstance(model, PriceAwareAllocator):
        # Get price matrix from model
        price_matrix = model.get_prices_matrix()
        if price_matrix is not None:
            # Convert to tensor if it's numpy array
            if not isinstance(price_matrix, torch.Tensor):
                price_matrix = torch.tensor(price_matrix, dtype=torch.float32)
            return price_matrix
            
    # Fallback to using input data as prices
    return x


def calculate_asset_volatility(prices: torch.Tensor, lookback_period: int) -> torch.Tensor:
    """
    Calculate historical volatility for each asset
    
    Args:
        - `prices` (torch.Tensor): Price tensor with shape (time_steps, num_assets)
        - `lookback_period` (int): Number of periods to use for volatility calculation
        
    Returns:
        - torch.Tensor: Tensor of volatilities with shape (num_assets,)
    """
    # Need at least 2 observations to calculate returns
    if prices.shape[0] < 2:
        return torch.ones(prices.shape[1])
        
    # Calculate returns - prices are already log prices, so use change
    returns = change(prices, lag=1)
    
    # Use only the most recent lookback_period returns
    if returns.shape[0] > lookback_period:
        recent_returns = returns[-lookback_period:]
    else:
        recent_returns = returns
    
    # Calculate standard deviation of returns
    vol = torch.std(recent_returns, dim=0, unbiased=False)
    
    # Replace zeros with a small value to avoid division by zero
    vol = torch.where(vol > 0, vol, torch.tensor(EPSILON))
    
    return vol


def calculate_portfolio_volatility(weights: torch.Tensor, returns: torch.Tensor) -> float:
    """
    Estimate portfolio volatility based on historical returns and weights
    
    Args:
        - `weights` (torch.Tensor): Weight tensor with shape (num_assets,) or (time_steps, num_assets)
        - `returns` (torch.Tensor): Returns tensor with shape (time_steps, num_assets)
        
    Returns:
        - float: Portfolio volatility as a scalar
    """
    # Ensure weights is a vector (last weight vector if it's a matrix)
    if weights.dim() > 1:
        weights = weights[-1]
        
    # Calculate weighted returns
    weighted_returns = returns @ weights
    
    # Calculate volatility
    portfolio_vol = torch.std(weighted_returns)
    
    return portfolio_vol


class TimeSeriesMomentumModel(BaseAllocator):
    """
    Time Series Momentum strategy implementation.
    Based on the research from "Time Series Momentum" by Moskowitz, Ooi and Pedersen.
    
    The strategy:
    1. Uses past returns to predict future returns
    2. Can incorporate roll returns for futures
    3. Supports position scaling over the holding period
    4. Optional mean reversion filter
    """
    
    def __init__(
        self,
        lookback: int,
        holding_period: int,
        threshold: float = 0.0,
        use_roll_returns: bool = False,
        scale_positions: bool = True,
        use_mean_reversion: bool = False,
        mean_reversion_lookback: int | None = None,
    ):
        """
        Args:
            - `lookback` (int): Number of days to look back for momentum calculation
            - `holding_period` (int): Number of days to hold positions
            - `threshold` (float): Minimum return threshold for taking positions
            - `use_roll_returns` (bool): Whether to use roll returns instead of total returns
            - `scale_positions` (bool): Whether to scale positions over the holding period
            - `use_mean_reversion` (bool): Whether to add mean reversion filter
            - `mean_reversion_lookback` (int | None): Lookback period for mean reversion filter
        """
        self.lookback = lookback
        self.holding_period = holding_period
        self.threshold = threshold
        self.use_roll_returns = use_roll_returns
        self.scale_positions = scale_positions
        self.use_mean_reversion = use_mean_reversion
        self.mean_reversion_lookback = mean_reversion_lookback or (lookback + 10)
        
        # Need enough data for calculations
        super().__init__(num_observations=max(lookback, mean_reversion_lookback or 0))

    def _calculate_returns(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Calculate returns based on configuration
        
        Args:
            - `prices` (torch.Tensor): Price tensor to calculate returns from
            
        Returns:
            - torch.Tensor: Calculated returns
        """
        # Ensure prices is 2D
        if prices.dim() == 1:
            prices = prices.unsqueeze(-1)
            
        if self.use_roll_returns:
            # For futures: calculate roll returns
            # This is a simplified version - in practice you'd need actual futures data
            returns = change(prices, lag=1)
            roll_returns = F.avg_pool1d(
                returns.unsqueeze(0),  # Add batch dimension
                kernel_size=252,
                stride=1,
                padding=251
            ).squeeze(0)  # Remove batch dimension
            return F.pad(roll_returns, (0, 0, 0, 1))  # Pad time dimension
        else:
            # Regular returns over lookback period
            return change(prices, lag=self.lookback)

    def _generate_signals(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Generate trading signals from returns
        
        Args:
            - `returns` (torch.Tensor): Returns tensor to generate signals from
            
        Returns:
            - torch.Tensor: Generated trading signals (-1, 0, 1)
        """
        signals = torch.zeros_like(returns)
        
        # Long signal when return > threshold
        signals[returns > self.threshold] = 1
        
        # Short signal when return < -threshold
        signals[returns < -self.threshold] = -1
        
        return signals

    def _apply_mean_reversion_filter(
        self, 
        signals: torch.Tensor, 
        prices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean reversion filter to signals
        
        Args:
            - `signals` (torch.Tensor): Trading signals to filter
            - `prices` (torch.Tensor): Price tensor for mean reversion calculation
            
        Returns:
            - torch.Tensor: Filtered trading signals
        """
        if not self.use_mean_reversion:
            return signals
            
        # Ensure prices is 2D
        if prices.dim() == 1:
            prices = prices.unsqueeze(-1)
            
        # Calculate returns for mean reversion period
        mr_returns = change(prices, lag=self.mean_reversion_lookback)
        
        # Only keep momentum signals that align with mean reversion
        filtered_signals = signals.clone()
        filtered_signals[(signals > 0) & (mr_returns > 0)] = 0
        filtered_signals[(signals < 0) & (mr_returns < 0)] = 0
        
        return filtered_signals

    def _scale_positions(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Scale positions over the holding period
        
        Args:
            - `signals` (torch.Tensor): Trading signals to scale
            
        Returns:
            - torch.Tensor: Scaled position weights
        """
        if not self.scale_positions:
            return signals
            
        weights = torch.zeros_like(signals)
        
        # For each day, build position over holding period
        for i in range(self.holding_period):
            shifted = F.pad(signals[:-i] if i > 0 else signals, (0, 0, i, 0))
            weights = weights + shifted
        
        # Normalize weights
        weights = weights / self.holding_period
        return weights

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate position weights based on momentum signals.
        
        Args:
            - `x` (torch.Tensor): Price tensor of shape (time,) or (time, assets)
            
        Returns:
            - torch.Tensor: Portfolio weights tensor between -1 and 1
        """
        # Store original shape
        original_dim = x.dim()
        
        # Ensure input is 2D
        if original_dim == 1:
            x = x.unsqueeze(-1)
            
        # Calculate returns
        returns = self._calculate_returns(x)
        
        # Generate signals
        signals = self._generate_signals(returns)
        
        # Apply mean reversion filter if enabled
        signals = self._apply_mean_reversion_filter(signals, x)
        
        # Scale positions if enabled
        weights = self._scale_positions(signals)
        
        # Return to original shape if input was 1D
        if original_dim == 1:
            weights = weights.squeeze(-1)
            
        return weights


class SimpleTimeSeriesMomentumModel(BaseAllocator):
    def __init__(self, lookback: int, hedge_market: bool = False):
        """
        Simple implementation of time series momentum strategy.
        
        Args:
            - `lookback` (int): Number of days to look back for return calculation
            - `hedge_market` (bool): Whether to neutralize market factor by subtracting average return
        """
        self.lookback = lookback
        self.hedge_market = hedge_market
        super().__init__(num_observations=lookback)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate position weights based on momentum signals.
        
        Args:
            - `x` (torch.Tensor): Price tensor of shape (time, assets)
            
        Returns:
            - torch.Tensor: Portfolio weights tensor
        """
        returns = change(x, lag=self.lookback)
        if self.hedge_market:
            market_return = returns.mean(dim=1)
            returns = returns - market_return.unsqueeze(1)
        return to_weights_matrix(returns)


class PriceChannelMomentumModel(BaseAllocator):
    """
    Price Channel Momentum strategy implementation.
    
    Strategy logic:
    1. Calculates price channel using either close prices only or high/low prices
    2. Takes long position (+1) when price breaks above upper channel
    3. Takes short position (-1) when price breaks below lower channel
    4. Holds position until price reaches channel midpoint
    """
    
    def __init__(
        self,
        lookback: int = 14,
        bounds: float = 0.8,
        use_high_low: bool = False
    ):
        """
        Args:
            - `lookback` (int): Period for calculating channel bounds
            - `bounds` (float): Width of the channel as percentage (0.0-1.0) from midpoint
            - `use_high_low` (bool): Whether to use high/low prices instead of just close
        """
        self.lookback = lookback
        self.bounds = bounds
        self.use_high_low = use_high_low
        super().__init__(num_observations=lookback)
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate position weights based on price channel breakouts.
        
        Args:
            - `x` (torch.Tensor): Price tensor of shape (time, assets) if use_high_low=False
               or (time, assets, 3) with high/low/close if use_high_low=True
            
        Returns:
            - torch.Tensor: Portfolio weights tensor between -1 and 1
        """
        if self.use_high_low:
            # Handle 3D input (high, low, close)
            high = x[..., 0]
            low = x[..., 1]
            close = x[..., 2]
        else:
            # Use same prices for high/low/close
            high = close = x
            low = x  # Use same tensor for low to avoid None
        
        # Calculate channel bounds directly with tensors
        upper, lower = price_channel(
            high=high,
            low=low,  # Now always tensor instead of None
            lookback=self.lookback,
            bounds=self.bounds
        )
        
        # Calculate midpoint and generate signals
        midpoint = (upper + lower) / 2
        positions = multivariate_barrier(
            x=close,
            upper_bound=upper,
            lower_bound=lower,
            neutral_level=midpoint
        )
        
        # Normalize positions and maintain tensor device
        return positions / x.shape[1]

class VolatilityTargetingWrapper(BaseAllocator):
    """
    A wrapper class that applies volatility targeting to any allocator model.
    
    Volatility targeting aims to maintain a constant level of portfolio volatility by
    adjusting position sizes. This improves Sharpe ratio, reduces turnover, and manages risk.
    
    Source: https://www.algos.org/p/breaking-down-momentum-strategies
    
    Args:
        - `base_model` (BaseAllocator): The underlying allocation model to wrap
        - `volatility_period` (int): Lookback period for volatility calculation
        - `target_volatility` (float): Annual volatility target (e.g., 0.15 for 15%)
        - `max_leverage` (float): Maximum allowed leverage
        - `penalize_high_vol` (bool): Whether to additionally penalize high volatility assets
        - `T` (int): Number of days in a year (default: 365)
    """
    
    def __init__(
        self,
        base_model: BaseAllocator,
        volatility_period: int = 63,  # ~3 months of trading days
        target_volatility: float = 0.15,  # 15% annual volatility target
        max_leverage: float = 2.0,
        penalize_high_vol: bool = True,
        T: int = 365
    ):
        self.base_model = base_model
        self.volatility_period = volatility_period
        # Convert annual target to daily using specified T
        self.target_volatility = target_volatility / (T ** 0.5)  
        self.max_leverage = max_leverage
        self.penalize_high_vol = penalize_high_vol
        
        # Need at least as much data as the base model + volatility period
        required_observations = max(
            base_model.num_observations, 
            volatility_period
        )
        super().__init__(num_observations=required_observations)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply volatility targeting to the base model's weights
        
        Args:
            - `x` (torch.Tensor): Input tensor (can be prices or other data)
            
        Returns:
            Volatility-targeted weights
        """
        # Get base model weights
        base_weights = self.base_model(x)
        
        # Get price data for volatility calculations
        price_data = get_price_data(self.base_model, x)
        
        # Calculate asset volatilities
        asset_vols = calculate_asset_volatility(price_data, self.volatility_period)
        
        # Optionally penalize high volatility assets
        if self.penalize_high_vol:
            # Inverse volatility weighting
            vol_adjustment = 1.0 / (asset_vols + EPSILON)
            # Normalize
            vol_adjustment = vol_adjustment / vol_adjustment.sum()
            
            # Adjust base weights by volatility
            # Multiply by sign to preserve direction
            adjusted_weights = base_weights * torch.sign(base_weights) * vol_adjustment
            adjusted_weights = to_weights_matrix(adjusted_weights)
        else:
            adjusted_weights = base_weights
            
        # Calculate portfolio volatility based on recent returns
        if price_data.shape[0] > self.volatility_period + 1:
            # Calculate returns
            returns = change(price_data, lag=1)
            recent_returns = returns[-self.volatility_period:]
            
            # Use the most recent weights for volatility calculation
            last_weights = adjusted_weights
            if adjusted_weights.dim() > 1:
                last_weights = adjusted_weights[-1]
                
            portfolio_vol = calculate_portfolio_volatility(
                last_weights, 
                recent_returns
            )
            
            # Calculate scaling factor to achieve target volatility
            if portfolio_vol > 0:
                scaling_factor = self.target_volatility / portfolio_vol
                # Apply max leverage constraint
                scaling_factor = min(scaling_factor, self.max_leverage)
                
                # Apply scaling
                final_weights = adjusted_weights * scaling_factor
            else:
                final_weights = adjusted_weights
        else:
            final_weights = adjusted_weights
            
        return final_weights


class InformationDiscretenessFilter(BaseAllocator):
    """
    Filter assets based on Information Discreteness to focus on sustained price moves.
    
    Information Discreteness (ID) measures how evenly distributed price changes are.
    High ID means returns are concentrated in a few large moves.
    Low ID means returns are more evenly distributed.
    
    For momentum strategies, we prefer lower ID assets as they represent more sustained trends.
    
    Source: https://www.algos.org/p/breaking-down-momentum-strategies
    
    Parameters:
        base_model: The underlying allocation model
        lookback_period: Period for calculating ID
        id_threshold: Maximum allowed ID value (higher = more permissive)
        min_assets: Minimum number of assets to keep after filtering
    """
    
    ...  # TODO: implement


class RegimeSwitchingModel(BaseAllocator):
    """
    Switches between momentum and mean-reversion strategies based on market regime.
    
    Regime detection uses a simple method: if the market is above its moving average,
    it's considered a bull market (positive autocorrelation), suitable for momentum.
    If it's below, it's a bear market (negative autocorrelation), suitable for mean-reversion.
    
    Source: https://www.algos.org/p/breaking-down-momentum-strategies
    
    Parameters:
        momentum_model: Model to use in bull markets
        mean_reversion_model: Model to use in bear markets
        regime_period: Period for detecting market regime
        benchmark_index: Which asset to use as market benchmark (default: first asset)
        inverse_signals: Whether to invert signals in opposite regime rather than switching models
    """
    
    ...  # TODO: implement

class BetaFilteredMomentumModel(BaseAllocator):
    """
    Momentum strategy that filters out high-beta assets.
    
    From the article: "It is good practice to filter out the top decile assets by beta. 
    This is a rule that holds true for many other alphas. Even when ignoring the additional 
    turnover high beta assets tend to introduce, they still underperform on average and 
    increase the portfolio risk unnecessarily."
    
    Source: https://www.algos.org/p/breaking-down-momentum-strategies
    
    Parameters:
        lookback_period: Period for momentum calculation
        beta_measurement_period: Period for beta calculation
        beta_filter_threshold: Percentile to use as cutoff for high beta (e.g., 0.9 for top 10%)
        num_positions: Number of positions to hold
    """
    ...  # TODO: implement

