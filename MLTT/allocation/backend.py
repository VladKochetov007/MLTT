"""
Data -> technical features -> ML allocator (fitting to autograd sharpe or other) -> weights
-> backtesting -> gradient
"""
from torch import Tensor, abs, cat, sum, ones, any, stack, zeros, roll, zeros_like

from warnings import warn

from MLTT.utils import EPSILON, change, CACHE_SIZE
from MLTT.cache import conditional_lru_cache


@conditional_lru_cache(maxsize=CACHE_SIZE)
def add_neutral_weight(x):
    """
    Adds a neutral weight to each asset.

    Args:
        - `x` (Tensor): A tensor of shape `(batch_size, n_tradable)`,
            where `batch_size` is the number of samples and
            `n_tradable` is the number of tradable assets.

    Returns:
        Tensor: A tensor of shape `(batch_size, n_tradable+1)`,
            where `batch_size` is the number of samples,
            `n_tradable` is the number of tradable assets and
            the last column contains neutral weights,
            in which the sum of each row is strictly equal to 1.
    """
    sums = sum(abs(x), dim=1)
    neutral = 1 - sums

    return cat((x, neutral.unsqueeze(1)), dim=1)

@conditional_lru_cache(maxsize=CACHE_SIZE)
def add_neutral_price(x):
    """
    Adds a neutral price to each asset.

    Args:
        - `x` (Tensor): A tensor of shape `(batch_size, n_tradable)`,
            where `batch_size` is the number of samples and
            `n_tradable` is the number of tradable assets.

    Returns:
        Tensor: A tensor of shape `(batch_size, n_tradable+1)`,
            where `batch_size` is the number of samples,
            `n_tradable` is the number of tradable assets and
            the last column contains neutral prices.
    """
    return cat((x, ones(x.shape[0], 1, device=x.device)), dim=1)

@conditional_lru_cache(maxsize=CACHE_SIZE)
def _validate_weights(weights):
    """
    Calls `warn` if some weights are invalid (>1 in sum).

    Args:
        - `weights` (Tensor): Weights matrix of shape: `(batch_size, n_tradable)`.
    """
    s = sum(abs(weights), dim=1) - EPSILON
    if any(s > 1):
        warn("invalid weights in allocation")

@conditional_lru_cache(maxsize=CACHE_SIZE)
def _process_backtest_inputs(
        weights: Tensor,
        prices: Tensor,
        prices_change: Tensor | None = None,
        check_weights: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Common input processing for allocation functions.

    Args:
        - `weights` (Tensor): Weights matrix
        - `prices` (Tensor): Prices matrix  
        - `prices_change` (Tensor): Logarithmic returns of prices
        - `check_weights` (bool): Whether to validate weights

    Returns:
        tuple[Tensor, Tensor, Tensor]: Processed weights, prices and price changes
    """
    weights, prices = reshape_predictions(weights, prices)
    
    if check_weights:
        _validate_weights(weights)

    prices = add_neutral_price(prices)
    
    if prices_change is None:
        prices_change = change(prices)
    else:
        prices_change = stack((
            prices_change, 
            zeros(prices_change.shape[0], 1, 
                  device=prices_change.device)
        ), dim=1)
        
    weights = add_neutral_weight(weights)
    
    return weights, prices, prices_change


@conditional_lru_cache(maxsize=CACHE_SIZE)
def allocation_quote_profit(weights: Tensor,
                            prices: Tensor,
                            prices_change: Tensor | None = None, 
                            commission: float = 0.0,
                            check_weights: bool = True):
    """
    Function calculates profit from quote weights without any adjustment in time.
    Note: This function provides approximate results and should only be used for rough 
    strategy evaluation.

    No market impact or slippage is considered.

    Args:
        - `weights` (Tensor): Weights matrix of shape: `(batch_size, n_tradable)`.
            Sum of absolute values in each row is `<=1` (negative values used for short positions).
            Needs to be already shifted to the next time step to avoid leakage in backtesting.
        - `prices` (Tensor): Prices matrix of shape: `(batch_size, n_tradable)`.
        - `prices_change` (Tensor): Logarithmic returns of prices.
            Shape: `(batch_size, n_tradable)`.
        - `commission` (float): Commission rate in range `[0, 1]`.
        - `check_weights` (bool): Whether to validate input weights.
            If the weights are invalid, function will issue a warning.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - Gross returns before commissions
            - Commission costs as positive relative values
    """
    weights, prices, prices_change = _process_backtest_inputs(
        weights,
        prices, 
        prices_change, 
        check_weights)

    change = sum(weights * prices_change, dim=1)
    
    commission_abs = zeros_like(change)
    if commission:
        rolled_quote_weights = roll(weights, shifts=1, dims=0)
        quote_weights_diff = weights - rolled_quote_weights
        commission_abs = commission * sum(abs(quote_weights_diff), dim=1)

    return change, commission_abs


@conditional_lru_cache(maxsize=CACHE_SIZE)
def reshape_predictions(weights: Tensor, prices: Tensor) -> tuple[Tensor, Tensor]:
    """
    Reshapes one-dim array to two-dim array if needed.
    
    Args:
        - `weights` (Tensor): model output weights.
            Shape: `(batch_size, n_tradable)`.
        - `prices` (Tensor): prices matrix.
            Shape: `(batch_size, n_tradable)`.

    Returns:
        Tensor: weights matrix of shape: `(batch_size, n_tradable)`.
        Tensor: prices matrix of shape: `(batch_size, n_tradable)`.
    """
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    if len(prices.shape) == 1:
        prices = prices.reshape(-1, 1)
    return weights, prices
