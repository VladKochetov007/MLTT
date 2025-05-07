from torch import Tensor, zeros_like, cumsum, roll
from MLTT.allocation.backend import allocation_quote_profit
from MLTT.utils import CACHE_SIZE
from MLTT.allocation.allocators import CapitalAllocator

from MLTT.cache import conditional_lru_cache

class BTResult:
    """
    Represents a backtest result with logarithmic performance metrics
    """
    _expenses_log: Tensor
    _gross_log_change: Tensor
    _net_log_change: Tensor
    _log_equity: Tensor | None
    _gross_equity: Tensor | None
    _weights: Tensor | None

    def __init__(self,
                 gross_log_change: Tensor,
                 expenses_log: Tensor | None = None,
                 weights: Tensor | None = None) -> None:
        """
        Args:
            - `gross_log_change` (Tensor): Logarithmic change of equity.
            - `expenses_log` (Tensor): Logarithmic expenses in range `[0, +inf)`.
            - `weights` (Tensor): Weights of the portfolio.
        """
        if expenses_log is None:
            expenses_log = zeros_like(gross_log_change)
        self._expenses_log = expenses_log
        self._gross_log_change = gross_log_change
        self._net_log_change = gross_log_change - self._expenses_log
        self._weights = weights
        self._log_equity = None  # Lazy initialization
        self._gross_equity = None  # Lazy initialization

    @property
    def log_equity(self) -> Tensor:
        if self._log_equity is None:
            self._log_equity = cumsum(self.net_change, dim=0)
        return self._log_equity
        
    @property
    def gross_equity(self) -> Tensor:
        if self._gross_equity is None:
            self._gross_equity = cumsum(self.gross_change, dim=0)
        return self._gross_equity

    @property
    def expenses_log(self) -> Tensor:
        return self._expenses_log

    @property
    def net_change(self) -> Tensor:
        return self._net_log_change

    @property
    def gross_change(self) -> Tensor:
        return self._gross_log_change

    @property
    def weights(self) -> Tensor | None:
        return self._weights

    def __getitem__(self, key: slice | int) -> 'BTResult':
        """
        Implements Python's slice syntax for BTResult objects.
        Allows using obj[start:end] notation.
        """
        if isinstance(key, int):
            start = key
            end = key + 1
        else:
            start = key.start if key.start is not None else 0
            end = key.stop if key.stop is not None else len(self.gross_change)
            
        sliced = self.__class__(
            self.gross_change[start:end],
            self.expenses_log[start:end],
            self.weights[start:end] if self.weights is not None else None
        )
        if self._log_equity is not None:
            sliced._log_equity = self._log_equity[start:end]
        if self._gross_equity is not None:
            sliced._gross_equity = self._gross_equity[start:end]
        return sliced
    

@conditional_lru_cache(maxsize=CACHE_SIZE)
def backtest(
    weights: Tensor,
    prices: Tensor | None = None,
    prices_change: Tensor | None = None,
    commission: float = 0.0,
    check_weights: bool = True,
    save_weights: bool = False
) -> BTResult:
    """
    Function calculates profit from quote weights without any adjustment in time.

    Args:
        - `weights` (Tensor): Weights matrix of shape: `(batch_size, n_tradable)`.
            Sum of absolute values in each row is `<=1` (negative values used for short positions).
            Needs to be already shifted to the next time step to avoid leakage in backtesting.
        - `prices` (Tensor): Log-prices matrix of shape: `(batch_size, n_tradable)`.
        - `prices_change` (Tensor): change of log-prices values in range `(-inf, +inf)` aka log-return.
            Shape: `(batch_size, n_tradable)`. If this argument is provided,
            the `prices` argument is not necessary and has no effect on the function's
            behavior. This is more efficient because calculating prices from price
            changes is computationally more expensive than just multiplying them.
        - `commission` (float): Commission rate in range `[0, 1]`.
        - `check_weights` (bool): Whether to validate input weights.
            If the weights are invalid, function will issue a warning.
        - `save_weights` (bool): Whether to save predicted weights. If False saves memory.
    
    Returns:
        BTResult: backtest summary with `1-d` equity curve and more accurate percentage change.
    """
    gross_change, commission_abs = allocation_quote_profit(weights, prices, prices_change, commission, check_weights)
    return BTResult(
        gross_log_change=gross_change,
        expenses_log=commission_abs,
        weights=weights if save_weights else None
    )


@conditional_lru_cache(maxsize=CACHE_SIZE)
def backtest_model(
        model: CapitalAllocator,
        prices: Tensor,
        prediction_info: Tensor | None = None,
        prices_change: Tensor | None = None,
        check_weights: bool = True,
        commission: float = 0.0,
        save_weights: bool = False
    ) -> BTResult:
    """
    Function to test a given model on a set of features and price data.
    The function takes a model and inputs, and returns a backtest result.

    Args:
        - `model` (CapitalAllocator): Model to be tested. It gives
            a portfolio distribution on each moment.
            The model should have a `min_observations` attribute,
            which is minimum number of observations in a slice.
        - `prices` (Tensor): The prices used to backtest the model.
            Shape: `(batch_size, n_tradable)`.
        - `prediction_info` (Tensor): Input data for the model, usually prices.
            If not provided, `prices` is used.
            Shape: `(batch_size, *n_information)`.
        - `prices_change` (Tensor): Price changes used to backtest the model.
            Shape: `(batch_size, n_tradable)`.
            Only one of `prices` or `prices_change` is required.
            If both are provided, `prices` is ignored.
        - `check_weights` (bool): Whether to check the weights of the model.
            Defaults to True.
        - `commission` (float): Commission to be paid on each trade.
            Defaults to 0.0.
        - `save_weights` (bool): Whether to save predicted weights. If False saves memory.
            Defaults to False.

    Returns:
        BTResult: The result of the backtesting process.
            The result is a curve of equity and percentage change of equity.

    Note:
        - `batch_size` is a number of observations in input data.
        - `n_tradable` is a number of assets.
        - `n_information` is a number of features.
    """
    if prediction_info is None:
        prediction_info = prices
        
    predictions = model(prediction_info)
    weights = roll(predictions, shifts=1, dims=0)
    weights[0] = 0

    return backtest(weights=weights,
                    prices=prices,
                    prices_change=prices_change,
                    check_weights=check_weights,
                    commission=commission,
                    save_weights=save_weights)
