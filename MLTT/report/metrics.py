from abc import ABC, abstractmethod

import torch
from numpy import sqrt, nan
from scipy.stats import skew, kurtosis

from warnings import warn
from MLTT.allocation.backtesting import BTResult
from MLTT.utils import CACHE_SIZE
from MLTT.cache import conditional_lru_cache

# Abstract base class for all metrics
class Metric(ABC):
    def __init__(self, **params):
        """
        Initializes the metric with given parameters.

        Args:
            - `params` (dict): Dictionary of parameters needed for the metric calculation
            - `T` (int): Number of periods in a year (365 for daily, 12 for monthly, etc)
        """
        self.params = params

    @abstractmethod
    def calculate(self, backtest_result: BTResult) -> float:
        """
        Abstract method to calculate the metric.

        Args:
            - `backtest_result` (BTResult): A backtest result object

        Returns:
            float: The calculated metric
        """
        pass

    def update_params(self, **new_params):
        """
        Updates the metric's parameters.

        Args:
            - `new_params` (dict): Dictionary of new parameters to update
        """
        self.params.update(new_params)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property for the metric's name.

        Returns:
            str: The name of the metric
        """
        pass

    @classmethod
    @abstractmethod
    def format_value(cls, value: float) -> str:
        """
        Abstract method to format the metric value.

        Args:
            - `value` (float): The metric value to format. Example for mean profit: 0.55

        Returns:
            str: The formatted metric value. Example for mean profit: "55%"
        """
        pass

    def __call__(self, backtest_result: BTResult) -> float:
        return self.calculate(backtest_result)


# Concrete implementations of different metrics
class AnnualMean(Metric):
    @property
    def name(self) -> str:
        return f'Annual Mean (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        return torch.mean(backtest_result.net_change) * T

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'

class MonthlyMean(Metric):
    @property
    def name(self) -> str:
        return f'Monthly Mean (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        return torch.mean(backtest_result.net_change) * T / 12

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualStd(Metric):
    @property
    def name(self) -> str:
        return f'Annual Standard Deviation (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        return torch.std(backtest_result.net_change) * sqrt(T)

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualSharpe(Metric):
    @property
    def name(self) -> str:
        return f'Annual Sharpe Ratio (Annualization: {self.params.get("T", 1)}, Risk Free Rate: {self.params.get("risk_free_rate", 0) * 100}%)'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        rf = self.params.get('risk_free_rate', 0)
        mean_return = AnnualMean(T=T).calculate(backtest_result)
        std_dev = AnnualStd(T=T).calculate(backtest_result)
        return (mean_return - rf) / std_dev

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.2f}'


class DownsideStd(Metric):
    @property
    def name(self) -> str:
        return 'Downside Standard Deviation'

    def calculate(self, backtest_result: BTResult) -> float:
        return torch.std(backtest_result.net_change[backtest_result.net_change < 0])   
    
    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualDownsideStd(Metric):
    @property
    def name(self) -> str:
        return f'Annual Downside Standard Deviation (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        return DownsideStd().calculate(backtest_result) * sqrt(T)
    
    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualSortino(Metric):
    @property
    def name(self) -> str:
        return f'Annual Sortino Ratio (Annualization: {self.params.get("T", 1)}, Risk Free Rate: {self.params.get("risk_free_rate", 0) * 100}%)'

    def calculate(self, backtest_result: BTResult) -> float:
        rf = self.params.get('risk_free_rate', 0)
        mean_return = AnnualMean(**self.params).calculate(backtest_result)
        downside_std = AnnualDownsideStd(**self.params).calculate(backtest_result)
        return (mean_return - rf) / downside_std

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.2f}'


class MaxDrawdown(Metric):
    @property
    def name(self) -> str:
        return 'Maximum Drawdown'

    def calculate(self, backtest_result: BTResult) -> float:
        # Use torch.cummax to get the cumulative maximum values
        running_max, _ = torch.cummax(backtest_result.log_equity, dim=0)
        # Calculate drawdowns as difference between running max and current value
        drawdowns = running_max - backtest_result.log_equity
        # Return the maximum drawdown
        return torch.max(drawdowns)
    
    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AverageDrawdown(Metric):
    @property
    def name(self) -> str:
        return 'Average Drawdown'

    def calculate(self, backtest_result: BTResult) -> float:
        # Use torch.cummax to get the cumulative maximum values
        running_max, _ = torch.cummax(backtest_result.log_equity, dim=0)
        # Calculate drawdowns as difference between running max and current value
        drawdowns = running_max - backtest_result.log_equity
        # Return the average drawdown
        return torch.mean(drawdowns)

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualCalmar(Metric):
    @property
    def name(self) -> str:
        return f'Annual Calmar Ratio (Annualization: {self.params.get("T", 1)}, Risk Free Rate: {self.params.get("risk_free_rate", 0) * 100}%)'

    def calculate(self, backtest_result: BTResult) -> float:
        rf = self.params.get('risk_free_rate', 0)
        mean_return = AnnualMean(**self.params).calculate(backtest_result)
        max_dd = MaxDrawdown().calculate(backtest_result)
        return (mean_return - rf) / max_dd

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.2f}'


class ValueAtRisk(Metric):
    @property
    def name(self) -> str:
        return f'Value at Risk (confidence={self.params.get("var_confidence", 0.95) * 100}%)'

    def calculate(self, backtest_result: BTResult) -> float:
        confidence = self.params.get('var_confidence', 0.95)
        return torch.quantile(backtest_result.net_change, 1 - confidence)

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{-value*100:.2f}%'


class ConditionalValueAtRisk(Metric):
    @property
    def name(self) -> str:
        return f'Conditional Value at Risk (confidence={self.params.get("var_confidence", 0.95) * 100}%)'

    def calculate(self, backtest_result: BTResult) -> float:
        var = ValueAtRisk(**self.params).calculate(backtest_result)
        tail_losses = backtest_result.net_change[backtest_result.net_change <= var]
        return torch.mean(tail_losses)
    
    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{-value*100:.2f}%'


class MedianDrawdown(Metric):
    @property
    def name(self) -> str:
        return 'Median Drawdown'

    def calculate(self, backtest_result: BTResult) -> float:
        return torch.median(torch.cummax(backtest_result.log_equity, dim=0)[0] - backtest_result.log_equity)

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualAlpha(Metric):
    def __init__(self, benchmark: torch.Tensor, **params):
        super().__init__(**params)
        self.benchmark = benchmark

    @property
    def name(self) -> str:
        return f'Annual Alpha (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 1)
        
        avg = AnnualMean(T=T)
        
        # Calculate annual mean returns for data and benchmark
        annual_mean_data = avg.calculate(backtest_result)
        annual_mean_benchmark = avg.calculate(self.benchmark)
        
        # Calculate and return annual alpha as the difference of annual means
        return annual_mean_data - annual_mean_benchmark

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'


class AnnualBeta(Metric):
    def __init__(self, benchmark: torch.Tensor, **params):
        super().__init__(**params)
        self.benchmark = benchmark

    @property
    def name(self) -> str:
        return f'Annual Beta (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        if len(backtest_result.net_change) != len(self.benchmark):
            warn("Strategy data and benchmark must have the same length")
            return float('nan')
        covariance = torch.cov(backtest_result.net_change, self.benchmark)[0][1]
        benchmark_variance = torch.var(self.benchmark)
        return covariance / benchmark_variance

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.2f}'


class UlcerIndex(Metric):
    @property
    def name(self) -> str:
        return 'Ulcer Index'

    def calculate(self, backtest_result: BTResult) -> float:
        drawdowns = torch.cummax(backtest_result.log_equity, dim=0)[0] - backtest_result.log_equity
        return torch.sqrt(torch.mean(drawdowns**2))

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.2f}'


class Skewness(Metric):
    @property
    def name(self) -> str:
        return 'Skewness'

    def calculate(self, backtest_result: BTResult) -> float:
        return skew(backtest_result.net_change.cpu())

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.4f}'


class Kurtosis(Metric):
    @property
    def name(self) -> str:
        return 'Excess Kurtosis'

    def calculate(self, backtest_result: BTResult) -> float:
        return kurtosis(backtest_result.net_change.cpu())

    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value:.4f}'


def format_duration(value: float) -> str:
    """
    Convert duration in seconds to a human-readable string.
    """
    if value < 1/1000:  # Less than 1ms
        return f'{value*1000000:.2f} μs'
    elif value < 1:  # Less than 1s
        return f'{value*1000:.2f} ms'
    elif value < 60:  # Less than 1m
        return f'{value:.2f} s'
    elif value < 3600:  # Less than 1h
        return f'{value/60:.2f} m'
    elif value < 86400:  # Less than 1d
        hours = int(value/3600)
        minutes = int((value % 3600)/60)
        return f'{hours}h {minutes}m'
    else:
        days = int(value/86400)
        hours = int((value % 86400)/3600)
        return f'{days}d {hours}h'
    
def years_to_seconds(years: float) -> float:
    return years * 365 * 24 * 60 * 60


def calculate_drawdown_durations(log_equity: torch.Tensor) -> torch.Tensor:
    cum_max = torch.cummax(log_equity, dim=0)[0]
    drawdowns = cum_max - log_equity
    is_drawdown = drawdowns > 0

    # Находим переходы между состояниями просадки
    diff = torch.diff(is_drawdown.int())
    starts = torch.where(diff == 1)[0] + 1
    ends = torch.where(diff == -1)[0] + 1

    # Обрабатываем незавершенную просадку в конце
    if is_drawdown[-1]:
        ends = torch.cat([ends, torch.tensor([len(is_drawdown)], device=ends.device)])

    # Выравниваем длины массивов и конвертируем в годы
    min_len = min(len(starts), len(ends))
    return (ends[:min_len] - starts[:min_len])

class AverageDrawdownDuration(Metric):
    @property
    def name(self) -> str:
        return 'Average Drawdown Duration'

    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 365)
        durations = calculate_drawdown_durations(backtest_result.log_equity) / T
        return years_to_seconds(torch.mean(durations.float())) if len(durations) > 0 else 0.0

    @classmethod
    def format_value(cls, value: float) -> str:
        return format_duration(value)

class MaxDrawdownDuration(Metric):
    @property
    def name(self) -> str:
        return 'Max Drawdown Duration'
    
    def calculate(self, backtest_result: BTResult) -> float:
        T = self.params.get('T', 365)
        durations = calculate_drawdown_durations(backtest_result.log_equity) / T
        return years_to_seconds(torch.max(durations.float())) if len(durations) > 0 else 0.0

    @classmethod
    def format_value(cls, value: float) -> str:
        return format_duration(value)

class AveragePositionDuration(Metric):
    @property
    def name(self) -> str:
        return 'Average Position Duration'

    def calculate(self, backtest_result: BTResult) -> float:
        """
        Calculate the average duration of positions.
        
        !!! IMPORTANT: data array actually not used in this metric. 
        All calculations are based on positions array from initialization.
        """
        weights = backtest_result.weights
        if weights is None:
            return 0.0
            
        # For multi-asset weights, we'll consider a position active if any asset has non-zero weight
        is_long_position = torch.any(weights > 0, axis=1) if len(weights.shape) > 1 else weights > 0
        is_short_position = torch.any(weights < 0, axis=1) if len(weights.shape) > 1 else weights < 0

        # Find the indices where long positions start and end
        long_starts = torch.where(torch.diff(is_long_position.int()) == 1)[0] + 1
        long_ends = torch.where(torch.diff(is_long_position.int()) == -1)[0] + 1

        # If the last period is a long position, add the end index
        if is_long_position[-1]:
            long_ends = torch.cat([long_ends, torch.tensor([len(is_long_position)], device=long_ends.device)])

        # Find the indices where short positions start and end
        short_starts = torch.where(torch.diff(is_short_position.int()) == 1)[0] + 1
        short_ends = torch.where(torch.diff(is_short_position.int()) == -1)[0] + 1

        # If the last period is a short position, add the end index
        if is_short_position[-1]:
            short_ends = torch.cat([short_ends, torch.tensor([len(is_short_position)], device=short_ends.device)])

        # Calculate long position durations
        long_durations = long_ends - long_starts if len(long_starts) > 0 and len(long_ends) > 0 else torch.tensor([])

        # Calculate short position durations
        short_durations = short_ends - short_starts if len(short_starts) > 0 and len(short_ends) > 0 else torch.tensor([])

        # Combine long and short durations
        all_durations = torch.cat((long_durations, short_durations))

        if len(all_durations) > 0:
            T = self.params.get('T', 365)
            return years_to_seconds(torch.mean(all_durations.float()) / T)
        else:
            return 0.0

    @classmethod
    def format_value(cls, value: float) -> str:
        return format_duration(value)

class DailyTurnover(Metric):
    @property
    def name(self) -> str:
        return f'Daily Turnover (Annualization: {self.params.get("T", 1)})'

    def calculate(self, backtest_result: BTResult) -> float:
        weights = backtest_result.weights  # Shape (n_observations, n_assets)
        if weights is None:
            return nan
        T = self.params.get('T', 1)

        if weights.shape[0] < 2:
            return 0.0  # Not enough data to compute turnover

        # Compute absolute weight changes
        turnover = torch.sum(torch.abs(torch.diff(weights, dim=0)), dim=1)

        # Average turnover over time and annualize
        avg_turnover = torch.mean(turnover)

        return avg_turnover * T / 365
    
    @classmethod
    def format_value(cls, value: float) -> str:
        return f'{value * 100:.2f}%'

ALL_METRICS = [
    AnnualMean(T=365),
    MonthlyMean(T=365),
    AnnualStd(T=365),
    AnnualSharpe(T=365, risk_free_rate=0.02),
    AnnualSortino(T=365, risk_free_rate=0.02),
    AnnualCalmar(T=365, risk_free_rate=0.02),
    MaxDrawdown(),
    AverageDrawdown(),
    MedianDrawdown(),
    ValueAtRisk(var_confidence=0.95),
    ConditionalValueAtRisk(var_confidence=0.95),
    UlcerIndex(),
    Skewness(),
    Kurtosis(),
    AverageDrawdownDuration(T=365),
    MaxDrawdownDuration(T=365),
    AveragePositionDuration(T=365),
    DailyTurnover(T=365)
]
