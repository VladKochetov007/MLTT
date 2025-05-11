import torch
from scipy.stats import ttest_1samp
from dataclasses import dataclass

from MLTT.allocation import CapitalAllocator
from MLTT import backtest_model
from warnings import warn


def _random_prices(shape: tuple[int, int]) -> torch.Tensor:
    prices_matrix = torch.cumsum(torch.randn(*shape), dim=0)
    prices_matrix += torch.randn(1, shape[1])
    return prices_matrix

class ConsistencyCheckResult:
    """Container for leak check results"""
    __slots__ = ['_has_leaks', '_discrepancies', '_tolerance']
    
    def __init__(self, has_leaks: bool, discrepancies: list[tuple[int, float]], tolerance: float):
        self._has_leaks = has_leaks
        self._discrepancies = tuple(discrepancies)
        self._tolerance = tolerance

    @property
    def has_leaks(self) -> bool:
        """Whether any inconsistencies were detected"""
        return self._has_leaks
        
    @property
    def discrepancies(self) -> tuple[tuple[int, float], ...]:
        """Tuple of (timestamp, weight_diff) pairs"""
        return self._discrepancies
        
    @property
    def tolerance(self) -> float:
        """Configured tolerance threshold used in check"""
        return self._tolerance
        
    @property
    def timestamps(self) -> tuple[int, ...]:
        """Timestamps with detected discrepancies"""
        return tuple(t for t, _ in self._discrepancies)
    def __repr__(self) -> str:
        status = "❌ INCONSISTENT" if self.has_leaks else "✅ CONSISTENT"
        details = (
            f"Threshold: {self.tolerance:.1e}\n"
            f"Total discrepancies: {len(self.discrepancies)}\n"
        )
        if self.has_leaks:
            details += f"First 3 timestamps: {self.timestamps[:3]}\n"
        return f"{status} WEIGHTS\n{details}"


class WeightConsistencyChecker:
    """
    Checks for weight inconsistencies at transition points between time windows.
    Detects data leaks by comparing model's weight allocations at overlapping timestamps.
    
    Args:
        - `prediction_info` (torch.Tensor): Input tensor for predictions (n_samples, n_features)
        - `window_size` (int | None): History window size for predictions. None = use all history
        - `step` (int): Step size between validation points
        - `tolerance` (float): Maximum allowed weight difference threshold
    
    Example:
        >>> checker = WeightConsistencyChecker(data, window_size=50)
        >>> has_leaks, discrepancies = checker.run_check(model)
        >>> checker.print_report(discrepancies)
    """
    
    def __init__(
        self,
        prediction_info: torch.Tensor,
        window_size: int | None = None,
        step: int = 1,
        tolerance: float = 1e-10
    ):
        self.data = prediction_info
        self.window_size = window_size
        self.step = step
        self.tolerance = tolerance

    def _validate_params(self, min_obs: int) -> None:
        """Check input parameters validity"""
        if min_obs >= self.data.shape[0]:
            raise ValueError(
                f"Insufficient data ({self.data.shape[0]}) "
                f"for model's minimum observations ({min_obs})"
            )
            
        if self.window_size and self.window_size <= min_obs + 1:
            warn(f"Window size ({self.window_size}) smaller than model's stability minimum ({min_obs + 2})")

    def _prepare_data(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get data windows for timestamp t and t-1"""
        if self.window_size is None:
            return self.data[:t-1], self.data[:t]
            
        start_prev = max(0, t-1 - self.window_size)
        start_curr = max(0, t - self.window_size)
        return (
            self.data[start_prev:t-1],
            self.data[start_curr:t]
        )

    def _compare_predictions(
        self, 
        t: int,
        model: CapitalAllocator
    ) -> tuple[int, float] | None:
        """Compare weight allocations at transition point t"""
        data_prev, data_curr = self._prepare_data(t)
        
        weights_prev = model.predict(data_prev)
        weights_curr = model.predict(data_curr)
        
        if weights_curr.shape[0] < 2 or weights_prev.shape[0] < 1:
            return None
            
        max_diff = torch.max(torch.abs(weights_prev[-1] - weights_curr[-2])).item()
        return (t, max_diff) if max_diff > self.tolerance else None

    def check(self, model: CapitalAllocator) -> ConsistencyCheckResult:
        """Execute consistency check and return result object"""
        min_obs = model.min_observations
        self._validate_params(min_obs)
        
        discrepancies = []
        for t in range(min_obs + 1, self.data.shape[0], self.step):
            if diff := self._compare_predictions(t, model):
                discrepancies.append(diff)
                
        return ConsistencyCheckResult(
            has_leaks=len(discrepancies) > 0,
            discrepancies=discrepancies,
            tolerance=self.tolerance
        )

@dataclass(frozen=True)
class LeakTestResult:
    """Represents results of data leak check using random data"""
    pvalue: float
    data_shape: tuple[int, int] | None = None
    used_external_data: bool = False
    signif: float = 0.05
    
    @property
    def has_leaks(self) -> bool:
        """Indicates if potential data leaks were detected"""
        return self.pvalue < self.signif
    
    def __repr__(self) -> str:
        status = "🔴 LEAKS DETECTED" if self.has_leaks else "🟢 NO LEAKS"
        return (
            f"{status}\n"
            f"Significance level: {self.signif:.3f}\n"
            f"P-value: {self.pvalue:.4f}\n"
            f"Data shape: {self.data_shape or 'external_data'}\n"
        )

class RandomLeakTester:
    """
    Performs data leak validation using random/supplied market data
    
    Args:
        - `model` (CapitalAllocator): Strategy model to validate
        - `data_shape` (tuple[int, int] | None): Optional shape for generated data (n_samples, n_assets)
        - `external_data` (torch.Tensor | None): Optional pre-generated data tensor to use instead of random
        - `prediction_info` (torch.Tensor | None): Prediction data template for model input
        - `signif` (float): Statistical test threshold (default: 0.05)
    """
    
    def __init__(
        self,
        model: CapitalAllocator,
        data_shape: tuple[int, int] | None = None,
        external_data: torch.Tensor | None = None,
        prediction_info: torch.Tensor | None = None,
        signif: float = 0.05
    ):
        if not 0 < signif < 1:
            raise ValueError("Significance level must be between 0 and 1")
            
        self.signif = signif
        if not data_shape and not external_data:
            raise ValueError("Must provide either data_shape or external_data")
            
        self.model = model
        self.data_shape = data_shape
        self.external_data = external_data
        self.prediction_info = prediction_info
        self._test_result: LeakTestResult | None = None

    def run_test(self) -> LeakTestResult:
        """Execute full leak test pipeline and return results"""
        prices = self._generate_data()
        returns = self._run_backtest(prices)
        return self._analyze_results(returns, prices.shape)

    def _generate_data(self) -> torch.Tensor:
        """Generate random walk data if no external data provided"""
        if self.external_data is not None:
            return self.external_data
        return _random_prices(self.data_shape)

    def _run_backtest(self, prices: torch.Tensor) -> torch.Tensor:
        """Execute backtest and return strategy returns"""
        result = backtest_model(
            self.model,
            prices,
            prediction_info=self.prediction_info,
            commission=0
        )
        return result.gross_change

    def _analyze_results(self, returns: torch.Tensor, data_shape: tuple[int, ...]) -> LeakTestResult:
        """Calculate statistics and generate final report"""
        returns_np = returns.cpu().numpy() if isinstance(returns, torch.Tensor) else returns
        pvalue = ttest_1samp(returns_np, 0).pvalue
        
        return LeakTestResult(
            pvalue=pvalue,
            data_shape=data_shape,
            used_external_data=self.external_data is not None,
            signif=self.signif
        )

def check_random_data_leaks(
    model: CapitalAllocator,
    data_shape: tuple[int, int] = (1000, 20),
    external_data: torch.Tensor | None = None,
    prediction_info: torch.Tensor | None = None,
    signif: float = 0.05
) -> LeakTestResult:
    """
    Validate strategy for data leaks using random/supplied market data.
    
    Args:
        - `model` (CapitalAllocator): Strategy model to validate
        - `data_shape` (tuple[int, int]): Optional shape for generated data (n_samples, n_assets)
        - `external_data` (torch.Tensor | None): Optional pre-generated data tensor to use instead of random
        - `prediction_info` (torch.Tensor | None): Prediction data template for model input
        - `signif` (float): Statistical significance threshold (default: 0.05)
        
    Returns:
        LeakTestResult: Results of the leak test including p-value and assessment
    """
    return RandomLeakTester(
        model=model,
        data_shape=data_shape,
        external_data=external_data,
        prediction_info=prediction_info,
        signif=signif
    ).run_test()
