import torch

from datetime import datetime
import numpy as np

from MLTT.utils import apply_columnwise, CACHE_SIZE
from MLTT.cache import conditional_lru_cache


@conditional_lru_cache(CACHE_SIZE)
def sma(close_log: torch.Tensor, period: int) -> torch.Tensor:
    """
    Computes Simple Moving Average using cumulative sum approach.
    Handles edge cases where window size exceeds available data.
    
    Args:
        - `close_log` (torch.Tensor): Input tensor of shape (T, N) where T is time dimension
        - `period` (int): SMA window size (must be positive)
        
    Returns:
        Tensor: SMA with same shape as input, dtype float32
    """
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if close_log.ndim != 2:
        raise ValueError(f"Input must be 2D tensor (T, N), got shape {close_log.shape}")
    
    T, N = close_log.shape
    if T == 0:
        return close_log.clone()
    
    cumsum = torch.cumsum(close_log, dim=0)
    
    if period > T:
        window = torch.arange(1, T+1, device=close_log.device, dtype=torch.float32).view(-1, 1)
        return (cumsum / window).to(close_log.dtype)
    
    # Calculate moving sum using shifted cumulative sums
    shifted = torch.zeros_like(cumsum)
    shifted[period:] = cumsum[:-period]
    moving_sum = cumsum - shifted
    
    # Calculate dynamic window sizes (clipped at period)
    window = torch.arange(1, T+1, device=close_log.device, dtype=torch.float32).view(-1, 1)
    window = torch.minimum(window, torch.tensor(period, dtype=torch.float32, device=close_log.device))
    
    return (moving_sum / window).to(close_log.dtype)


@conditional_lru_cache(CACHE_SIZE)
def ema(close_log: torch.Tensor, period: int) -> torch.Tensor:
    """
    Computes Exponential Moving Average (EMA) of a time series using vectorized PyTorch ops.

    Args:
        - `close_log` (torch.Tensor): Time series (T, N), where T — time, N — series.
        - `period` (int): Smoothing period.

    Returns:
        torch.Tensor: EMA of same shape as input.
    """
    alpha = 2 / (period + 1)
    close_log = close_log.float()
    ema = torch.zeros_like(close_log)
    ema[0] = close_log[0]  # init first value to be same

    # recursive EMA computation, vectorized over columns
    for t in range(1, close_log.shape[0]):
        ema[t] = alpha * close_log[t] + (1 - alpha) * ema[t - 1]
    
    return ema

@conditional_lru_cache(CACHE_SIZE)
def stdev(series: torch.Tensor, period: int) -> torch.Tensor:
    """
    Calculates rolling standard deviation of a time series.

    Args:
        - `series` (torch.Tensor): (T, N) tensor where T — time, N — different series (columns).
        - `period` (int): Rolling window size.

    Returns:
        torch.Tensor: (T, N) rolling standard deviation. First few rows will be NaN or zero due to padding.
    """
    T = series.shape[0]
    if period > T:
        raise ValueError("Period can't be greater than time dimension")

    # pad the front so that output shape stays (T, N)
    pad = period - 1
    padded = torch.nn.functional.pad(series.T.unsqueeze(1), (pad, 0), mode='replicate')  # (N, 1, T + pad)

    # unfold to get rolling windows
    unfolded = padded.unfold(dimension=2, size=period, step=1)  # (N, 1, T, period)
    
    # calc std over last dim (window)
    std = unfolded.std(dim=-1)  # (N, 1, T)
    
    return std.squeeze(1).T  # -> (T, N)


@conditional_lru_cache(CACHE_SIZE)
def _kalman_muth_ewma_volatility(returns, 
                                mu: float = 0.0, 
                                lambda_: float = 0.05,
                                tau_psi: float = 0.02, 
                                tau_eta: float = 0.1,
                                init_state_variance: float = 1.0,
                                ):
    """
    Kalman filter for estimating volatility using a state-space model
    with mean-reversion in the latent log-variance process.

    Muth's EWMA model (1960).
    Source: The elements of quantitative investing (2024), Giuseppe A. Paleologo
    
    Args:
        - `returns` (torch.Tensor): Array of returns.
        - `mu` (float): Long-term mean of variance (log-space).
        - `lambda_` (float): Mean-reversion strength.
        - `tau_psi` (float): Std of process noise.
        - `tau_eta` (float): Std of observation noise.
        - `init_state_variance` (float): Initial state variance (log-space) for the Kalman filter.

    Returns:
        - volatility_estimates (np.ndarray): Estimated volatilities (std).
        - log_variance_estimates (np.ndarray): Estimated log-variances.
    """
    n = len(returns)
    
    # Initialize state variables
    x_pred = mu  # initial estimate of log variance
    P_pred = init_state_variance  # initial state variance

    # Storage
    x_filt = torch.zeros(n)
    P_filt = torch.zeros(n)
    sigma_est = torch.zeros(n)

    # Precompute constants
    gamma = -1.2704  # E[log(χ²_1)] ≈ -1.27
    R = tau_eta ** 2  # Observation noise variance
    Q = tau_psi ** 2  # Process noise variance

    for t in range(n):
        # Observation: y_t = log(r_t^2) - gamma
        y_t = torch.log(returns[t]**2 + 1e-8) - gamma  # small value added to avoid log(0)

        # --- Prediction step ---
        x_pred = (1 - lambda_) * x_pred + lambda_ * mu
        P_pred = (1 - lambda_)**2 * P_pred + Q

        # --- Update step ---
        K = P_pred / (P_pred + R)  # Kalman gain
        x_upd = x_pred + K * (y_t - x_pred)
        P_upd = (1 - K) * P_pred

        # Save results
        x_filt[t] = x_upd
        P_filt[t] = P_upd
        sigma_est[t] = torch.exp(x_upd / 2)

        # Prepare for next step
        x_pred = x_upd
        P_pred = P_upd

    return sigma_est, x_filt

@conditional_lru_cache(CACHE_SIZE)
def kalman_muth_volatility(
        returns, 
        mu: float = 0.0, 
        lambda_: float = 0.05,
        tau_psi: float = 0.02, 
        tau_eta: float = 0.1,
        init_state_variance: float = 1.0,
    ):
    if returns.dim() == 1:
        return _kalman_muth_ewma_volatility(returns, mu, lambda_, tau_psi, tau_eta, init_state_variance)[0]
    return apply_columnwise(returns, lambda x: _kalman_muth_ewma_volatility(x, mu, lambda_, tau_psi, tau_eta, init_state_variance)[0])


@conditional_lru_cache(CACHE_SIZE)
def z_score(series: torch.Tensor, period: int, mean: float | None = None) -> torch.Tensor:
    if mean is None:
        return (series - sma(series, period)) / stdev(series, period)
    else:
        return (series - mean) / stdev(series, period)


@conditional_lru_cache(CACHE_SIZE)
def vwap(prices: torch.Tensor, volumes: torch.Tensor, period: int) -> torch.Tensor:
    """
    Computes Volume-Weighted Average Price with dynamic window handling.
    Implements both cumulative and rolling window approaches.
    
    Args:
        - `prices` (torch.Tensor): Tensor of shape (T,) or (T, N) containing price values
        - `volumes` (torch.Tensor): Tensor of shape (T,) or (T, N) with corresponding volumes
        - `period` (int): Lookback window size for VWAP calculation
        
    Returns:
        Tensor: VWAP values with same shape as input
    """
    if prices.shape != volumes.shape:
        raise ValueError(f"Prices and volumes must have same shape. Got {prices.shape} vs {volumes.shape}")
    
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    
    # Calculate price-volume product and cumulative sums
    pv = prices * volumes
    cum_pv = torch.cumsum(pv, dim=0)
    cum_vol = torch.cumsum(volumes, dim=0)
    
    # Handle full-period windows using rolling difference
    if len(prices) > period:
        shifted_pv = torch.zeros_like(cum_pv)
        shifted_pv[period:] = cum_pv[:-period]
        
        shifted_vol = torch.zeros_like(cum_vol)
        shifted_vol[period:] = cum_vol[:-period]
        
        window_pv = cum_pv - shifted_pv
        window_vol = cum_vol - shifted_vol
    else:
        window_pv = cum_pv
        window_vol = cum_vol
    
    # Add epsilon to prevent division by zero
    epsilon = torch.finfo(prices.dtype).eps
    vwap_values = window_pv / (window_vol + epsilon)
    
    # Maintain original dtype
    return vwap_values.to(prices.dtype)


@conditional_lru_cache(CACHE_SIZE)
def rsi(close_log: torch.Tensor, period: int) -> torch.Tensor:
    """
    Calculates the Relative Strength Index (RSI) of a time series.

    Args:
        - `close_log` (torch.Tensor): Tensor of shape (T, N) containing price values
        - `period` (int): Lookback window size for RSI calculation
        
    Returns:
        Tensor: RSI values with same shape as input
    """
    T, N = close_log.shape

    delta = close_log[1:] - close_log[:-1]  # (T-1, N)

    gain = torch.clamp(delta, min=0)
    loss = -torch.clamp(delta, max=0)

    avg_gain = torch.zeros((T - 1, N), device=close_log.device)
    avg_loss = torch.zeros((T - 1, N), device=close_log.device)

    avg_gain[period - 1] = gain[:period].mean(dim=0)
    avg_loss[period - 1] = loss[:period].mean(dim=0)

    for t in range(period, T - 1):
        avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gain[t]) / period
        avg_loss[t] = (avg_loss[t - 1] * (period - 1) + loss[t]) / period

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)

    # Padding to match the input length
    rsi_full = torch.zeros((T, N), device=close_log.device)
    rsi_full[period:] = rsi[period - 1:]  # first period values are default 0

    return rsi_full



class GlobalTimeFeatures:
    def __init__(self, timestamp: torch.Tensor):
        # Convert timestamp tensor to datetime objects
        self.datetime = np.array([
            datetime.fromtimestamp(ts.item())
            for ts in timestamp / 1000
        ])
    
    def day_of_week(self) -> torch.Tensor:
        # Returns tensor with values 0 (Monday) through 6 (Sunday)
        return torch.tensor([
            dt.weekday() 
            for dt in self.datetime
        ], dtype=torch.long).unsqueeze(-1)
    
    def day_of_month(self) -> torch.Tensor:
        # Returns tensor with values 1-31
        return torch.tensor([
            dt.day 
            for dt in self.datetime
        ], dtype=torch.long).unsqueeze(-1)
    
    def month_of_year(self) -> torch.Tensor:
        # Returns tensor with values 1-12
        return torch.tensor([
            dt.month 
            for dt in self.datetime
        ], dtype=torch.long).unsqueeze(-1)
    
    def hour_of_day(self) -> torch.Tensor:
        # Returns tensor with values 0-23
        return torch.tensor([
            dt.hour 
            for dt in self.datetime
        ], dtype=torch.long).unsqueeze(-1)