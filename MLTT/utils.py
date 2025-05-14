import torch

from MLTT.cache import conditional_lru_cache

EPSILON = 1e-6
CACHE_SIZE = 128

@conditional_lru_cache(maxsize=CACHE_SIZE)
def change(prices, lag: int = 1) -> torch.Tensor:
    """
    Computes absolute change in prices.

    Args:
        - `prices` (torch.Tensor): A tensor of shape `(batch_size, n_tradable)`.
        - `lag` (int): The number of periods to lag the prices by.
    Returns:
        torch.Tensor: Absolute change in prices. Shape: `(batch_size, n_tradable)`.
    """
    prices = torch.as_tensor(prices)

    pad = prices[0].detach().repeat(lag, 1)
    shifted_prices = torch.cat((pad, prices), dim=0)

    return shifted_prices[lag:] - shifted_prices[:-lag]

def apply_columnwise(s: torch.Tensor, func: callable) -> torch.Tensor:
    """
    Apply a function column-wise to a tensor.

    Result will be cached.

    Args:
        s (torch.Tensor): The input tensor.
        func (callable): The function to apply column-wise.
    """
    if s.dim() == 1:
        return func(s)

    res = torch.zeros_like(s)

    for j in range(s.shape[1]):
        res[:, j] = func(s[:, j])

    return res

@conditional_lru_cache(maxsize=CACHE_SIZE)
def to_weights_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a matrix to have each row sum to one when considering absolute values
    
    Parameters
    ----------
    x : torch.Tensor
        Input matrix
        
    Returns
    -------
    torch.Tensor
        Normalized matrix where each row sums to 1
    """
    # Convert to tensor if needed
    x = torch.as_tensor(x)
    
    # If x is 1D, just use to_weights directly
    if x.dim() == 1:
        return to_weights(x)
    
    # Sum absolute values along dim 1 (columns) for each row
    row_abs_sum = torch.sum(torch.abs(x), dim=1, keepdim=True)
    
    # Avoid division by zero
    row_abs_sum = torch.where(row_abs_sum == 0, torch.ones_like(row_abs_sum), row_abs_sum)
    
    # Normalize each row
    return x / row_abs_sum

@conditional_lru_cache(maxsize=CACHE_SIZE)
def to_weights(x):
    x = torch.as_tensor(x)
    return x / torch.sum(torch.abs(x))

@conditional_lru_cache(maxsize=CACHE_SIZE)
def alpha_beta(series: torch.Tensor, benchmark: torch.Tensor) -> tuple[float, float]:
    """
    Calculate alpha and beta for a series against a benchmark
    
    Parameters
    ----------
    series : torch.Tensor
        Returns series to evaluate
    benchmark : torch.Tensor
        Benchmark returns series
    
    Returns
    -------
    tuple[float, float]
        Alpha and beta values
    """
    # Convert inputs to tensors
    series = torch.as_tensor(series, dtype=torch.float32)
    benchmark = torch.as_tensor(benchmark, dtype=torch.float32)
    
    # Calculate covariance matrix
    stacked = torch.stack([benchmark, series])
    cov_matrix = torch.cov(stacked)
    
    # Extract values
    var_benchmark = cov_matrix[0, 0]
    cov_series_benchmark = cov_matrix[0, 1]
    
    # Calculate beta
    beta = cov_series_benchmark / var_benchmark
    
    # Calculate alpha (mean of series - beta * mean of benchmark)
    alpha = torch.mean(series) - beta * torch.mean(benchmark)
    
    return alpha.item(), beta.item()

@conditional_lru_cache(maxsize=CACHE_SIZE)
def mad_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize data using Median Absolute Deviation (MAD) method
    
    x_norm = (x - median(x)) / MAD
    where MAD = median(|x - median(x)|)
    
    Parameters
    ----------
    x : torch.Tensor
        Input data
    
    Returns
    -------
    torch.Tensor
        Normalized data
    """
    # Convert to tensor if needed
    x = torch.as_tensor(x)
    
    # Calculate median
    median = torch.median(x)
    
    # Calculate MAD
    mad = torch.median(torch.abs(x - median))
    
    # Handle case where MAD is zero to avoid division by zero
    if mad == 0:
        return torch.zeros_like(x)
    
    # Normalize
    return (x - median) / mad

def ensure_tensor(*arrays, device=None) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Convert arrays to PyTorch tensors with consistent float32 dtype
    
    Parameters
    ----------
    arrays : array-like
        Arrays to convert
    device : torch.device, optional
        Device to place tensors on
        
    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Converted tensor(s) with float32 dtype
    """
    tensors = []
    
    for array in arrays:
        if isinstance(array, torch.Tensor):
            tensor = array.float()  # Ensure float32 dtype
        else:
            tensor = torch.as_tensor(array, dtype=torch.float32)  # Convert to float32
            
        if device is not None:
            tensor = tensor.to(device)
            
        tensors.append(tensor)
        
    return tensors[0] if len(tensors) == 1 else tuple(tensors)

@conditional_lru_cache(maxsize=CACHE_SIZE)
def univariate_barrier(x: torch.Tensor, 
                      upper_bound: torch.Tensor | float, 
                      lower_bound: torch.Tensor | float,
                      neutral_level: torch.Tensor | float | None = 0) -> torch.Tensor:
    """
    Generates trading signals based on dynamic upper and lower barriers.
    
    Takes a time series and generates trading signals when the values cross
    specified upper and lower bounds. Implements a mean reversion strategy:
    - Generates -1 (short) when value exceeds upper bound
    - Generates +1 (long) when value falls below lower bound 
    - Maintains previous position when value is between bounds
    - If neutral_level is provided:
        Exits position (0) when value crosses back through neutral level
    - If neutral_level is None:
        Exits position only when opposite signal is generated
        (e.g. short position is closed only when price falls below lower bound)
    
    Parameters
    ----------
    x : torch.Tensor
        Input time series values
    upper_bound : torch.Tensor or float
        Upper barrier threshold, can be dynamic (tensor) or static (float)
    lower_bound : torch.Tensor or float
        Lower barrier threshold, can be dynamic (tensor) or static (float)
    neutral_level : torch.Tensor or float or None, default 0
        Level at which positions are exited, can be dynamic or static.
        If None, positions are held until opposite signal is generated.
        
    Returns
    -------
    torch.Tensor
        Tensor of -1, 0, +1 trading signals
    """
    # Convert inputs to tensors if needed
    x = torch.as_tensor(x)
    
    # Ensure x is at least 1D
    if x.dim() == 0:
        x = x.unsqueeze(0)
    
    signal = torch.zeros_like(x)
    prev_signal = 0
    
    # Convert scalar bounds to tensors
    if isinstance(upper_bound, (int, float)):
        upper_bound = torch.full_like(x, upper_bound)
    # Convert bounds to match time dimension if needed
    elif isinstance(upper_bound, torch.Tensor) and upper_bound.shape[0] == 1:
        upper_bound = upper_bound.expand_as(x)
        
    if isinstance(lower_bound, (int, float)):
        lower_bound = torch.full_like(x, lower_bound)
    elif isinstance(lower_bound, torch.Tensor) and lower_bound.shape[0] == 1:
        lower_bound = lower_bound.expand_as(x)
        
    if neutral_level is not None:
        if isinstance(neutral_level, (int, float)):
            neutral_level = torch.full_like(x, neutral_level)
        elif isinstance(neutral_level, torch.Tensor) and neutral_level.shape[0] == 1:
            neutral_level = neutral_level.expand_as(x)

    for i in range(x.shape[0]):
        if x[i] > upper_bound[i]:
            signal[i] = -1
        elif x[i] < lower_bound[i]:
            signal[i] = 1
        else:
            signal[i] = prev_signal

        # Exit positions based on neutral level if provided
        if neutral_level is not None:
            if signal[i] == -1 and x[i] <= neutral_level[i]:
                signal[i] = 0
            elif signal[i] == 1 and x[i] >= neutral_level[i]:
                signal[i] = 0
        # Otherwise exit only on opposite signals
        else:
            if signal[i] == -1 and x[i] < lower_bound[i]:
                signal[i] = 1  # Direct switch to long
            elif signal[i] == 1 and x[i] > upper_bound[i]:
                signal[i] = -1  # Direct switch to short

        prev_signal = signal[i]

    return signal

@conditional_lru_cache(maxsize=CACHE_SIZE)
def multivariate_barrier(x: torch.Tensor, 
                      upper_bound: torch.Tensor | float, 
                      lower_bound: torch.Tensor | float,
                      volatility: torch.Tensor | float = 1,
                      neutral_level: torch.Tensor | float | None = 0) -> torch.Tensor:
    """
    Performs univariate barrier calculations across multiple columns/series.
    
    Parameters
    ----------
    x : torch.Tensor
        2D input tensor with shape (time, features)
    upper_bound : torch.Tensor or float
        Upper barrier threshold(s)
    lower_bound : torch.Tensor or float
        Lower barrier threshold(s)
    volatility : torch.Tensor or float, default 1
        Volatility scaling for each series
    neutral_level : torch.Tensor or float or None, default 0
        Levels at which positions are exited
        
    Returns
    -------
    torch.Tensor
        2D tensor of -1, 0, +1 trading signals with shape (time, features)
    """
    # Convert inputs to tensors
    x = torch.as_tensor(x)
    
    # Handle single dimension input
    if x.dim() == 1:
        return univariate_barrier(x, upper_bound, lower_bound, neutral_level)
    
    # Create output tensor
    signal = torch.zeros_like(x)
    
    # Handle scalar bounds for all time steps
    if isinstance(upper_bound, (int, float)):
        upper_bound = torch.full_like(x, upper_bound)
    elif upper_bound.dim() == 1 and upper_bound.shape[0] == 1:
        upper_bound = upper_bound.expand_as(x)
    
    if isinstance(lower_bound, (int, float)):
        lower_bound = torch.full_like(x, lower_bound)
    elif lower_bound.dim() == 1 and lower_bound.shape[0] == 1:
        lower_bound = lower_bound.expand_as(x)
    
    if isinstance(volatility, (int, float)):
        volatility = torch.full((1, x.shape[1]), volatility)
    elif volatility.dim() == 1:
        volatility = volatility.unsqueeze(0)
    
    if neutral_level is not None:
        if isinstance(neutral_level, (int, float)):
            neutral_level = torch.full((1, x.shape[1]), neutral_level)
        elif neutral_level.dim() == 1:
            neutral_level = neutral_level.unsqueeze(0)
    
    # Process each column separately
    for j in range(x.shape[1]):
        # Extract bounds for this column, adjusted by volatility
        col_upper = upper_bound[:, j] * volatility[:, j]
        col_lower = lower_bound[:, j] * volatility[:, j]
        
        # Handle neutral level
        col_neutral = None
        if neutral_level is not None:
            col_neutral = neutral_level[:, j]
            
        # Apply univariate barrier to this column
        signal[:, j] = univariate_barrier(
            x[:, j], col_upper, col_lower, col_neutral
        )
        
    return signal


@conditional_lru_cache(maxsize=CACHE_SIZE)
def calculate_betas(asset_returns: torch.Tensor, 
                   benchmark_returns: torch.Tensor,
                   beta_measurement_period: int = None) -> torch.Tensor:
    """
    Calculate beta coefficients for assets relative to benchmark.
    
    Args:
        asset_returns: Tensor of asset returns with shape (time_steps, num_assets)
        benchmark_returns: Tensor of benchmark returns with shape (time_steps, 1)
        beta_measurement_period: Number of recent periods to use for beta calculation.
                                If None, use all available data.
    
    Returns:
        Tensor of beta coefficients with shape (num_assets,)
    """
    # Default beta is 1.0
    num_assets = asset_returns.shape[1]
    betas = torch.ones(num_assets)
    
    # Use all available data if beta_measurement_period is not specified
    if beta_measurement_period is None or beta_measurement_period > len(asset_returns):
        beta_measurement_period = len(asset_returns)
    
    # Only calculate betas if we have enough data
    if len(asset_returns) > 0:
        # Get the most recent data for beta calculation
        end_idx = len(asset_returns)
        start_idx = end_idx - beta_measurement_period
        
        # Calculate beta for each asset
        for i in range(num_assets):
            asset_i_returns = asset_returns[start_idx:end_idx, i]
            bench_i_returns = benchmark_returns[start_idx:end_idx, 0]
            
            # Calculate covariance and variance for beta
            cov = torch.mean((asset_i_returns - torch.mean(asset_i_returns)) * 
                            (bench_i_returns - torch.mean(bench_i_returns)))
            var = torch.var(bench_i_returns, unbiased=False)
            
            # Avoid division by zero
            if var > 0:
                betas[i] = cov / var

    return betas
