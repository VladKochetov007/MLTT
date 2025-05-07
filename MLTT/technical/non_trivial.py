import torch
import torch.nn.functional as F
from MLTT.utils import ensure_tensor, CACHE_SIZE
from MLTT.cache import conditional_lru_cache


@conditional_lru_cache(maxsize=CACHE_SIZE)
def price_channel(high: torch.Tensor,
                   low: torch.Tensor = None, 
                   lookback: int = 14, 
                   bounds: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate price channel with upper and lower bounds.
    
    The channel is calculated by finding the highest high and lowest low over the lookback period.
    The width of the channel is adjusted to be bounds% of the midpoint.
    
    Args:
        - `high` (torch.Tensor): High prices tensor of shape (n_samples,) or (n_samples, n_assets)
        - `low` (torch.Tensor): Low prices tensor of shape (n_samples,) or (n_samples, n_assets). If None, uses close prices from high tensor
        - `lookback` (int): Period for calculating channel bounds
        - `bounds` (float): Width of the channel as percentage (0.0-1.0) from midpoint
            e.g. 0.8 means ±40% from midpoint
            
    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of (upper_bound, lower_bound) tensors matching input shape
        First lookback-1 values will be NaN
    """
    high, low = ensure_tensor(high, low)
    if low is None:
        low = high
        
    # Handle both single asset and multi-asset cases
    original_shape = high.shape
    if high.dim() == 1:
        high = high.view(-1, 1)
        low = low.view(-1, 1)
    
    n_samples, n_assets = high.shape
    
    # Initialize output tensors with NaN
    resistance = torch.full_like(high, float('nan'))
    support = torch.full_like(low, float('nan'))
    
    # Calculate valid range values using max_pool1d for each asset
    for i in range(n_assets):
        high_pad = high[:, i].view(1, 1, -1)
        low_pad = low[:, i].view(1, 1, -1)
        
        valid_resistance = -F.max_pool1d(-high_pad, kernel_size=lookback, stride=1).view(-1)
        valid_support = F.max_pool1d(low_pad, kernel_size=lookback, stride=1).view(-1)
        
        # Assign valid values starting from lookback-1 index
        resistance[lookback-1:, i] = valid_resistance
        support[lookback-1:, i] = valid_support
    
    # Calculate midpoint and range
    midpoint = (resistance + support) / 2
    channel_range = resistance - support
    
    # Adjust channel width based on bounds parameter
    upper_bound = midpoint + (channel_range * bounds) / 2
    lower_bound = midpoint - (channel_range * bounds) / 2
    
    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        upper_bound = upper_bound.view(-1)
        lower_bound = lower_bound.view(-1)
    
    return upper_bound, lower_bound