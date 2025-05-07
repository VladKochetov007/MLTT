from warnings import warn
from functools import wraps

import torch

from MLTT.utils import mad_normalize, ensure_tensor, CACHE_SIZE
from MLTT.cache import conditional_lru_cache


BPS_NORM = 1e4  # 1 basis point is 1/10_000 of the price

def handle_relative_price(func):
    """
    Decorator to handle relative price calculations
    
    Args:
        func: Function that returns a tensor of shape (batch_size,) or (batch_size, n_levels)
        
    Returns:
        Wrapped function that handles relative price calculations
    """
    @wraps(func)
    def wrapper(*args, mid_price=None, relative=False, **kwargs):
        result = func(*args, **kwargs)
        if relative and mid_price is not None:
            if mid_price.dim() == 2:
                mid_price = mid_price.squeeze(-1)
            if result.dim() > mid_price.dim():
                mid_price = mid_price.unsqueeze(-1)
            result = result / mid_price - 1
        return result
    return wrapper

@conditional_lru_cache(maxsize=CACHE_SIZE)
def weighted_average(values, weights):
    """
    Calculate simple weighted average along last dimension
    """
    values, weights = ensure_tensor(values, weights)
    return (values * weights).sum(dim=-1) / (weights.sum(dim=-1) + 1e-10)

@conditional_lru_cache(maxsize=CACHE_SIZE)
def convert_to_quote_volumes(prices, volumes):
    """
    Convert base volumes to quote volumes
    """
    return volumes * prices

@conditional_lru_cache(maxsize=CACHE_SIZE)
def create_volume_mask(volumes, target):
    """
    Create mask for volumes up to target
    """
    cum_vol = torch.cumsum(volumes, dim=-1)
    return cum_vol <= target

@conditional_lru_cache(maxsize=CACHE_SIZE)
def calculate_masked_vwap(prices, volumes, mask):
    """
    Calculate VWAP using masked prices and volumes
    """
    masked_prices = prices * mask
    masked_volumes = volumes * mask
    return weighted_average(masked_prices, masked_volumes)

@conditional_lru_cache(maxsize=CACHE_SIZE)
@handle_relative_price
def vamp(bid_prices, ask_prices, bid_volumes, ask_volumes, target_quote_volume):
    """
    Calculate Volume-Adjusted Mid Price (VAMP) based on target quote volume threshold(s)
    
    Args:
        bid_prices: tensor of shape (batch_size, depth) 
        ask_prices: tensor of shape (batch_size, depth)
        bid_volumes: bid volumes tensor of shape (batch_size, depth)
        ask_volumes: ask volumes tensor of shape (batch_size, depth)
        target_quote_volume: target volume threshold(s) in quote currency (e.g. USD)
            Can be a single number or list/array/tensor of thresholds
        
    Returns:
        VAMP tensor of shape:
            - (batch_size,) if target_quote_volume is a single number
            - (batch_size, n_levels) if target_quote_volume is a sequence

    Citation:
         Martin, Payton and Line Jr., William and Feng, Yuxin and Yang, Yunfan and Zheng, 
            Sharon and Qi, Susan and Zhu, Beiming, Mind the Gaps: Short-Term Crypto Price 
            Prediction (December 15, 2022). Available at SSRN:
            https://ssrn.com/abstract=4351947 or http://dx.doi.org/10.2139/ssrn.4351947 
    """
    bid_prices, ask_prices, bid_volumes, ask_volumes, target_quote_volume = ensure_tensor(
        bid_prices, ask_prices, bid_volumes, ask_volumes, target_quote_volume)
    if target_quote_volume.dim() == 0:
        target_quote_volume = target_quote_volume.unsqueeze(0)
    
    quote_bid_volumes = convert_to_quote_volumes(bid_prices, bid_volumes)
    quote_ask_volumes = convert_to_quote_volumes(ask_prices, ask_volumes)
    
    vamps = []
    for target_vol in target_quote_volume:
        bid_mask = create_volume_mask(quote_bid_volumes, target_vol)
        ask_mask = create_volume_mask(quote_ask_volumes, target_vol)
        
        vwap_bid = calculate_masked_vwap(bid_prices, quote_bid_volumes, bid_mask)
        vwap_ask = calculate_masked_vwap(ask_prices, quote_ask_volumes, ask_mask)
        
        vamp_ = (vwap_bid + vwap_ask) / 2
        vamps.append(vamp_)
    
    vamps = torch.stack(vamps, dim=-1)
    return vamps.squeeze() if len(target_quote_volume) == 1 else vamps

@conditional_lru_cache(maxsize=CACHE_SIZE)
def mid_price(bid_prices, ask_prices):
    """
    Calculate mid price between bid and ask prices
    """
    bid, ask = ensure_tensor(bid_prices, ask_prices)
    return (bid + ask) / 2

@conditional_lru_cache(maxsize=CACHE_SIZE)
def spread(bid_prices, ask_prices, mid_price=None, relative=False):
    bid, ask, mp = ensure_tensor(bid_prices, ask_prices, mid_price)
    if relative:
        return (ask - bid) / mp
    return ask - bid

@conditional_lru_cache(maxsize=CACHE_SIZE)
def imbalance_lob(bid_volumes, ask_volumes):
    """
    Calculate limit order book imbalance for each level.

    Imbalance = (bid_volumes) / (total_volumes) 
    In range [0, 1] where 0 means absolute balance and 1 means buying pressure.
    
    Args:
        bid_volumes: bid volumes tensor of shape (batch_size, depth)
        ask_volumes: ask volumes tensor of shape (batch_size, depth)
        
    Returns:
        Tensor of shape (batch_size, depth) where each element [i,j] represents
        limit order book imbalance calculated using volumes from levels 0 to j for batch element i
    """
    bid, ask = ensure_tensor(bid_volumes, ask_volumes)
    cumsum_bid = torch.cumsum(bid, dim=-1)
    cumsum_ask = torch.cumsum(ask, dim=-1)
    
    denominator = cumsum_bid + cumsum_ask
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    
    return cumsum_bid / denominator

@conditional_lru_cache(maxsize=CACHE_SIZE)
def weighted_mid_price(bid_prices, ask_prices, imbalance_lob):
    """
    Calculate weighted mid price from bid and ask prices and imbalance.

    Args:
        bid_prices: tensor of shape (batch_size, depth)
        ask_prices: tensor of shape (batch_size, depth)
        imbalance_lob: tensor of shape (batch_size, depth)

    Returns:
        Tensor of shape (batch_size, depth)
    """
    bid_prices, ask_prices, imbalance_lob = ensure_tensor(bid_prices, ask_prices, imbalance_lob)
    return bid_prices * (1 - imbalance_lob) + ask_prices * imbalance_lob

@conditional_lru_cache(maxsize=CACHE_SIZE)
def reduce_wap(values, weights):
    """
    Calculate weighted average price (common logic for VWAP and TWAP)
    
    For each position n in output, calculates weighted average of first n elements 
    from values using first n elements from weights as weights.
    """
    if values.dim() != weights.dim():
        warn(f"Values and weights have different dimensions. values: {values.shape}, weights: {values.shape}")
    
    values_cumsum = torch.cumsum(values * weights, dim=-1)
    weights_cumsum = torch.cumsum(weights, dim=-1)
    weights_cumsum = torch.where(weights_cumsum == 0, torch.ones_like(weights_cumsum), weights_cumsum)
    
    return values_cumsum / weights_cumsum

@conditional_lru_cache(maxsize=CACHE_SIZE)
def vwap_orderbook_levels(prices, volumes, mid_price=None, relative=False):
    prices, volumes, mid_price = ensure_tensor(prices, volumes, mid_price)
    vwap_ = reduce_wap(prices, volumes)
    if relative:
        vwap_ /= mid_price
        vwap_ -= 1
    return vwap_

@conditional_lru_cache(maxsize=CACHE_SIZE)
def all_features(bid_prices, 
                 ask_prices, 
                 bid_volumes, 
                 ask_volumes, 
                 vamp_levels=1e6,
                 normalize_vamp: bool = True,
                 mad_deviations: float = 15,
                 relative: bool = True):
    """
    Calculate all orderbook features
    
    Args:
        bid_prices: tensor of shape (batch_size, depth)
        ask_prices: tensor of shape (batch_size, depth) 
        bid_volumes: tensor of shape (batch_size, depth)
        ask_volumes: tensor of shape (batch_size, depth)
        vamp_levels: target quote volume for VAMP calculation (list[float] or float)
            Can be a single number or list/array/tensor of quote volumes. 
            Default is 1e6 (1M USD e.g.)
        normalize_vamp: whether to apply MAD normalization to VAMP (default: True)
        mad_deviations: number of MADs to use as threshold for outliers (default: 15)
        relative: whether to return relative values (default: True)
    Returns:
        List of tensors containing orderbook features:
        - mid price (shape: batch_size, depth)
        - imbalance (shape: batch_size, depth)
        - weighted mid price (shape: batch_size, depth)
        - spread (shape: batch_size, depth)
        - VAMP (shape: batch_size, n_levels)
        - VWAP (shape: batch_size, depth)
    """
    bid_prices, ask_prices, bid_volumes, ask_volumes = ensure_tensor(
        bid_prices, ask_prices, bid_volumes, ask_volumes)
    
    # Calculate mid price (shape: batch_size, depth)
    mp = mid_price(bid_prices, ask_prices)
    
    # Calculate spread in bps
    spread_ = spread(bid_prices, ask_prices, mp[:, 0].unsqueeze(-1), relative=relative)
    if relative:
        spread_ *= BPS_NORM
    
    # Calculate limit order book imbalance
    imb_lob = imbalance_lob(bid_volumes, ask_volumes)
    
    # Calculate weighted mid price
    wmp = weighted_mid_price(bid_prices, ask_prices, imb_lob)
    
    # Calculate VAMP for multiple levels
    vamp_ = vamp(bid_prices, ask_prices, bid_volumes, ask_volumes,
                mid_price=mp[:, 0], target_quote_volume=vamp_levels, relative=relative)
    if relative:
        vamp_ *= BPS_NORM
    
    # Normalize VAMP if requested
    if normalize_vamp:
        vamp_ = mad_normalize(vamp_, deviations=mad_deviations, axis=1)
    
    # Calculate VWAP for orderbook levels
    vwap = vwap_orderbook_levels(bid_prices, bid_volumes, mid_price=mp[:, 0].unsqueeze(-1), relative=relative)
    if relative:
        vwap *= BPS_NORM

    # Return list of features
    return [
        mp,          # [batch_size, depth]
        imb_lob,     # [batch_size, depth]
        wmp,         # [batch_size, depth]
        spread_,     # [batch_size, depth] 
        vamp_,       # [batch_size, n_levels]
        vwap,        # [batch_size, depth]
    ]
