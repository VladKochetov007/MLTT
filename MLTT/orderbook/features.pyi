import torch
import numpy as np
from typing import Callable

def _ensure_tensor(*args: (torch.Tensor| None)) -> list[torch.Tensor]: ...

def mid_price(bid_prices: torch.Tensor,
             ask_prices: torch.Tensor) -> torch.Tensor: ...

def spread(bid_prices: torch.Tensor,
          ask_prices: torch.Tensor,
          mid_price: torch.Tensor | None = None,
          relative: bool = False) -> torch.Tensor: ...

def imbalance_lob(bid_volumes: torch.Tensor,
                     ask_volumes: torch.Tensor) -> torch.Tensor: ...

def weighted_mid_price(bid_prices: torch.Tensor,
                      ask_prices: torch.Tensor,
                      imbalance_lob: torch.Tensor) -> torch.Tensor: ...

def weighted_average(values: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor: ...

def vamp(bid_prices: torch.Tensor,
         ask_prices: torch.Tensor,
         bid_volumes: torch.Tensor,
         ask_volumes: torch.Tensor,
         target_quote_volume: float | list[float] | torch.Tensor,
         mid_price: torch.Tensor | None = None,
         relative: bool = False) -> torch.Tensor: ...

def reduce_wap(values: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor: ...

def vwap_orderbook_levels(prices: torch.Tensor,
                         volumes: torch.Tensor,
                         mid_price: torch.Tensor | None = None,
                         relative: bool = False) -> torch.Tensor: ...

def _process_weighted_price(func: Callable,
                          price: torch.Tensor,
                          volume: torch.Tensor,
                          window: int,
                          mp: torch.Tensor) -> torch.Tensor: ...

def all_features(bid_prices: torch.Tensor,
                ask_prices: torch.Tensor,
                bid_volumes: torch.Tensor,
                ask_volumes: torch.Tensor,
                vamp_levels: float | list[float] = 1e6,
                quote_asset_volume: torch.Tensor | None = None,
                normalize_vamp: bool = True,
                mad_deviations: float = 15
                ) -> list[torch.Tensor]: ...

def handle_relative_price(func: Callable) -> Callable: ...

def convert_to_quote_volumes(prices: torch.Tensor, volumes: torch.Tensor) -> torch.Tensor: ...

def create_volume_mask(volumes: torch.Tensor, target: float) -> torch.Tensor: ...

def calculate_masked_vwap(prices: torch.Tensor, 
                         volumes: torch.Tensor, 
                         mask: torch.Tensor) -> torch.Tensor: ...
