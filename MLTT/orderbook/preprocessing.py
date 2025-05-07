# TODO: SWITCH TO TIMESCALEDB
import torch
import polars as pl
import pandas as pd
import pickle

from dataclasses import dataclass

@dataclass
class Orderbook:
    bid_prices: torch.Tensor
    bid_sizes: torch.Tensor
    ask_prices: torch.Tensor
    ask_sizes: torch.Tensor
    received_time: torch.Tensor

    @classmethod
    def from_parquet(cls, path, levels=10):
        orderbook = pl.read_parquet(path)
        return orderbook_to_vectors(orderbook, levels)

    def to_pickle(self, path, verbose=True):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if verbose:
            print(f"Orderbook saved successfully to {path}")
    
    @classmethod
    def from_pickle(cls, path, verbose=True):
        with open(path, "rb") as f:
            orderbook = pickle.load(f)
        if verbose:
            print(f"Orderbook loaded successfully from {path}")
        return orderbook


def parse_tensor_column(column, strip=None):
    """
    Parse a Polars Series into tensors, truncating to `strip` length if provided.
    Fills missing values with appropriate defaults.

    Args:
        column (pl.Series): Polars column with tensor-like data in string format.
        strip (int, optional): Maximum elements in each tensor. Defaults to None.

    Returns:yi
        torch.Tensor: Stacked tensor of all rows.
    """
    arrays = column.to_numpy().astype(str)  # Convert to numpy strings
    tensor_data = []
    
    for arr in arrays:
        values = arr.split(",")[:strip]
        # Convert values to float, replacing invalid with 0
        float_values = []
        for x in values:
            try:
                float_values.append(float(x))
            except ValueError:
                float_values.append(0.0)
                
        # Pad with zeros if not enough values
        if strip and len(float_values) < strip:
            if "price" in column.name:
                # For prices, repeat last valid price
                last_valid = float_values[-1] if float_values else 0.0
                float_values.extend([last_valid] * (strip - len(float_values)))
            else:
                # For sizes, pad with zeros
                float_values.extend([0.0] * (strip - len(float_values)))
                
        tensor_data.append(torch.tensor(float_values, dtype=torch.float32))
        
    return torch.stack(tensor_data)

def orderbook_to_vectors(orderbook: pl.DataFrame, levels: int = 10):
    """
    Convert order book data into a dictionary of tensors.
    Handles missing or invalid data by filling with appropriate values.

    Args:
        orderbook (pl.DataFrame): Polars DataFrame containing:
            - 'bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes'.
        levels (int): Maximum number of price/size levels.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing parsed tensors.
    """
    # Convert timestamps to nanoseconds
    received_time_ns = pd.to_datetime(orderbook["received_time"]).astype('int64')

    return Orderbook(
        bid_prices=parse_tensor_column(orderbook["bid_prices"], strip=levels),
        bid_sizes=parse_tensor_column(orderbook["bid_sizes"], strip=levels),
        ask_prices=parse_tensor_column(orderbook["ask_prices"], strip=levels),
        ask_sizes=parse_tensor_column(orderbook["ask_sizes"], strip=levels),
        received_time=torch.as_tensor(received_time_ns.to_numpy(), dtype=torch.int64)
    )
