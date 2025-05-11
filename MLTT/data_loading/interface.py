# TODO: SWITCH TO TIMESCALEDB
from MLTT.data_loading.dumper import dump_data
from MLTT.data_loading.formatter import format_files, sort_dates

import os
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from MLTT.utils import EPSILON

def load_and_format(tickers: list[str] | None,
                    interval: str,
                    ask: bool = True,
                    asset_class='spot',
                    path='../../data') -> None:
    """Loads financial data from Binance exchange into local storage and 
        formats it for further analysis.

    Args:
        - `tickers` (list[str]): List of ticker symbols to download data for. 
            Defaults to all available USDT tickers.
        - `interval` (str): Time interval of the data, e.g. '1m', '1h', '1d'.
        - `ask` (bool): Whether to ask for confirmation before downloading 
            new data. Defaults to True.
        - `asset_class` (str): spot, um (for USD-M futures) or 
            cm (for coin-M futures). Defaults to 'spot'.
        - `path` (str): Path to the directory where to dump the data. 
            Defaults to '../../data'.
    """
    if not ask or input("Are you actually want to download new data? (y/n)").lower() == 'y':
        dump_data(tickers,
                  interval,
                  path,
                  asset_class=asset_class,
                  data_type='klines')
        only_format(interval, path)

def only_format(interval, path='../../data', overwrite=False, verbose=False):
    format_files(
        os.path.join(path, 'spot', 'monthly'),
        interval, os.path.join(path, interval), 
        verbose=verbose, overwrite=overwrite
    )
    sort_dates(os.path.join(path, interval))

def save_symbols(symbols, file_path):
    """
    Saves a list of strings (symbols) to a file (file_path).
    Each string is written on a new line in the file.

    Args:
        - `symbols` (list): List of strings to save.
        - `file_path` (str): Path to the file for writing.
    """
    try:
        with open(file_path, 'w') as file:
            for symbol in sorted(symbols):
                file.write(symbol + '\n')
        print(f"Symbols successfully saved to file: {file_path}")
    except Exception as e:
        print(f"Error saving symbols to file: {e}")

def load_symbols(file_path):
    """
    Loads a list of strings from a file (file_path).
    Each line in the file is treated as a separate element in the list.

    Args:
        - `file_path` (str): Path to the file for reading.

    Returns:
        list: List of strings loaded from the file.
    """
    try:
        with open(file_path, 'r') as file:
            symbols = [line.strip() for line in file.readlines()]
        print(f"Successfully loaded {len(symbols)} symbols from file: {file_path}")
        return sorted(symbols)
    except Exception as e:
        print(f"Error loading symbols from file: {e}")
        return []

async def fetch_ohlcv(exchange, sym, timeframe):
    try:
        ohlcv = await exchange.fetch_ohlcv(sym, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return sym, df
    except Exception as e:
        print(f"Error fetching data for {sym}: {e}")
        return sym, None

async def download_ohlcv(exchange_name: str, symbol: str | list[str], timeframe: str):
    """
    Asynchronously downloads OHLCV data from the specified exchange for one or multiple pairs.

    Args:
        - `exchange_name` (str): Name of the exchange (e.g. 'binance', 'kraken').
        - `symbol` (str | list[str]): Trading pair or list of pairs (e.g. 'BTC/USDT' or ['BTC/USDT', 'ETH/USDT']).
        - `timeframe` (str): Time interval (e.g. '1m', '1h', '1d').

    Returns:
        dict: Dictionary containing OHLCV data for each pair.
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({'enableRateLimit': True})

    try:
        if isinstance(symbol, str):
            symbol = [symbol]

        tasks = [fetch_ohlcv(exchange, sym, timeframe) for sym in symbol]
        results = await asyncio.gather(*tasks)

        data = {sym: df for sym, df in results if df is not None}

    finally:
        await exchange.close()

    return data

def ccxt_ohlcv(exchange_name: str, symbol: str | list[str], timeframe: str):
    """
    Downloads OHLCV data from the specified exchange using CCXT.

    Args:
        - `exchange_name` (str): Name of the exchange (e.g. 'binance', 'kraken').
        - `symbol` (str | list[str]): Trading pair or list of pairs.
        - `timeframe` (str): Time interval.

    Returns:
        dict: Dictionary containing OHLCV data for each pair.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(download_ohlcv(exchange_name, symbol, timeframe))


def _convert_timestamp(ts: int | float) -> pd.Timestamp:
    """Convert timestamp with automatic unit detection"""
    try:
        return pd.to_datetime(ts, unit='ms')
    except pd.errors.OutOfBoundsDatetime:
        return pd.to_datetime(ts, unit='ns')

def check_linearity(timestamps: np.ndarray, tolerance: float = 0.05) -> bool:
    """Check if timestamps have constant intervals within given tolerance."""
    if len(timestamps) < 2:
        return True

    deltas = np.diff(timestamps)
    
    max_step = np.max(deltas)
    min_step = np.min(deltas)
    is_linear = max_step - min_step <= 2*tolerance*np.median(deltas)
    return bool(is_linear)

def check_klines_timeline(
        klines: np.ndarray, 
        deep: bool = True, 
        verbose: bool = True,
        linear: bool = True,
        tolerance: float = 0.05
    ) -> bool:
    if deep:
        timestamps = klines[:, :, 0]
        reference = timestamps[0]
        timeline_passed = all(np.array_equal(ts, reference) for ts in timestamps)
        passed = timeline_passed

        linear_passed = True
        if timeline_passed and linear:
            linear_passed = check_linearity(reference, tolerance)
            passed = passed and linear_passed
    else:
        corrected_starts = klines[:, 0, 0]
        corrected_ends = klines[:, -1, 0]
        passed = len(np.unique(corrected_starts)) == len(np.unique(corrected_ends)) == 1

    if verbose:
        if deep:
            timeline_status = "passed ✅" if timeline_passed else "failed ❌"
            print(f"Same timeline check {timeline_status}")

            if linear and timeline_passed:
                lin_status = "passed ✅" if linear_passed else "failed ❌"
                print(f"Linear timeline check {lin_status}")
        else:
            passed_str = "passed ✅" if passed else "failed ❌"
            print(f"Same timeline check {passed_str}")

        if passed:
            try:
                start_ts = klines[0, 0, 0]
                end_ts = klines[0, -1, 0]
                start_date = _convert_timestamp(start_ts)
                end_date = _convert_timestamp(end_ts)

                print(f"\nStart date: {start_date.strftime('%d %b %Y %H:%M:%S.%f')[:-3]}")
                print(f"End date: {end_date.strftime('%d %b %Y %H:%M:%S.%f')[:-3]}")
            except Exception as e:
                print(f"\nError converting timestamps: {e}")
                print(f"Raw start timestamp: {start_ts}")
                print(f"Raw end timestamp: {end_ts}")

    return passed
