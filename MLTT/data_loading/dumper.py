# TODO: SWITCH TO TIMESCALEDB
from datetime import datetime
import os
import json
import csv
from warnings import warn

import numpy as np
from binance_historical_data import BinanceDataDumper
from tqdm.auto import tqdm


BEGIN_DATE = datetime(2016, 1, 1).date()
END_DATE   = datetime.today().date()
CACHE_FILE = 'line_counts_cache.json'


def dump_data(tickers: list[str],
              interval: str,
              path: str,
              asset_class="spot",
              data_type="klines") -> None:
    """
    Dumps historical market data from Binance exchange and saves it locally.

    Args:
        - `tickers` (list[str]): List of trading pairs to download 
            (e.g. ['BTCUSDT', 'ETHUSDT'])
        - `interval` (str): Timeframe of the data ('1s', '1m', '1h', '1d', etc.)
        - `path` (str): Directory path where to save the downloaded data
        - `asset_class` (str, optional): Type of market data:
            - 'spot' for spot markets
            - 'um' for USD-M futures
            - 'cm' for COIN-M futures
            Defaults to "spot".
        - `data_type` (str, optional): Type of data to download:
            - 'klines' for candlestick data
            - 'aggTrades' for aggregated trades
            - 'trades' for raw trades
            Defaults to "klines".
    """
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=path,
        asset_class=asset_class,
        data_type=data_type,
        data_frequency=interval,
    )

    data_dumper.dump_data(
        tickers=tickers,
        date_start=BEGIN_DATE,
        date_end=END_DATE,
        is_to_update_existing=False,
    )

def count_lines(file_path: str) -> int:
    """
    Counts number of lines in a file.

    Args:
        - `file_path` (str): Path to the file

    Returns:
        - `int`: Number of lines in the file
    """
    counter = 0

    with open(file_path, 'r') as f:
        for _ in f.readlines():
            counter += 1
    return counter

def min_lines(files: list[str], update_cache=True) -> tuple[int, dict[str, int]]:
    """
    Finds minimum number of lines across multiple files.
    Uses caching to avoid recounting files that haven't changed.

    Args:
        - `files` (list[str]): List of file paths to check
        - `update_cache` (bool, optional): Whether to update the cache. 
            Defaults to True.

    Returns:
        - `tuple[int, dict[str, int]]`: Tuple containing:
            - Minimum number of lines across all files
            - Dictionary mapping file paths to their line counts
    """
    if update_cache:
        cache = {}
    else:
        cache = _load_cache(CACHE_FILE)
    counts = _get_counts(files, cache)
    _save_cache(CACHE_FILE, counts)

    min_num = _get_min_count(counts)
    return min_num, counts


def _load_cache(cache_file: str) -> dict:
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _get_counts(files: list[str], cache: dict) -> dict:
    print("Counting lines in files...")
    counts = {file: cache.get(file, 0) for file in files}
    for file in tqdm(files):
        if file not in cache:
            counts[file] = count_lines(file)
            cache[file] = counts[file]
    return counts


def _save_cache(cache_file: str, cache: dict) -> None:
    with open(cache_file, 'w') as f:
        json.dump(cache, f)


def _get_min_count(counts: dict) -> int:
    return min(counts.values())


def load_klines_from_file(file_path: str, load_cols: list[int], skiprows: int) -> np.ndarray:
    """
    Loads specific columns from a CSV file containing market data.

    Args:
        - `file_path` (str): Path to the CSV file
        - `load_cols` (list[int]): List of column indices to load
        - `skiprows` (int): Number of rows to skip from the beginning

    Returns:
        - `np.ndarray`: Array containing the requested columns
    """
    data = np.loadtxt(
        file_path,
        delimiter=',',
        dtype=np.float64,
        usecols=load_cols,
        skiprows=skiprows,
    )
    return data


def _load_min_csv_files(files: list[str], 
                        load_cols: list[int],
                        update_cache=True,
                        equals=True) -> np.ndarray | list[np.ndarray]:
    """
    Loads multiple CSV files and ensures they have consistent dimensions.

    Args:
        - `files` (list[str]): List of CSV file paths to load
        - `load_cols` (list[int]): Column indices to load from each file
        - `update_cache` (bool, optional): Whether to update line count cache. 
            Defaults to True.
        - `equals` (bool, optional): If True, ensures all loaded arrays have same length. 
            Defaults to True.

    Returns:
        - `np.ndarray | list[np.ndarray]`: Either:
            - Single numpy array with shape (n_files, min_length, n_columns) if equals=True
            - List of arrays with potentially different lengths if equals=False
    """
    min_len, lines = min_lines(files, update_cache=update_cache)

    n_columns = None
    for file_path in files:
        with open(file_path, 'r') as f:
            header = next(csv.reader(f))
            if n_columns is None:
                n_columns = len(header)
            elif len(header) != n_columns:
                raise ValueError(f"Different number of columns in file {file_path}")

    print("Loading files...")

    shape = (len(files), min_len-1, len(load_cols))

    if equals:
        values = np.empty(shape, dtype=np.float64)
    else:
        values = [[] for _ in range(len(files))]

    for i, file_path in enumerate(tqdm(files)):
        skiprows = lines[file_path] - min_len + 1 if equals else 1
        data = load_klines_from_file(file_path, load_cols, skiprows)
        values[i] = data

    return values

def load_csv_files_from_directory(directory: str,
                                  pairs_to_load: list[str] | None = None,
                                  load_cols: list[int] | None = None,
                                  update_cache: bool = True,
                                  equals: bool = True,
                                  close_only: bool | None = None) -> tuple[np.ndarray | list[np.ndarray], list[str]]:
    """
    Loads market data from CSV files in a directory.

    Args:
        - `directory` (str): Directory containing the CSV files
        - `pairs_to_load` (list[str] | None, optional): Specific trading pairs to load. 
            If None, loads all. 
        - `load_cols` (list[int] | None, optional): Specific columns to load from each file.
            Common columns are:
            - 0: OpenTime
            - 1: Open price
            - 2: High price
            - 3: Low price
            - 4: Close price
            - 5: Volume
            - 6: Close time
            - 7: Quote volume
            - 8: Number of trades
            - 9: Taker buy base volume
            - 10: Taker buy quote volume
            If None, loads all columns.
        - `update_cache` (bool, optional): Whether to update line count cache. 
            Defaults to True.
        - `equals` (bool, optional): If True, ensures all loaded data has same length. 
            Defaults to True.

    Returns:
        - `tuple[np.ndarray | list[np.ndarray], list[str]]`: Tuple containing:
            - Array or list of arrays with loaded data
            - List of successfully loaded trading pairs
    """
    if close_only:
        load_cols = [4]
        warn("close_only is deprecated. Use load_cols instead.")

    pairs_to_load = sorted(pairs_to_load) if pairs_to_load else None
    files = sorted(os.listdir(directory))

    file_names = []
    loaded = []
    try:
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            pair = filename.split('_')[0]
            if pairs_to_load and pair not in pairs_to_load:
                continue
            file_names.append(filename)
            loaded.append(pair)

        paths = [os.path.join(directory, filename) for filename in file_names]
        return _load_min_csv_files(paths, load_cols, update_cache, equals), loaded
    except KeyboardInterrupt:
        # Return partial results if keyboard interrupt occurs
        if file_names:  # If we have any loaded files
            paths = [os.path.join(directory, filename) for filename in file_names]
            return _load_min_csv_files(paths, load_cols, update_cache, equals), loaded
        raise  # Re-raise if no files were loaded
