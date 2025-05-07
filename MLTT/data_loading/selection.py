# TODO: SWITCH TO TIMESCALEDB
import numpy as np


def extract_time(data: np.ndarray, symbols: list[str], position: int) -> dict[str, int]:
    """Extracts timestamps from specific position in data array.

    Args:
        data (np.ndarray): Array containing market data
        symbols (list[str]): List of trading pair symbols
        position (int): Index position to extract timestamp from (0 for first, -1 for last)

    Returns:
        dict[str, int]: Dictionary mapping symbols to their timestamps
    """
    result = {}
    for i, start in enumerate(data[:, position, 0]):
        result[symbols[i]] = int(start)
    return result


def get_timestamps(klines: np.ndarray, symbols: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Gets first and last timestamps for each trading pair.

    Args:
        klines (np.ndarray): Array containing market data
        symbols (list[str]): List of trading pair symbols

    Returns:
        tuple[dict[str, int], dict[str, int]]: Two dictionaries containing:
            - Last timestamps for each symbol
            - First timestamps for each symbol
    """
    if not isinstance(klines, np.ndarray):
        raise ValueError(
            f"klines must be np.ndarray, but got {type(klines)}."
            "You should use `equals=True` when loading data from files")

    if klines.ndim != 3:
        raise ValueError(f"klines must be 3-dimensional array, but got {klines.ndim}-dimensional")

    last_infos = extract_time(klines, symbols, -1)
    first_infos = extract_time(klines, symbols, 0)

    return last_infos, first_infos


def find_incorrect_symbols(last_infos: dict[str, int],
                         first_infos: dict[str, int]) -> list[str]:
    """Identifies symbols with mismatched timestamps.

    Args:
        last_infos (dict[str, int]): Dictionary of last timestamps
        first_infos (dict[str, int]): Dictionary of first timestamps

    Returns:
        list[str]: List of symbols that have inconsistent timestamps
    """
    starts = []
    ends = []
    incorrect_symbols = []

    for symbol_, timestamp_line in last_infos.items():
        ends.append(int(timestamp_line))
        starts.append(int(first_infos[symbol_]))

    max_end_time = max(ends)
    max_start_time = max(starts)

    for symbol_, timestamp_line in last_infos.items():
        start = int(first_infos[symbol_])
        if timestamp_line != max_end_time or start != max_start_time:
            incorrect_symbols.append(symbol_)

    return incorrect_symbols


def filter_symbols_by_correct_timestamp(klines: np.ndarray,
                                      symbols: list[str]) -> list[str]:
    """Filters out symbols with inconsistent timestamps.

    Args:
        klines (np.ndarray): Array containing market data
        symbols (list[str]): List of trading pair symbols

    Returns:
        list[str]: List of symbols with consistent timestamps
    """
    last_infos, first_infos = get_timestamps(klines, symbols)
    incorrect_symbols = find_incorrect_symbols(last_infos, first_infos)
    correct_symbols = set(symbols) - set(incorrect_symbols)

    return list(correct_symbols)


def select_symbols_by_observations(min_data_length: int,
                                 klines: np.ndarray,
                                 symbols: list[str]) -> list[str]:
    """Selects symbols that have enough historical data.

    Args:
        min_data_length (int): Minimum required number of observations
        klines (np.ndarray): Array containing market data
        symbols (list[str]): List of trading pair symbols

    Returns:
        list[str]: List of symbols that meet the minimum data requirement
    """
    correct_symbols = []

    for symbol, kline in zip(symbols, klines):
        if kline.shape[0] >= min_data_length:
            correct_symbols.append(symbol)

    return correct_symbols


def max_information_length(depth_curve: list[int],
                         information_criterion=None) -> int:
    """Determines optimal data length based on information criterion.

    Args:
        depth_curve (list[int]): List of available data lengths
        information_criterion (callable, optional): Custom function to evaluate information content.
            If None, uses default criterion: depth_curve[i]*(n_pairs-i)

    Returns:
        int: Optimal data length that maximizes information content
    """
    n_pairs = len(depth_curve)

    if information_criterion is None:
        information_criterion = lambda i: depth_curve[i]*(n_pairs-i)

    informs = [information_criterion(i) for i in range(n_pairs)]

    return depth_curve[np.argmax(informs)]

def frames_lengths(klines: list[np.ndarray]) -> list[int]:
    """Gets lengths of all data frames.

    Args:
        klines (list[np.ndarray]): List of arrays containing market data

    Returns:
        list[int]: List of lengths for each array
    """
    return [klines[i].shape[0] for i in range(len(klines))]


def filter_symbols_by_maximum_information(klines: list[np.ndarray],
                                        symbols: list[str],
                                        information_criterion=None) -> list[str]:
    """Filters symbols to maximize information content while maintaining data consistency.

    Args:
        klines (list[np.ndarray]): List of arrays containing market data
        symbols (list[str]): List of trading pair symbols
        information_criterion (callable, optional): Custom function to evaluate information content

    Returns:
        list[str]: Filtered list of symbols that optimize information content
    """
    lengths = frames_lengths(klines)

    min_data_length = max_information_length(sorted(lengths), information_criterion)
    filtered_symbols = select_symbols_by_observations(min_data_length, klines, symbols)

    return filtered_symbols


def prune_klines(klines: list[np.ndarray]) -> np.ndarray:
    """Combines multiple kline arrays into single array with consistent length.

    Args:
        klines (list[np.ndarray]): List of arrays containing market data

    Returns:
        np.ndarray: Combined array with shape (n_symbols, min_length, n_features)

    Note:
        This operation requires approximately 2x the memory of input data
    """
    min_len = min(kline.shape[0] for kline in klines)
    dst = np.empty((len(klines), min_len, klines[0].shape[1]), dtype=klines[0].dtype)
    for i, kline in enumerate(klines):
        dst[i, -kline.shape[0]:] = kline[-min_len:]
        klines[i] = None
    return dst


def remove_symbols(klines: np.ndarray, symbols: list[str], correct_symbols: list[str]):
    """Removes all symbols from `klines` that are not in `correct_symbols`.

    Args:
        `klines` (np.ndarray): Data array with k-lines information. Shape: `(n_symbols, n_observations, n_features)`
        `symbols` (list[str]): Symbols or identifiers for each data row. Length: `n_symbols`
        `correct_symbols` (list[str]): List of symbols that should be present in `klines` after removal. Length: `<=n_symbols`
    """
    for i in range(len(symbols)-1, -1, -1):
        if symbols[i] not in correct_symbols:
            del klines[i]
            symbols.pop(i)

def make_klines_array(klines: list[np.ndarray],
                      symbols: list[str],
                      information_criterion=None) -> tuple[np.ndarray, list[str]]:
    """Creates unified array from multiple kline arrays, ensuring data consistency.

    Args:
        klines (list[np.ndarray]): List of arrays containing market data
        symbols (list[str]): List of trading pair symbols
        information_criterion (callable, optional): Custom function to evaluate information content

    Returns:
        tuple[np.ndarray, list[str]]: Tuple containing:
            - Combined array with consistent timestamps and optimal information content
            - List of symbols included in the final array
    """
    corrected_symbols = filter_symbols_by_maximum_information(klines, symbols, information_criterion)
    remove_symbols(klines, symbols, corrected_symbols)
    symbols = corrected_symbols
    klines = prune_klines(klines)
    corrected_symbols = filter_symbols_by_correct_timestamp(klines, symbols)
    indices = [symbols.index(symbol) for symbol in corrected_symbols]
    return klines[indices], corrected_symbols
