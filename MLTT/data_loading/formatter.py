# TODO: SWITCH TO TIMESCALEDB
import os
import glob
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool


def format_files(source: str, interval: str, out_path: str, verbose: bool = False, overwrite: bool = False) -> None:
    """Formats downloaded Binance data files into a more convenient structure.

    Args:
        - `source` (str): Path to directory containing raw downloaded data
        - `interval` (str): Timeframe of the data ('1m', '1h', etc.)
        - `out_path` (str): Directory where to save formatted files
        - `verbose` (bool, optional): Whether to print progress messages. 
            Defaults to False.
        - `overwrite` (bool, optional): Whether to overwrite existing files. 
            Defaults to False.
    """
    data_folder = f'{source}/klines'
    output_folder = out_path
    os.makedirs(output_folder, exist_ok=True)

    for asset in tqdm(glob.glob(f'{data_folder}/*/{interval}')):
        asset_name = asset.split('/')[-2]
        output_file = os.path.join(output_folder, f'{asset_name}_{interval}.csv')

        if os.path.exists(output_file) and not overwrite:
            if verbose:
                print(f"Skipping: {output_file}")
            continue

        if verbose:
            print(f"Started: {output_file}")

        with open(output_file, 'w') as f_out:
            f_out.write("OpenTime,Open,High,Low,Close,Volume,CloseTime,QuoteAssetVolume,Trades,TakerBuyBaseAssetVolume,TakerBuyQuoteAssetVolume\n")

            for file in os.listdir(asset):
                file_path = os.path.join(asset, file)
                with open(file_path, 'r') as f_in:
                    next(f_in)  # skip header
                    for line in f_in:
                        line = line.rsplit(',', 1)[0] + '\n'
                        f_out.write(line)

        if verbose:
            print(f"Processed: {output_file}")

def process_file(asset: str) -> None:
    """Sorts a single CSV file by timestamp.

    Args:
        asset (str): Path to CSV file to process
    """
    df = pd.read_csv(asset)
    df = df.sort_values('OpenTime')
    df.to_csv(asset, index=False)

def sort_dates(source: str) -> None:
    """Sorts all CSV files in a directory by timestamp using parallel processing.

    Args:
        - `source` (str): Directory containing CSV files to sort
    """
    assets = glob.glob(f'{source}/*.csv')
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, assets), total=len(assets)))
