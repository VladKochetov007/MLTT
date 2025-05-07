import torch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

from warnings import warn

from MLTT.report.metrics import ALL_METRICS, Metric
from MLTT.allocation.backtesting import BTResult
from IPython.display import Markdown


class Report:
    def __init__(self, values: dict[str, float]):
        """
        Initializes the Report object with a dictionary of metric values.

        Args:
            - `values` (dict[str, float]): A dictionary where keys are metric names (str) and values are metric values (float)
        """
        self.values = values

    def as_dict(self) -> dict[str, float]:
        """
        Returns the metric values as a dictionary.

        Returns:
            dict[str, float]: A dictionary where keys are metric names (str) and values are metric values (float)
        """
        return self.values.copy()

    def as_series(self) -> pd.Series:
        """
        Returns the metric values as a pandas Series.

        Returns:
            pd.Series: A pandas Series where index is metric names (str) and values are metric values (float)
        """
        return pd.Series(self.values)
    
    def as_markdown(self) -> str:
        """
        Returns the metric values as a markdown table.
        
        Returns:
            Markdown: IPython.display.Markdown object containing the formatted table
        """
        return generate_markdown_table(self.as_series())

class StrategyReporter:
    def __init__(self, metrics: list[Metric]):
        """
        Initializes the StrategyReporter object with ordered metrics.

        Args:
            - `metrics` (list[Metric]): A list of metric instances
        """
        self.metrics = metrics

    def compose_report(self, backtest_result: BTResult, mode: str = 'visual') -> Report:
        """
        Composes a dictionary of metric values for the given backtest result.

        Args:
            - `backtest_result` (BTResult): A backtest result object
            - `mode` (str): mode of report composition:
                - `visual` - human readable report
                - `numeric` - report with raw values for each metric

        Returns:
            Report: A Report object containing the metric values
        """
        report = {}
        for metric in self.metrics:
            value = metric.calculate(backtest_result)
            if mode == 'visual':
                value = metric.format_value(value)
            report[metric.name] = value
        return Report(report)

    def update_metric_params(self, metric_name: str, **new_params):
        """
        Updates the parameters of a specific metric by name.

        Args:
            - `metric_name` (str): The name of the metric to update
            - `new_params` (dict): Dictionary of new parameters to update
        """
        for metric in self.metrics:
            if metric.name == metric_name:
                metric.update_params(**new_params)
                return
        raise ValueError(f"Metric '{metric_name}' not found in the report.")

    def update_all_metrics_params(self, **new_params):
        """
        Updates parameters for all metrics at once.

        Args:
            - `new_params` (dict): Dictionary of new parameters to update in all metrics
        """
        for metric in self.metrics:
            metric.update_params(**new_params)


def generate_markdown_table(series):
    """
    Generate a markdown-formatted table from a pandas Series.

    Args:
        - `series` (pd.Series): A pandas Series where the index contains metric names and the values
                            are the corresponding metric values

    Returns:
        Markdown: IPython.display.Markdown object containing the formatted table
    """
    # Generate the markdown table header
    markdown_table = "| Metric | Value |\n|--------|-------|\n"

    # Iterate through all items in the series and add them to the table
    for metric, value in series.items():
        if value is not None:
            markdown_table += f"| {metric:<36} | {value:<9} |\n"

    return Markdown(markdown_table)
    
def plot_graphs(backtest_result: BTResult, 
                start_index: int,
                resample_period: str = "ME",
                figsize: tuple[int, int] = (20, 15),
                height_ratios: list[int] = [3, 1, 1],
                timeframe: timedelta | None = None):
    """
    Create a comprehensive visualization of strategy performance metrics.
    
    Args:
        - `backtest_result` (BTResult): The backtest result object containing performance data
        - `start_index` (int): Starting index for the plots
        - `resample_period` (str): Resampling frequency for monthly data ("ME" for month end)
        - `figsize` (tuple[int, int]): Figure size as (width, height)
        - `height_ratios` (list[int]): Height ratios for the subplots
        - `timeframe` (timedelta | None): Time interval between data points
    """
    sum_equity = backtest_result.log_equity
    running_max = torch.cummax(sum_equity, dim=0).values
    drawdown = running_max - sum_equity

    # Create date index ending today with specified timeframe
    end_date = datetime.now()
    if timeframe is None:
        timeframe = timedelta(days=len(backtest_result.net_change))
    
    equity_series = pd.Series(
        backtest_result.net_change.numpy(),
        index=pd.date_range(end=end_date, periods=len(backtest_result.net_change), freq=timeframe)
    )
    month_profits = equity_series.resample(resample_period).sum()

    plt.style.use('dark_background')
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=height_ratios)

    plot_equity_and_running_max(gs, sum_equity, running_max, start_index)
    plot_drawdown(gs, drawdown, start_index)
    plot_monthly_profits(gs, month_profits, start_index)

    plt.tight_layout()
    plt.show()

def plot_equity_and_running_max(gs, sum_equity, running_max, start_index):
    """
    Plot the logarithmic equity curve and running maximum.
    
    Args:
        - `gs` (matplotlib.gridspec.GridSpec): Grid specification for subplot arrangement
        - `sum_equity` (torch.Tensor): Logarithmic equity curve data
        - `running_max` (torch.Tensor): Running maximum values
        - `start_index` (int): Starting index for the plot
    """
    plt.subplot(gs[0])
    plt.plot(sum_equity[start_index:], label='Log Equity')
    plt.plot(running_max[start_index:], label='Running Max', linestyle='--')
    plt.title('Equity and Running Max Over Time')
    plt.legend()
    plt.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels

def plot_drawdown(gs, drawdown, start_index):
    """
    Plot the drawdown over time.
    
    Args:
        - `gs` (matplotlib.gridspec.GridSpec): Grid specification for subplot arrangement
        - `drawdown` (torch.Tensor): Drawdown values
        - `start_index` (int): Starting index for the plot
    """
    plt.subplot(gs[1])
    plt.plot(drawdown[start_index:], label='Drawdown', color='red')
    plt.title('Drawdown Over Time')
    plt.legend()
    plt.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels

def plot_monthly_profits(gs, month_profits, start_index):
    """
    Plot monthly profits as a bar chart.
    
    Args:
        - `gs` (matplotlib.gridspec.GridSpec): Grid specification for subplot arrangement
        - `month_profits` (pd.Series): Monthly profit data
        - `start_index` (int): Starting index for the plot
    """
    colors = ['#00ff0069' if profit >= 0 else '#ff000044' for profit in month_profits]
    months = month_profits.index.strftime('%b %Y').tolist()
    plt.subplot(gs[2])
    plt.bar(months[start_index:], month_profits[start_index:], color=colors[start_index:])
    plt.title('Monthly Profits')
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)

def write_report_to_file(filename: str, backtest_result: BTResult, resample_period: str = "ME", timeframe: timedelta | None = None):
    """
    Generate a comprehensive strategy report and write it to a file.
    
    Args:
        - `filename` (str): Path to the output file
        - `backtest_result` (BTResult): Backtest result data to analyze
        - `resample_period` (str): Resampling frequency for monthly data ("ME" for month end)
        - `timeframe` (timedelta | None): Time interval between data points
    """
    with open(filename, mode="w") as file:
        write_general_summary(file, backtest_result)
        write_monthly_summaries(file, backtest_result, resample_period, timeframe)

def write_general_summary(file, backtest_result: BTResult):
    """
    Write a general summary of the strategy's performance.
    
    Args:
        - `file` (file-like object): File object to write to
        - `backtest_result` (BTResult): Backtest result data to analyze
    """
    reporter = StrategyReporter(ALL_METRICS)
    report = reporter.compose_report(backtest_result, mode='visual')
    print("General summary:\n", file=file)
    print(report.as_markdown().data, file=file)

def write_monthly_summaries(file, backtest_result: BTResult, resample_period: str, timeframe: timedelta | None = None):
    """
    Write monthly performance summaries.
    
    Args:
        - `file` (file-like object): File object to write to
        - `backtest_result` (BTResult): Backtest result data to analyze
        - `resample_period` (str): Resampling frequency for monthly data ("ME" for month end)
        - `timeframe` (timedelta | None): Time interval between data points
    """
    # Create date index ending today with specified timeframe
    end_date = datetime.now()
    if timeframe is None:
        timeframe = timedelta(days=len(backtest_result.net_change))
    start_date = end_date - timeframe
    
    equity_series = pd.Series(
        backtest_result.net_change.numpy(),
        index=pd.date_range(end=end_date, periods=len(backtest_result.net_change), freq='D')
    )
    monthly_changes = equity_series.resample(resample_period)

    reporter = StrategyReporter(ALL_METRICS)
    for month, changes in monthly_changes:
        monthly_result = BTResult(
            gross_log_change=torch.tensor(changes.values),
            expenses_log=torch.zeros_like(torch.tensor(changes.values)),
            weights=None
        )
        report = reporter.compose_report(monthly_result, mode='visual')
        print("Month:", month.strftime('%B %Y'), file=file)
        print("Portfolio summary this month:\n", file=file)
        print(report.as_markdown().data, file=file)

def monthly_report(backtest_result: BTResult,
                   filename: str | None = None,
                   resample_period: str = "ME",
                   figsize: tuple[int, int] = (20, 15),
                   timeframe: timedelta | None = None):
    """Generate a complete monthly performance report."""
    start_index = 0
    if filename:
        write_report_to_file(filename, backtest_result, resample_period, timeframe)
    plot_graphs(backtest_result, start_index, resample_period, figsize, timeframe=timeframe)
