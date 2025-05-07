import torch
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import math
from scipy.stats import gaussian_kde


from copy import deepcopy

from MLTT.allocation import backtest


def walk_forward_index(start_idx: int,
                       end_idx: int,
                       IS_len: int,
                       OOS_len: int) -> tuple[list[slice], list[slice]]:
    """
    Walk-forward function for index-based data.

    Args:
        - `start_idx` (int): Starting index for the walk-forward validation
        - `end_idx` (int): Ending index for the walk-forward validation
        - `IS_len` (int): Length of the in-sample period (training set)
        - `OOS_len` (int): Length of the out-of-sample period (testing set)
        
    Returns:
        tuple[list[slice], list[slice]]: A tuple containing two lists: in-sample slices and out-sample slices
    """
    step = OOS_len

    IS_ranges = []
    OOS_ranges = []
    current_idx = start_idx

    while current_idx + IS_len + OOS_len <= end_idx:
        IS_end = current_idx + IS_len
        OOS_end = IS_end + OOS_len

        IS_ranges.append(slice(current_idx, IS_end))
        OOS_ranges.append(slice(IS_end, OOS_end))

        current_idx += step

    return IS_ranges, OOS_ranges


def _sub_slice(indices: slice, shift: int):
    start_idx = max(0, indices.start + shift)
    return slice(start_idx, indices.stop + shift, indices.step)

def _sub_indices(indices: slice, shift: int):
    return [_sub_slice(idx, shift) for idx in indices]


def default_backtest(commission=0, subtract_one=False):
    def _bt_shift_pred_change(predictions, prices):
        if subtract_one:
            predictions -= 1
        # Replace NumPy roll with PyTorch roll
        weights = torch.roll(predictions, shifts=1, dims=0)

        return backtest(weights, prices, commission=commission)
    return _bt_shift_pred_change


class MetricDistribution:
    """
    A class to represent a collection of metric distributions and provide
    functionality to plot the distributions using either Matplotlib or Plotly.

    Attributes:
        - `_names` (list[str]): The names of the metrics
        - `_data` (list[list[float]]): The list of observations for each metric

    Methods:
        - `plot(n_rows=1, kde=False, subplots=True, lib='matplotlib')`: 
          Plots the distributions of the metrics using either Matplotlib or Plotly
    """

    def __init__(self,
                 metric_names: list[str],
                 observations: list[list[float]]):
        """
        Initializes the MetricDistribution with metric names and corresponding observations.

        Args:
            - `metric_names` (list[str]): A list of metric names
            - `observations` (list[list[float]]): A list of lists, where each inner list contains observations for the respective metric

        Raises:
            ValueError: If the length of `metric_names` and `observations` is not equal
        """
        if len(metric_names) != len(observations):
            raise ValueError(
                "Length of `metric_names` and `observations`"
                f" parameters should be equal. Got: {len(metric_names)} and {len(observations)}"
            )
        self._names = metric_names
        self._data = observations
        self._n_metrics = len(self._names)

    def plot(self,
             n_rows=1,
             kde=False,
             subplots=True,
             lib="matplotlib",
             kde_sigma: float=5.0):
        """
        Plots the metric distributions in a grid format, either using Matplotlib or Plotly,
        with optional Kernel Density Estimation (KDE).

        Args:
            - `n_rows` (int): The number of rows to organize the plots into (default is 1)
            - `kde` (bool): Whether to include Kernel Density Estimation (KDE) in the plots (default is False)
            - `subplots` (bool): Whether to plot each metric in a separate subplot (default is True)
            - `lib` (str): The plotting library to use. Accepts either 'matplotlib' or 'plotly' (default is 'matplotlib')
            - `kde_sigma` (float): Number of standard deviations to include in the KDE (default is 5.0)

        Raises:
            ValueError: If `lib` is not 'matplotlib' or 'plotly'
        """
        self._n_cols = math.ceil(self._n_metrics / n_rows)

        self._validate_lib(lib)
        self._kde_sigma = kde_sigma

        if lib == "matplotlib":
            self._plot_matplotlib(n_rows, kde, subplots)

        elif lib == "plotly":
            self._plot_plotly(n_rows, kde, subplots)

    def _plot_matplotlib(self, n_rows, kde, subplots):
        if subplots:
            fig, axs = plt.subplots(n_rows,
                                    self._n_cols,
                                    figsize=(self._n_cols * 5, n_rows * 5))
            axs = axs.ravel()  # Flatten the axis array if it's 2D
            for i, ax in enumerate(axs[:self._n_metrics]):
                sns.histplot(self._data[i], bins=20, kde=kde, ax=ax, color='b', alpha=0.7)
                ax.set_title(self._names[i])
                ax.grid(True)
            for j in range(i+1, len(axs)):
                fig.delaxes(axs[j])  # Remove unused subplots
        else:
            plt.figure(figsize=(10, 6))
            for i in range(self._n_metrics):
                sns.histplot(self._data[i], bins=20, kde=kde, label=self._names[i], alpha=0.6)
            plt.title("Metric Distributions")
            plt.grid(True)
            plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_plotly(self, n_rows, kde, subplots):
        if subplots:
            fig = sp.make_subplots(rows=n_rows, cols=self._n_cols, subplot_titles=self._names)
            for i in range(self._n_metrics):
                row = i // self._n_cols + 1
                col = i % self._n_cols + 1
                # Plot histogram
                fig.add_trace(go.Histogram(x=self._data[i], name=self._names[i], opacity=0.7), row=row, col=col)
                # Plot KDE if enabled
                if kde:
                    kde_values = self._calculate_kde(self._data[i])
                    fig.add_trace(go.Scatter(x=kde_values['x'], y=kde_values['y'], mode='lines', name=f"{self._names[i]} KDE"), row=row, col=col)
        else:
            fig = go.Figure()
            for i in range(self._n_metrics):
                fig.add_trace(go.Histogram(x=self._data[i], name=self._names[i], opacity=0.7))
                if kde:
                    kde_values = self._calculate_kde(self._data[i])
                    fig.add_trace(go.Scatter(x=kde_values['x'], y=kde_values['y'], mode='lines', name=f"{self._names[i]} KDE"))
            fig.update_layout(barmode='overlay', title="Metric Distributions")

        fig.update_layout(
            title="Metric Distributions",
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay' if not subplots else None,
            template="plotly_dark"
        )
        fig.show()

    def _validate_lib(self, lib):
        if lib not in ["matplotlib", "plotly"]:
            raise ValueError("Invalid plotting library. Please use 'matplotlib' or 'plotly'.")

    def _calculate_kde(self, data, num_points=1000):
        """
        Helper function to calculate the KDE for a given dataset using SciPy.

        Parameters:
        -----------
        data : list[float]
            The data for which KDE will be calculated.
        num_points : int, optional
            Number of points used for estimating the KDE curve (default is 1000).

        Returns:
        --------
        kde_values : dict
            A dictionary containing the x and y values for the KDE plot.
        """
        # Convert to numpy if data is a PyTorch tensor
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        kde = gaussian_kde(data)

        # Calculate mean and std - use torch if data is tensor, otherwise use numpy
        if isinstance(data, torch.Tensor):
            mean = torch.mean(data)
            std = torch.std(data)
            mean = mean.item()
            std = std.item()
        else:
            mean = sum(data) / len(data)
            std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

        lower_bound = mean - self._kde_sigma * std
        higher_bound = mean + self._kde_sigma * std

        # Use torch linspace if available, otherwise use numpy
        x_values = torch.linspace(lower_bound, higher_bound, num_points).numpy()
        y_values = kde(x_values)
        return {'x': x_values, 'y': y_values}


class WalkForward:  # TODO: complete refactoring needed. Check log-prices compatibility.
    def __init__(self,
                 model,
                 in_sample_len,
                 out_sample_len,
                 save_models=False,
                 drop='first',
                 train_params=None,
                 model_backtest=None):
        """
        Initialize WalkForward class.

        :param model: Model to be used.
        :param in_sample_len: Length of in-sample period for each training step.
        :param out_sample_len: Length of out-sample period for each validation step.
        :param save_models: Flag to save models after each step.
        :param drop: Strategy to drop excess elements ('first' or 'last').
        :param train_params: Additional parameters for .fit() method.
        :param model_backtest: Adapting function for backtesting the model predictions.
            Should use `predictions` (not shifted) and `prices` arguments.
            Returns percentage change of equity curve
        """
        self.model = model
        self.in_sample_len = in_sample_len
        self.out_sample_len = out_sample_len
        self.save_models = save_models
        self.train_params = train_params if train_params is not None else {}
        self.drop = drop
        self._equities = []

        if not model_backtest:
            model_backtest = default_backtest()

        self._backtester = model_backtest

    def _sample_indices(self, X):
        total_len = len(X)

        n_drop = self.in_sample_len % self.out_sample_len

        if self.drop == 'first':
            start_idx = n_drop
            end_idx = total_len
        elif self.drop == 'last':
            start_idx = 0
            end_idx = total_len - n_drop
        else:
            raise ValueError('Drop strategy should be either "first" or "last".')

        self._start_idx = start_idx
        self._end_idx = end_idx
        self._n_drop = n_drop

        return walk_forward_index(start_idx,
                                  end_idx,
                                  self.in_sample_len,
                                  self.out_sample_len)

    def _validate_input(self, Y):
        if isinstance(Y, torch.Tensor):
            if torch.isnan(Y).any():
                raise ValueError('Y should not contain NaN values.')
        else:
            # Convert to torch tensor if not already
            Y_tensor = torch.tensor(Y.values if hasattr(Y, 'values') else Y)
            if torch.isnan(Y_tensor).any():
                raise ValueError('Y should not contain NaN values.')

    def run(self,
            X, Y,
            model=None,
            shift=0):
        """
        Run walk-forward validation with specified in-sample and out-sample lengths.

        :param X: Features in DataFrame format.
        :param Y: Target variable in DataFrame or Series format.
        :param model: Model to be used.
        :param shift: Shift of the walk-forward validation periods.
        :return: PyTorch tensor or pandas DataFrame. First column contains real values,
                 second column contains predictions of the model.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y)

        in_sample_indices, out_sample_indices = self._sample_indices(X)

        in_sample_indices = _sub_indices(in_sample_indices, shift)
        out_sample_indices = _sub_indices(out_sample_indices, shift)

        self._validate_input(Y)

        if model is not None:
            self.model = model

        result = torch.full((len(Y),), float('nan'))

        for in_sample_idx, out_sample_idx in zip(in_sample_indices, out_sample_indices):
            X_train, Y_train = X.iloc[in_sample_idx], Y.iloc[in_sample_idx]
            X_test, Y_test = X.iloc[out_sample_idx], Y.iloc[out_sample_idx]

            model = deepcopy(self.model)
            model.fit(X_train, Y_train, **self.train_params)
            pred = model.predict(X_test)
            
            # Convert pred to tensor if it's not already
            if not isinstance(pred, torch.Tensor):
                pred = torch.tensor(pred)
            
            # Fill the result tensor
            result[out_sample_idx] = pred

        return result

    def distributions(self,
                      X, Y,
                      metrics: list[callable],
                      prices: torch.Tensor,
                      n: int = 30,
                      save_equity: bool = True) -> MetricDistribution:
        """
        Runs multiple walk-forward validations with different shifts and computes specified metrics.

        :param metrics: List of callable metrics: BTResult -> float.
        :param n: Number of different shifts.
        :param save_equity: Flag to save percentage change of equity after each shift.
            Data should be saved in `self.equities`
        :return: List of lists. Each sublist contains values of a metric for each shift.
        """
        results = [[] for _ in metrics]
        self._equities = []

        # Using only part of InSample as data model depends
        shift = self.in_sample_len // n

        for i in range(n):
            predictions = self.run(X, Y, shift=i * shift)
            pct_change_equity = self._backtester(predictions, prices)
            for j, metric in enumerate(metrics):
                results[j].append(metric(pct_change_equity))
            if save_equity:
                self._equities.append(pct_change_equity)
        names = [metric.__name__ for metric in metrics]
        return MetricDistribution(names, results)

    @property
    def equities(self):
        return deepcopy(self._equities)
