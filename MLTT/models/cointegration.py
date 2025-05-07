#TODO: Rewrite this models using torch
from dataclasses import dataclass

import numpy as np
from tqdm import auto
import torch

from .kalman import RegressionModel
from .mean_reversion import ZScoreBarriersModel
from MLTT.allocation.allocators import BaseAllocator
from MLTT.utils import to_weights_matrix


@dataclass
class RegressionIndices:
    endog_idx: int
    exog_indices: list[int]


def _get_exog_indices(index_map: np.ndarray) -> list[RegressionIndices]:
    """
    Generates a mapping of regression variables (endog) and factors (exog).

    Args:
        - `index_map` (numpy.ndarray): A matrix where each row represents a regression model.
            The first column is the index of the variable, and the rest of
            the columns are indicators (0 or 1) of whether the endogenous
            variable at the corresponding index uses the exogenous variable
            at the same index as a predictor.

    Returns:
        - `list[RegressionIndices]`: A list of tuples, where each
            tuple contains the index of the endogenous variable and a
            list of the indices of the exogenous variables.
    """
    indices = []
    for i in range(index_map.shape[0]):
        exog_row = index_map[i]
        endog_idx = int(index_map[i, 0])
        exog_row_indices = np.flatnonzero(exog_row[1:])
        indices.append(RegressionIndices(endog_idx, exog_row_indices.tolist()))

    return indices


def _get_endog_indices(index_map: np.ndarray | list[RegressionIndices]) -> list[int]:
    if isinstance(index_map, np.ndarray):
        index_map = _get_exog_indices(index_map)

    endog_indices = []
    for indices in index_map:
        endog_indices.append(indices.endog_idx)

    return endog_indices


def update_step(x: np.ndarray, models: list[RegressionModel], exog_indices) -> list[np.ndarray]:
    """
    Updates the regression coefficients for each regression model using the given data point.

    Args:
        - `x` (numpy.ndarray): The data point to use for updating the regression coefficients
        - `models` (list[RegressionModel]): A list of regression models to update
        - `exog_indices` (list[RegressionIndices]): A list of tuples,
            where each tuple contains the index of the endogenous
            variable and a list of the indices of the exogenous variables

    Returns:
        - list[numpy.ndarray]: A list of the updated regression coefficients
            for each regression model
    """
    states: list[np.ndarray] = []

    for i, (endog_idx, exog_index) in enumerate(exog_indices):
        states.append(
            models[i].update(
                x[exog_index],
                x[endog_idx]
            )
        )
    return states


def all_combinations(n_exog) -> np.ndarray: # returns indices matrix
    """
    Generates a matrix of all possible combinations of endogenous and exogenous variables.

    Args:
        - `n_exog` (int): The number of exogenous variables

    Returns:
        - numpy.ndarray: A matrix where each row
            represents a regression model.
            The first column is the index of the endogenous
            variable, and the rest of the columns are the
            indices of the exogenous variables
    """
    exog_matrix = np.ones((n_exog, n_exog))
    exog_matrix -= np.eye(n_exog)

    endog_idx = np.arange(n_exog).reshape(-1, 1)

    return np.hstack([endog_idx, exog_matrix])


def regress_predictions(prices_matrix, models, exog_indices, use_tqdm=True, save_states=True):
    """
    Generates and saves the predictions and states matrices.

    Args:
        - `prices_matrix` (numpy.ndarray): A matrix of prices for
            each asset in the portfolio
        - `models` (list[RegressionModel]): A list of regression models
        - `exog_indices` (list[RegressionIndices]): A list of
            tuples, where each tuple contains the index of the endogenous
            variable and a list of the indices of the exogenous variables
        - `use_tqdm` (bool): Whether to use tqdm for progress bar
        - `save_states` (bool): Whether to save the historical states

    Returns:
        - tuple[numpy.ndarray, list[numpy.ndarray]]: A tuple containing
            the predictions matrix and a list of states matrices
    """
    predictions = np.zeros((prices_matrix.shape[0], len(exog_indices)))
    states: list[np.ndarray] = [
        np.zeros((prices_matrix.shape[0], len(indices.exog_indices)))
        for indices in exog_indices
    ]

    iterator = enumerate(models)
    if use_tqdm:
        iterator = auto.tqdm(iterator, total=len(models))

    # using GPU-accelerated functions
    for i, model in iterator:
        exog = prices_matrix[:, exog_indices[i].exog_indices]
        endog = prices_matrix[:, exog_indices[i].endog_idx]
        predictions[:, i] = model.run(exog, endog)
        if save_states:
            states[i] = model.states


    return predictions, states


def predictions_by_coefficients(prices_matrix, coefficients, indices):
    """
    Calculate predictions using coefficient matrices.
    
    Args:
        - `prices_matrix` (numpy.ndarray): Matrix of prices
        - `coefficients` (numpy.ndarray): Coefficient values
        - `indices` (numpy.ndarray): Indices to use from prices matrix
        
    Returns:
        - numpy.ndarray: Predicted values
    """
    return np.sum(prices_matrix[:, indices] * coefficients, axis=1)


def many_hot_decode(dim_raw_matrix: int, states: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Transforms a matrix of coefficients from the size of the
    indices to the size of the raw matrix of observations.

    Args:
        - `dim_raw_matrix` (int): The size of the raw matrix of observations
        - `states` (numpy.ndarray): A matrix of coefficients of shape (n_observations, n_exog)
        - `indices` (numpy.ndarray): A matrix of indices of shape (n_exog,)

    Returns:
        - numpy.ndarray: A matrix of coefficients of shape (n_observations, dim_raw_matrix)
    """
    raw = np.zeros((states.shape[0], dim_raw_matrix))
    raw[:, indices] = states
    return raw


class VectorRegressionModel:
    def __init__(self,
                 exog_indices: list[RegressionIndices] | np.ndarray,
                 models: list[RegressionModel],
                 use_tqdm=True) -> None:
        """
        Initialize a vector regression model with multiple individual regression models.
        
        Args:
            - `exog_indices` (list[RegressionIndices] | np.ndarray): Indices of exogenous variables
            - `models` (list[RegressionModel]): Regression models for each endogenous variable
            - `use_tqdm` (bool): Whether to use tqdm progress bars
        """
        if not isinstance(exog_indices, list):
            exog_indices = _get_exog_indices(exog_indices)

        self.indices = exog_indices
        self.models = models
        self.use_tqdm = use_tqdm

    def update(self, x: np.ndarray) -> list[np.ndarray]:
        """
        Update all regression models with new data.
        
        Args:
            - `x` (numpy.ndarray): New data point for updating models
            
        Returns:
            - list[numpy.ndarray]: List of updated model states
        """
        return update_step(x, self.models, self.indices)

    def run(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Runs the vector regression model on the given data.

        Args:
            - `x` (numpy.ndarray): An array of shape (n_observations, n_variables)
                containing the data to be used for prediction

        Returns:
            - tuple[numpy.ndarray, list[numpy.ndarray]]: A tuple containing the predictions
            and the states of each regression model

            - `predictions` (numpy.ndarray): An array of
              shape (n_observations, n_exog) containing
              the predictions for each regression model
            - `states` (list[numpy.ndarray]): A list of
              arrays of shape (n_observations, n_exog) containing
              the states of each regression model
        """
        predictions, states = regress_predictions(
            x,
            self.models,
            self.indices,
            use_tqdm=self.use_tqdm
        )
        return predictions, states

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with all regression models for a single observation.
        
        Args:
            - `x` (numpy.ndarray): Input data for prediction
            
        Returns:
            - numpy.ndarray: Predictions for each model
        """
        predictions = np.zeros(len(self.models))

        for i, model in enumerate(self.models):
            prediction = model.predict(x)
            if prediction.ndim != 1 or prediction.shape[0] != 1:
                raise ValueError(
                    f"Predictions must be 1D, got {prediction.shape}-dim. Error in model {i}"
                )

            predictions[i] = prediction.item()

        return predictions

    def fit(self, x: np.ndarray):
        """
        Fit all regression models on historical data.
        
        Args:
            - `x` (numpy.ndarray): Historical data for training models
        """
        for model, indices in auto.tqdm(zip(self.models, self.indices), total=len(self.models)):
            endog = x[:, indices.endog_idx]
            exog = x[:, indices.exog_indices]

            model.fit(exog, endog)

class OneLegIndexSignalModel(BaseAllocator):
    def __init__(self,
                 z_score_model: ZScoreBarriersModel,
                 vector_regression: VectorRegressionModel) -> None:
        """
        A model that combines regression and z-score for signal generation.
        
        Args:
            - `z_score_model` (ZScoreBarriersModel): Model for generating signals based on z-scores
            - `vector_regression` (VectorRegressionModel): Model for predicting fair prices through regression
        """
        self._z_model = z_score_model
        self._regression = vector_regression
        self._endog_idx = _get_endog_indices(vector_regression.indices)
        self.num_observations = 1

    def fit(self, x: np.ndarray) -> None:
        """
        Fits all regression models on the data.

        Args:
            - `x` (np.ndarray): Matrix with log-prices for regression
                of shape `(n_observations, n_tradable)`
        """
        self._regression.fit(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generates base weights using log-prices.

        Args:
            - `x` (np.ndarray): Matrix with log-prices
                Shape: `(n_observations, n_tradable)`

        Returns:
            - np.ndarray: Weights in base assets
                Shape: `(n_observations, n_tradable)`
        """
        prediction, _ = self._regression.run(x)
        spreads = x - prediction
        signals = self._z_model.predict(spreads)
        return signals


class TorchRefitModel(BaseAllocator):
    def __init__(self, 
                 model: torch.nn.Module,
                 lookback: int = 5,
                 n_refit: int = 100,
                 n_refit_data: int = 500,
                 device: str = "cpu",
                 lr: float = 0.01,
                 refit_epochs: int = 3):
        """
        A model that periodically refits a PyTorch neural network model.
        
        Args:
            - `model` (torch.nn.Module): PyTorch model to use for predictions
            - `lookback` (int): Window size for input features
            - `n_refit` (int): How often to refit the model (in steps)
            - `n_refit_data` (int): How many historical data points to use when refitting
            - `device` (str): Device to run model on ("cpu" or "cuda")
            - `lr` (float): Learning rate for optimizer
            - `refit_epochs` (int): Number of epochs to train when refitting
        """
        super().__init__()
        # The model is moved to the specified device
        self.model = model.to(device)
        # Other parameters are initialized
        self.lookback = lookback
        self.n_refit = n_refit
        self.n_refit_data = n_refit_data
        self.device = device
        self.lr = lr
        self.refit_epochs = refit_epochs
        
        # The optimizer and loss function are initialized
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
        # Other variables are initialized
        self.step_counter = 0
        self.history = torch.tensor([], device=device)
        self.last_processed_idx = 0
        self.initial_padding = None  # For storing initial padding

    def _prepare_window(self, data: torch.Tensor) -> torch.Tensor:
        """
        Creates sliding windows directly in tensors.
        
        Args:
            - `data` (torch.Tensor): Input data tensor
            
        Returns:
            - torch.Tensor: Tensor with sliding windows
        """
        return data.unfold(0, self.lookback, 1).flatten(start_dim=1)

    def _refit_model(self, data: torch.Tensor):
        """
        Fully tensor-based training pipeline.
        
        Args:
            - `data` (torch.Tensor): Historical data for training
        """
        self.model.train()
        
        # Data preparation
        X = self._prepare_window(data[:-1])
        y = data[self.lookback-1:-1]
        
        # Training loop
        for _ in range(self.refit_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """
        Predict using model with periodical refitting.
        
        Args:
            - `x` (np.ndarray): Input data tensor
            
        Returns:
            - torch.Tensor: Predicted weights
        """
        if x.shape[0] < self.lookback:
            raise ValueError(f"Need at least {self.lookback} observations, got {x.shape[0]}")
            
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
        # Padding initialization on the first call
        if self.history.numel() == 0:
            self.initial_padding = torch.zeros(
                (self.lookback-1, x.shape[1]), 
                device=self.device
            )
            self.history = torch.cat([self.initial_padding, x_tensor], dim=0)
            self.last_processed_idx = self.lookback - 1
        else:
            self.history = torch.cat([self.history, x_tensor], dim=0)
        
        start_idx = self.last_processed_idx + 1
        end_idx = self.history.size(0)
        
        predictions = []
        for i in range(start_idx, end_idx):
            if (i - self.lookback) % self.n_refit == 0 and i >= self.n_refit_data:
                train_data = self.history[i-self.n_refit_data:i]
                self._refit_model(train_data)
            
            window = self.history[i-self.lookback:i].view(-1)
            
            with torch.no_grad():
                self.model.eval()
                pred = self.model(window)
                current_price = self.history[i]
                predictions.append(pred - current_price)
        
        self.last_processed_idx = end_idx - 1
        
        # Assemble the full result with padding
        full_predictions = torch.cat([
            self.initial_padding if self.initial_padding is not None else torch.tensor([]),
            torch.stack(predictions) if predictions else torch.tensor([], device=self.device)
        ])
        
        # Trim to the size of the input data
        result = full_predictions[-x.shape[0]:]
        
        # Reset padding after the first call
        self.initial_padding = None
        
        return to_weights_matrix(result)


# This is a simple linear model for fair price prediction
class SimpleLinearFairPrice(torch.nn.Module):
    def __init__(self, n_assets: int, lookback: int):
        """
        Initialize a simple linear model for fair price prediction.
        
        Args:
            - `n_assets` (int): Number of assets in the portfolio
            - `lookback` (int): Number of historical observations to use for prediction
        """
        super().__init__()
        self.linear = torch.nn.Linear(n_assets * lookback, n_assets)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            - `x` (torch.Tensor): Input tensor of shape (batch_size, n_assets * lookback)
            
        Returns:
            - torch.Tensor: Predicted fair prices
        """
        # Add squeeze to remove batch dimension
        return self.linear(x).squeeze(0)
