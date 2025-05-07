from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor, ones, zeros_like
import scipy.stats as ss
from scipy.optimize import minimize
from MLTT.utils import ensure_tensor


class RegressionModel(ABC):
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """
        Predict output for input data.
        
        Args:
            - `x` (Tensor): Input data
            
        Returns:
            - Tensor: Predicted values
        """
        ...
        
    @abstractmethod
    def fit(self, x: Tensor, y: Tensor):
        """
        Fit model to training data.
        
        Args:
            - `x` (Tensor): Input features
            - `y` (Tensor): Target values
        """
        ...

    @abstractmethod
    def update(self, x: Tensor, y: Tensor):
        """
        Update model with new data point.
        
        Args:
            - `x` (Tensor): Input features
            - `y` (Tensor): Target value
        """
        ...

    @abstractmethod
    def run(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Run model on data sequence.
        
        Args:
            - `x` (Tensor): Input features sequence
            - `y` (Tensor): Target values sequence
            
        Returns:
            - Tensor: Predictions
        """
        ...

    @abstractmethod
    def calibrate(self, x: Tensor, y: Tensor):
        """
        Calibrate model parameters.
        
        Args:
            - `x` (Tensor): Input features
            - `y` (Tensor): Target values
        """
        ...

class AdaptiveLinearRegression(RegressionModel):
    """
    This class implements adaptive linear regression with a Kalman filter.

    Attributes:
        - `num_features` (int): Number of features in the data
        - `Q` (Tensor): State transition noise covariance matrix (diagonal)
        - `R` (Tensor): Measurement noise covariance matrix (diagonal)
        - `F` (float): Forgetting factor (0 <= F <= 1)
        - `P` (Tensor): Initial covariance matrix
        - `state` (Tensor): Initial regression coefficients
        - `l2_reg` (float): L2 regularization parameter (default: 0)
    """

    def __init__(self, Q, R, F=0.999, P=None, state=None, l2_reg=0, save_states=True):
        """
        Initializes the AdaptiveLinearRegression object.

        Args:
            - `Q` (Tensor): State transition noise covariance matrix (diagonal)
            - `R` (Tensor): Measurement noise covariance matrix
            - `F` (float): Forgetting factor (0 <= F <= 1).
                Brian Lai and Dennis S. Bernstein (arXiv:2404.10914v1)
            - `P` (Tensor | None): Initial covariance matrix. Defaults to None
            - `state` (Tensor | None): Initial regression coefficients. Defaults to None
            - `l2_reg` (float): L2 regularization parameter. Defaults to 0
            - `save_states` (bool): Whether to save the historical state of the Kalman filter. Defaults to True
        """
        self.num_features = Q.shape[0]
        self.Q = Q
        self.R = R
        self.F = F
        if P is None:
            P = Q/2 # Initial covariance (identity matrix)
            # by the paper Q-P need to be positive definite
        if state is None:
            state = ones(self.num_features)/self.num_features  # Initial coefficients
        self.P = P
        self.state = state
        self.l2_reg = l2_reg

        self.pdf = ss.norm.pdf
        self.log_likelihood = 0

        self.save_states = save_states
        self.states = None

    def predict(self, x: Tensor) -> Tensor:
        """
        Predicts the target value for a given input data point.

        Args:
            - `x` (Tensor): Input data point of shape (num_features,)

        Returns:
            - Tensor: Predicted target value
        """
        return x @ self.state

    def update(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Updates the regression coefficients based on a new data point.

        Args:
            x (Tensor): Input data point (shape: (num_features,)).
            y (float): Target value for the data point.

        Returns:
            Tensor: Updated regression coefficients.
        """
        # using numba function
        self.P, self.state = update(
            self.P,
            self.state,
            x,
            y,
            self.Q,
            self.R,
            self.F,
            l2_reg=self.l2_reg,
        )

        return self.state

    def run(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Runs the Kalman filter on the input data.

        Args:
            x (Tensor): Input data (shape: (num_observations, num_features)).
            y (Tensor): Target values (shape: (num_observations,)).

        Returns:
            Tensor: Array of predicted target values.
                Shape: (num_observations,)
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must have the same number of observations,"
                f"but got x.shape={x.shape} and y.shape={y.shape}"
            )

        self.log_likelihood = 0

        # using numba function
        states, self.P, predictions = adaptive_linear_regression(
            self.Q,
            self.R,
            self.F,
            self.P,
            self.state,
            x,
            y,
            l2_reg=self.l2_reg,
            save_states=self.save_states,
        )
        if self.save_states:
            self.state = states[-1]
            self.states = states

        return predictions

    def fit(self, x: Tensor, y: Tensor):
        """
        Fits the model to the training data using ordinary least squares.

        Args:
            x (Tensor): Training data (shape: (num_observations, num_features)).
            y (Tensor): Target values (shape: (num_observations,)).
        """
        # Handle target values with a single column (if needed)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()  # Reshape to 1D array

        self.state = np.linalg.lstsq(x, y, rcond=None)[0]
        self.run(x, y)

    def calibrate(self, x: Tensor, y: Tensor): # TODO: debug. This method does not converge
        """
        Calibrates the model using Maximum Likelihood Estimation (MLE).

        Args:
            x (Tensor): Training data (shape: (num_observations, num_features)).
            y (Tensor): Target values (shape: (num_observations,)).
        """
        def neg_log_likelihood(params):
            Q = np.diag(params[:self.num_features])
            R = np.diag(params[self.num_features:])
            self.Q = Q
            self.R = R
            self.fit(x, y)
            self.run(x, y)
            return -self.log_likelihood

        initial_params = np.concatenate((np.diag(self.Q), np.diag(self.R)))

        res = minimize(
            neg_log_likelihood,
            initial_params,
            method="L-BFGS-B",
            bounds=[(1e-10, None) for _ in range(2 * self.num_features)],
            tol=1e-6,
        )
        self.Q = np.diag(res.x[:self.num_features])
        self.R = np.diag(res.x[self.num_features:])
        if res.success:
            self.log_likelihood = -res.fun
        else:
            raise RuntimeError(f"Optimization failed: {res.message}")

def update(P: Tensor, state: Tensor, x: Tensor, y: Tensor, Q: Tensor, R: Tensor, F: float, l2_reg: float):
    # Convert inputs to tensors if they are numpy arrays
    P, state, x, y, Q, R = ensure_tensor(P, state, x, y, Q, R)

    # Predict target value for the input data point 
    y_hat = x @ state

    # Calculate the innovation (prediction error)
    v = y - y_hat

    # Update the covariance matrix with the forgetting factor
    P = F * P + Q

    # Calculate the Kalman gain
    S = x.T @ P @ x + R
    K = P @ x.T @ S.inverse()  # Kalman gain

    # Update the regression coefficients
    state += K.T * v - l2_reg * state

    y_hat = x @ state

    return P, state, y_hat

def adaptive_linear_regression(
        Q: Tensor, 
        R: Tensor,
        F: float, 
        P: Tensor, 
        state: Tensor, 
        x: Tensor, 
        y: Tensor, 
        l2_reg: float = 0, 
        save_states: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Implements adaptive linear regression with a Kalman filter.

    Args:
        Q (torch.Tensor or numpy.ndarray): State transition noise covariance matrix (diagonal).
        R (torch.Tensor or numpy.ndarray): Measurement noise covariance matrix (diagonal).
        F (float): Forgetting factor (0 <= F <= 1).
        P (Tensor): Initial covariance matrix.
        state (Tensor): Initial regression coefficients.
        x (Tensor): Input data (shape: (num_observations, num_features)).
        y (Tensor): Target values (shape: (num_observations,)).
        l2_reg (float, optional): L2 regularization parameter. Defaults to 0.
        save_states (bool, optional): Whether to save the historical state of the Kalman filter. Defaults to False.

    Returns:
        - torch.Tensor: Array of updated regression coefficients.
            Shape: (num_observations, num_features)
        - torch.Tensor: Updated covariance matrix.
        - torch.Tensor: Array of predicted target values.
    """
    # Convert inputs to tensors if they are numpy arrays
    Q, R, P, state, x, y = ensure_tensor(Q, R, P, state, x, y)

    num_observations = x.shape[0]
    states = zeros_like(x)
    predictions = zeros_like(y)

    for i in range(num_observations):
        Pi = P.clone()
        statei = state.clone()
        P, state, predictions[i] = update(Pi, statei, x[i].clone(), y[i], Q, R, F, l2_reg=l2_reg)
        if save_states:
            states[i] = state

    return states, P, predictions
