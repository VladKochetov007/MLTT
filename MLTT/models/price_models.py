#TODO: make geometric brownian motion usual brownian motion (log-scale)
"""
Interface and implementations for price models that can be calibrated to market data
and used for price simulation.
"""
from abc import ABC, abstractmethod
from scipy.optimize import minimize

import torch
from MLTT.utils import change

class PriceModel(ABC):
    """
    Base class for all price models.
    """

    def __init__(self):
        self.generated_paths = None
        self.time_steps_generated = None

    @abstractmethod
    def simulate_paths(self, num_paths: int, time_steps: int) -> torch.Tensor:
        """
        Simulate multiple paths of the asset price.
        
        Args:
            - `num_paths` (int): Number of paths to simulate
            - `time_steps` (int): Number of time steps in simulation
            
        Returns:
            - torch.Tensor: Simulated paths as a tensor of shape (time_steps + 1, num_paths)
        """
        pass

    @abstractmethod
    def fit(self, historical_prices: torch.Tensor) -> None:
        """
        Fit the model parameters based on historical price data.
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data as a tensor
        """
        pass


class GBMModel(PriceModel):
    """
    Classical Geometric Brownian Motion model with constant drift and volatility.
    """
    
    def __init__(self, initial_price, drift=None, volatility=None):
        """
        Initialize Geometric Brownian Motion model.
        
        Args:
            - `initial_price` (float): Initial price of the asset
            - `drift` (float | None): Expected return (drift) of the asset per time step
            - `volatility` (float | None): Volatility of the asset per time step
        """
        super().__init__()
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility

    def simulate_paths(self, num_paths: int, time_steps: int) -> torch.Tensor:
        """
        Simulate multiple price paths using Geometric Brownian Motion.
        
        Args:
            - `num_paths` (int): Number of paths to simulate
            - `time_steps` (int): Number of time steps in simulation
            
        Returns:
            - torch.Tensor: Simulated paths as a tensor of shape (time_steps + 1, num_paths)
        """
        paths = torch.zeros((time_steps + 1, num_paths))
        paths[0] = self.initial_price

        for t in range(1, time_steps + 1):
            random_shocks = torch.randn(num_paths)
            paths[t] = paths[t - 1] * torch.exp(
                (self.drift - 0.5 * self.volatility ** 2) +
                self.volatility * random_shocks
            )
            
        self.generated_paths = paths
        self.time_steps_generated = torch.arange(time_steps + 1)
        return paths

    def fit(self, historical_prices: torch.Tensor) -> None:
        """
        Fit drift and volatility parameters using historical prices.
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data
        """
        log_returns = torch.diff(torch.log(historical_prices))
        self.drift = torch.mean(log_returns)
        self.volatility = torch.std(log_returns)


class HestonModel(PriceModel):  # TODO: Check if Heston model is correct and fits to the market data
    """
    Heston stochastic volatility model.
    """
    
    def __init__(self, initial_price, v0, kappa, theta, sigma, rho):
        """
        Initialize Heston stochastic volatility model.
        
        Args:
            - `initial_price` (float): Initial price of the asset
            - `v0` (float): Initial variance
            - `kappa` (float): Rate of mean reversion
            - `theta` (float): Long-term variance
            - `sigma` (float): Volatility of volatility
            - `rho` (float): Correlation between asset returns and variance
        """
        super().__init__()
        self.initial_price = initial_price
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def simulate_paths(self, num_paths: int, time_steps: int) -> torch.Tensor:
        """
        Simulate multiple price paths using Heston stochastic volatility model.
        
        Args:
            - `num_paths` (int): Number of paths to simulate
            - `time_steps` (int): Number of time steps in simulation
            
        Returns:
            - torch.Tensor: Simulated paths as a tensor of shape (time_steps + 1, num_paths)
        """
        dt = 1 / time_steps
        S = torch.full((time_steps + 1, num_paths), self.initial_price)
        v = torch.full((time_steps + 1, num_paths), self.v0)

        for t in range(1, time_steps + 1):
            Z = torch.randn(num_paths, 2)
            Z = torch.cat([Z, -Z], dim=1)
            
            v[t] = torch.maximum(
                v[t - 1] + 
                self.kappa * (self.theta - v[t - 1]) * dt + 
                self.sigma * torch.sqrt(v[t - 1] * dt) * Z[:, 1], 
                0
            )
            
            exponent = (self.kappa - 0.5 * v[t - 1]) * dt + torch.sqrt(v[t - 1] * dt) * Z[:, 0]
            exponent = torch.clip(exponent, -700, 700)
            S[t] = S[t - 1] * torch.exp(exponent)
        
        self.generated_paths = S
        self.time_steps_generated = torch.arange(time_steps + 1)
        return S

    def _log_likelihood(self, params, historical_prices):
        """
        Helper method for MLE fitting.
        
        Args:
            - `params` (list): Model parameters [v0, kappa, theta, sigma, rho]
            - `historical_prices` (torch.Tensor): Historical price data
            
        Returns:
            - float: Negative log-likelihood (for minimization)
        """
        v0, kappa, theta, sigma, rho = params
        self.v0, self.kappa, self.theta, self.sigma, self.rho = params
        
        simulated_path = self._simulate_calibration_path(historical_prices)
        
        log_returns_hist = torch.diff(torch.log(historical_prices))
        log_returns_sim = torch.diff(torch.log(simulated_path))
        
        residuals = log_returns_hist - log_returns_sim
        log_likelihood = -0.5 * torch.sum(residuals ** 2)
        
        return -log_likelihood
    
    def _simulate_calibration_path(self, historical_prices):
        """
        Simulate a single path for calibration purposes.
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data to match
            
        Returns:
            - torch.Tensor: Simulated price path for calibration
        """
        time_steps = len(historical_prices)
        dt = 1 / time_steps
        path = torch.zeros(time_steps)
        path[0] = historical_prices[0]
        v = torch.zeros(time_steps)
        v[0] = self.v0

        Z = torch.randn(time_steps-1, 2)
        Z = torch.cat([Z, -Z], dim=1)

        for t in range(1, time_steps):
            v[t] = torch.maximum(
                v[t-1] + 
                self.kappa * (self.theta - v[t-1]) * dt + 
                self.sigma * torch.sqrt(v[t-1] * dt) * Z[t-1, 1], 
                0
            )
            
            exponent = (self.kappa - 0.5 * v[t-1]) * dt + torch.sqrt(v[t-1] * dt) * Z[t-1, 0]
            exponent = torch.clip(exponent, -700, 700)
            path[t] = historical_prices[t-1] * torch.exp(exponent)

        return path

    def fit(self, historical_prices: torch.Tensor) -> None:
        """
        Fit Heston model parameters using maximum likelihood estimation.
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data
        """
        initial_params = [self.v0, self.kappa, self.theta, self.sigma, self.rho]
        
        bounds = [
            (1e-6, None),  # v0: variance > 0
            (1e-6, None),  # kappa: mean-reversion speed > 0
            (1e-6, None),  # theta: long-term variance > 0
            (1e-6, None),  # sigma: vol of vol > 0
            (-1, 1)        # rho: correlation in [-1, 1]
        ]

        result = minimize(
            self._log_likelihood, 
            initial_params,
            args=(historical_prices,),
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        if result.success:
            self.v0, self.kappa, self.theta, self.sigma, self.rho = result.x
        else:
            raise RuntimeError("Failed to fit Heston model parameters!")

            
class HistoricalPriceMotion(PriceModel):
    """
    Model that simulates price paths by sampling from historical price changes.
    """
    
    def __init__(self, initial_price, historical_prices=None):
        """
        Initialize historical price motion model.
        
        Args:
            - `initial_price` (float): Initial price of the asset
            - `historical_prices` (torch.Tensor | None): Historical price data to fit the model
        """
        super().__init__()
        self.initial_price = initial_price
        self.changes = None
        if historical_prices is not None:
            self.fit(historical_prices)
    
    def fit(self, historical_prices):
        """
        Calculate historical price changes for sampling.
        
        Args:
            - `historical_prices` (torch.Tensor): Historical price data
        """
        self.changes = change(historical_prices)
    
    def simulate_paths(self, num_paths, time_steps):
        """
        Simulate multiple paths by sampling historical changes.
        
        Args:
            - `num_paths` (int): Number of paths to simulate
            - `time_steps` (int): Number of time steps to simulate
            
        Returns:
            - torch.Tensor: Simulated paths tensor of shape (time_steps + 1, num_paths)
        """
        if self.changes is None:
            raise ValueError("Model must be fit before simulating paths")
            
        # Sample changes with replacement
        changes = torch.randint(
            low=0,
            high=len(self.changes),
            size=(time_steps, num_paths),
            replace=True
        )
        
        # Calculate cumulative product of (1 + changes)
        paths = self.initial_price * torch.cumprod(1 + changes, axis=0)
        
        # Add initial price as first row
        paths = torch.vstack([
            torch.full((1, num_paths), self.initial_price),
            paths
        ])
        
        self.generated_paths = paths
        self.time_steps_generated = torch.arange(time_steps + 1)
        
        return paths
