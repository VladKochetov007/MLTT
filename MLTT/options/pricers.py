"""
Module for pricing binary outcome events based on price models.

More about PolyMarket-like predictions
"""

import torch
from abc import ABC, abstractmethod
from ..models.price_models import PriceModel


class BinaryPricer(ABC):
    """
    Base class for binary outcome pricers.
    """
    
    def __init__(self, model: PriceModel, num_paths: int = 10000, time_steps: int = 252):
        """
        :param model: Price model instance for simulations
        :param num_paths: Number of paths to simulate for Monte Carlo
        :param time_steps: Number of time steps in simulation
        """
        self.model = model
        self.num_paths = num_paths
        self.time_steps = time_steps
        self._cached_paths = None
        
    def simulate_paths(self):
        """
        Simulate paths using the price model.
        """
        if self._cached_paths is None:
            self._cached_paths = self.model.simulate_paths(self.num_paths, self.time_steps)
        return self._cached_paths
    
    def clear_cache(self):
        """
        Clear cached paths.
        """
        self._cached_paths = None
        
    @abstractmethod
    def estimate_probability(self, strike: float) -> float:
        """
        Estimate probability of the event.
        
        :param strike: Strike price level
        :return: Probability estimate
        """
        pass


class TouchProbabilityPricer(BinaryPricer):
    """
    Estimates probability of price touching strike before expiration.
    """
    
    def estimate_probability(self, strike: float) -> float:
        paths = self.simulate_paths()
        crosses = torch.any(paths >= strike, axis=0)
        return torch.mean(crosses)


class ExpirationProbabilityPricer(BinaryPricer):
    """
    Estimates probability of price being above strike at expiration.
    """
    
    def estimate_probability(self, strike: float) -> float:
        paths = self.simulate_paths()
        final_prices = paths[-1]
        above_strike = final_prices >= strike
        return torch.mean(above_strike)
