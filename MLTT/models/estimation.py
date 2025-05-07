from torch import Tensor
from scipy.stats import johnsonsu
from pyriemann.utils.covariance import covariance_mest

from MLTT.utils import CACHE_SIZE
from MLTT.cache import conditional_lru_cache

class JohnsonSU:
    """
    Johnson SU distribution class.

    Args:
        - `mu` (float): Location parameter
        - `sigma` (float): Scale parameter
        - `alpha` (float): Skewness parameter
        - `beta` (float): Shape parameter
    """

    def __init__(self, mu: float, sigma: float, alpha: float, beta: float):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def fit(x: Tensor, /) -> 'JohnsonSU':
        """
        Fit the Johnson SU distribution to data.

        Args:
            - `x` (Tensor): The data to fit the distribution to

        Returns:
            - JohnsonSU: The fitted distribution
        """
        mu, sigma, alpha, beta = johnsonsu.fit(x)
        return JohnsonSU(mu, sigma, alpha, beta)

    @property
    def params(self) -> tuple:
        return self.mu, self.sigma, self.alpha, self.beta

    def pdf(self, x: Tensor) -> Tensor:
        """
        Compute the pdf of a Johnson SU distribution.

        Args:
            - `x` (Tensor): The points to evaluate the pdf at

        Returns:
            - Tensor: The pdf evaluated at x
        """
        return johnsonsu.pdf(x, self.mu, self.sigma, self.alpha, self.beta)

    def cdf(self, x: Tensor) -> Tensor:
        """
        Compute the cdf of a Johnson SU distribution.

        Args:
            - `x` (Tensor): The points to evaluate the cdf at

        Returns:
            - Tensor: The cdf evaluated at x
        """
        return johnsonsu.cdf(x, self.mu, self.sigma, self.alpha, self.beta)

    def sample(self, size: int) -> Tensor:
        """
        Generate random samples from the Johnson SU distribution.
        
        Args:
            - `size` (int): Number of samples to generate
            
        Returns:
            - Tensor: Random samples from the distribution
        """
        return johnsonsu.rvs(self.mu, self.sigma, self.alpha, self.beta, size=size)


@conditional_lru_cache(maxsize=CACHE_SIZE)
def robust_covariance(X: Tensor, m_estimator: str = 'tyl', q: float = 0, tol: float = 10e-3, max_iter: int = 100) -> Tensor:
    """
    Computes a robust estimate of the covariance matrix of a multivariate
    time-series using a fixed-point algorithm and a M-estimator.

    Args:
        - `X` (Tensor): A multivariate time-series of shape (n_channels, n_times)
        - `m_estimator` (str): The type of M-estimator to use:
            - "hub" for Huber's adaptive trimmed mean
            - "stu" for Student-t's robust mean
            - "tyl" for Tyler's resistant mean
        - `q` (float): A parameter of Huber's M-estimator, determines the trade-off
            between robustness to outliers and closeness to the sample mean.
            If q=0, equivalent to Tyler's estimator, if q=1, equivalent to sample covariance
        - `tol` (float): The tolerance for stopping the fixed-point iteration
        - `max_iter` (int): The maximum number of iterations to perform

    Returns:
        - Tensor: A robust estimate of the covariance matrix of shape (n_channels, n_channels)
    """
    return covariance_mest(X, m_estimator=m_estimator, q=q, tol=tol, n_iter_max=max_iter)
