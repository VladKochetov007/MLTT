"""
Alpha factor generation and evaluation package.

This module provides functionality for generating alpha factors, evaluating their performance,
and visualizing the results. It includes operations for creating alpha factors from strings,
computing information coefficients, and visualizing alpha factors.

Note:
    - This module is currently not ready for production use AT ALL.
"""

from MLTT.alphas.base import Operation, AlphaFactor
from MLTT.alphas.data import AlphaData
from MLTT.alphas.performance import compute_ic
from MLTT.alphas.generation import generate_random_alpha, optimize_alpha_population
from MLTT.alphas.operations import InputFeature, TsRank, TsMin, TsMax, TsCorrPearson, TsCorrSpearman
from MLTT.alphas.visualization import visualize_alpha, print_alpha_tree

__all__ = [
    'Operation',
    'AlphaFactor',
    'AlphaData',
    'compute_ic',
    'generate_random_alpha',
    'optimize_alpha_population',
    'InputFeature',
    'TsRank',
    'TsMin',
    'TsMax',
    'TsCorrPearson',
    'TsCorrSpearman',
    'visualize_alpha',
    'print_alpha_tree',
]
