"""
Functions for generating alpha factors.
"""
import random
from typing import Callable

from torch import Tensor

from .alpha import Operation, AlphaFactor
from .operations import (
    InputFeature, TsRank, TsMin, TsMax,
    TsCorrPearson, TsCorrSpearman
)
from .data import AlphaData
from .performance import compute_ic


def generate_random_alpha(
    data: AlphaData,
    max_depth: int = 3,
    operations: list[type[Operation]] = [TsRank, TsMin, TsMax, TsCorrPearson, TsCorrSpearman],
    window_range: tuple[int, int] = (5, 30)
) -> AlphaFactor:
    """Generate random alpha factor.
    
    Args:
        - `data` (AlphaData): Input data
        - `max_depth` (int): Maximum depth of operation tree
        - `operations` (list[type[Operation]]): List of available operations
        - `window_range` (tuple[int, int]): Range for window parameters
    
    Returns:
        - `AlphaFactor`: Generated alpha factor
    """
    def _generate_operation(depth: int) -> Operation:
        if depth >= max_depth:
            # At max depth, only allow input features
            feature_idx = random.randint(0, data.n_features - 1)
            return InputFeature(feature_idx, data.feature_names[feature_idx])
        
        op_class = random.choice(operations)
        
        if op_class == InputFeature:
            feature_idx = random.randint(0, data.n_features - 1)
            return InputFeature(feature_idx, data.feature_names[feature_idx])
        
        elif op_class == TsRank:
            input_op = _generate_operation(depth + 1)
            window = random.randint(*window_range)
            return TsRank(input_op, window)
        
        elif op_class == TsMin:
            input_op = _generate_operation(depth + 1)
            window = random.randint(*window_range)
            return TsMin(input_op, window)
        
        elif op_class == TsMax:
            input_op = _generate_operation(depth + 1)
            window = random.randint(*window_range)
            return TsMax(input_op, window)
        
        elif op_class in [TsCorrPearson, TsCorrSpearman]:
            input_op1 = _generate_operation(depth + 1)
            input_op2 = _generate_operation(depth + 1)
            window = random.randint(*window_range)
            return op_class(input_op1, input_op2, window)
        
        else:
            raise ValueError(f"Unknown operation class: {op_class}")
    
    root = _generate_operation(0)
    return AlphaFactor(root, data.feature_names)


def optimize_alpha_population(
    data: AlphaData,
    returns: Tensor,
    population_size: int = 100,
    n_generations: int = 50,
    mutation_rate: float = 0.1,
    tournament_size: int = 5,
    fitness_fn: Callable[[Tensor, Tensor], float] = compute_ic
) -> list[tuple[AlphaFactor, float]]:
    """Optimize population of alpha factors using genetic algorithm.
    
    Args:
        - `data` (AlphaData): Input data
        - `returns` (Tensor): Forward returns
        - `population_size` (int): Size of population
        - `n_generations` (int): Number of generations
        - `mutation_rate` (float): Probability of mutation
        - `tournament_size` (int): Size of tournament for selection
        - `fitness_fn` (Callable[[Tensor, Tensor], float]): Function to compute fitness of alpha factor
    
    Returns:
        - `list[tuple[AlphaFactor, float]]`: List of (alpha factor, fitness) pairs, sorted by fitness
    """
    # Initialize population
    population = [
        generate_random_alpha(data)
        for _ in range(population_size)
    ]
    
    for generation in range(n_generations):
        # Evaluate fitness
        fitness_scores = [
            fitness_fn(alpha.compute(data.values), returns)
            for alpha in population
        ]
        
        # Sort by fitness
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select best individuals
        new_population = [pair[0] for pair in population_with_fitness[:population_size // 2]]
        
        # Fill rest of population with tournament selection and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament = random.sample(population_with_fitness, tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            
            # Mutation
            if random.random() < mutation_rate:
                new_alpha = generate_random_alpha(data)
            else:
                new_alpha = winner
            
            new_population.append(new_alpha)
        
        population = new_population
    
    # Final evaluation
    final_fitness = [
        fitness_fn(alpha.compute(data.values), returns)
        for alpha in population
    ]
    
    # Sort and return results
    results = list(zip(population, final_fitness))
    results.sort(key=lambda x: x[1], reverse=True)
    return results