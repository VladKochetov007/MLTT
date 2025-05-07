"""
Functions for generating and optimizing alpha factors.
"""
import os
from dataclasses import dataclass

import pandas as pd
from torch import Tensor

from MLTT.alphas.base import AlphaFactor
from MLTT.alphas.data import AlphaData
from MLTT.alphas.generation import generate_random_alpha
from MLTT.alphas.performance import compute_ic
from MLTT.alphas.visualization import visualize_alpha, print_alpha_tree


@dataclass
class AlphaResult:
    """Container for alpha generation result."""
    
    alpha: AlphaFactor
    ic: float
    stats: dict[str, float]
    values: Tensor

def generate_and_evaluate_alphas(
    data: AlphaData,
    returns: Tensor,
    n_alphas: int = 50,
    max_depth: int = 5,
    window_range: tuple[int, int] = (12, 48),
    save_dir: str | None = None
) -> list[AlphaResult]:
    """Generate and evaluate multiple alpha factors.
    
    Args:
        - `data` (AlphaData): Input data
        - `returns` (Tensor): Forward returns for IC calculation
        - `n_alphas` (int): Number of alphas to generate
        - `max_depth` (int): Maximum depth of alpha trees
        - `window_range` (tuple[int, int]): Range for window parameters
        - `save_dir` (str | None): Directory to save visualizations
        
    Returns:
        list[AlphaResult]: List of alpha results, sorted by absolute IC value
    """
    results = []
    
    print(f"\nGenerating {n_alphas} alpha factors...")
    for i in range(n_alphas):
        print(f"\nAlpha {i+1}/{n_alphas}")
        
        # Generate alpha
        alpha = generate_random_alpha(
            data=data,
            max_depth=max_depth,
            window_range=window_range
        )
        
        # Print formula
        print("Formula:", alpha.to_string())
        
        # Compute values
        values = alpha.compute(data.values)
        
        # Compute IC
        ic = compute_ic(values, returns)
        print(f"IC: {ic:.4f}")
        
        # Save result
        results.append(AlphaResult(alpha, ic, {}, values))
        
        # Save visualization if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            visualize_alpha(
                alpha,
                filename=os.path.join(save_dir, f"alpha_{i+1}"),
                format="png",
                view=False
            )
    
    # Sort by absolute IC value
    results.sort(key=lambda x: abs(x.ic), reverse=True)
    
    return results


def print_alpha_results(
    results: list[AlphaResult],
    n_best: int = 10,
    save_path: str | None = None
) -> None:
    """Print results of alpha generation.
    
    Args:
        - `results` (list[AlphaResult]): List of alpha results
        - `n_best` (int): Number of best alphas to show
        - `save_path` (str | None): Path to save results CSV
    """
    print(f"\nTop {n_best} alphas by IC:")
    for i, result in enumerate(results[:n_best], 1):
        print(f"\n{i}. IC: {result.ic:.4f}")
        print("Formula:", result.alpha.to_string())
        print("Statistics:")
        for name, value in result.stats.items():
            print(f"  {name}: {value:.4f}")
        print("\nTree structure:")
        print_alpha_tree(result.alpha)
    
    if save_path:
        # Create DataFrame with results
        df = pd.DataFrame([
            {
                'rank': i + 1,
                'ic': r.ic,
                'formula': r.alpha.to_string(),
                **{f'stat_{k}': v for k, v in r.stats.items()}
            }
            for i, r in enumerate(results)
        ])
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")


def find_best_alpha(
    data: AlphaData,
    returns: Tensor,
    n_alphas: int = 50,
    max_depth: int = 5,
    window_range: tuple[int, int] = (12, 48),
    save_dir: str | None = "alpha_results",
    n_best: int = 10
) -> AlphaResult:
    """Generate multiple alphas and find the best one.
    
    Args:
        - `data` (AlphaData): Input data
        - `returns` (Tensor): Forward returns for IC calculation
        - `n_alphas` (int): Number of alphas to generate
        - `max_depth` (int): Maximum depth of alpha trees
        - `window_range` (tuple[int, int]): Range for window parameters
        - `save_dir` (str | None): Directory to save results
        - `n_best` (int): Number of best alphas to show
        
    Returns:
        AlphaResult: Best alpha result
    """
    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vis_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = None
    
    # Generate and evaluate alphas
    results = generate_and_evaluate_alphas(
        data=data,
        returns=returns,
        n_alphas=n_alphas,
        max_depth=max_depth,
        window_range=window_range,
        save_dir=vis_dir
    )
    
    # Print and save results
    if save_dir:
        csv_path = os.path.join(save_dir, "results.csv")
    else:
        csv_path = None
    
    print_alpha_results(results, n_best=n_best, save_path=csv_path)
    
    return results[0]  # Return best alpha