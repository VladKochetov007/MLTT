import torch
from torch import Tensor

def compute_ic(alpha: Tensor, returns: Tensor, method: str = "spearman") -> float:
    """Compute Information Coefficient between alpha and returns.
    
    Args:
        - `alpha` (Tensor): Alpha factor values
        - `returns` (Tensor): Forward returns
        - `method` (str): Correlation method ("pearson" or "spearman")
    
    Returns:
        - `float`: Information Coefficient value
    """
    if method == "spearman":
        alpha_ranks = torch.argsort(torch.argsort(alpha)).float()
        return_ranks = torch.argsort(torch.argsort(returns)).float()
        
        # Center the ranks
        alpha_ranks = alpha_ranks - alpha_ranks.mean()
        return_ranks = return_ranks - return_ranks.mean()
        
        # Compute correlation
        num = (alpha_ranks * return_ranks).sum()
        den = torch.sqrt((alpha_ranks * alpha_ranks).sum() * (return_ranks * return_ranks).sum())
        
        return (num / (den + 1e-8)).item()
    
    elif method == "pearson":
        # Center the values
        alpha_centered = alpha - alpha.mean()
        returns_centered = returns - returns.mean()
        
        # Compute correlation
        num = (alpha_centered * returns_centered).sum()
        den = torch.sqrt((alpha_centered * alpha_centered).sum() * (returns_centered * returns_centered).sum())
        
        return (num / (den + 1e-8)).item()
    
    else:
        raise ValueError(f"Unknown correlation method: {method}")
