"""
Data structures for alpha factors.
"""
from dataclasses import dataclass

from torch import Tensor


@dataclass
class AlphaData:
    """Container for alpha factor data."""
    
    values: Tensor  # Shape: (n_observations, n_features)
    feature_names: list[str]
    
    def __post_init__(self):
        assert len(self.feature_names) == self.values.shape[1], \
            "Feature names length must match number of features: " \
            f"{len(self.feature_names)} != {self.values.shape[1]}"
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    def get_feature_idx(self, name: str) -> int:
        """Get index of feature by name."""
        try:
            return self.feature_names.index(name)
        except ValueError:
            raise ValueError(f"Feature {name} not found in data")
