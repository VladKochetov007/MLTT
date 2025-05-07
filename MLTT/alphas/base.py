"""
Base classes for alpha factor representation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from torch import Tensor


class Operation(ABC):
    """Base class for all operations in alpha factor trees."""
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply operation to input tensor."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert operation to string representation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_string(cls, s: str) -> 'Operation':
        """Create operation from string representation."""
        pass


@dataclass
class AlphaFactor:
    """Represents a complete alpha factor as a tree of operations."""
    
    root: Operation
    feature_names: list[str]
    
    def compute(self, data: Tensor) -> Tensor:
        """Compute alpha factor values for given input data.
        
        Args:
            - `data` (Tensor): Input tensor of shape (n_observations, n_features)
            
        Returns:
            - `Tensor`: Tensor of shape (n_observations,) with alpha factor values
        """
        assert data.shape[1] == len(self.feature_names), \
            f"Expected {len(self.feature_names)} features, got {data.shape[1]}"
        return self.root.forward(data)
    
    def to_string(self) -> str:
        """Convert alpha factor to string representation."""
        return f"{self.root.to_string()}|{','.join(self.feature_names)}"
    
    @classmethod
    def from_string(cls, s: str) -> 'AlphaFactor':
        """Create alpha factor from string representation."""
        from .operations import Operation, InputFeature
        
        op_str, features_str = s.split('|')
        feature_names = features_str.split(',')
        
        # Create alpha factor
        alpha = cls(
            root=Operation.from_string(op_str),
            feature_names=feature_names
        )
        
        # Fix feature indices in InputFeature operations
        def fix_indices(op: Operation) -> None:
            if isinstance(op, InputFeature):
                op.feature_idx = feature_names.index(op.feature_name)
            for field in fields(op):
                value = getattr(op, field)
                if isinstance(value, Operation):
                    fix_indices(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, Operation):
                            fix_indices(item)
        
        fix_indices(alpha.root)
        return alpha 