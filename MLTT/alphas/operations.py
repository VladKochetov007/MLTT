"""
Operations that can be used in alpha factor trees.
"""
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from MLTT.alphas.base import Operation


@dataclass
class InputFeature(Operation):
    """Represents an input feature from the data."""
    
    feature_idx: int
    feature_name: str
    
    def forward(self, x: Tensor) -> Tensor:
        return x[:, self.feature_idx]
    
    def to_string(self) -> str:
        return f"${self.feature_name}"
    
    @classmethod
    def from_string(cls, s: str) -> 'InputFeature':
        assert s.startswith("$")
        name = s[1:]
        # Note: feature_idx will be set by AlphaFactor when deserializing
        return cls(0, name)


@dataclass
class TsRank(Operation):
    """Compute cross-sectional ranks over a rolling window."""
    
    input_op: Operation
    window: int
    
    def forward(self, x: Tensor) -> Tensor:
        values = self.input_op.forward(x)
        n = len(values)
        result = torch.zeros_like(values)
        
        # Compute rolling ranks
        for i in range(self.window - 1, n):
            window_data = values[i - self.window + 1:i + 1]
            result[i] = torch.argsort(torch.argsort(window_data)).float()[-1]
        
        # Fill initial values
        result[:self.window - 1] = result[self.window - 1]
        return result
    
    def to_string(self) -> str:
        return f"TSRANK({self.input_op.to_string()},{self.window})"
    
    @classmethod
    def from_string(cls, s: str) -> 'TsRank':
        assert s.startswith("TSRANK(") and s.endswith(")")
        inner = s[7:-1]
        op_str, window_str = inner.rsplit(",", 1)
        return cls(
            Operation.from_string(op_str),
            int(window_str)
        )


@dataclass
class TsMin(Operation):
    """Compute rolling minimum."""
    
    input_op: Operation
    window: int
    
    def forward(self, x: Tensor) -> Tensor:
        values = self.input_op.forward(x)
        
        # Pad the input to maintain the same length
        pad_size = self.window - 1
        padded_values = F.pad(values.unsqueeze(0).unsqueeze(0), 
                            (pad_size, 0), mode='replicate')
        
        # Apply min pooling
        return F.max_pool1d(-padded_values, self.window, stride=1).squeeze() * -1
    
    def to_string(self) -> str:
        return f"TSMIN({self.input_op.to_string()},{self.window})"
    
    @classmethod
    def from_string(cls, s: str) -> 'TsMin':
        assert s.startswith("TSMIN(") and s.endswith(")")
        inner = s[6:-1]
        op_str, window_str = inner.rsplit(",", 1)
        return cls(
            Operation.from_string(op_str),
            int(window_str)
        )


@dataclass
class TsMax(Operation):
    """Compute rolling maximum."""
    
    input_op: Operation
    window: int
    
    def forward(self, x: Tensor) -> Tensor:
        values = self.input_op.forward(x)
        
        # Pad the input to maintain the same length
        pad_size = self.window - 1
        padded_values = F.pad(values.unsqueeze(0).unsqueeze(0), 
                            (pad_size, 0), mode='replicate')
        
        # Apply max pooling
        return F.max_pool1d(padded_values, self.window, stride=1).squeeze()
    
    def to_string(self) -> str:
        return f"TSMAX({self.input_op.to_string()},{self.window})"
    
    @classmethod
    def from_string(cls, s: str) -> 'TsMax':
        assert s.startswith("TSMAX(") and s.endswith(")")
        inner = s[6:-1]
        op_str, window_str = inner.rsplit(",", 1)
        return cls(
            Operation.from_string(op_str),
            int(window_str)
        )


@dataclass
class TsCorrPearson(Operation):
    """Compute Pearson correlation over a rolling window."""
    
    input_op1: Operation
    input_op2: Operation
    window: int
    
    def forward(self, x: Tensor) -> Tensor:
        values1 = self.input_op1.forward(x)
        values2 = self.input_op2.forward(x)
        
        # Get device of input tensor
        device = x.device
        
        # Ensure values are on the same device
        values1 = values1.to(device)
        values2 = values2.to(device)
        
        # Initialize result tensor
        n = len(values1)
        result = torch.zeros(n, device=device)
        
        # Pad the beginning with the first correlation value
        for i in range(self.window - 1):
            # Use available data for initial windows
            v1 = values1[:i+1]
            v2 = values2[:i+1]
            
            # Center the values
            v1 = v1 - v1.mean()
            v2 = v2 - v2.mean()
            
            # Compute correlation
            num = (v1 * v2).sum()
            den = torch.sqrt((v1 * v1).sum() * (v2 * v2).sum())
            
            result[i] = num / (den + 1e-8)
        
        # Compute rolling correlation for the rest
        for i in range(self.window - 1, n):
            v1 = values1[i - self.window + 1:i + 1]
            v2 = values2[i - self.window + 1:i + 1]
            
            # Center the values
            v1 = v1 - v1.mean()
            v2 = v2 - v2.mean()
            
            # Compute correlation
            num = (v1 * v2).sum()
            den = torch.sqrt((v1 * v1).sum() * (v2 * v2).sum())
            
            result[i] = num / (den + 1e-8)
        
        return result
    
    def to_string(self) -> str:
        return f"TSCORRP({self.input_op1.to_string()},{self.input_op2.to_string()},{self.window})"
    
    @classmethod
    def from_string(cls, s: str) -> 'TsCorrPearson':
        assert s.startswith("TSCORRP(") and s.endswith(")")
        inner = s[8:-1]
        op1_str, rest = inner.split(",", 1)
        op2_str, window_str = rest.rsplit(",", 1)
        return cls(
            Operation.from_string(op1_str),
            Operation.from_string(op2_str),
            int(window_str)
        )


@dataclass
class TsCorrSpearman(Operation):
    """Compute Spearman correlation over a rolling window."""
    
    input_op1: Operation
    input_op2: Operation
    window: int
    
    def forward(self, x: Tensor) -> Tensor:
        values1 = self.input_op1.forward(x)
        values2 = self.input_op2.forward(x)
        
        # Get device of input tensor
        device = x.device
        
        # Ensure values are on the same device
        values1 = values1.to(device)
        values2 = values2.to(device)
        
        # Initialize result tensor
        n = len(values1)
        result = torch.zeros(n, device=device)
        
        # Pad the beginning with the first correlation value
        for i in range(self.window - 1):
            # Use available data for initial windows
            v1 = values1[:i+1]
            v2 = values2[:i+1]
            
            # Convert to ranks
            r1 = torch.argsort(torch.argsort(v1)).float()
            r2 = torch.argsort(torch.argsort(v2)).float()
            
            # Center the ranks
            r1 = r1 - r1.mean()
            r2 = r2 - r2.mean()
            
            # Compute correlation
            num = (r1 * r2).sum()
            den = torch.sqrt((r1 * r1).sum() * (r2 * r2).sum())
            
            result[i] = num / (den + 1e-8)
        
        # Compute rolling correlation for the rest
        for i in range(self.window - 1, n):
            v1 = values1[i - self.window + 1:i + 1]
            v2 = values2[i - self.window + 1:i + 1]
            
            # Convert to ranks
            r1 = torch.argsort(torch.argsort(v1)).float()
            r2 = torch.argsort(torch.argsort(v2)).float()
            
            # Center the ranks
            r1 = r1 - r1.mean()
            r2 = r2 - r2.mean()
            
            # Compute correlation
            num = (r1 * r2).sum()
            den = torch.sqrt((r1 * r1).sum() * (r2 * r2).sum())
            
            result[i] = num / (den + 1e-8)
        
        return result
    
    def to_string(self) -> str:
        return f"TSCORRS({self.input_op1.to_string()},{self.input_op2.to_string()},{self.window})"
    
    @classmethod
    def from_string(cls, s: str) -> 'TsCorrSpearman':
        assert s.startswith("TSCORRS(") and s.endswith(")")
        inner = s[8:-1]
        op1_str, rest = inner.split(",", 1)
        op2_str, window_str = rest.rsplit(",", 1)
        return cls(
            Operation.from_string(op1_str),
            Operation.from_string(op2_str),
            int(window_str)
        )