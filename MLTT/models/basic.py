from MLTT.allocation import BaseAllocator
from MLTT.utils import to_weights_matrix

from torch import Tensor, sign, roll


def sign_model(cls):
    """
    A decorator that converts model's signals to -1, 0, or 1.
    Modifies the predict method of any allocator model to output sign values.
    
    Usage:
        @sign_model
        class MyModel(BaseAllocator):
            ...
    """
    original_predict = cls.predict
    
    def new_predict(*args, **kwargs) -> Tensor:
        # Get predictions from original method
        predictions = original_predict(*args, **kwargs)
        # Convert to -1, 0, 1 signals
        signals = sign(predictions)
        return to_weights_matrix(signals)
        
    cls.predict = new_predict
    return cls

@sign_model
class LeakModel(BaseAllocator):
    """
    A model that "leaks" future information by looking one step ahead.
    For each asset:
    - Takes long position if price will increase next day
    - Takes short position if price will decrease next day
    
    Warning: This model uses future information and should only be used for research/testing purposes.
    """

    def __init__(self) -> None:
        super().__init__(num_observations=1)

    def predict(self, x: Tensor) -> Tensor:
        # Calculate next day's returns by shifting prices back
        next_prices = roll(x, shifts=-1, dims=0)
        future_returns = (next_prices - x)
        return future_returns
