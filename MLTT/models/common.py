from torch import Tensor
from MLTT.allocation import CapitalAllocator
from MLTT.cache import conditional_lru_cache
from MLTT.utils import CACHE_SIZE
import copy


@conditional_lru_cache(maxsize=CACHE_SIZE)
def treshold_change(allocator: CapitalAllocator, threshold: float = 0.2, deepcopy: bool = True) -> CapitalAllocator:
    """
    Decorator for CapitalAllocator that prevents allocation changes unless they exceed a threshold.
    
    Args:
        - `allocator` (CapitalAllocator): The capital allocator to decorate. Stays unchanged. Operates with tensors.
        - `threshold` (float): The minimum sum of absolute differences required to change the allocation
        - `deepcopy` (bool): Whether to deepcopy the allocator. If False, the allocator will be modified in place.
        
    Returns:
        - CapitalAllocator: A decorated CapitalAllocator with threshold-based change prevention
    """
    # Create a copy if requested to avoid modifying original
    decorated_alloc = copy.deepcopy(allocator) if deepcopy else allocator
    
    # Preserve original predict method before any modifications
    original_predict = decorated_alloc.predict
    
    def predict_with_threshold(x: Tensor) -> Tensor:
        # Use preserved original implementation, not current method
        base_allocation = original_predict(x)
        last_allocation = None

        for i in range(base_allocation.shape[0]):
            if last_allocation is None:
                last_allocation = base_allocation[i]
                continue
            
            if (base_allocation[i] - last_allocation).abs().sum() < threshold:
                base_allocation[i] = last_allocation
            else:
                last_allocation = base_allocation[i]

        return base_allocation
    
    decorated_alloc.predict = predict_with_threshold
    return decorated_alloc
