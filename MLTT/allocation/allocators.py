from abc import ABC
from torch import Tensor, mean

from allocation_o2 import CapitalAllocator # type: ignore


class BaseAllocator(CapitalAllocator, ABC):
    def __init__(self,
                 num_observations: int = 0) -> None:
        self.num_observations = num_observations

    @property
    def min_observations(self) -> int:
        return self.num_observations


class PriceAwareAllocator(CapitalAllocator, ABC):
    def get_prices_matrix(self) -> Tensor:
        ...


class BlendingModel(CapitalAllocator):
    def __init__(self, models: list[CapitalAllocator], weights: Tensor) -> None:
        self.models = models
        self.weights = weights

    @property
    def min_observations(self) -> int:
        return max([model.min_observations for model in self.models])

    def predict(self, x: Tensor) -> Tensor:
        predictions = [model.predict(x) for model in self.models]
        return mean(predictions, dim=0, weights=self.weights)  # TODO: check axis
