from abc import ABC, abstractmethod
from .IDataset import IDataset


class IProcessor (ABC):
    @abstractmethod
    def process(self, dataset: IDataset):
        pass
