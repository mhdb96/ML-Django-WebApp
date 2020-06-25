from abc import ABC, abstractmethod


class IModel (ABC):
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def setParameters(self):
        pass
