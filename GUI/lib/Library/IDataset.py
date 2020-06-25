from abc import ABC, abstractmethod


class IDataset (ABC):
    @abstractmethod
    def getDataset(self):
        pass

    @abstractmethod
    def getParameters(self):
        pass

    @abstractmethod
    def getFeatures(self):
        pass

    @abstractmethod
    def getClasses(self):
        pass

    @abstractmethod
    def getPath(self):
        pass
