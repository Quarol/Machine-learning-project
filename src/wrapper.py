from abc import ABC, abstractmethod

class MultiLabelWrapper(ABC):
    @abstractmethod
    def fit(self, X, Y): pass
    
    @abstractmethod
    def predict(self, X): pass