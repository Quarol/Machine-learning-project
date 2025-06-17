from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiLabelWrapper(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass