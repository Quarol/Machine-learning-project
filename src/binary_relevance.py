import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone

from src.wrapper import MultiLabelWrapper

class BinaryRelevanceWrapper(MultiLabelWrapper):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        self.classifier_ = MultiOutputClassifier(clone(self.base_classifier))
        self.classifier_.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)