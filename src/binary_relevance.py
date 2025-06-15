import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone

from src.wrapper import MultiLabelWrapper

class BinaryRelevanceWrapper(MultiLabelWrapper):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifier = MultiOutputClassifier(clone(base_classifier))

    def fit(self, X, Y):
        self.classifier.fit(X, Y)

    def predict(self, X):
        return self.classifier.predict(X)