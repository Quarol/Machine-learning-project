import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from src.wrapper import MultiLabelWrapper

class LabelPowersetWrapper(MultiLabelWrapper):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.label_encoder = LabelEncoder()
        self.classifier = None

    def fit(self, X, Y):
        Y_combined = [''.join(str(int(y)) for y in row) for row in Y]
        y_encoded = self.label_encoder.fit_transform(Y_combined)
        self.classifier = clone(self.base_classifier)
        self.classifier.fit(X, y_encoded)

    def predict(self, X):
        y_pred_encoded = self.classifier.predict(X)
        Y_str = self.label_encoder.inverse_transform(y_pred_encoded)
        return np.array([[int(bit) for bit in s] for s in Y_str])