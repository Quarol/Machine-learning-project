import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

from src.wrapper import MultiLabelWrapper

class LabelPowersetWrapper(MultiLabelWrapper):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_combined = [''.join(str(int(i)) for i in row) for row in y]
        y_encoded = self.label_encoder_.fit_transform(y_combined)
        self.classifier_ = clone(self.base_classifier)
        self.classifier_.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.classifier_.predict(X)
        y_str = self.label_encoder_.inverse_transform(y_pred_encoded)
        return np.array([[int(bit) for bit in s] for s in y_str])