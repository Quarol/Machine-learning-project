import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from src.wrapper import MultiLabelWrapper

class LabelPowersetWrapper(MultiLabelWrapper):
    def __init__(self, max_iter=100, C=1.0, penalty='l2', solver='lbfgs', l1_ratio=None, random_state=None):
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_combined = [''.join(str(int(i)) for i in row) for row in y]
        y_encoded = self.label_encoder_.fit_transform(y_combined)

        lr_kwargs = {
            'max_iter': self.max_iter,
            'C': self.C,
            'penalty': self.penalty,
            'solver': self.solver,
            'random_state': self.random_state
        }
        if self.penalty == 'elasticnet':
            lr_kwargs['l1_ratio'] = self.l1_ratio

        self.classifier_ = LogisticRegression(**lr_kwargs)
        self.classifier_.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.classifier_.predict(X)
        y_str = self.label_encoder_.inverse_transform(y_pred_encoded)
        return np.array([[int(bit) for bit in s] for s in y_str])