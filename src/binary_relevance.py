import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

from src.wrapper import MultiLabelWrapper

class BinaryRelevanceWrapper(MultiLabelWrapper):
    def __init__(self, max_iter=100, C=1.0, penalty='l2', solver='lbfgs', l1_ratio=None, random_state=None):
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
    def fit(self, X, y):
        clf_kwargs = dict(
            max_iter=self.max_iter,
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            random_state=self.random_state
        )
        if self.penalty == 'elasticnet':
            clf_kwargs['l1_ratio'] = self.l1_ratio
        
        base_clf = LogisticRegression(**clf_kwargs)
        self.classifier_ = MultiOutputClassifier(base_clf)
        self.classifier_.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)