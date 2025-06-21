import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

from src.wrapper import MultiLabelWrapper

class ClassifierChainsWrapper(MultiLabelWrapper):
    def __init__(self, max_iter=100, C=1.0, penalty='l2', solver='lbfgs', l1_ratio=None, random_state=42, order='random'):
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.order = order

    def fit(self, X, y):
        clf_params = {
            'max_iter': self.max_iter,
            'C': self.C,
            'penalty': self.penalty,
            'solver': self.solver,
            'random_state': self.random_state,
        }
        if self.penalty == 'elasticnet' and self.solver == 'saga':
            clf_params['l1_ratio'] = self.l1_ratio

        base_clf = LogisticRegression(**clf_params)

        self.chain_ = ClassifierChain(
            base_estimator=base_clf,
            order=self.order,
            random_state=self.random_state
        )
        self.chain_.fit(X, y)
        return self

    def predict(self, X):
        return self.chain_.predict(X)