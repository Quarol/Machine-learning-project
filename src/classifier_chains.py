import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.base import clone

from src.wrapper import MultiLabelWrapper

class ClassifierChainsWrapper(MultiLabelWrapper):
    def __init__(self, base_classifier, n_chains=3, random_state=42):
        self.base_classifier = base_classifier
        self.n_chains = n_chains
        self.random_state = random_state

    def fit(self, X, y):
        self.chains_ = [
            ClassifierChain(
                clone(self.base_classifier),
                order='random',
                random_state=self.random_state + i
            )
            for i in range(self.n_chains)
        ]
        for chain in self.chains_:
            chain.fit(X, y)
        return self

    def predict(self, X):
        y_pred_agg = sum(chain.predict(X) for chain in self.chains_)
        return (y_pred_agg / self.n_chains >= 0.5).astype(int)