import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

from src.binary_relevance import BinaryRelevanceWrapper
from src.label_powerset import LabelPowersetWrapper
from src.classifier_chains import ClassifierChainsWrapper

from src.load_dataset import load_dataset
from src.wrapper import MultiLabelWrapper

if __name__ == "__main__":
    max_iter = 1000
    base_estimator = LogisticRegression(max_iter=max_iter)

    models: dict[MultiLabelWrapper] = {
        'binary_relevance': BinaryRelevanceWrapper(base_estimator),
        'label_powerset': LabelPowersetWrapper(base_estimator),
        'classifier_chains': ClassifierChainsWrapper(base_estimator, n_chains=3)
    }

    X_train, X_test, y_train, y_test = load_dataset()

    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{name}.joblib')