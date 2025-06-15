import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

from src.binary_relevance import BinaryRelevanceWrapper
from src.label_powerset import LabelPowersetWrapper
from src.classifier_chains import ClassifierChainsWrapper

def get_data():
    sample_size = 1000
    feature_size = 2048
    label_size = 10
    X = np.random.rand(sample_size, feature_size)
    Y = np.random.randint(0, 2, size=(sample_size, label_size))
    return X, Y


if __name__ == "__main__":
    X, Y = get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    max_iter = 1000
    base_estimator = LogisticRegression(max_iter=max_iter)

    models = {
        'binary_relevance': BinaryRelevanceWrapper(base_estimator),
        'label_powerset': LabelPowersetWrapper(base_estimator),
        'classifier_chains': ClassifierChainsWrapper(base_estimator, n_chains=3)
    }

    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, Y_train)
        joblib.dump(model, f'models/{name}.joblib')