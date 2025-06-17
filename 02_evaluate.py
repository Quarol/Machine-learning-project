import joblib

from src.load_dataset import load_dataset
from src.wrapper import MultiLabelWrapper

if __name__ == "__main__":
    models: dict[MultiLabelWrapper] = {
        'binary_relevance': joblib.load('models/binary_relevance.joblib'),
        'label_powerset': joblib.load('models/label_powerset.joblib'),
        'classifier_chains': joblib.load('models/classifier_chains.joblib')
    }
    X_train, X_test, y_train, y_test = load_dataset()
    