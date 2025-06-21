import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

from src.binary_relevance import BinaryRelevanceWrapper
from src.label_powerset import LabelPowersetWrapper
from src.classifier_chains import ClassifierChainsWrapper

from src.load_dataset import load_dataset
from src.wrapper import MultiLabelWrapper
from sklearn.metrics import f1_score


def evaluate_model(model, X_val, y_val, macro=True):
    metric_type = 'macro' if macro else 'micro'
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average=metric_type, zero_division=0)


if __name__ == "__main__":
    X_train_full, X_test, y_train_full, y_test = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    label_frequencies = np.sum(y_train, axis=0)
    chains_order = np.argsort(-label_frequencies)

    max_iter_options = [100, 250, 500, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000]
    n_chains_options = [5, 10, 25, 50]

    best_models = {}

    label_frequencies = np.sum(y_train, axis=0)
    chains_order = np.argsort(-label_frequencies)

    # --- Binary Relevance ---
    best_score = -np.inf
    best_model = None
    for max_iter in max_iter_options:
        base_clf = LogisticRegression(max_iter=max_iter)
        model = BinaryRelevanceWrapper(base_clf)
        model.fit(X_train_scaled, y_train)
        score = evaluate_model(model, X_val_scaled, y_val)
        print(f"BR max_iter={max_iter}, Macro F1={score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
    best_models['binary_relevance'] = (best_model, best_score)

    # --- Label Powerset ---
    best_score = -np.inf
    best_model = None
    for max_iter in max_iter_options:
        base_clf = LogisticRegression(max_iter=max_iter)
        model = LabelPowersetWrapper(base_clf)
        model.fit(X_train_scaled, y_train)
        score = evaluate_model(model, X_val_scaled, y_val)
        print(f"LP max_iter={max_iter}, Macro F1={score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
    best_models['label_powerset'] = (best_model, best_score)

    # --- Classifier Chains ---
    best_score = -np.inf
    best_model = None
    for max_iter in max_iter_options:
        for n_chains in n_chains_options:
            base_clf = LogisticRegression(max_iter=max_iter)
            model = ClassifierChainsWrapper(base_clf, n_chains=n_chains, order=chains_order)
            model.fit(X_train_scaled, y_train)
            score = evaluate_model(model, X_val_scaled, y_val)
            print(f"CC max_iter={max_iter}, n_chains={n_chains}, Macro F1={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
    best_models['classifier_chains'] = (best_model, best_score)

    # Zapis najlepszych modeli i skalera
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    for name, (model, score) in best_models.items():
        print(f"Saving best {name} model with Macro F1={score:.4f}")
        joblib.dump(model, f'models/{name}_best.joblib')

    print("Validation done. Best models saved.")
