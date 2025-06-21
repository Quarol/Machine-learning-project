import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, jaccard_score
#from skmultilearn.model_selection import IterativeStratification
import joblib
import os
import itertools

from src.binary_relevance import BinaryRelevanceWrapper
from src.label_powerset import LabelPowersetWrapper
from src.classifier_chains import ClassifierChainsWrapper

from src.load_dataset import load_dataset
from src.wrapper import MultiLabelWrapper
from src.iterative_stratification import IterativeStratification


def print_metrics(metrics):
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"Jaccard Score: {metrics['jaccard_score']:.4f}")


def br_and_lp_param_combinations(penalties: list[str], solvers: list[str], max_iters: list[int], Cs: list[float], l1_ratios: list[float]) -> list[dict]:
    combinations = []
    for penalty, solver, max_iter, C in itertools.product(penalties, solvers, max_iters, Cs):
        # filter invalid combos
        if penalty == 'none' and solver not in ['lbfgs', 'newton-cg', 'saga']:
            continue
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            continue
        if penalty == 'elasticnet' and solver != 'saga':
            continue
        if penalty == 'l2' and solver not in ['newton-cg', 'lbfgs', 'liblinear', 'saga']:
            continue
        
        param_dict = {
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'C': C,
        }
        # Only add l1_ratio if elasticnet
        if penalty == 'elasticnet':
            for l1_ratio in l1_ratios:
                param_dict_with_l1 = param_dict.copy()
                param_dict_with_l1['l1_ratio'] = l1_ratio
                combinations.append(param_dict_with_l1)
        else:
            # No l1_ratio needed
            param_dict['l1_ratio'] = None
            combinations.append(param_dict)

    return combinations


def cc_param_combinations(penalties: list[str], solvers: list[str], max_iters: list[int], Cs: list[float], l1_ratios: list[float], orders: list[str]) -> list[dict]:
    base_combos = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, l1_ratios)
    cc_combos = []

    for combo in base_combos:
        for order in orders:
            combo_with_order = combo.copy()
            combo_with_order['order'] = order
            cc_combos.append(combo_with_order)

    return cc_combos


def multilabel_CV(wrapper_cls, X, y, scaler, n_splits=5, **wrapper_kwargs):
    skf = IterativeStratification(n_splits=n_splits, order=1)

    macro_f1_scores = []
    micro_f1_scores = []
    hamming_losses = []
    subset_accuracies = []
    jaccard_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = wrapper_cls(**wrapper_kwargs)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)

        macro_f1_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        micro_f1_scores.append(f1_score(y_val, y_pred, average='micro', zero_division=0))
        hamming_losses.append(hamming_loss(y_val, y_pred))
        subset_accuracies.append(accuracy_score(y_val, y_pred))
        jaccard_scores.append(jaccard_score(y_val, y_pred, average='samples'))

    return {
        'macro_f1': np.mean(macro_f1_scores),
        'micro_f1': np.mean(micro_f1_scores),
        'hamming_loss': np.mean(hamming_losses),
        'subset_accuracy': np.mean(subset_accuracies),
        'jaccard_score': np.mean(jaccard_scores),
    }


def grid_search_wrapper(wrapper_cls, X, y, scaler, cv_splits, param_combinations):
    best_score = -np.inf
    best_model = None
    best_params = None

    for params in param_combinations:
        metrics = multilabel_CV(wrapper_cls, X, y, scaler, cv_splits, **params)
        macro_f1 = metrics['macro_f1']
        print(f"Params: {params}")
        print_metrics(metrics)
        print()

        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params
            best_model = wrapper_cls(**params)
            best_model.fit(scaler.fit_transform(X), y)

    return best_model, best_score, best_params


def binary_relevance_grid_search(X_full, y_full, scaler, CV_SPLITS, br_param_grid, folder):
    print("=== Grid Search Binary Relevance ===")
    br_model, br_score, br_params = grid_search_wrapper(BinaryRelevanceWrapper, X_full, y_full, scaler, CV_SPLITS, br_param_grid)
    print(f"Best BR params: {br_params}, Best CV Macro F1: {br_score:.4f}")
    joblib.dump(br_model, f'{folder}/binary_relevance.joblib')
    with open(f'{folder}/binary_relevance_params.txt', 'w') as f:
        for k, v in br_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"CV Macro F1: {br_score:.4f}\n")


def label_powerset_grid_search(X_full, y_full, scaler, CV_SPLITS, lp_param_grid, folder):
    print("=== Grid Search Label Powerset ===")
    lp_model, lp_score, lp_params = grid_search_wrapper(LabelPowersetWrapper, X_full, y_full, scaler, CV_SPLITS, lp_param_grid)
    print(f"Best LP params: {lp_params}, Best CV Macro F1: {lp_score:.4f}")
    joblib.dump(lp_model, f'{folder}/label_powerset_best.joblib')
    with open(f'{folder}/label_powerset_params.txt', 'w') as f:
        for k, v in lp_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"CV Macro F1: {lp_score:.4f}\n")


def classifier_chains_grid_search(X_full, y_full, scaler, CV_SPLITS, cc_param_grid, folder):
    print("=== Grid Search Classifier Chains ===")
    cc_model, cc_score, cc_params = grid_search_wrapper(ClassifierChainsWrapper, X_full, y_full, scaler, CV_SPLITS, cc_param_grid)
    print(f"Best CC params: {cc_params}, Best CV Macro F1: {cc_score:.4f}")
    joblib.dump(cc_model, f'{folder}/classifier_chains_best.joblib')
    with open(f'{folder}/classifier_chains_params.txt', 'w') as f:
        for k, v in cc_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"CV Macro F1: {cc_score:.4f}\n")


if __name__ == "__main__":
    input_directory = input('Enter the path to the dataset directory: ').strip()

    X_full, X_test, y_full, y_test = load_dataset()
    scaler = StandardScaler()

    CV_SPLITS = 5

    penalties = ['l1', 'l2', 'elasticnet', 'none']
    solvers = ['liblinear', 'lbfgs', 'saga', 'newton-cg']
    max_iters = [250, 500, 750]
    Cs = [0.01, 0.1, 1.0, 10]
    ratios = [0.1, 0.5, 0.9]

    orders = ['random']  # or add deterministic orders like [list(range(y_full.shape[1]))]

    br_param_grid = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, ratios)
    lp_param_grid = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, ratios)
    cc_param_grid = cc_param_combinations(penalties, solvers, max_iters, Cs, ratios, orders)

    folder = f'models/{input_directory}'
    os.makedirs(folder, exist_ok=True)
    joblib.dump(scaler, f'{folder}/scaler.joblib')

    print("Starting grid search...")

    binary_relevance_grid_search(X_full, y_full, scaler, CV_SPLITS, br_param_grid, folder)
    label_powerset_grid_search(X_full, y_full, scaler, CV_SPLITS, lp_param_grid, folder)
    #classifier_chains_grid_search(X_full, y_full, scaler, CV_SPLITS, cc_param_grid, folder)

    print("Grid search complete. Best models saved.")
