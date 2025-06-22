import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, jaccard_score
import joblib
import os
import itertools
import warnings
import pandas as pd

from src.binary_relevance import BinaryRelevanceWrapper
from src.label_powerset import LabelPowersetWrapper
from src.classifier_chains import ClassifierChainsWrapper
from src.load_dataset import load_dataset
from src.iterative_stratification import IterativeStratification


def print_metrics(metrics):
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"Jaccard Score: {metrics['jaccard_score']:.4f}")


def add_random_state_to_param_grid(param_grid, random_state):
    for param_dict in param_grid:
        param_dict['random_state'] = random_state
    return param_grid

def br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, l1_ratios):
    combinations = []
    
    for penalty, solver, max_iter, C in itertools.product(penalties, solvers, max_iters, Cs):
        if penalty == 'none' and solver not in ['lbfgs', 'newton-cg', 'saga']:
            continue
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            continue
        if penalty == 'elasticnet':
            if solver != 'saga' or not l1_ratios:
                continue
        if penalty == 'l2' and solver not in ['newton-cg', 'lbfgs', 'liblinear', 'saga']:
            continue

        base_params = {
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'C': C,
        }

        if penalty == 'elasticnet':
            for l1_ratio in l1_ratios:
                combo = base_params.copy()
                combo['l1_ratio'] = l1_ratio
                combinations.append(combo)
        else:
            base_params['l1_ratio'] = None
            combinations.append(base_params)

    return combinations


def cc_param_combinations(penalties, solvers, max_iters, Cs, l1_ratios, orders):
    base_combos = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, l1_ratios)
    cc_combos = []

    for combo in base_combos:
        for order in orders:
            combo_with_order = combo.copy()
            combo_with_order['order'] = order
            cc_combos.append(combo_with_order)

    return cc_combos


def multilabel_CV(wrapper_cls, X, y, scaler, n_splits=5, **wrapper_kwargs):
    random_state = wrapper_kwargs.get('random_state', None)
    skf = IterativeStratification(n_splits=n_splits, order=1, random_state=random_state)

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


def grid_search_wrapper(wrapper_cls, X, y, scaler, cv_splits, param_combinations, results_filename=None):
    best_score = -np.inf
    best_params = None
    records = []

    for params in param_combinations:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            metrics = multilabel_CV(wrapper_cls, X, y, scaler, cv_splits, **params)

            warning_msgs = [str(warning.message) for warning in w]

        macro_f1 = metrics['macro_f1']
        print(f"Params: {params}")
        print_metrics(metrics)
        print()

        record = {
            **params,
            **metrics,
            'warnings': warning_msgs,
        }
        records.append(record)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_params = params

    if results_filename:
        df = pd.DataFrame(records)
        df.to_csv(results_filename, index=False)

    return best_score, best_params


def grid_search_and_save(wrapper_cls, X_full, y_full, CV_SPLITS, param_grid, folder, model_name):
    print(f"=== Grid Search {model_name} ===")
    results_file = os.path.join(folder, f"{model_name.lower().replace(' ', '_')}_grid_search_results.csv")

    temp_scaler = StandardScaler()  # used for CV only
    best_score, best_params = grid_search_wrapper(
        wrapper_cls, X_full, y_full, temp_scaler, CV_SPLITS, param_grid, results_filename=results_file
    )

    print(f"Best {model_name} params: {best_params}, Best CV Macro F1: {best_score:.4f}")

    # Now fit final model using best_params and final scaler on all training data
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_full)

    final_model = wrapper_cls(**best_params)
    final_model.fit(X_scaled, y_full)

    # Save final model and fitted scaler
    model_path =  os.path.join(folder, f"{model_name.lower().replace(' ', '_')}.joblib")
    scaler_path = os.path.join(folder, f"{model_name.lower().replace(' ', '_')}_scaler.joblib")
    params_path = os.path.join(folder, f"{model_name.lower().replace(' ', '_')}_params.txt")

    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)

    with open(params_path, 'w') as f:
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"CV Macro F1: {best_score:.4f}\n")


if __name__ == "__main__":
    input_directory = input('Enter the path to the dataset directory: ').strip()

    X_full, X_test, y_full, y_test = load_dataset()

    CV_SPLITS = 5
    penalties = ['l1', 'l2']
    solvers = ['liblinear']
    max_iters = [100, 250, 500]
    Cs = [0.1, 1.0, 10]
    ratios = []
    orders = ['random']
    random_state = 123

    br_param_grid = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, ratios)
    lp_param_grid = br_and_lp_param_combinations(penalties, solvers, max_iters, Cs, ratios)
    cc_param_grid = cc_param_combinations(penalties, solvers, max_iters, Cs, ratios, orders)

    br_param_grid = add_random_state_to_param_grid(br_param_grid, random_state)
    lp_param_grid = add_random_state_to_param_grid(lp_param_grid, random_state)
    cc_param_grid = add_random_state_to_param_grid(cc_param_grid, random_state)

    folder = f'models/{input_directory}'
    os.makedirs(folder, exist_ok=True)

    print("Starting grid search...")

    grid_search_and_save(BinaryRelevanceWrapper, X_full, y_full, CV_SPLITS, br_param_grid, folder, "binary_relevance")
    grid_search_and_save(LabelPowersetWrapper, X_full, y_full, CV_SPLITS, lp_param_grid, folder, "label_powerset")
    grid_search_and_save(ClassifierChainsWrapper, X_full, y_full, CV_SPLITS, cc_param_grid, folder, "classifier_chains")

    print("Grid search complete. Best models and scalers saved.")
