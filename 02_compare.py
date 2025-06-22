import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import joblib
import pandas as pd
from sklearn.metrics import f1_score
from src.load_dataset import load_dataset


def compute_per_instance_f1(y_true, y_pred, average_type):
    """Compute per-instance F1 scores using macro or micro averaging."""
    scores = []
    for i in range(len(y_true)):
        try:
            score = f1_score(
                np.atleast_1d(y_true[i]), 
                np.atleast_1d(y_pred[i]),
                average=average_type,
                zero_division=0
            )
            scores.append(score)
        except ValueError:
            scores.append(0.0)
    return np.array(scores)


def compute_all_f1_scores(y_true, y_preds):
    """Compute per-instance macro and micro F1 scores for each model."""
    f1_scores = {'macro': {}, 'micro': {}}
    for model_name, y_pred in y_preds.items():
        f1_scores['macro'][model_name] = compute_per_instance_f1(y_true, y_pred, 'macro')
        f1_scores['micro'][model_name] = compute_per_instance_f1(y_true, y_pred, 'micro')
    return f1_scores


def run_pairwise_tests(metric_dict):
    """Run Wilcoxon signed-rank test with Bonferroni correction."""
    model_names = list(metric_dict.keys())
    comparisons, p_values, raw_results = [], [], []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            vec1, vec2 = metric_dict[m1], metric_dict[m2]

            stat, p = wilcoxon(vec1, vec2, alternative='two-sided')
            comparisons.append((m1, m2))
            p_values.append(p)
            raw_results.append({
                'models': (m1, m2),
                'p': p,
                'medians': (np.median(vec1), np.median(vec2)),
                'means': (np.mean(vec1), np.mean(vec2))
            })

    reject, adj_p_values, _, _ = multipletests(p_values, method='bonferroni')

    adj_results = []
    for idx, comp in enumerate(comparisons):
        adj_results.append({
            'models': comp,
            'adj_p': adj_p_values[idx],
            'significant': adj_p_values[idx] < 0.05
        })

    return raw_results, adj_results


def report_results(metric_name, raw_results, adj_results):
    """Print formatted test results for a specific F1 metric."""
    print(f"\n{'='*50}")
    print(f"Statistical Comparison Results: {metric_name.upper()} F1 Score")
    print(f"{'='*50}")

    print("\nPairwise Comparisons (Raw):")
    for res in raw_results:
        m1, m2 = res['models']
        print(f"{m1} vs {m2}:")
        print(f"  p-value = {res['p']:.4f}")
        print(f"  Medians: {res['medians'][0]:.4f} vs {res['medians'][1]:.4f}")
        print(f"  Means:   {res['means'][0]:.4f} vs {res['means'][1]:.4f}")

    print("\nPairwise Comparisons (Bonferroni-adjusted):")
    for res in adj_results:
        m1, m2 = res['models']
        sig = ' (significant)' if res['significant'] else ''
        print(f"{m1} vs {m2}:")
        print(f"  adj-p = {res['adj_p']:.4f}{sig}")

    significant_pairs = [res['models'] for res in adj_results if res['significant']]
    if significant_pairs:
        print("\nSignificant Differences Found:")
        for pair in significant_pairs:
            print(f"- {pair[0]} vs {pair[1]}")
    else:
        print("\nNo Significant Differences Found")


def print_f1_score_table(f1_scores, y_true, y_preds):
    """Print a table of per-instance and global F1 scores."""
    rows = []

    for model_name in y_preds.keys():
        row = {
            'Model': model_name,
            'Per-instance Macro-F1': np.mean(f1_scores['macro'][model_name]),
            'Per-instance Micro-F1': np.mean(f1_scores['micro'][model_name]),
            'Global Macro-F1': f1_score(y_true, y_preds[model_name], average='macro', zero_division=0),
            'Global Micro-F1': f1_score(y_true, y_preds[model_name], average='micro', zero_division=0)
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n====================== F1 SCORE SUMMARY ======================")
    print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    directory_name = input('Enter the path to the dataset directory: ')
    if not directory_name:
        print("No directory provided. Exiting.")
        exit(1)

    print("\nResearch Hypothesis: At least one model performs significantly better in terms of macro or micro F1 score.")
    print("Statistical Test: Wilcoxon signed-rank test on per-instance F1 scores (Bonferroni-adjusted).\n")

    models = {
        'binary_relevance': joblib.load(f'models/{directory_name}/binary_relevance.joblib'),
        'label_powerset': joblib.load(f'models/{directory_name}/label_powerset.joblib'),
        'classifier_chains': joblib.load(f'models/{directory_name}/classifier_chains.joblib')
    }
    scalers = {
        'binary_relevance': joblib.load(f'models/{directory_name}/binary_relevance_scaler.joblib'),
        'label_powerset': joblib.load(f'models/{directory_name}/label_powerset_scaler.joblib'),
        'classifier_chains': joblib.load(f'models/{directory_name}/classifier_chains_scaler.joblib')
    }

    X_train, X_test, y_train, y_test = load_dataset()

    X_test_scaled = {
        model_name: scalers[model_name].transform(X_test)
        for model_name in models
    }

    y_preds = {
        model_name: models[model_name].predict(X_test_scaled[model_name])
        for model_name in models
    }

    f1_scores = compute_all_f1_scores(y_test, y_preds)

    print_f1_score_table(f1_scores, y_test, y_preds)

    for metric_type in ['macro', 'micro']:
        raw, adj = run_pairwise_tests(f1_scores[metric_type])
        report_results(f"{metric_type} F1", raw, adj)