import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import joblib
from sklearn.metrics import f1_score

from src.load_dataset import load_dataset

def compute_per_instance_f1(y_true, y_pred, average_type):
    """Compute per-instance F1 scores for macro or micro averaging"""
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
            scores.append(0.0)  # Handle cases with no predicted labels
    return np.array(scores)

def compute_all_f1_scores(y_true, y_preds):
    """Compute both macro and micro F1 scores for all models"""
    f1_scores = {
        'macro': {},
        'micro': {}
    }
    
    for model_name, y_pred in y_preds.items():
        f1_scores['macro'][model_name] = compute_per_instance_f1(
            y_true, y_pred, 'macro'
        )
        f1_scores['micro'][model_name] = compute_per_instance_f1(
            y_true, y_pred, 'micro'
        )
    
    return f1_scores

def run_pairwise_tests(metric_dict):
    """Run pairwise Wilcoxon tests and adjust for multiple comparisons"""
    model_names = list(metric_dict.keys())
    comparisons = []
    p_values = []
    raw_results = []
    
    # Perform all pairwise comparisons
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            vec1 = metric_dict[model1]
            vec2 = metric_dict[model2]
            
            # Wilcoxon signed-rank test
            stat, p = wilcoxon(vec1, vec2, alternative='two-sided')
            comparisons.append((model1, model2))
            p_values.append(p)
            
            # Store raw results
            median1 = np.median(vec1)
            median2 = np.median(vec2)
            mean1 = np.mean(vec1)
            mean2 = np.mean(vec2)
            raw_results.append({
                'models': (model1, model2),
                'p': p,
                'medians': (median1, median2),
                'means': (mean1, mean2)
            })
    
    # Adjust p-values using Bonferroni correction
    reject, adj_p_values, _, _ = multipletests(p_values, method='bonferroni')
    
    # Prepare adjusted results
    adj_results = []
    for idx, comp in enumerate(comparisons):
        adj_results.append({
            'models': comp,
            'adj_p': adj_p_values[idx],
            'significant': adj_p_values[idx] < 0.05
        })
    
    return raw_results, adj_results

def report_results(metric_name, raw_results, adj_results):
    """Print formatted test results for a specific metric"""
    print(f"\n{'='*50}")
    print(f"Statistical Comparison Results: {metric_name.upper()} F1 Score")
    print(f"{'='*50}")
    
    # Print raw results
    print("\nPairwise Comparisons (Raw):")
    for res in raw_results:
        m1, m2 = res['models']
        print(f"{m1} vs {m2}:")
        print(f"  p-value = {res['p']:.4f}")
        print(f"  Medians: {res['medians'][0]:.4f} vs {res['medians'][1]:.4f}")
        print(f"  Means:   {res['means'][0]:.4f} vs {res['means'][1]:.4f}")
    
    # Print adjusted results
    print("\nPairwise Comparisons (Bonferroni-adjusted):")
    for res in adj_results:
        m1, m2 = res['models']
        print(f"{m1} vs {m2}:")
        print(f"  adj-p = {res['adj_p']:.4f} {'(significant)' if res['significant'] else ''}")
    
    # Print summary
    significant_pairs = [res['models'] for res in adj_results if res['significant']]
    if significant_pairs:
        print("\nSignificant Differences Found:")
        for pair in significant_pairs:
            print(f"- {pair[0]} vs {pair[1]}")
    else:
        print("\nNo Significant Differences Found")

if __name__ == "__main__":
    directory_name = input('Enter the path to the dataset directory: ')
    if not directory_name:
        print("No directory provided. Exiting.")
        exit(1)

    # Load models and data
    models = {
        'binary_relevance': joblib.load(f'models/{directory_name}/binary_relevance.joblib'),
        'label_powerset': joblib.load(f'models/{directory_name}/label_powerset.joblib'),
        'classifier_chains': joblib.load(f'models/{directory_name}/classifier_chains.joblib')
    }
    scaler = joblib.load(f'models/{directory_name}/scaler.joblib')
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_dataset()
    X_test = scaler.transform(X_test)

    # Get predictions
    y_preds = {
        'binary_relevance': models['binary_relevance'].predict(X_test),
        'label_powerset': models['label_powerset'].predict(X_test),
        'classifier_chains': models['classifier_chains'].predict(X_test)
    }

    # Compute all F1 scores
    f1_scores = compute_all_f1_scores(y_test, y_preds)
    
    # Run tests and report results for each metric
    for metric_type in ['macro', 'micro']:
        # Run statistical tests
        raw, adj = run_pairwise_tests(f1_scores[metric_type])
        
        # Report results
        report_results(f"{metric_type} F1", raw, adj)