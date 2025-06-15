import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score, hamming_loss


def display_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    metrics = {
    'BR': {
        'f1_micro': f1_score(y_test, y_pred_br, average='micro'),
        'hamming_loss': hamming_loss(y_test, y_pred_br)
    },
    'CC': {
        'f1_micro': f1_score(y_test, y_pred_cc, average='micro'),
        'hamming_loss': hamming_loss(y_test, y_pred_cc)
    },
    'LP': {
        'f1_micro': f1_score(y_test, y_pred_lp, average='micro'),
        'hamming_loss': hamming_loss(y_test, y_pred_lp)
    }
    }
    data1 = np.array([1, 2, 3, 4, 5])
    data2 = np.array([2, 3, 4, 5, 6])

    t_stat, p_value_t = ttest_rel(data1, data2)
    print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_t}")

    w_stat, p_value_w = wilcoxon(data1, data2)
    print(f"Wilcoxon signed-rank test: W-statistic = {w_stat}, p-value = {p_value_w}")