import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score, hamming_loss
import joblib



if __name__ == "__main__":
    models = {
        'binary_relevance': joblib.load('models/binary_relevance.joblib'),
        'label_powerset': joblib.load('models/label_powerset.joblib'),
        'classifier_chains': joblib.load('models/classifier_chains.joblib')
    }
