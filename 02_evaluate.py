import joblib

if __name__ == "__main__":
    models = {
        'binary_relevance': joblib.load('models/binary_relevance.joblib'),
        'label_powerset': joblib.load('models/label_powerset.joblib'),
        'classifier_chains': joblib.load('models/classifier_chains.joblib')
    }