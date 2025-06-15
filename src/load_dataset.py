import torch
import pandas as pd


def load_dataset():
    annotations = ('data/breast-level-annotations.csv')
    
    train_features = torch.load('data/training_features_cut_image.pt')
    test_features = torch.load('data/test_features_cut_image.pt')

    X_train, X_test, y_train, y_test = None, None, None, None
    return X_train, X_test, y_train, y_test