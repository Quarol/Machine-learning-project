import torch
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

LABELS = [
    'Mass',
    'Suspicious Calcification',
    'Asymmetry',
    'Focal Asymmetry',
    'Global Asymmetry',
    'Architectural Distortion',
    'Skin Thickening',
    'Skin Retraction',
    'Nipple Retraction',
    'Suspicious Lymph Node'
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

def encode_labels(label_str):
    label_list = ast.literal_eval(label_str)
    binary_vector = np.zeros(len(LABELS), dtype=int)
    
    for label in label_list:
        if label in LABEL_TO_INDEX:
            binary_vector[LABEL_TO_INDEX[label]] = 1
    return binary_vector


def load_dataset():
    annotations_df = pd.read_csv('dataset/finding_annotations.csv')
    annotations_df = annotations_df.drop_duplicates(subset=['image_id']).reset_index(drop=True)
    annotations_df = annotations_df[annotations_df['finding_categories'] != "['No Finding']"]

    train_annotations_df = annotations_df[annotations_df['split'] == 'training'].reset_index(drop=True)
    test_annotations_df = annotations_df[annotations_df['split'] == 'test'].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=LABELS)
    y_train = mlb.fit_transform(train_annotations_df['finding_categories'].apply(ast.literal_eval))
    y_test = mlb.transform(test_annotations_df['finding_categories'].apply(ast.literal_eval))
    
    train_features_pt = torch.load('dataset/training_features_cut_image.pt')
    test_features_pt = torch.load('dataset/test_features_cut_image.pt')
    tensor_X_train = torch.stack(train_features_pt)
    tesnor_X_test = torch.stack(test_features_pt)
    X_train = tensor_X_train.numpy()
    X_test = tesnor_X_test.numpy()

    return X_train, X_test, y_train, y_test