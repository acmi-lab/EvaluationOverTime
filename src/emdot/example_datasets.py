"""Example datasets to be used in evaluation over time."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def get_toy_breast_cancer_data():
    # construct dummy data as input to EotDataset
    data = load_breast_cancer()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    df['timept'] = np.random.randint(0, 10, size=(len(df)))
    df['Patient ID'] = list(range(len(df)))

    # arbitrarily create some binary features
    df['large radius'] = (df['mean radius'] > 15).astype(int)
    df['large texture'] = (df['mean texture'] > 20).astype(int)
    
    col_dict = {
        "numerical_features": list(data['feature_names']),  # all features are numerical
        "all_features": list(data['feature_names']) + ['large radius', 'large texture'],
        "label_cols": ['target'],
        "time_col": 'timept',
        "ID": "Patient ID"
    }
    return df, col_dict
