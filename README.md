# Model Evaluation On Medical Datasets (EMDOT)

This repository is the official implementation of the EMDOT evaluation framework described in _[Model Evaluation On Medical Datasets Over Time](https://arxiv.org/abs/2211.07165)_. 

### Installation

First, create a conda environment:

```
conda create --name <env_name> python==3.8.3
conda activate <env_name>
```

Then, install the `emdot` package:

```
git clone https://github.com/acmi-lab/EvaluationOverTime.git
cd EvaluationOverTime/src
pip install .
```

### Quickstart

Below is a code snippet demonstrating how to use this package. For a more complete introduction, **please see [tutorial.ipynb](tutorial.ipynb)**. To view documentation, simply run `help(...)` on the class or object of interest.

```
from emdot.eot_experiment import EotExperiment
from emdot.example_datasets import get_toy_breast_cancer_data
from emdot.models.LR import LR
from emdot.models.GBDT import GBDT
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emdot.utils import plot_metric_vs_time

## First, specify model parameters and experiment.

model_name = 'LR'
model_class = LR
hyperparam_grid = {"C": [10.0**e for e in np.arange(-2, 5)], "max_iter": [500]}

seed_list = [1, 2, 3]

dataset_name = 'toy_data'
label = 'target'
training_regime = 'sliding_window'
df, col_dict = get_toy_breast_cancer_data()

experiment = EotExperiment(
    dataset_name = dataset_name,
    df = df, 
    col_dict = col_dict, 
    label = label,
    model_class = model_class,
    model_name = model_name,
    hyperparam_grid = hyperparam_grid,
    training_regime = training_regime,
    initial_train_window = (0, 4),
    train_end = 7,
    test_end = 9,
    train_prop = 0.5,
    val_prop = 0.25,
    test_prop = 0.25,
    window_length = 5,
    time_unit = 'Year',
    model_folder = './model_info')

## Next, run the experiment.
result_df, model_info = experiment.run_experiment(seed_list, eval_metric="auc")
```

### Inner workings

Below is a diagram of how the `emdot` package works under the hood:

![System Diagram](img/system_diagram.png)

