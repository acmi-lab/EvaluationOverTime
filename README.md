# Model Evaluation On Medical Datasets (EMDOT)

This repository is the official implementation of the EMDOT evaluation framework described in _Model Evaluation On Medical Datasets_. 

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

Below is a code snippet demonstrating how to use this package. For more details, please see `tutorial.ipynb`.
```
## First, specify experiment and model parameters.
max_iter = 500
training_regime = 'sliding_window'
seed_list = [1, 2, 3]

model_name = 'LR'

dataset_name = 'toy_data'
label = 'target'
training_regime = 'sliding_window'

df, col_dict = get_toy_breast_cancer_data()

# model class and hyperparameter grids
models = {
    "LR": {
        "model_class": LR,
        "search_space": {
            "C": [10.0**e for e in np.arange(-2, 5)],
        }
    },
    "GBDT": {
        "model_class": GBDT,
        "search_space": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        }
    },
}
model_class = models[model_name]['model_class']
hyperparam_grid = models[model_name]['search_space']
hyperparam_grid['max_iter'] = [max_iter]

# experiment setup
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

## Finally, save the results.
result_name = f"{dataset_name}_{max_iter}_{label}_{model_name}_{seed_num}_{training_regime}.csv"

result_folder = f'result/{model_name}/{dataset_name}'
if not os.path.exists('result'):
    os.makedirs(result_folder)

result_df.to_csv(f"./{result_folder}/{result_name}")

if model_name == "LR":  # save coefficients if logistic regression
    model_info.to_csv(f"./model_info/{model_name}/{dataset_name}/{result_name}")

```

### Inner workings

Below is a diagram of how the `emdot` package works under the hood:

![System Diagram](img/system_diagram.png)
