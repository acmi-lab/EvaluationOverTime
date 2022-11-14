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

Please see [tutorial.ipynb](tutorial.ipynb).

### Inner workings

Below is a diagram of how the `emdot` package works under the hood:

![System Diagram](img/system_diagram.png)
