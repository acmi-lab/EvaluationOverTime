import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm

from emdot.models.ExpModel import ExpModel
from emdot.eot_dataset import EotDataset
from emdot.eot_evaluator import EotEvaluator

# # import warnings filter
# from warnings import simplefilter
# # ignore all future warnings
# simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None

class EotExperiment:
    def __init__(
        self, 
        dataset_name: str,
        df: pd.DataFrame, 
        col_dict: dict, 
        label: str,
        model_class: ExpModel,
        model_name: str,
        hyperparam_grid: dict,
        training_regime: str,
        initial_train_window: Tuple[int, int],
        train_end: int,
        test_end: int,
        train_prop: float,
        val_prop: float,
        test_prop: float,
        window_length: int,
        time_unit: str = 'Year',
        model_folder: Optional[str] = './model_info'):
        """Constructor for EotExperiment.
        
        Args:
            dataset_name: name of dataset 
            df: pre-processed dataframe, which includes all of the columns mentioned in col_dict.
                All categorical variables should be converted to dummies, however numerical features 
                do not need to be normalized. This is to ensure that the same scaling factors and offsets 
                from the training data are applied to the validation and test sets.
            col_dict: dictionary of the following format:
                {
                    "numerical_features": ['numerical_col1', 'numerical_col2', ...],   # names of all numerical features
                    "all_features": ['feature_col1', 'feature_col2', ...],             # names of all features in df, including numerical features. make sure to exclude outcomes.
                    "label_cols": ['label_col1', 'label_col2', ...],                   # names of column(s) corresponding to labels
                    "time_col": "Year of diagnosis",                                   # name of column corresponding to timestamp (year, month, etc.)
                    "ID": "Patient ID"                                                 # unique identifier for each patient
                }
            label: name of the column in the dataframe to be used as the label (e.g. "mortality", "MORTALITY_180D")
            model_class: model class to evaluate. Should implement the models.ExpModel interface
            model_name: name of model class (e.g. "LR", "GBDT")
            hyperparam_grid: dictionary containing grid of hyperparameters to search over 
                (e.g. {"C": [0.01, 0.1, 1]})
            training_regime: training regime to use in evaluation of performance over time 
                (e.g. "all_historical", "sliding_window", "all_historical_subsampling")
            initial_train_window: tuple with the first timepoint and last timepoint (inclusive) 
                of the FIRST in-sample time window to be used for training
                (e.g. (1975, 1978) would be a four-year time range.)
                The oldest model is trained on data in this time range, and subsequent models are trained on newer data.
            train_end: the latest timepoint of the entire in-sample time range
            test_end: the latest timepoint of the entire out-of-sample time range
            train_prop: proportion of in-sample data to use as training set
            val_prop: proportion of in-sample data to use as validation set
            test_prop: proportion of in-sample data to use as test set
            window_length: number of timepoints in sliding window.
                If doing sliding window evaluation: should equal train_end_t - train_start_t + 1.
                If doing subsampling with all-historical: this is used to compute the number of samples to 
                    subsample so that it is comparable to the sliding window evaluation. 
                    Thus, set window_length to be the same as used in sliding window evaluation.
            time_unit: unit of time that timepoints are in. Used for display purposes.
            model_folder: path to folder to save trained models. If None, models will not be saved.
        """

        self.dataset_name = dataset_name
        self.df = df
        self.col_dict = col_dict
        self.label = label
        self.model_class = model_class
        self.model_name = model_name
        self.hyperparam_grid = hyperparam_grid
        self.training_regime = training_regime
        self.initial_train_window = initial_train_window
        self.train_end = train_end
        self.test_end = test_end
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.window_length = window_length - 1
        self.time_unit = time_unit
        self.model_folder = model_folder
        self.subsample =  'subsampling' in training_regime

        # Checks for window lengths and initial train windows
        if self.training_regime == 'sliding_window': 
            if self.window_length != self.initial_train_window[1] - self.initial_train_window[0]:
                print(f'Attention: Your initial training window {self.initial_train_window} does not match your window length {self.window_length}.')
        if self.training_regime == 'all_historical_subsampling':
            assert self.initial_train_window[1] - self.window_length >= self.initial_train_window[0], \
                'The number of timepoints in the subsampling window length should at most as large as the training window.'

    def run_experiment(self, seed_list, eval_metric="auc"):
        """Runs an experiment evaluating model performance over time.
        
        Args:
            seed_list: list of random seeds
            eval_metric: name of metric to use for selecting the best model in grid search.
                Should correspond to a key of the dictionary output by ExpModel.evaluate(x, y)

        Returns:
            result_df: dataframe containing results of experiment
            model_info: coefficient information of the trained models, if exists. Otherwise, None.
        """
        model_info = []
        test_results = []
        for seed in tqdm(seed_list, desc="Seed", leave=False):
            for train_end_t in tqdm(range(self.initial_train_window[1], self.train_end + 1), 
                                    desc="Train {self.time_unit}", leave=False):
                ## Calculate left bound of in-sample time range
                if self.training_regime in ["all_historical", "all_historical_subsampling"]:
                    train_start_t = self.initial_train_window[0]
                else: 
                    assert self.training_regime == "sliding_window", "Invalid training regime provided."
                    train_start_t = train_end_t - self.window_length
                
                ## Initialize the dataset
                dataset_object = EotDataset(
                    df_processed = self.df,
                    col_dict = self.col_dict,
                    label=self.label,
                    train_start_t=train_start_t,
                    train_end_t=train_end_t,
                    seed=seed,
                    dataset_name=self.dataset_name,
                    train_prop=self.train_prop,
                    val_prop=self.val_prop,
                    test_prop=self.test_prop,
                    window_length=self.window_length,
                    subsample=self.subsample,
                )
                    
                ## Initialize the evaluator
                evaluator = EotEvaluator(
                    dataset=dataset_object,
                    method=self.training_regime,
                    model=self.model_class,
                    model_name=self.model_name,
                    search_space=self.hyperparam_grid
                )
                
                ## Fit the model, searching over the hyperparam_grid for best validation performance.
                # results are stored in eva.best_results, eva.best_model, eva.best_hparams.
                evaluator.fit_best_model(eval_metric=eval_metric)
                evaluator.save_best_model(model_folder=self.model_folder)  # save model to folder
                
                # get coefficients
                coefs = evaluator.get_coef_dict()
                if coefs is not None:
                    model_info.append(coefs)
                
                # collect and save results on in-sample test sets
                test_results.append(evaluator.in_sample_test())
                
                # collect and save results on out-of-sample test sets
                for test_t in tqdm(range(train_end_t + 1, self.test_end + 1), 
                                      desc=f"Test {self.time_unit}", leave=False):
                    test_results.append(evaluator.out_sample_test(test_t))

        result_df = pd.DataFrame(test_results)
        result_df["staleness"] = result_df["test_end"] - result_df["train_end"]

        if model_info:
            model_info = pd.DataFrame(model_info)
        else:
            model_info = None

        return result_df, model_info
