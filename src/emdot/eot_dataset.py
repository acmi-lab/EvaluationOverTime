"""EotDataset class.

Given a pre-processed dataset, this class prepares the data to be used in the 
evaluation over time (EOT) pipeline.
"""

import copy
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional

class EotDataset:
    def __init__(self, 
        df_processed: pd.DataFrame,
        col_dict: dict,
        label: str,
        train_start_t: int,
        train_end_t: int, 
        seed: int, 
        dataset_name: str = "SEER_breast", 
        train_prop: float = 0.8, 
        val_prop: float = 0.1, 
        test_prop: float = 0.1, 
        window_length: Optional[int] = None, 
        subsample: bool = False,
        ):
        """Initialize the EotDataset object
        
        Args:
            df_processed: pre-processed dataframe, which includes all of the columns mentioned in col_dict.
                all categorical variables should have been converted to dummies, but numerical values can remain unnormalized.
            col_dict: dictionary of the following format:
                {
                    "numerical_features": ['numerical_col1', 'numerical_col2', ...],   # names of all numerical features
                    "all_features": ['feature_col1', 'feature_col2', ...],             # names of all features in df, including numerical features. make sure to exclude outcomes.
                    "label_cols": ['label_col1', 'label_col2', ...],                   # names of column(s) corresponding to labels
                    "time_col": "Year of diagnosis",                                   # name of column corresponding to timestamp (year, month, etc.)
                    "ID": "Patient ID"                                                 # unique identifier for each patient
                }
            label: name of the column in the dataframe to be used as the label (e.g. "mortality", "MORTALITY_180D")
            train_start_t: starting time point of in-sample time range (inclusive)
            train_end_t: end time point of in-sample time range (inclusive; comes from use of pd.Series.between() function)
            seed: generated seed for dataset spiting and subsampled (to improve reproducibility)
            dataset_name: name of the dataset (e.g. "SEER_breast, OPTN_liver")
            train_prop: proportion of training set in in-sample time range
            val_prop: proportion of validation set in in-sample time range
            test_prop: proportion of in-sample testing set in in-sample time range
            window_length: length of sliding window
                if doing sliding window evaluation: should equal train_end_t - train_start_t.
                if doing subsampling with all-historical: this is used to compute the number of samples to subsample
                    so that it is comparable to the sliding window evaluation. Thus, set window_length to be the same
                    as used in sliding window evaluation.
            subsample: boolean variable to indicate applying subsampling or not
        """
        
        assert train_end_t >= train_start_t
        
        self.df = df_processed.copy()
        self.col_dict = copy.deepcopy(col_dict)

        self.label = label
        self.train_start_t = train_start_t
        self.train_end_t = train_end_t
        self.seed = seed
        self.dataset_name = dataset_name
        self.train_prop = train_prop
        self.val_size = val_prop
        self.test_size = test_prop
        self.window_length = window_length
        self.subsample = subsample
        
        ## Initialize scaler function for normalizing numeric features
        self.scaler = StandardScaler()
        
        ## Filter for in-sample data
        mask = self.df[self.col_dict["time_col"]].between(self.train_start_t, self.train_end_t)
        self.df_in_sample = self.df[mask]
        
        ## Split dataset to train-validation-test
        self.df_train, self.df_val, self.df_test_in_sample = self.__train_val_test_split()
        
        ## Apply subsample if required
        if self.subsample:
            assert(window_length), f"window_length is not provided for subsample"
            
            ### Calculate subsample fraction
            year_start = self.train_end_t - self.window_length
            mask = self.df_in_sample[self.col_dict["time_col"]].between(year_start, self.train_end_t)
            num_sample = self.df_in_sample[mask].shape[0]
            subsample_frac = num_sample * self.train_prop / self.df_train.shape[0]
            

            ### Apply subsampling
            self.__subsample_train_and_val(subsample_frac)
        
        # if time is listed as a feature, remove it
        if self.col_dict["time_col"] in self.col_dict["all_features"]:
            self.col_dict["all_features"].remove(self.col_dict["time_col"])
        if self.col_dict["time_col"] in self.col_dict["numerical_features"]:
            self.col_dict["numerical_features"].remove(self.col_dict["time_col"])
        
        # scale numerical features
        self.df_train = self.__normalize(self.df_train, fit=True)
        self.df_val = self.__normalize(self.df_val)
        self.df_test_in_sample = self.__normalize(self.df_test_in_sample)
    
    def __train_val_test_split(self):
        """Splits in-sample data into training, validation and testing sets.
        
        Returns:
            train, validation, and in-sample test sets
        """
        num_sample = self.df_in_sample.shape[0]
        shuffled_index = np.arange(num_sample)
        np.random.seed(self.seed)
        np.random.shuffle(shuffled_index)

        num_train = int(num_sample * self.train_prop)
        num_val = int(num_sample * self.val_size)

        train_index = shuffled_index[:num_train]
        val_index = shuffled_index[num_train: num_train + num_val]
        test_index = shuffled_index[num_train + num_val:]
        
        df_train = self.df_in_sample.iloc[train_index, :]
        df_val = self.df_in_sample.iloc[val_index, :]
        df_test_in_sample = self.df_in_sample.iloc[test_index, :]
        
        ## Keep the in_sample test data as the last year of in_sample range
        df_test_in_sample = df_test_in_sample[
            df_test_in_sample[self.col_dict["time_col"]] == self.train_end_t
        ]
        assert df_test_in_sample.shape[0] > 0, "No in-sample test record on last year"

        return df_train, df_val, df_test_in_sample

        
    def __subsample_train_and_val(self, subsample_frac: float):
        """Subsamples in-sample training set and validation sets.
        
        Args:
            subsample_frac: fraction of data to be subsampled
        """
        ### Compute subsampling in training set
        num_train_sample = self.df_train.shape[0]
        shuffled_train_index = np.arange(num_train_sample)
        np.random.seed(self.seed)
        np.random.shuffle(shuffled_train_index)
        
        num_train_subsample = int(num_train_sample * subsample_frac)
        train_subsample_index = shuffled_train_index[:num_train_subsample]
        self.df_train = self.df_train.iloc[train_subsample_index, :]
        
        ### Compute subsampling in validation set
        num_val_sample = self.df_val.shape[0]
        shuffled_val_index = np.arange(num_val_sample)
        np.random.seed(self.seed)
        np.random.shuffle(shuffled_val_index)
        
        num_val_subsample = int(num_val_sample * subsample_frac)
        val_subsample_index = shuffled_val_index[:num_val_subsample]
        self.df_val = self.df_val.iloc[val_subsample_index, :]
        
    def __normalize(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ):
        """Normalizes the data using the StandardScaler object in self.scaler.

        Args:
            df: dataframe to normalize
            fit: boolean value to indicate fitting the scaler with the data
                Normally, set to True for training set and set to False for validation and testing set

        Returns:
            df: dataframe after
        """
        numerical_cols = self.col_dict["numerical_features"]
        
        if len(numerical_cols) == 0:
            return df
        
        if fit:
            df.loc[:, numerical_cols] = self.scaler.fit_transform(df.loc[:, numerical_cols])
        else:
            df.loc[:, numerical_cols] = self.scaler.transform(df.loc[:, numerical_cols])
        return df
    
    def __df_to_numpy(self, df: pd.DataFrame):
        """Given the data, extract the features and labels, and return in Numpy arrays.
        
        Args:
            df: dataframe to process

        Returns:
            X: features in Numpy array
            y: labels in Numpy array

        """
        X = df[self.col_dict["all_features"]].to_numpy()
        y = df[self.label].to_numpy()
        return X, y
    
    def __df_feature_label(self, df: pd.DataFrame):
        """Given the data, extracts the features and labels, and return in dataframe.
        
        Args:
            df: dataframe to process

        Returns:
            df_X: features in dataframe
            df_y: labels in dataframe

        """
        df_X = df[self.col_dict["all_features"]]
        df_y = df[self.label]
        return df_X, df_y
    
    def get_train(self, return_df=True):
        """Returns the training set either in dataframe or numpy.
        
        Args:
            return_df: boolean variable indicates returning dataframe or not

        Returns:
            features and labels for training set either in dataframe or numpy
        """
        if return_df:
            return self.__df_feature_label(self.df_train)
        else:
            return self.__df_to_numpy(self.df_train)
        
    def get_val(self, return_df=True):
        """Returns the validation dataset either in dataframe or numpy.

        Args:
            return_df: boolean variable indicates returning dataframe or numpy array

        Returns:
            features and labels for validation set either in dataframe or numpy
        
        """
        if return_df:
            return self.__df_feature_label(self.df_val)
        else:
            return self.__df_to_numpy(self.df_val)
        
    def get_test_in_sample(self, return_df=True):
        """Returns the in-sample testing dataset either in dataframe or numpy.
        
        Args:
            return_df: boolean variable indicates returning dataframe or numpy array

        Returns:
            features and labels for in-sample testing set either in dataframe or numpy
        
        """
        if return_df:
            return self.__df_feature_label(self.df_test_in_sample)
        else:
            return self.__df_to_numpy(self.df_test_in_sample)
        
    def get_test_out_sample(
        self,
        test_start_year: int, 
        test_end_year: Optional[int] = None, 
        return_df: bool = True
    ):
        """Returns the out-of-sample testing dataset either in dataframe or numpy format.
        
        Args:
            test_start_year: starting time point for out-sample test
            test_end_year: end time point for out-sample test
                If None, set to the same as test_start_year
            return_df: boolean variable indicates returning dataframe or numpy array

        Returns:
            features and labels for out-sample testing set either in dataframe or numpy
        """
        if test_end_year is None:
            test_end_year = test_start_year
            
        assert test_start_year > self.train_end_t
        assert test_start_year <= test_end_year

        ## Initialize outsample dataset
        df_test_out_sample = self.df[self.df[self.col_dict["time_col"]].between(test_start_year, test_end_year)]

        ## exclude year
        df_test_out_sample = df_test_out_sample.drop(columns=self.col_dict["time_col"])
    
        ## Normalize data
        df_test_out_sample = self.__normalize(df_test_out_sample)

        if return_df:
            return self.__df_feature_label(df_test_out_sample)
        else:
            return self.__df_to_numpy(df_test_out_sample)
        
    def get_dataset_info(self):
        """Gets basic information of dataset.
        
        Returns:
            dictionary containing dataset name, seed, size of training set and size of validation set
        """
        return {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "train_size": self.df_train.shape[0],
            "val_size": self.df_val.shape[0]
        }
    
    def get_feature_list(self):
        """Gets features to be used in the model.
        
        Returns:
            list of names of features

        """
        return list(self.df_train.drop(columns=self.col_dict["label_cols"]).columns)
