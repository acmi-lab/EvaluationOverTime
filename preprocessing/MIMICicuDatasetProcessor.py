import os
import json
import yaml

import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm.autonotebook import tqdm

class MIMICicuDatasetProcessor:
    def __init__(self, label, dataset_name="MIMIC_icu", keep_first_only=True):
        self.label = label
        self.dataset_name = dataset_name
        self.keep_first_only = keep_first_only
        
        self.data_dir = os.path.abspath("/data/yuwenc/data")
        
        self.df_raw = self.__read_raw_data()
        self.df_processed = self.df_raw.copy()
        
        if keep_first_only:
            self.keep = "keep_first"
        else:
            self.keep = "keep_all"
        
        ## Remove data in 2021
        self.df_processed = self.df_processed[~(self.df_processed["admit_year"] == 2021)]
        
        ## Convert datetime column
        self.df_processed["dod"] = pd.to_datetime(self.df_processed["dod"], errors='coerce')
        self.df_outtime = pd.to_datetime(self.df_processed["outtime"])
        
        ## Read feature yaml
        self.feature_yaml = self.__read_feature_yaml()
        
        categorical = self.__dict_to_tuples(self.feature_yaml['categorical_features'])
        numerical = self.__dict_to_tuples(self.feature_yaml['numerical_features'])
        label = self.__dict_to_tuples(self.feature_yaml['labels'])
        
        ## Select only the columns we need.
        cat_cols = list(self.feature_yaml['categorical_features'].keys())
        num_cols = list(self.feature_yaml['numerical_features'].keys())
        label_cols = list(self.feature_yaml['labels'].keys())
        self.df_processed = self.df_processed[cat_cols + num_cols + label_cols]
        
        ## Process categorical features
        for f, kwargs in tqdm(categorical, desc='Process categorical features'):
            self.df_processed = self.__process_categorical(self.df_processed, f, **kwargs)
        
        ## Process numerical features
        for f, kwargs in tqdm(numerical, desc='Process numerical features'):
            self.df_processed, _ = self.__process_numerical(self.df_processed, f, **kwargs)
            
        ## Fill nan with mean
        self.__num_fill_mean(num_cols)
            
        ## Create prediction targets
        self.df_processed = self.__process_labels(self.df_processed, label_cols)
        
        self.df_processed.apply(
            pd.to_numeric
        )  # This conversion should raise an error if there are strings.
        assert not np.isnan(self.df_processed).any().any()
        assert np.isfinite(self.df_processed).all().all()
        
        ### Reorder column name
        self.df_processed = self.df_processed.reindex(sorted(self.df_processed.columns), axis=1)
        
        self.feature_set = categorical
        
        ## Create processed feature dict
        self.processed_feature_dict = {
            "processed_numerical_feature": num_cols,
            "processed_all_feature": list(self.df_processed.drop(columns=["los_3", "los_7", "in_icu_mortality"], errors='ignore').columns),
            "processed_label": ["los_3", "los_7", "in_icu_mortality"],
            "year_label": "admit_year",
            "time_info": None,
            "ID": "subject_id"
        }
        
        ## Save dataset
        self.__save_processed_data()
        
    def __read_raw_data(self):
        raw_data_dir = os.path.join(self.data_dir, self.dataset_name, f"{self.dataset_name}_cate_raw")
        return pd.read_csv(os.path.join(raw_data_dir, f"{self.dataset_name}_cate_raw.csv"), index_col=0)
    
    def __read_feature_yaml(self):
        
        feature_dir = os.path.join(self.data_dir, self.dataset_name, "feature_yaml_cate.yaml")
        return yaml.load(open(feature_dir))
        
    def __dict_to_tuples(self, d):
        ret = []
        for k, v in d.items():
            if v is None:
                v = {}
            ret.append((k, v))
        return ret
    
    def __process_categorical(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Processes categorical features by converting them to one-hot-encoded columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.

        feature : str
            Name of categorical column to process.
        """
        # Make sure column is string type.
        df[feature] = df[feature].astype(str)

        ohe_feature = pd.get_dummies(df[feature], prefix=feature)
        ohe_feature = self.__drop_constant(ohe_feature)

        assert len(ohe_feature.columns.tolist()) == len(set(ohe_feature.columns.tolist()))

        df = pd.concat([df, ohe_feature], axis=1)
        df = df.drop(columns=feature)

        return df
    
    def __drop_constant(self, df):
        """
        Drop any constant-value columns.
        """
        # print(df.columns.tolist())
        return df.loc[:, (df != df.iloc[0]).any()]
    
    def __process_numerical(
        self,
        df: pd.DataFrame,
        feature: str,
        special_values: List[int] = [],
        blank_value: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Processes numerical features by separating continuous values from special encoding
        values, and creating one-hot columns for the special encodings.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.

        feature : str
            Name of numerical column to process.

        special_values : List[int], default = []
            List of numerical values which have special significance (and shouldn't be
            treated in a continuous fashion). A one-hot column will be created for each of
            these special encoding values.

        blank_value : Optional[int], default = None
            Integer value that corresponds to a missing entry. If a value is provided here,
            it should also be included in the `special_values` argument as well.
        """
        # Sometimes a missing value is encoded as the string 'Blank(s),' so we manually convert it to
        # the corresponding number specified by the SEER documentation.
        if blank_value:
            df[feature] = df[feature].replace("Missing", blank_value)

        # Convert the column to numeric in case it isn't already.
        df[feature] = pd.to_numeric(df[feature])

        # Create new col with special values only, and np.nan otherwise (get_dummies will ignore nan).
        df_special_vals_only = df[feature].where(
            df[feature].isin(special_values), other=np.nan
        )
        df_special_vals_ohe = pd.get_dummies(df_special_vals_only, prefix=feature)
        df_special_vals_ohe = self.__drop_constant(df_special_vals_ohe)
        
        # Replace the original feature column with a new version where the special values
        # are set to 0.
        df[feature] = df[feature].where(~df[feature].isin(special_values), other=0)

        df = pd.concat([df, df_special_vals_ohe], axis=1)

        return df, df_special_vals_ohe.columns.values
    
    def __process_labels(self, df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
        
        if self.label == "los_3":
            df[self.label] = np.where(df["los"] > 3, 1, 0)
        if self.label == "los_7":
            df[self.label] = np.where(df["los"] > 7, 1, 0)
        if self.label == "in_icu_mortality":
            df[self.label] = np.where((~df["dod"].isna()) & ((self.df_outtime - df["dod"]) <= pd.Timedelta(days=1)) & ((self.df_outtime - df["dod"]) >= pd.Timedelta(days=0)), 1, 0)
        
        df = df.drop(columns=label_cols)
        
        return df
    
    def __num_fill_mean(self, num_cols):
        for f in num_cols:
            mean_tmp = self.df_processed[f].mean(skipna=True)
            self.df_processed[f] = self.df_processed[f].fillna(mean_tmp)
    
    def __save_processed_data(self):
        
        processed_data_dir = os.path.join(self.data_dir, self.dataset_name, f"{self.dataset_name}_cate_processed")
        
        processed_feature_dir = os.path.join(processed_data_dir, f"{self.dataset_name}_cate_feature_{self.label}_{self.keep}.json")
        with open(processed_feature_dir, "w+") as f:
            json.dump(self.processed_feature_dict, f)
            f.close()

        self.df_processed.to_pickle(os.path.join(processed_data_dir, f"{self.dataset_name}_cate_processed_{self.label}_{self.keep}.pkl")) 
    
    def get_raw_data(self):
        return self.df_raw
    
    def get_processed_data(self):
        return self.processed_data
    
    def get_processed_feature(self):
        return self.feature_set