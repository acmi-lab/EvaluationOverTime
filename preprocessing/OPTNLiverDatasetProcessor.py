# This file contains pipeline for OPTN (liver) dataset aimed at mortality prediction. The class is fed by raw liver dataset and output processed dataset

# - input:
#     -- dataset_name: raw OPTN (liver) dataset

# - internal function:
#     -- __read_raw_data(): read raw data information as dataframe
#     -- __read_feature_dict(): read dictionary of features
#     -- __combine_primary_secondary_diagnoses(): helper function to combine primary and secondary diganoses data
#     -- __save_processed_data(): save processed data as pickle file

# - output interface:
#     -- get_raw_data(): get the raw dataset
#     -- get_numerical_nan_feature_list(): get list of numerical features that contain nan
#     -- get_column_set(): get set of features in the dataframe
#     -- get_processed_data(): get dataframe of processed dataset
#     -- get_processed_feature(): get list of features that will be used for evaluation
        
import os
import json
import yaml

import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm.autonotebook import tqdm

class OPTNLiverDatasetProcessor:
    def __init__(self, label, dataset_name="OPTN_liver", keep_first_only=False):
        assert label in ["MORTALITY_1y", "MORTALITY_90D", "MORTALITY_30D", "MORTALITY_180D"]
        
        self.label = label
        self.dataset_name = dataset_name
        self.keep_first_only = keep_first_only
        
        self.data_dir = os.path.abspath("/data/yuwenc/data")
        
        self.df_raw = self.__read_raw_data()
        self.df_processed = self.df_raw.copy()
        
        ## Keep first record if specified
        if self.keep_first_only:
            self.keep = "keep_first"
            self.df_processed = self.df_processed.loc[self.df_processed.groupby("WL_ID_CODE")["WAITING_TIME_UPDATE2"].idxmin()]
        else:
            self.keep = "keep_all"
            
        # Add year information
        self.df_processed["CHG_DATE"] = pd.to_datetime(self.df_processed["CHG_DATE"])
        self.df_processed["YEAR"] = self.df_processed["CHG_DATE"].dt.year
        
        ## Read dictionary of feature list
        self.feature_dict = self.__read_feature_dict()
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
            self.df_processed, new_numerial = self.__process_numerical(self.df_processed, f, **kwargs)
        
        ### Combine diagnosis features
        self.df_processed = self.__combine_primary_secondary_diagnoses()
        
        self.__num_fill_mean(num_cols)
           
        # Assert that all values are numeric and there aren't any NaN / infinite values.
        self.df_processed.apply(
            pd.to_numeric
        )  # This conversion should raise an error if there are strings.
        assert not np.isnan(self.df_processed).any().any()
        assert np.isfinite(self.df_processed).all().all()
        
        ### Reorder column name
        self.df_processed = self.df_processed.reindex(sorted(self.df_processed.columns), axis=1)
        
        self.feature_set = categorical + numerical
        
        ## Create processed feature dict
        self.processed_feature_dict = {
            "processed_numerical_feature": num_cols,
            "processed_all_feature": list(self.df_processed.drop(columns=label_cols).columns),
            "processed_label": ["MORTALITY_1y", "MORTALITY_180D", "MORTALITY_90D", "MORTALITY_30D"],
            "year_label": "YEAR",
            "time_info": None,
            "ID": "WL_ID_CODE"
        }
        
        ## Remove samples that are labelled as -1
        self.df_processed = self.df_processed[self.df_processed[self.label] != -1]
        
        ### Save dataset
        self.__save_processed_data()
    
    def __read_raw_data(self):
        
        raw_data_dir = os.path.join(self.data_dir, self.dataset_name, f"{self.dataset_name}_cate_raw")
        return pd.read_csv(os.path.join(raw_data_dir, f"{self.dataset_name}_cate_raw.csv"), index_col=0)
    
    def __read_feature_dict(self):
        feature_dir = os.path.join(self.data_dir, self.dataset_name, "feature_dict_cate.json")
        
        with open(feature_dir, "r") as f:
            feature_dict = json.load(f)
            f.close()
        
        return feature_dict
    
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
    
    def __combine_primary_secondary_diagnoses(self):
        diags = [c for c in self.df_processed if c.startswith("DGN")]
        for diag in list(set([diag.split('_')[-1] for diag in diags])):
            c = 0
            if ("DGN_TCR_" + diag) in self.df_processed:
                c += self.df_processed["DGN_TCR_" + diag]
            if ("DGN2_TCR_" + diag) in self.df_processed:
                c += self.df_processed["DGN2_TCR_" + diag]
            self.df_processed["DGNC_" + diag] = c.clip(lower=0, upper=1)

        return self.df_processed.drop(
            [c for c in self.df_processed if (c.startswith("DGN_") or c.startswith("DGN2"))], axis=1
        )
    
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
            df[feature] = df[feature].replace("Blank(s)", blank_value)

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
    
    def get_numerical_nan_feature_list(self):
        return self.numerical_nan
    
    def get_column_set(self):
        return self.column_set
    
    def get_processed_data(self):
        return self.processed_data
    
    def get_processed_feature(self):
        return self.feature_set
    
    def get_year_attribute(self):
        return "YEAR"
    
    def get_ID_attribute(self):
        return "WL_ID_CODE"
