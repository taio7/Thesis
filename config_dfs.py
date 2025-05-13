import pandas as pd
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "ready_dfs"
WITH_NAN_PATH= DATA_PATH/ "with_nans"
NO_NAN_PATH= DATA_PATH/ "no_nans"

"""Selects with or nonans data, creates the input for 
models with args to select columns, controled in 
model script."""


def select_dfs(with_nans=None):
    if with_nans==True:
        path= WITH_NAN_PATH
    else:
        path= NO_NAN_PATH

    train= pd.read_parquet( path/ "train_ready.parquet")
    masked_df= pd.read_parquet(path / "masked_df_ready.parquet")
    dev= pd.read_parquet(path / "dev_gold_ready.parquet")
    masked_positions= pd.read_parquet(path / "masked_positions_ready.parquet")
    
    return train, masked_df, dev, masked_positions 

def select_cols(train, masked_df, dev, use_fam=None, use_fam_aug=None):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fam_cols = ["language_family"]
    
    selected= gb_columns.copy()

    if use_fam==True:
        selected+= fam_cols
    if use_fam_aug==True:
        selected_columns += [col for col in train.columns if col.startswith("IE_") or col.startswith("AUSTRO_")]
    else:
        selected=selected
    
    X_train= train[selected]
    y_train=train[gb_columns]
    X_dev=masked_df[selected]
    y_dev=dev[gb_columns]

    return X_train, y_train, X_dev, y_dev
