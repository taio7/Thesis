import pandas as pd
import numpy as np
from pathlib import Path


BASE_PATH = Path(__file__).parent
#DATA_PATH = BASE_PATH / "ready_dfs"
DATA_PATH = BASE_PATH / "final_dfs"
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

    train= pd.read_parquet( path/ "train_aug_ready.parquet")
    masked_df= pd.read_parquet(path / "masked_df_aug_ready.parquet")
    dev= pd.read_parquet(path / "dev_aug_ready.parquet")
    masked_positions= pd.read_parquet(path / "masked_positions_aug_ready.parquet")
    
    return train, masked_df, dev, masked_positions


def select_cols(train, masked_df, dev, use_fam=None, use_fam_aug=None, use_loc=None, use_k_loc_clusters=None, use_db_loc_clusters=None, use_hdb_loc_clusters=None, MODEL=None):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fams_1hot_cols= [col for col in train.columns if col.startswith("fam_")]
    topfams_1hot_cols= [col for col in train.columns if col.startswith("top_")]
    fam_cols = ["language_family"]
    lat_lon_cols= ["language_latitude", "language_longitude"]
    k_cluster=['kmeans_cluster']
    db_cluster=['dbscan_cluster']
    hdb_cluster=['hdbscan_cluster']

    
    if MODEL== "HGB":
        fam_cols= fams_1hot_cols
    else:
        fam_cols = fam_cols
    
    selected= gb_columns.copy()

    if use_fam==True:
        selected+= fam_cols
    if use_fam_aug==True:
        selected += topfams_1hot_cols
    if use_loc==True:
        selected+=lat_lon_cols
    if use_k_loc_clusters==True:
        selected+=k_cluster
    if use_db_loc_clusters==True:
        selected+=db_cluster
    if use_hdb_loc_clusters==True:
        selected+=hdb_cluster
    else:
        selected=selected
    
    X_train= train[selected]
    y_train=train[gb_columns]
    X_dev=masked_df[selected]
    y_dev=dev[gb_columns]

    return X_train, y_train, X_dev, y_dev


