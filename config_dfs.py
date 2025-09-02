import pandas as pd
import numpy as np
from pathlib import Path
import pickle

BASE_PATH = Path(__file__).parent
#print("BASE_PATH is:", BASE_PATH.resolve())

#DATA_PATH = BASE_PATH / "ready_dfs"
DATA_PATH = BASE_PATH / "final_dfs"
WITH_NAN_PATH= DATA_PATH/ "with_nans"
NO_NAN_PATH= DATA_PATH/ "no_nans"

"""Selects with or nonans data, creates the input for 
models with args to select columns, controled in 
model script."""

def decode_language_info(dfs, encoder_file):

    with open(encoder_file, "rb") as f:
        encoders = pickle.load(f)

    for df in dfs:
        if "language_name" in df.columns and "language_name" in encoders:
            df["language_name"] = encoders["language_name"].inverse_transform(df["language_name"])
        if "id" in df.columns and "id" in encoders:
            df["id"] = encoders["id"].inverse_transform(df["id"])

    return dfs

def select_dfs(with_nans=None):
    if with_nans==True:
        path= WITH_NAN_PATH
    else:
        path= NO_NAN_PATH

    train= pd.read_parquet( path/ "train_aug_ready.parquet")
    masked_df= pd.read_parquet(path / "masked_df_aug_ready.parquet")
    dev= pd.read_parquet(path / "dev_aug_ready.parquet")
    masked_positions= pd.read_parquet(path / "masked_positions_aug_ready.parquet")
    test_masked_df= pd.read_parquet(path / "test_masked_df_ready.parquet")
    test=pd.read_parquet(path / "test_gold_ready.parquet")
    test_masked_positions = pd.read_parquet(path / "test_masked_positions_ready.parquet")
    dev_to_train= pd.read_parquet(path / "dev_to_train_ready.parquet")

    if with_nans is False:
        encoder_file = path / "label_encoders.pkl"
        train, masked_df, dev, test, dev_to_train = decode_language_info(
            [train, masked_df, dev, test, dev_to_train], encoder_file
        )
    
    return train, masked_df, dev, masked_positions, test_masked_df, test, test_masked_positions, dev_to_train


def select_cols(train, masked_df, dev, test, masked_df_test, dev_to_train, use_fam=None, use_fam_aug=None, 
                use_loc=None, use_k_loc_clusters=None, use_db_loc_clusters=None, use_hdb_loc_clusters=None, 
                use_12k_cl=None, use_12db_cl=None, use_12hdb_cl=None, 
                use_24k_cl=None, use_24db_cl=None, use_24hdb_cl=None,
                MODEL=None, final_run=None):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fams_1hot_cols= [col for col in train.columns if col.startswith("fam_")]
    topfams_1hot_cols= [col for col in train.columns if col.startswith("top_")]
    fam_cols = ["language_family"]
    lat_lon_cols= ["language_latitude", "language_longitude"]
    k_cluster=['kmeans_cluster']
    db_cluster=['dbscan_cluster']
    hdb_cluster=['hdbscan_cluster']
    k12_cl=["12_kmeans_cluster"]
    db12_cl=["12_dbscan_cluster"]
    hdb12_cl=["12_hdbscan_cluster"]
    k24_cl=["24_kmeans_cluster"]
    db24_cl=["24_dbscan_cluster"]
    hdb24_cl=["24_hdbscan_cluster"]
    
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
    if use_12k_cl==True:
        selected+=k12_cl
    if use_12db_cl==True:
        selected+=db12_cl
    if use_12hdb_cl==True:
        selected+=hdb12_cl
    if use_24k_cl==True:
        selected+=k24_cl
    if use_24db_cl==True:
        selected+=db24_cl
    if use_24hdb_cl==True:
        selected+=hdb24_cl
    else:
        selected=selected
    if final_run:
        train_combined = pd.concat([train, dev_to_train], axis=0)
        X_train = train_combined[selected].copy()
        y_train = train_combined[gb_columns].copy()
    else:
        X_train = train[selected].copy()
        y_train = train[gb_columns].copy()

    X_test= masked_df_test[selected].copy()
    y_test= test[gb_columns].copy()
    
    X_dev=masked_df[selected].copy()
    y_dev=dev[gb_columns].copy()

    return X_train, y_train, X_dev, y_dev, X_test, y_test

train, masked_df, dev, masked_positions, test_masked_df, test, test_masked_posistions, dev_to_train= select_dfs(with_nans=False)



path = Path("c:/Users/theod/Desktop/code/grambank/final_dfs/no_nans")
print(list(path.glob("*.parquet")))
