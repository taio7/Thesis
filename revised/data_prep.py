import pandas as pd 
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq 
"""Load data, 
save train, dev_gold and test parquet files 
in preped dir."""

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH= BASE_PATH/ "preped"

def data_load():
    df_train = pd.read_csv(DATA_PATH / "train.tsv", sep="\t")
    df_dev = pd.read_csv(DATA_PATH / "dev.tsv", sep="\t")
    df_test = pd.read_csv(DATA_PATH / "test_gold_rand.tsv", sep="\t")
    gb_columns = [col for col in df_train.columns if col.startswith("GB")] #use from train, all have the same column headers
    fam_cols= ["language_family"] 
    lat_lon_cols= ["language_latitude", "language_longitude"]
    return df_train, df_dev, df_test, gb_columns, fam_cols, lat_lon_cols




def main():
    df_train, df_dev, df_test, gb_columns, fam_cols, lat_lon_cols = data_load()
    assert not set(df_train['id']).intersection(set(df_dev['id'])) #no same lgs in train and dev 
    assert not set(df_train['id']).intersection(set(df_test['id']))
    assert not set(df_dev['id']).intersection(set(df_test['id'])) 


#ensure boolean types
    df_train[gb_columns] = df_train[gb_columns].astype("boolean")
    df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")
    df_train[fam_cols] = df_train[fam_cols].astype("category")
    df_dev[fam_cols] = df_dev[fam_cols].astype("category")

#"mode"  "all False"  "none" "all True"  "impute"

    #strategy= "mode"
    #
    #print(set(df_train['id']).intersection(df_dev['id'])) #check for data leak
    
    #assert not df_dev[gb_columns].isnull().values.any()
    #assert not df_train[gb_columns].isnull().values.any()
    #
    # print(df_dev.head())
#dont mask here so i do it in the model script, and save the files there?
    #masked_dev, masked_positions= mask_dev(df_dev, gb_columns, mask_ratio= 0.3, seed=30)
    
    #masked_df_dev.to_csv(OUTPUT_PATH / f"masked_dev.csv", sep="\t", index=False)  #save 
    
    df_train.to_parquet(OUTPUT_PATH / "train.parquet")
    df_dev.to_parquet(OUTPUT_PATH / "dev_gold.parquet")
    df_test.to_parquet(OUTPUT_PATH / "test.parquet")

    """table2 = pq.read_table('example.parquet')
    pq.read_table('example.parquet', columns=['one', 'three']) 

    to maintain additional idx column data
    pq.read_pandas"""

    print("saved files to", OUTPUT_PATH)

if __name__ == "__main__":
    main()