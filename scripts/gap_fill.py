import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"




def data_load():
    df_train = pd.read_csv(DATA_PATH / "train.tsv", sep="\t")
    df_dev = pd.read_csv(DATA_PATH / "dev.tsv", sep="\t")
    df_test = pd.read_csv(DATA_PATH / "test_gold_rand.tsv", sep="\t")
    gb_columns = [col for col in df_train.columns if col.startswith("GB")] #use from train, all have the same column headers
    
    return df_train, df_dev, df_test, gb_columns


#fill missing values with most common
             #gb, the filled, where mode is taken from 
def mode_gap_fill(cols, df, from_df):
    for col in cols:
        mode=from_df[col].mode(dropna=True)  #find the most common value per col (not NAN)
        if not mode.empty:
            df[col]= df[col].fillna(mode[0]).astype("boolean")
    return df 

def all_f_fill(cols, df, value=False):
    for col in cols:
        df[col]= df[col].fillna(value).astype("boolean")
    return df

def all_t_fill(cols, df, value=True):
    for col in cols:
        df[col]= df[col].fillna(value).astype("boolean")
    return df

def no_fill(cols, df):
    df[cols]= df[cols].astype("boolean")
    return df