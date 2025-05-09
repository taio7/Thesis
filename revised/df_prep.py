from gap_fill_meta import mode_gap_fill, no_fill, all_f_fill, all_t_fill, imputer, fill_strat
import pandas as pd 
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq 
from sklearn.preprocessing import LabelEncoder
import pickle 

"""Load data from parquet files, and save two versions:
with nans/no nans encoded: (train, masked_dev, dev, masked_pos)
Save encoders to later decode predictions for eval"""
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "preped"
OUTPUT_PATH= BASE_PATH/ "ready_dfs"
WITH_NAN_PATH= OUTPUT_PATH/ "with_nans"
NO_NAN_PATH= OUTPUT_PATH/ "no_nans"


"""def aug_with_selected_fams():
    returns the df with extra columns eg IE one hot encoded
    and is then passed into preprocess"""


def preprocess_with_nans(train, dev, test, strategy= None, mask_ratio= 0.3, seed=42):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fam_cols = ["language_family"]

    train= fill_strat("mode", train, gb_columns)  #dont fill dev and mask eval
    assert not train[gb_columns].isnull().values.any()
    train[gb_columns] = train[gb_columns].astype("boolean")
    dev[gb_columns] = dev[gb_columns].astype("boolean")
    train[fam_cols] = train[fam_cols].astype("category")
    dev[fam_cols] = dev[fam_cols].astype("category") 

    masked_df, masked_positions= mask_dev(dev, gb_columns, mask_ratio=mask_ratio, seed=seed)


    train.to_parquet(WITH_NAN_PATH / "train_ready.parquet")
    masked_df.to_parquet(WITH_NAN_PATH / "masked_df_ready.parquet")
    dev.to_parquet(WITH_NAN_PATH / "dev_gold_ready.parquet")
    masked_positions.to_parquet(WITH_NAN_PATH / "masked_positions_ready.parquet")
    #test.to_parquet(WITH_NAN_PATH / "test_ready.parquet")

    print(f"Saved preprocessed data to {WITH_NAN_PATH}")

def bool_to_cat(df, gb_columns):
    #turns boolean to str and fills gaps with "unk"
    df_cat= df.copy()
    for col in gb_columns:
        df_cat[col]= df_cat[col].map({True: "True", False: "False"})
        df_cat[col]= df_cat[col].fillna("unk")
    return df_cat

def label_encode_df(df, lat_lon_cols):
    df_enc= df.copy()
    encoders= {}  #save each col encoder in dict to reuse 
    for col in df.columns:
        if col in lat_lon_cols:
            continue
        le= LabelEncoder()
        df_enc[col]= le.fit_transform(df_enc[col]) #one encoder for each feat and transform vals
        encoders[col]= le  #{col:encoder}
    
    return df_enc, encoders

def transform_w_encoder(df, encoders):
    #encodes columns that the encoder dict has, returns the encoded df 
    df_enc= df.copy()
    for col in df.columns:
        if col in encoders:
            df_enc[col]= encoders[col].transform(df_enc[col])# use same encoders 
    return df_enc

def preprocess_no_nans(train, dev, test, strategy="mode", mask_ratio=0.3, seed=30):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fam_cols = ["language_family"]
    lat_lon_cols= ['language_latitude', 'language_longitude']

    train= fill_strat("mode", train, gb_columns)  #dont fill dev and mask eval
    assert not train[gb_columns].isnull().values.any()

    train[gb_columns] = train[gb_columns].astype("boolean")
    dev[gb_columns] = dev[gb_columns].astype("boolean")
    train[fam_cols] = train[fam_cols].astype("category")
    dev[fam_cols] = dev[fam_cols].astype("category")

    #MAP TO CAT
    df_train_cat = bool_to_cat(train, gb_columns)
    df_dev_cat= bool_to_cat(dev, gb_columns)
    

#Combine data so encoders learn all possible values and labels and then split again
    unk_row = pd.DataFrame([["unk"] * len(df_train_cat.columns)], columns=df_train_cat.columns)
    combined_for_enc=pd.concat([df_train_cat, df_dev_cat, unk_row], ignore_index=True)
#exclude lat-lon from encoding and transforming

    all_data_enc, encoders= label_encode_df(combined_for_enc, lat_lon_cols)
    
    df_train_enc= all_data_enc.iloc[:len(df_train_cat)]
    df_dev_enc= all_data_enc.iloc[len(df_train_cat):-1]

    masked_df, masked_positions= mask_dev(dev, gb_columns, mask_ratio=mask_ratio, seed=seed)
    masked_df_dev = bool_to_cat(masked_df, gb_columns)
    masked_df_dev_enc= transform_w_encoder(masked_df_dev, encoders)

    df_train_enc.to_parquet(NO_NAN_PATH / "train_ready.parquet")
    masked_df_dev_enc.to_parquet(NO_NAN_PATH / "masked_df_ready.parquet")
    df_dev_enc.to_parquet(NO_NAN_PATH / "dev_gold_ready.parquet")
    masked_positions.to_parquet(NO_NAN_PATH / "masked_positions_ready.parquet")
    #test.to_parquet(NO_NAN_PATH / "test_ready.parquet")

    print(f"Saved enc preprocessed data to {NO_NAN_PATH}")
    return encoders

def mask_dev(df_dev, gb_columns, mask_ratio=None, seed=None):
    np.random.seed(seed) #
    mask_df= df_dev.copy() #copy whole, need metadata for interpretable in prediction df
    mask_rec_for_eval= pd.DataFrame(False, index=df_dev.index, columns=gb_columns) #create to store the df of masked for evaluation
    for col in gb_columns:
        use_indx= mask_df[col].dropna().index.tolist() #list of indices on non empty vals
        n_mask= int(len(use_indx)*mask_ratio) #number of indices that will be masked
        mask_indx= np.random.choice(use_indx, size=n_mask, replace=False) #randomize number of selected indices and make them Fasle 
        mask_df.loc[mask_indx, col]= np.nan #locate the indexed feature and replacw with 'nan
        mask_rec_for_eval.loc[mask_indx, col] = True #marking the masked positions
    return mask_df, mask_rec_for_eval 

def main():
    train = pd.read_parquet(DATA_PATH / "train.parquet")
    dev = pd.read_parquet(DATA_PATH / "dev_gold.parquet")
    test = pd.read_parquet(DATA_PATH / "test.parquet")

    preprocess_with_nans(train, dev, test, strategy="mode", mask_ratio=0.3, seed=30)

    encoders= preprocess_no_nans(train, dev, test, strategy="mode", mask_ratio=0.3, seed=30)

    with open(NO_NAN_PATH/ "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    """read: 
    with open(NO_NAN_PATH / "label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)"""

if __name__ == "__main__":
    main()