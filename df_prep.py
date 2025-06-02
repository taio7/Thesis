from gap_fill_meta import mode_gap_fill, no_fill, all_f_fill, all_t_fill, imputer, fill_strat
import pandas as pd 
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq 
from sklearn.preprocessing import LabelEncoder
import pickle 
from augment_dfs import kmeans_clusters, other_clusters


"""Load data from parquet files, and save two versions:
with nans/no nans encoded: (train, masked_dev, dev, masked_pos)
Save encoders to later decode predictions for eval"""
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "preped"
OUTPUT_PATH= BASE_PATH/ "final_dfs"
WITH_NAN_PATH= OUTPUT_PATH/ "with_nans"
NO_NAN_PATH= OUTPUT_PATH/ "no_nans"
TEMP= BASE_PATH/ "temp"

def save_files(path, train_aug, dev_aug, masked_df_aug, masked_positions):
    train_aug.to_parquet(path / "train_aug_ready.parquet")
    dev_aug.to_parquet(path / "dev_aug_ready.parquet")
    masked_df_aug.to_parquet(path / "masked_df_aug_ready.parquet")
    masked_positions.to_parquet(path / "masked_positions_aug_ready.parquet")
    print(f"Saved preprocessed data to {path}")

def aug_with_selected_fams(df, train, fam_cols):
    """returns the df with extra columns eg IE one hot encoded, 
    takes 6 most populated families from train, keeps lgfam original column"""
    df=df.copy()
    #print(df[fam_cols])  #returns df with lgfam column, [95r,1]
    #print(df["language_family"])  #returns series [95r]
#take top 6 families from train
    top_fams= train[fam_cols[0]].value_counts().nlargest(6) #6 top populated families 
    top_fams_idx= top_fams.index.tolist() 

    if "OTHER" not in df["language_family"].cat.categories:
        df["language_family"] = df["language_family"].cat.add_categories(["OTHER"])
    #placeholder aug df, if it meets condition return it, else fill non top with other 
    df["augm"]=df["language_family"].apply(lambda x: x if x in top_fams_idx else "OTHER")
    fams_1hot= pd.get_dummies(df["augm"], prefix="top_fams", dtype=bool)
    print(fams_1hot.columns.tolist())
    df=pd.concat([df, fams_1hot], axis=1)
    df.drop(columns=["augm"], inplace=True)
    return df 

def one_hot_enc_all_fams(train, dev, fam_cols):
    train_only_fams= train[fam_cols[0]].copy()  #series, idx family 
    #hierarchical index, keys as outermost level 
    combined= pd.concat([train, dev], keys=["train", "dev"])
    combined_1hot_enc= pd.get_dummies(combined, columns=fam_cols, prefix="fam", dtype=bool)
    
    train= combined_1hot_enc.loc["train"].copy()
    dev= combined_1hot_enc.loc["dev"].copy()
    
    for i, fam in train_only_fams.items():
        fam_col= f"fam_{fam}"
        assert fam_col in train.columns
        assert train.at[i, fam_col]== True

    return train, dev 


def preprocess_with_nans(train, dev, test, strategy= None, mask_ratio= 0.3, seed=42):
    gb_columns = [col for col in train.columns if col.startswith("GB")]
    fam_cols = ["language_family"]


    train= fill_strat("mode", train, gb_columns)  #dont fill dev and mask eval
    assert not train[gb_columns].isnull().values.any()
    train[gb_columns] = train[gb_columns].astype("boolean")
    dev[gb_columns] = dev[gb_columns].astype("boolean")
    train[fam_cols] = train[fam_cols].astype("category")
    dev[fam_cols] = dev[fam_cols].astype("category") 
    #print(train.index)
    #print(dev.index)
    #aug_with_selected_fams(train, fam_cols)
    #dev_top_fams=aug_with_selected_fams(dev, train, fam_cols)
    
    train_base= train.copy()
    dev_base= dev.copy()

    train_1hot_fams, dev_1hot_fams= one_hot_enc_all_fams(train, dev, fam_cols)
    train_top_1hot_fams= aug_with_selected_fams(train, train, fam_cols)
    dev_top_1hot_fams= aug_with_selected_fams(dev, train, fam_cols)
    
    fams_1hot_cols= [col for col in train_1hot_fams.columns if col.startswith("fam_")]
    topfams_1hot_cols= [col for col in train_top_1hot_fams.columns if col.startswith("top_")]
   

#add lost family column after one hot encoding, for checks  
    train_1hot_fams["language_family"] = train_base["language_family"]
    dev_1hot_fams["language_family"] = dev_base["language_family"]

    train_aug = pd.concat([train, train_1hot_fams[fams_1hot_cols], train_top_1hot_fams[topfams_1hot_cols]], axis=1)
    dev_aug = pd.concat([dev, dev_1hot_fams[fams_1hot_cols], dev_top_1hot_fams[topfams_1hot_cols]], axis=1)


#CHECHS
    base_cols = train_base.columns.tolist()
    
    new_cols = fams_1hot_cols + topfams_1hot_cols
    
    assert train_aug[base_cols].equals(train_base[base_cols])
    assert dev_aug[base_cols].equals(dev_base[base_cols])

    assert train_aug.index.equals(train_base.index)
    assert dev_aug.index.equals(dev_base.index)
#apply mask to aug and base 

    masked_df, masked_positions= mask_dev(dev, gb_columns, mask_ratio=mask_ratio, seed=seed)
    masked_df_aug, masked_positions_aug= mask_dev(dev_aug, gb_columns, mask_ratio=mask_ratio, seed=seed)
    return train_aug, dev_aug, masked_df_aug, masked_positions_aug
    


def bool_to_cat(df, gb_columns):
    #turns boolean to str and fills gaps with "unk"
    df_cat= df.copy()
    for col in gb_columns:
        df_cat[col]= df_cat[col].map({True: "True", False: "False"})
        df_cat[col]= df_cat[col].fillna("unk")
    return df_cat

def label_encode_df(df, lat_lon_cols, topfams_1hot_cols):
    df_enc= df.copy()
    encoders= {}  #save each col encoder in dict to reuse 
    for col in df.columns:
        if col in lat_lon_cols or col in topfams_1hot_cols:
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

    train_base= train.copy()                               #has no nans
    dev_base= dev.copy()                                   #has nans
    train_top_1hot_fams= aug_with_selected_fams(train, train, fam_cols)  #adds 7 cols (top fams+other), returns full df
    dev_top_1hot_fams= aug_with_selected_fams(dev, train, fam_cols)
    topfams_1hot_cols= [col for col in train_top_1hot_fams.columns if col.startswith("top_")]
    #MAP TO CAT, only gb cols
    df_train_cat = bool_to_cat(train, gb_columns)
    df_dev_cat= bool_to_cat(dev, gb_columns)
    

#Combine data so encoders learn all possible values and labels per columns and then split again
    unk_row = pd.DataFrame([["unk"] * len(df_train_cat.columns)], columns=df_train_cat.columns)
    combined_for_enc=pd.concat([df_train_cat, df_dev_cat, unk_row], ignore_index=True)
#exclude lat-lon and top1hotfams from encoding and transforming

    all_data_enc, encoders= label_encode_df(combined_for_enc, lat_lon_cols, topfams_1hot_cols)
    
    df_train_enc= all_data_enc.iloc[:len(df_train_cat)]   #all columns int except lat/lon and topfams
    df_dev_enc= all_data_enc.iloc[len(df_train_cat):-1]
    df_dev_enc.index = dev.index
#add bool topfam columns to encoded dfs
    train_aug= pd.concat([df_train_enc, train_top_1hot_fams[topfams_1hot_cols]], axis=1)
    dev_aug= pd.concat([df_dev_enc, dev_top_1hot_fams[topfams_1hot_cols]],axis=1)
    
#convert topfamcols to int for consistency    
    train_aug[topfams_1hot_cols]=train_aug[topfams_1hot_cols].astype(int)
    dev_aug[topfams_1hot_cols]=dev_aug[topfams_1hot_cols].astype(int)


    masked_df, masked_positions= mask_dev(dev, gb_columns, mask_ratio=mask_ratio, seed=seed)
    masked_df_dev = bool_to_cat(masked_df, gb_columns)
    masked_df_dev_enc= transform_w_encoder(masked_df_dev, encoders)
    
#make masked aug 
    masked_df_aug_enc = pd.concat([masked_df_dev_enc, dev_top_1hot_fams[topfams_1hot_cols].astype(int)], axis=1)
    #after resplitting dev from combined idx change, so realign 
    
    df_dev_enc.index = dev.index  # maintain original alignment
    masked_positions.index = df_dev_enc.index
    assert df_dev_enc.index.equals(masked_positions.index), "index mismatch between dev and mask"
    assert df_train_enc.index.equals(masked_df_dev_enc.index) == False, "train and masked_dev should not have same index"
    print(masked_positions.values.sum())
#CHECK augmented 
    masked_positions.index = dev_aug.index
    assert dev_aug.index.equals(masked_positions.index), "index mismatch between dev and mask"
    assert train_aug.index.equals(masked_positions.index) == False, "train and masked_dev should not have same index"
    print(masked_positions.values.sum())
    assert list(train_aug.columns) == list(dev_aug.columns)
    
    
    return encoders, train_aug, dev_aug, masked_df_aug_enc, masked_positions

def mask_dev(df_dev, gb_columns, mask_ratio=None, seed=None):
    """takes in full df_dev with Obj/float*2/obj/ cat/ GB=bools, + <NA>
    and makes mask_df= meta+gb(f/t/masked as <NA> in selected indx)
    and masked_rec_for_eval= gb(all Fasle except masked positions=True)"""
    np.random.seed(seed) 
    mask_df= df_dev.copy() #copy whole, need metadata for interpretable in prediction df
    #print("Column dtypes in mask_dev:")
    #print(mask_df.dtypes)
    #print(mask_df)
    mask_rec_for_eval= pd.DataFrame(False, index=df_dev.index, columns=gb_columns, dtype=bool) #all False df 
    for col in gb_columns:
        use_indx= mask_df[col].dropna().index.tolist() #list of indices on non empty vals
        n_mask= int(len(use_indx)*mask_ratio) #number of indices that will be masked
        mask_indx= np.random.choice(use_indx, size=n_mask, replace=False) #randomize number of selected indices and make them Fasle 
        #  HOW many t/f per feature is masked 
        #original_vals = df_dev.loc[mask_indx, col]
        #print(f"{col} - masked values: {original_vals.value_counts(dropna=False).to_dict()}")
        
        
        mask_df.loc[mask_indx, col]= np.nan #locate the indexed feature and replacw with 'nan
        mask_rec_for_eval.loc[mask_indx, col] = True #marking the masked positions
    print(f"all masked cells: {mask_rec_for_eval.values.sum()}")
    #print("masked_positions index :", mask_rec_for_eval.index)

    return mask_df, mask_rec_for_eval 

def add_clusters(train, masked_dev, k_kwargs=None, db_kwargs=None, hdb_kwargs=None):
    train_kclustered, masked_df_kclustered=kmeans_clusters(train, masked_dev, method="kmeans", **(k_kwargs or {}))
    train_kdb_clustered, masked_df_kdb_clustered=other_clusters(train_kclustered, masked_df_kclustered, method="dbscan", **(db_kwargs or {}))
    train_kdbh_clustered, masked_df_kdbh_clustered=other_clusters(train_kdb_clustered, masked_df_kdb_clustered, method="hdbscan", **(hdb_kwargs or {}))
    return train_kdbh_clustered, masked_df_kdbh_clustered

def main():
    train = pd.read_parquet(DATA_PATH / "train.parquet")
    dev = pd.read_parquet(DATA_PATH / "dev_gold.parquet")
    test = pd.read_parquet(DATA_PATH / "test.parquet")

    wn_train_aug, wn_dev_aug, wn_masked_df_aug, wn_masked_positions_aug= preprocess_with_nans(train, dev, test, strategy="mode", mask_ratio=0.3, seed=30)

    encoders, train_aug, dev_aug, masked_df_aug_enc, masked_positions= preprocess_no_nans(train, dev, test, strategy="mode", mask_ratio=0.3, seed=30)

    with open(NO_NAN_PATH/ "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    k_kwargs= {"n_clusters":3, "random_state": 42}
    db_kwargs= {"eps":9.2, #max neighbor radius distance
                 "min_samples":5, #higher for denser clusters
                 "metric":'euclidean', 
                 "metric_params":None, 
                 "algorithm":'auto', 
                 "leaf_size":30, 
                 "p":None, 
                 "n_jobs":-1}
    hdb_kwargs= {"min_cluster_size": 25,
                "min_samples": None,
                "cluster_selection_method": "eom"}
    
    wn_train_kdbh_clustered, wn_masked_df_kdbh_clustered=add_clusters(wn_train_aug, wn_masked_df_aug, k_kwargs=k_kwargs, db_kwargs=db_kwargs, hdb_kwargs=hdb_kwargs)
    train_kdbh_clustered, masked_df_kdbh_clustered=add_clusters(train_aug, masked_df_aug_enc, k_kwargs=k_kwargs, db_kwargs=db_kwargs, hdb_kwargs=hdb_kwargs)
    metadata_cols=  [col for col in train.columns if not col.startswith("GB")]
    print(wn_train_kdbh_clustered.columns)
    print(wn_masked_df_kdbh_clustered.columns)
    print(train_kdbh_clustered.columns)
    print(masked_df_kdbh_clustered.columns)

    save_files(WITH_NAN_PATH, wn_train_kdbh_clustered, wn_dev_aug, wn_masked_df_kdbh_clustered, wn_masked_positions_aug)
    save_files(NO_NAN_PATH, train_kdbh_clustered, dev_aug, masked_df_kdbh_clustered, masked_positions)
    """read: 
    with open(NO_NAN_PATH / "label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)"""

if __name__ == "__main__":
    main()