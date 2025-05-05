import pandas as pd
from gap_fill_meta import data_load, mode_gap_fill, no_fill, all_f_fill, all_t_fill, imputer
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
RES_PATH = BASE_PATH / "results"
OUTPUT_PATH=  BASE_PATH / "pred_output"

def prep_dfs(df, gb_columns, fam_cols=None, loc_cols=None):
    columns = gb_columns.copy()
    if fam_cols:
        columns += fam_cols
    if loc_cols:
        columns += loc_cols
    return df[columns].copy()
#prep a df that has metadata added+gb_columns, so training input 


def fill_strat(strategy, df_train, df_dev, gb_columns):
    if strategy== "mode":
        df_train= mode_gap_fill(gb_columns, df_train, df_train)
        #df_dev= mode_gap_fill(gb_columns, df_dev, df_dev)
    elif strategy== "all False":
        df_train= all_f_fill(gb_columns, df_train, value=False)
        #df_dev= all_f_fill(gb_columns, df_dev, value=False)
    elif strategy== "all True":
        df_train= all_t_fill(gb_columns, df_train, value=True)
        #df_dev= all_t_fill(gb_columns, df_dev, value=True)
    elif strategy== "none":
        df_train= no_fill(gb_columns, df_train)
        #df_dev= no_fill(gb_columns, df_dev)
    elif strategy== "impute":
        df_train= imputer(gb_columns, df_train)
        #df_dev= imputer(gb_columns, df_dev)
    else:
        print("no such fill strategy")
    return df_train, df_dev

def bool_to_cat(df, gb_columns):
    df_cat= df.copy()
    for col in gb_columns:
        df_cat[col]= df_cat[col].map({True: "True", False: "False"})
        df_cat[col]= df_cat[col].fillna("unk")
    return df_cat

def label_encode_df(df):
    df_enc= df.copy()
    encoders= {}  #save each col encoder in dict to reuse 
    for col in df.columns:
        le= LabelEncoder()
        df_enc[col]= le.fit_transform(df_enc[col]) #one encoder for each feat and transform vals
        encoders[col]= le  #{col:encoder}
    
    return df_enc, encoders

def transform_w_encoder(df, encoders):
    df_enc= df.copy()
    for col in df.columns:
        df_enc[col]= encoders[col].transform(df_enc[col])# use same encoders 
    return df_enc

     
def decode_pred(df_pred, gb_columns, encoders):
    df_dec= df_pred.copy()
    for col in gb_columns:
        le= encoders[col]
        df_dec[col]= le.inverse_transform(df_pred[col])
    return df_dec 

def train_pred(X_train, y_train, X_dev, y_dev, dev_for_metadata):
    
#Multi_model=train one classifier per feature 

    #base_model = RandomForestClassifier(random_state=40)  #extend: separate decision tree per family/area  #baseline
    #base_model =KNeighborsClassifier(n_neighbors=5)                                                        #KN
    base_model =MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)                    #MLP

    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape
    y_pred_df.columns = y_dev.columns #add feature names to pred 
    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", dev_for_metadata["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev_for_metadata["language_name"].values)
    
    return y_pred_df, y_pred_interp

def eval(gb_columns, y_dev, y_pred_df, output_file):
    results=[]
# evaluation 
    for feature in gb_columns:
        re= classification_report(y_dev[feature], y_pred_df[feature], zero_division=0)
        results.append(f"feature: {feature}\n")
        results.append(re+"\n")
        results.append("-" * 40 + "\n")
        

#convert to int to calculate accuracy 
    y_dev_int = y_dev.astype(int)
    y_pred_int = y_pred_df.astype(int)

    sub_acc= accuracy_score(y_dev_int, y_pred_int)
    macro_prec= precision_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    macro_recall= recall_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    weighted_f1= f1_score(y_dev_int, y_pred_int, average="weighted", zero_division=0)
    macro_f1= f1_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    micro_f1=f1_score(y_dev_int, y_pred_int, average="micro", zero_division=0)

    res= (
        f"Subset accuracy:{sub_acc:.3f}\n"
        f"Macro-average precision::{macro_prec:.3f}\n"
        f"Macro-average recall:{macro_recall:.3f}\n"
        f"Weighted F1 score:{weighted_f1:.3f}\n"
        f"Macro-average F1 score:{macro_f1:.3f}\n"
        f"Micro-average F1 score:{micro_f1:.3f}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(res)
        f.writelines(results)
    print(f"eval file saved to {output_file}")

def masked_eval(gb_columns, y_dev, y_pred_df, mask_m, output_file):
    results=[]
# evaluation 
    assert list(y_pred_df.columns) == list(y_dev.columns), "pred and true columns dont match"

    for feature in gb_columns:
        mask = mask_m[feature] #the masked matrix 
        if mask.sum()== 0:
            continue #skip features per lang with nan vlaues 

        y_true = y_dev.loc[mask, feature].astype(int)
        y_pred = y_pred_df.loc[mask, feature].astype(int)

        re= classification_report(y_true, y_pred, zero_division=0)
        results.append(f"feature: {feature}\n")
        results.append(re+"\n")
        results.append("-" * 40 + "\n")
        

#convert to int to calculate accuracy 
    y_dev_int = y_dev.astype(int)
    y_pred_int = y_pred_df.astype(int)
    mask_flat= mask_m.values.flatten()

    masked_indices = np.where(mask_flat)[0]# where masked feats are 
    print(f"Number of masked: {len(masked_indices)}")

    y_true_flat = y_dev_int.values.flatten()[mask_flat]# use only masked positions for eval
    y_pred_flat = y_pred_int.values.flatten()[mask_flat]
    

    sub_acc= accuracy_score(y_true_flat, y_pred_flat)
    macro_prec= precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    macro_recall= recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    weighted_f1= f1_score(y_true_flat, y_pred_flat, average="weighted", zero_division=0)
    macro_f1= f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    micro_f1=f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)

    res= (
        f"Subset accuracy:{sub_acc:.3f}\n"
        f"Macro-average precision::{macro_prec:.3f}\n"
        f"Macro-average recall:{macro_recall:.3f}\n"
        f"Weighted F1 score:{weighted_f1:.3f}\n"
        f"Macro-average F1 score:{macro_f1:.3f}\n"
        f"Micro-average F1 score:{micro_f1:.3f}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(res)
        f.writelines(results)
    print(f"eval file saved to {output_file}")


# only maks where values are not nan 
def mask_dev(df_dev, gb_columns, mask_ratio= 0.2, seed=42):
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
    model= "MLP"
#"mode"  "all False"  "none" "all True"  "impute"
    strategy= "mode"
    df_train, df_dev, df_test, gb_columns, fam_cols = data_load()
    assert not set(df_train['id']).intersection(set(df_dev['id'])) #no same lgs in train and dev 
    assert not set(df_train['id']).intersection(set(df_test['id']))
    assert not set(df_dev['id']).intersection(set(df_test['id'])) 
    print(set(df_dev[fam_cols[0]]) - set(df_train[fam_cols[0]]))


#ensure boolean types
    df_train[gb_columns] = df_train[gb_columns].astype("boolean")
    df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")
#CREATE XTRAIN XDEV FILL AND ENCODE 
    df_train, _= fill_strat(strategy, df_train, df_dev, gb_columns)
#MAP TO CAT
    df_train = bool_to_cat(df_train, gb_columns)
    df_dev= bool_to_cat(df_dev, gb_columns)
    
#BUILD INPUT MATRIX+FAM
    X_train= prep_dfs(df_train, gb_columns, fam_cols=fam_cols, loc_cols=None)
    y_train = df_train[gb_columns]
    X_dev= prep_dfs(df_dev, gb_columns, fam_cols=fam_cols, loc_cols=None)
    y_dev = df_dev[gb_columns]
    

#Combine data so encoders learn all possible values and labels and then split again
    unk_row = pd.DataFrame([["unk"] * len(X_train.columns)], columns=X_train.columns)
    combined_for_enc=pd.concat([X_train, X_dev, unk_row], ignore_index=True)
    all_data_enc, encoders= label_encode_df(combined_for_enc)
    X_train_enc= all_data_enc.iloc[:len(X_train)]
    X_dev_enc= all_data_enc.iloc[len(X_train):-1]

#mAKE SURE SPLIT IS MADE CORRECTLY
    decoded_X_train = X_train_enc.copy()
    decoded_X_dev = X_dev_enc.copy()
    for col in decoded_X_train.columns:
        decoded_X_train[col] = encoders[col].inverse_transform(decoded_X_train[col])
        decoded_X_dev[col] = encoders[col].inverse_transform(decoded_X_dev[col])

# 4. Assert no change in content — comparing string values
    assert X_train.reset_index(drop=True).equals(decoded_X_train.reset_index(drop=True)), "Data leakage or mismatch in X_train!"
    assert X_dev.reset_index(drop=True).equals(decoded_X_dev.reset_index(drop=True)), "Data leakage or mismatch in X_dev!"
    
    #print(set(df_train['id']).intersection(df_dev['id'])) #check for data leak
#TRANSFORM INPUT 
    y_train_enc = transform_w_encoder(y_train, encoders)
    y_dev_enc = transform_w_encoder(y_dev, encoders)
    #assert not df_dev[gb_columns].isnull().values.any()
    #assert not df_train[gb_columns].isnull().values.any()
    #print(df_dev.head())

#MASK DEV 
    masked_df_dev, masked_positions= mask_dev(df_dev, gb_columns, mask_ratio= 0.3, seed=30)  #returns entire df with metadata
    masked_df_dev = bool_to_cat(masked_df_dev, gb_columns)  
    X_masked_dev = prep_dfs(masked_df_dev, gb_columns, fam_cols=fam_cols) #only keep relevant columns
    X_masked_dev_enc = transform_w_encoder(X_masked_dev, encoders) #encode etnire thing
    

#TRAIN and PRED 
    df_pred, y_pred_interp= train_pred(X_train_enc, y_train_enc, X_masked_dev_enc, y_dev_enc, df_dev)
#EVAL
    masked_eval(gb_columns, y_dev_enc, df_pred, masked_positions, RES_PATH/ f"{model}3_fam_results{strategy}.txt")
#DECODE
    y_pred_decoded = decode_pred(df_pred, gb_columns, encoders)
    

    
    y_pred_interp_decoded = y_pred_interp.copy()
    for col in gb_columns:
        y_pred_interp_decoded[col] = y_pred_decoded[col]
    for col in fam_cols:
        y_pred_interp.insert(2, col, df_dev[col].values)

    y_pred_interp_decoded.to_csv(OUTPUT_PATH/ f"{model}3_fam_output{strategy}.csv", sep="\t", index=False)


    #for col in gb_columns:  #is the model predicting just one val?
        #print(f"{col} - Unique predictions: {df_pred[col].nunique()}")

    #predictions are made on the masked dev set and then evaluated against the original dev set

if __name__ == "__main__":
    main()

