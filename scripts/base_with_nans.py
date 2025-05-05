import pandas as pd
from gap_fill import data_load, mode_gap_fill, no_fill, all_f_fill, all_t_fill, imputer
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


def fill_strat(strategy, df_train, gb_columns):
    #fills the selected gb columns with specified strategy RETURN entire df 
    if strategy== "mode":
        df_train= mode_gap_fill(gb_columns, df_train, df_train)
    elif strategy== "all False":
        df_train= all_f_fill(gb_columns, df_train, value=False)
    elif strategy== "all True":
        df_train= all_t_fill(gb_columns, df_train, value=True)
    elif strategy== "none":
        df_train= no_fill(gb_columns, df_train)
    elif strategy== "impute":
        df_train= imputer(gb_columns, df_train)
    else:
        print("no such fill strategy")
    return df_train

def train_pred(df_train, df_dev_masked, df_dev, gb_columns):
    X_train = df_train[gb_columns] #train on whole and targets are all features 
    y_train = df_train[gb_columns]  

    X_dev = df_dev_masked[gb_columns]
    y_dev = df_dev[gb_columns]
    
#Multi_model=train one classifier per feature 

    base_model = HistGradientBoostingClassifier(random_state=42)                      #HGB, supports Nan 

    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape
    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", df_dev["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", df_dev["language_name"].values)
    
    return y_pred_df, y_pred_interp

#evaluate on all features, for later whne i only have metadata as input 
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
        

#convert to int to calculate accuracy after removing nans otherwise error
    mask_flat= mask_m.values.flatten()

    masked_indices = np.where(mask_flat)[0]# where masked feats are 
    print(f"Number of masked: {len(masked_indices)}")

    y_true_flat = y_dev.values.flatten()[mask_flat].astype(int)# use only masked positions for eval
    y_pred_flat = y_pred_df.values.flatten()[mask_flat].astype(int)
    

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
    df_train, df_dev, df_test, gb_columns = data_load()
    assert not set(df_train['id']).intersection(set(df_dev['id'])) #no same lgs in train and dev 
    assert not set(df_train['id']).intersection(set(df_test['id']))
    assert not set(df_dev['id']).intersection(set(df_test['id'])) 


#ensure boolean types
    df_train[gb_columns] = df_train[gb_columns].astype("boolean")
    df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")

#"mode"  "all False"  "none" "all True"  "impute"

    strategy= "mode"
    df_train= fill_strat(strategy, df_train, gb_columns)  #dont fill dev and mask eval 
    #print(set(df_train['id']).intersection(df_dev['id'])) #check for data leak

    #assert not df_dev[gb_columns].isnull().values.any()
    #assert not df_train[gb_columns].isnull().values.any()
    #print(df_dev.head())
    masked_df_dev, masked_positions= mask_dev(df_dev, gb_columns, mask_ratio= 0.5, seed=30)
    
    masked_df_dev.to_csv(OUTPUT_PATH / f"masked_dev.csv", sep="\t", index=False)  #save 

#TRAIN and PRED 
    df_pred, y_pred_interp= train_pred(df_train, masked_df_dev, df_dev, gb_columns)

    
    
    """for col in gb_columns:  #is the model predicting just one val?
        print(f"{col} - Unique predictions: {df_pred[col].nunique()}")"""

    y_true= df_dev[gb_columns]
    #predictions are made on the masked dev set and then evaluated against the original dev set
    masked_eval(gb_columns, y_true, df_pred, masked_positions, RES_PATH/ f"HGB_m5_results{strategy}.txt")
    y_pred_interp.to_csv(OUTPUT_PATH/ f"HGB_m5_output{strategy}.csv", sep="\t", index=False)
if __name__ == "__main__":
    main()

