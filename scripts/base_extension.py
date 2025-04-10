import pandas as pd
from gap_fill import data_load, mode_gap_fill, no_fill, all_f_fill, all_t_fill, imputer
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
RES_PATH = BASE_PATH / "results"
OUTPUT_PATH=  BASE_PATH / "pred_output"



def fill_strat(strategy, df_train, df_dev, gb_columns):
    if strategy== "mode":
        df_train= mode_gap_fill(gb_columns, df_train, df_train)
        df_dev= mode_gap_fill(gb_columns, df_dev, df_dev)
    elif strategy== "all False":
        df_train= all_f_fill(gb_columns, df_train, value=False)
        df_dev= all_f_fill(gb_columns, df_dev, value=False)
    elif strategy== "all True":
        df_train= all_t_fill(gb_columns, df_train, value=True)
        df_dev= all_t_fill(gb_columns, df_dev, value=True)
    elif strategy== "none":
        df_train= no_fill(gb_columns, df_train)
        df_dev= no_fill(gb_columns, df_dev)
    elif strategy== "impute":
        df_train= imputer(gb_columns, df_train)
        df_dev= imputer(gb_columns, df_dev)
    else:
        print("no such fill strategy")
    return df_train, df_dev


def train_pred(df_train, df_dev, gb_columns):
    X_train = df_train[gb_columns] #train on whole and targets are all features 
    y_train = df_train[gb_columns]  

    X_dev = df_dev[gb_columns]
    y_dev = df_dev[gb_columns]

#Multi_model=train one classifier per feature 

    #base_model = RandomForestClassifier(random_state=40)  #extend: separate decision tree per family/area  #baseline
    #base_model =KNeighborsClassifier(n_neighbors=5)                                                        #KN
    #base_model =MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)                    #MLP
    base_model = HistGradientBoostingClassifier(random_state=40)                                            #HGB, supports Nan 

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

def masked_eval(gb_columns, y_dev, y_pred_df, output_file):
    results=[]
# evaluation 
   
    for feature in gb_columns:
        mask = ~y_dev[feature].isna() #not ydev rows with nan value 
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
    mask= y_dev_int.notna()

    y_true_flat = y_dev_int[mask].values.flatten()
    y_pred_flat = y_pred_int[mask].values.flatten()

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
    df_train, _= fill_strat(strategy, df_train, df_dev, gb_columns)  #dont fill dev and mask eval 
    print(set(df_train['id']).intersection(df_dev['id'])) #check for data leak

    #assert not df_dev[gb_columns].isnull().values.any()
    #assert not df_train[gb_columns].isnull().values.any()
    #print(df_dev.head())

    #ensure boolean types
    df_train[gb_columns] = df_train[gb_columns].astype("boolean")
    df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")


    df_pred, y_pred_interp= train_pred(df_train, df_dev, gb_columns)
    
    
    """for col in gb_columns:  #is the model predicting just one val?
        print(f"{col} - Unique predictions: {df_pred[col].nunique()}")"""

    y_true= df_dev[gb_columns]

    masked_eval(gb_columns, y_true, df_pred, RES_PATH/ f"HGB_results_nonfilldev{strategy}.txt")
    y_pred_interp.to_csv(OUTPUT_PATH/ f"HGB_output_{strategy}.csv", sep="\t", index=False)
if __name__ == "__main__":
    main()

