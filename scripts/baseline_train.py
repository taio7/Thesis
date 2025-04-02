import pandas as pd
from gap_fill import data_load, mode_gap_fill, no_fill, all_f_fill, all_t_fill
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"
RES_PATH = BASE_PATH / "results"



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
    else:
        print("no such fill strategy")
    return df_train, df_dev



def train_pred(df_train, df_dev, gb_columns):
    X_train = df_train[gb_columns] #train on whole and targets are all features 
    y_train = df_train[gb_columns]  

    X_dev = df_dev[gb_columns]
    y_dev = df_dev[gb_columns]

#Multi_model=train one classifier per feature 

    base_model = RandomForestClassifier(random_state=40)  #extend: separate decision tree per family/area
    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)

    assert y_pred_df.shape == y_dev.shape
    return y_pred_df

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

"""Subset accuracy 0.11578947368421053
Macro-average precision: 0.9215662290416539
Macro-average recall: 0.789148386775071
Weighted F1 score: 0.9654978871211904
Macro-average F1 score: 0.8297660066324019
Micro-average F1 score: 0.9736165349080865"""

def main():
    df_train, df_dev, df_test, gb_columns = data_load()
    assert not set(df_train['id']).intersection(set(df_dev['id'])) #no same lgs in train and dev 
    assert not set(df_train['id']).intersection(set(df_test['id']))
    assert not set(df_dev['id']).intersection(set(df_test['id'])) 


#ensure boolean types
    df_train[gb_columns] = df_train[gb_columns].astype("boolean")
    df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")

#"mode"  "all False"  "none" "all True"
    strategy= "none"
    df_train, df_dev= fill_strat(strategy, df_train, df_dev, gb_columns)
    
    #assert not df_dev[gb_columns].isnull().values.any()
    #assert not df_train[gb_columns].isnull().values.any()
    #print(df_dev.head())
    df_pred= train_pred(df_train, df_dev, gb_columns)
    
    y_true= df_dev[gb_columns]

    eval(gb_columns, y_true, df_pred, RES_PATH/ f"baseline_results_{strategy}.txt")

if __name__ == "__main__":
    main()