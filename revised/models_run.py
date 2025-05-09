import pandas as pd
from pathlib import Path
from config_dfs import select_cols, select_dfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.inspection import permutation_importance
from eval import masked_eval

MODEL= "RF"
WITH_NANS= True
USE_FAM= False 
USE_FAM_AUG= False

train, masked_df, dev, masked_positions= select_dfs(with_nans=WITH_NANS)
X_train, y_train, X_dev, y_dev= select_cols(train, masked_df, dev, use_fam=USE_FAM, use_fam_aug=USE_FAM_AUG)
#CHECHS
for col in X_train.columns:

    print(f"{col}: {dev[col].dtypes}")
#print("Train columns:", X_train.columns.tolist())
#print("NaNs in X_train:", X_train.isnull().values.any())

def train_pred(X_train, y_train, X_dev, y_dev, dev):
    gb_columns = [col for col in train.columns if col.startswith("GB")]

    base_model = HistGradientBoostingClassifier(random_state=42)                      #HGB, supports Nan 

    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape

    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", dev["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev["language_name"].values)

    return y_pred_df, y_pred_interp, gb_columns




def main():

    y_pred_df, y_pred_interp, gb_columns=train_pred(X_train, y_train, X_dev, y_dev, dev)
    masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file)
    y_pred_interp.to_csv(OUTPUT_PATH/ f"HGB_fam_m3_output{strategy}.csv", sep="\t", index=False)
    