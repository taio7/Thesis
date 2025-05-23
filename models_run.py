import pandas as pd
from pathlib import Path
from config_dfs import select_cols, select_dfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from model_helper import get_param_grid, do_grid_search, train_pred_chain, train_pred_indiv

from eval import masked_eval, see_feature_importance 


MODEL= "HGB"
USE_FAM= False
USE_FAM_AUG= False

BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
PRED_OUTPUT_PATH = BASE_PATH/ "new_pred_output"
output_file = RES_OUTPUT_PATH / f"{MODEL}ind_mr3_results.txt"

#set model variable for base model to be fed into train pred 
models_no_nans= ["RF", "KN", "MLP"]
if MODEL in models_no_nans:
    WITH_NANS=False
else:
    WITH_NANS=True


train, masked_df, dev, masked_positions= select_dfs(with_nans=WITH_NANS)
X_train, y_train, X_dev, y_dev= select_cols(train, masked_df, dev, use_fam=USE_FAM, use_fam_aug=USE_FAM_AUG, MODEL=MODEL)

#for col in masked_df.columns:
    
#    print(f"{col}: {masked_df[col].dtypes}")

#print(masked_positions.index)
#mask_count = (masked_positions == True).sum().sum()
#print("DEBUG: masked positions models.py:", mask_count)

#print("GLOBAL DEBUG mask_m.dtypes.unique():", masked_positions.dtypes.unique())
#CHECHS
"""for col in y_dev.columns:

    print(f"{col}: {y_dev[col].dtypes}")
print(len(y_dev.columns))"""
#print("Train columns:", X_train.columns.tolist())
#print("NaNs in X_train:", X_train.isnull().values.any())
def select_model(name):
    if name== "RF":
        return RandomForestClassifier(random_state=40,      
                                      class_weight="balanced",
                                      n_estimators=200,
                                      min_samples_split=30,
                                      max_features=0.5        #too slow?
                                      )
    elif name== "KN":
        return KNeighborsClassifier(algorithm="auto",
                                     metric="jaccard",
                                     n_neighbors=3,
                                     weights="distance"
                                     )
    elif name== "MLP":
        return MLPClassifier(random_state=42)
    elif name== "HGB":
        return HistGradientBoostingClassifier(random_state=42, 
                                              #l2_regularization=1.0, 
                                              #learning_rate=0.5, 
                                              #max_iter=200, 
                                              #max_leaf_nodes=15, 
                                              #min_samples_leaf=20, 
                                              #warm_start=True
                                              )
    else:
        raise ValueError("wrong model name")


    

def train_pred_multi(X_train, y_train, X_dev, y_dev, dev):
    gb_columns = [col for col in train.columns if col.startswith("GB")]

    base_model = select_model(MODEL)                      #HGB, supports Nan 

    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape

    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", dev["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev["language_name"].values)

    return y_pred_df, y_pred_interp, gb_columns, multi_model


        
def main():

    #y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_multi(X_train, y_train, X_dev, y_dev, dev)
    #y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_chain(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL))
    y_pred_df, y_pred_interp, gb_columns, _=train_pred_indiv(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL))
    #see_feature_importance(multi_model, X_dev, y_dev, gb_columns, masked_positions)
    masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file)
    
    y_pred_interp.to_csv(PRED_OUTPUT_PATH/ f"{MODEL}ind_mr3_output.csv", sep="\t", index=False)
    #best_model, best_params, best_score = do_grid_search(MODEL, X_train, y_train, select_model(MODEL))


if __name__=="__main__":
    main()
