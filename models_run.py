import pandas as pd
from pathlib import Path
from config_dfs import select_cols, select_dfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from model_helper import do_halving_search, get_param_grid_multi, do_grid_search, train_pred_chain, train_pred_indiv

from eval import masked_eval, see_feature_importance 


MODEL= "HGB"
USE_FAM= False
USE_FAM_AUG= False
USE_LOC= False
USE_K_LOC_CLUSTERS=True
USE_DB_LOC_CLUSTERS=False
USE_HDB_LOC_CLUSTERS=False

BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
CLUSTERS_PATH=RES_OUTPUT_PATH/ "clusters"

PRED_OUTPUT_PATH = BASE_PATH/ "new_pred_output"
pred_file = PRED_OUTPUT_PATH/ f"{MODEL}_k_mr3_output.csv"
output_file = CLUSTERS_PATH / f"{MODEL}_k_mr3_results.txt"

#set model variable for base model to be fed into train pred 
models_no_nans= ["RF", "KN", "MLP"]
if MODEL in models_no_nans:
    WITH_NANS=False
else:
    WITH_NANS=True


train, masked_df, dev, masked_positions= select_dfs(with_nans=WITH_NANS)
X_train, y_train, X_dev, y_dev= select_cols(train, masked_df, 
                                            dev, use_fam=USE_FAM, 
                                            use_fam_aug=USE_FAM_AUG, 
                                            use_loc=USE_LOC, 
                                            use_k_loc_clusters=USE_K_LOC_CLUSTERS, 
                                            use_db_loc_clusters=USE_DB_LOC_CLUSTERS, 
                                            use_hdb_loc_clusters=USE_HDB_LOC_CLUSTERS, MODEL=MODEL)

for col in X_train.columns:
    
    print(f"{col}: {X_train[col].dtypes}")

print("NaNs in X_train", X_train.isnull().any().any())
print("NaNs in y_train", y_train.isnull().any().any())
print("NaNs in X_dev", X_dev.isnull().any().any())
print("NaNs in y_dev", y_dev.isnull().any().any())

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
                                      #class_weight="balanced",
                                      #n_estimators=200,
                                      #min_samples_split=30,
                                      #max_features=0.5
                                      )
    elif name== "KN":
        return KNeighborsClassifier(algorithm="auto",  #copes withvery nonlinesr pockets of different lgs. small amount of neihgbors= loc makes good results, pockets of exceptions kn is good, adding clustering that info is destroyed 
                                     #metric="jaccard",  #turns everything to bool
                                     #n_neighbors=3,
                                     #weights="distance"
                                     )
    elif name== "MLP":
        return MLPClassifier(random_state=42,
                             #alpha=0.001,
                             #early_stopping=False,
                             #hidden_layer_sizes=100,
                             #learning_rate="constant",
                             #max_iter=200,
                             #solver="lbfgs"
                             )
    
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
#multioutput
    y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_multi(X_train, y_train, X_dev, y_dev, dev)

#classifier chain
    #y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_chain(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL))

#individual   
    #y_pred_df, y_pred_interp, gb_columns, _=train_pred_indiv(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL))

#RUN PRED
    
    masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file)
    
    y_pred_interp.to_csv(pred_file, sep="\t", index=False)

#PARAM SEARCH
    #best_model, best_params, best_score = do_grid_search(MODEL, X_train, y_train, select_model(MODEL))
    #do_halving_search(MODEL, X_train, y_train, select_model(MODEL))
#FEATURE IMPORTANCE MULTI 
    #see_feature_importance(multi_model, X_dev, y_dev, gb_columns, masked_positions)

if __name__=="__main__":
    main()