import pandas as pd
from pathlib import Path
from config_dfs import select_cols, select_dfs
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from model_helper import train_pred_chain, train_pred_indiv, train_pred_multi, most_freq_baseline
from param_search import do_grid_search, do_halving_search, hgb_grid_search, plot_feature_importance, summarize_input_feature_influence, plot_partial_dependence, explain_individual_models
from eval import masked_eval, see_feature_importance, compute_global_permutation_importance 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

 
MODEL= "KN"
USE_FAM= False  #raw family column
USE_FAM_AUG=False  #top 6 1hot encoded 
USE_LOC= False 
USE_K_LOC_CLUSTERS=False
USE_DB_LOC_CLUSTERS=False
USE_HDB_LOC_CLUSTERS=False
USE_12K_CLUSTERS=False
USE_12DB_CLUSTERS=False
USE_12HDB_CLUSTERS=False
USE_24K_CLUSTERS=False
USE_24DB_CLUSTERS=False
USE_24HDB_CLUSTERS=False

 
BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
CLUSTERS_PATH=RES_OUTPUT_PATH/ "clusters"

FAM_LOC= RES_OUTPUT_PATH/ "famloc"
FAM_PATH=RES_OUTPUT_PATH/ "fam"
LOC_PATH= RES_OUTPUT_PATH/"loc"
PARAMS_TRY= RES_OUTPUT_PATH/"try_params"
BASELINE_PATH=RES_OUTPUT_PATH/"baseline"
LAST_PATH= RES_OUTPUT_PATH/"last" #with optim params
TEST_PATH= RES_OUTPUT_PATH/ "test"

PRED_OUTPUT_PATH = BASE_PATH/ "new_pred_output"
pred_file = PRED_OUTPUT_PATH/ f"{MODEL}_bbest_mr3_output.csv"
#output_file = CLUSTERS_PATH / f"{MODEL}_results.txt"
output_file = LAST_PATH / f"{MODEL}_bbest_mr3_results.txt"

#set model variable for base model to be fed into train pred 
models_no_nans= ["RF", "KN", "MLP"]
if MODEL in models_no_nans:
    WITH_NANS=False
else:
    WITH_NANS=True

"""def run_models_with_clusters():
    models=["RF", "KN", "MLP", "HGB"]"""


train, masked_df, dev, masked_positions, test_masked_df, test, test_masked_posistions, dev_to_train= select_dfs(with_nans=WITH_NANS)
X_train, y_train, X_dev, y_dev, X_test, y_test= select_cols(train, masked_df, dev, test, test_masked_df, dev_to_train,use_fam=USE_FAM, 
                                            use_fam_aug=USE_FAM_AUG, 
                                            use_loc=USE_LOC, 
                                            use_k_loc_clusters=USE_K_LOC_CLUSTERS, 
                                            use_db_loc_clusters=USE_DB_LOC_CLUSTERS, 
                                            use_hdb_loc_clusters=USE_HDB_LOC_CLUSTERS, 
                                            use_12k_cl=USE_12K_CLUSTERS, use_12db_cl=USE_12DB_CLUSTERS, use_12hdb_cl=USE_12HDB_CLUSTERS, 
                                            use_24k_cl=USE_24K_CLUSTERS, use_24db_cl=USE_24DB_CLUSTERS, use_24hdb_cl=USE_24HDB_CLUSTERS,
                                            MODEL=MODEL, final_run=False)

gb_columns = [col for col in y_train.columns if col.startswith("GB")]
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
                                      class_weight="balanced",
                                      n_estimators=200,
                                      min_samples_split=30,
                                      max_features=1,
                                      max_depth=20
                                    
                                      
                                      #class_weight="balanced",
                                      #n_estimators=300,
                                      #min_samples_split=20,
                                      #max_features=0.5
                                      )
    elif name== "KN":
        return KNeighborsClassifier(algorithm="auto", 
                                     metric="hamming",  #hamming
                                     n_neighbors=3,
                                     weights="distance"
                                     )

        """algorithm="auto",  #copes withvery nonlinesr pockets of different lgs. small amount of neihgbors= loc makes good results, pockets of exceptions kn is good, adding clustering that info is destroyed 
                metric="jaccard",  #turns everything to bool 
                    n_neighbors=3, #3 best for raw loc, 
                    weights="distance"""
    elif name== "MLP":
        return MLPClassifier(random_state=42,
                             alpha=0.1,
                             early_stopping=True,
                             hidden_layer_sizes=70,                 
                             max_iter=300,
                             solver="lbfgs",
                             learning_rate="constant"
                             )
    
    elif name== "HGB":
        return HistGradientBoostingClassifier(random_state=42, 
                                              #l2_regularization=1.0, 
                                              #learning_rate=0.05, 
                                              #max_iter=200, 
                                              #max_leaf_nodes=15, 
                                              #min_samples_leaf=20, 
                                              #warm_start=True
                                              )
    
    else:
        raise ValueError("wrong model name")
    

    
def main():
#multioutput

    y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_multi(X_train, y_train, X_dev, y_dev, dev, model=select_model(MODEL), model_name=MODEL)
    #all_importances= see_feature_importance(multi_model, X_dev, y_dev, gb_columns, masked_positions)
#classifier chain
    #y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred_chain(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL), model_name=MODEL)

#individual   
    #y_pred_df, y_pred_interp, gb_columns, indiv_model=train_pred_indiv(X_train, y_train, X_dev, y_dev, dev, select_model(MODEL), model_name=MODEL)
    #all_importances= see_feature_importance(indiv_model, X_dev, y_dev, gb_columns, masked_positions)

#baseline

    #y_pred_df, y_pred_interp, gb_columns= most_freq_baseline(X_train, y_train, X_test, test, save_path=pred_file)
    #masked_eval(gb_columns, y_test, y_pred_df, test_masked_posistions, output_file)
#RUN PRED
    masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file, test_run=False, language_families=None)
    #masked_eval(gb_columns, y_test, y_pred_df, test_masked_posistions, output_file, test_run=True, language_families=test["language_family"])
    y_pred_interp.to_csv(pred_file, sep="\t", index=False)

#PARAM SEARCH
    #best_model, best_params, best_score = do_grid_search(MODEL, X_train, y_train, select_model(MODEL), file_name="famlocskf")
    #do_halving_search(MODEL, X_train, y_train, select_model(MODEL), file_name="all")
    """best_model, best_params, best_score = hgb_grid_search(model_name="HGB",
                                                         X_train=X_train,
                                                         y_train=y_train,
                                                         model=HistGradientBoostingClassifier(),
                                                         file_name="sensitivity"
                                                         )"""
#EXPLAIN FEAT IMPORTACNE
    #compute_global_permutation_importance(indiv_model, X_dev, y_dev, gb_columns)

    
if __name__=="__main__":
    main()