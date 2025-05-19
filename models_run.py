import pandas as pd
from pathlib import Path
from config_dfs import select_cols, select_dfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score

from sklearn.model_selection import GridSearchCV
from eval import masked_eval, see_feature_importance 


MODEL= "RF"
USE_FAM= False
USE_FAM_AUG= False

BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
PRED_OUTPUT_PATH = BASE_PATH/ "new_pred_output"
output_file = RES_OUTPUT_PATH / f"{MODEL}_new_mr3_results.txt"

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
                                              l2_regularization=1.0, 
                                              learning_rate=0.5, 
                                              max_iter=200, 
                                              max_leaf_nodes=15, 
                                              min_samples_leaf=20, 
                                              warm_start=True
                                              )
    else:
        raise ValueError("wrong model name")

def get_param_grid(MODEL=MODEL):
    #need estimator__ since models are wrapped with multioutput
    if MODEL== "RF":
        param_grid= {"estimator__n_estimators":[200], #n of decision trees, more prob better
                     "estimator__max_depth": [None, 10, 20], #how deep a tree grows, small=less overfit
                     "estimator__min_samples_split": [30], #higher= more reg
                     "estimator__max_features": [0.5, 1.0]}
    elif MODEL== "KN":
        param_grid= {"estimator__n_neighbors": [3, 5, 7], #exp more better but wasnt 
                     "estimator__weights": ["distance"], #distacne better
                     "estimator__metric": ["hamming", "jaccard"]
                     }
    elif MODEL== "MLP":
        param_grid= {"estimator__hidden_layer_sizes": [100, 150],
                     "estimator__solver":["lbfgs"], 
                     #"estimator__alpha" : [0.0001, 0.1],
                     #"estimator__learning_rate": ["constant", "adaptive"],
                     "estimator__max_iter": [300]
                     }
    elif MODEL== "HGB":
        param_grid= {"estimator__learning_rate": [0.01, 0.05],  #smaller prob better for sparse data: step size shrinkage 
                     "estimator__max_iter": [200, 300], #higher to go with smaller lr 
                     "estimator__max_leaf_nodes":[15, None], #tree complexity, prob better smaller: binary values,  no need for complexity
                     "estimator__warm_start":[True],
                     "estimator__min_samples_leaf": [20, 30], #each leaf represents more data, prob better
                     "estimator__l2_regularization": [0.0, 1.0] #reg on leaf weights penalize large change per pred 
                     }
    
    return param_grid

def do_grid_search(model_name, X_train, y_train):
    print(f"gridSearchCV for model:{model_name}")
    
    base_model = select_model(model_name)  
    multi_model = MultiOutputClassifier(base_model)
    param_grid = get_param_grid(model_name)
 
    scorer = make_scorer(f1_score, average="macro", zero_division=0) #for class imbalance, levels each feature 
    if model_name== "KN":
        metrics= param_grid.get("estimator__metric", [])
        if "jaccard" in metrics:
            print("using jaccard= bool conversion")
            X_train = X_train.replace(-1, 0).astype(bool)
            y_train = y_train.replace(-1, 0).astype(bool)

    grid = GridSearchCV(
        estimator=multi_model,
        param_grid=param_grid,
        scoring=scorer,   #which score it is optimizing for
        cv=3,         #n of cross validation folds
        n_jobs=-1,    #all CPU cores
        verbose=2    #log progress 
    )
    print("NaNs in X_train", X_train.isnull().any().any())
    print("NaNs in y_train", y_train.isnull().any().any())
    

    y_train = y_train.astype('bool')
    grid.fit(X_train, y_train)

    print(f"best params for {model_name}: {grid.best_params_}")
    print(f"best f1 macro {grid.best_score_:.4f}")
    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(RES_OUTPUT_PATH/ f"{MODEL}_gridsearch_log.csv", index=False)

    return grid.best_estimator_, grid.best_params_, grid.best_score_
    

def train_pred(X_train, y_train, X_dev, y_dev, dev):
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

    y_pred_df, y_pred_interp, gb_columns, multi_model=train_pred(X_train, y_train, X_dev, y_dev, dev)
    #see_feature_importance(multi_model, X_dev, y_dev, gb_columns, masked_positions)
    masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file)
    
    y_pred_interp.to_csv(PRED_OUTPUT_PATH/ f"{MODEL}_new_mr3_output.csv", sep="\t", index=False)
    #best_model, best_params, best_score = do_grid_search(MODEL, X_train, y_train)


if __name__=="__main__":
    main()
