from sklearn.metrics import make_scorer, f1_score
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.base import clone
import numpy as np


BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"


def train_pred_chain(X_train, y_train, X_dev, y_dev, dev, model):
    gb_columns = [col for col in y_train.columns if col.startswith("GB")]

    base_model =model                   

    chain_model = ClassifierChain(base_model)
    chain_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = chain_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape

    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", dev["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev["language_name"].values)

    return y_pred_df, y_pred_interp, gb_columns, chain_model

def train_pred_indiv(X_train, y_train, X_dev, y_dev, dev, model):
    gb_columns = [col for col in y_train.columns if col.startswith("GB")]
    preds=[]   #list of arrays, 1/feature = (209,95)
    base_model= model

    for col in gb_columns:
        model=clone(base_model)
        model.fit(X_train, y_train[col])
        preds.append(model.predict(X_dev))

    y_pred_df = pd.DataFrame(np.array(preds).T, columns=gb_columns, index=X_dev.index)  #transpose so that (95,209)
    assert y_pred_df.shape == y_dev.shape

    y_pred_interp= y_pred_df.copy().astype(bool)
    y_pred_interp.insert(0, "id", dev["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev["language_name"].values)

    return y_pred_df, y_pred_interp, gb_columns, None
    

def get_param_grid(MODEL=None):
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

def do_grid_search(model_name, X_train, y_train, model):
    print(f"gridSearchCV for model:{model_name}")
    
    base_model = model 
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
    results_df.to_csv(RES_OUTPUT_PATH/ f"{model_name}_gridsearch_log.csv", index=False)

    return grid.best_estimator_, grid.best_params_, grid.best_score_