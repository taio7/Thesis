from sklearn.metrics import make_scorer, f1_score
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.base import clone
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from collections import OrderedDict, Counter
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import shap
from shap import KernelExplainer, Explainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt


BASE_PATH = Path(__file__).parent
PARAM_PATH = BASE_PATH / "params"

skf=StratifiedKFold(n_splits=3)

def get_param_grid_indiv(MODEL=None):
    #need estimator__ since models are wrapped with multioutput
    if MODEL== "RF":
        param_grid= {"n_estimators":[100, 200, 300], #n of decision trees, more prob better
                     "max_depth": [None, 10, 20], #how deep a tree grows, small=less overfit
                     "min_samples_split": [20, 30, 40], #higher= more reg
                     "max_features": [0.5, 1.0, "sqrt"]}
    elif MODEL== "KN":
        param_grid= {"n_neighbors": [3, 5, 7, 14], #exp more better but wasnt 
                     "weights": ["distance", "uniform"], #distacne better
                     "metric": ["hamming", "jaccard", "euclidean", "manhattan"]
                     }
    elif MODEL== "MLP":
        param_grid= {"hidden_layer_sizes": [50, 100, 200],
                     "solver":["lbfgs", "adam"], 
                     "alpha" : [0.05, 0.1],
                     "learning_rate": ["constant", "adaptive"],
                     "max_iter": [200, 300, 500],
                     "early_stopping":[True, False]
                     }
    elif MODEL== "HGB":
        param_grid= {"learning_rate": [0.01, 0.05],  #smaller prob better for sparse data: step size shrinkage 
                     "max_iter": [200, 300], #higher to go with smaller lr 
                     "max_leaf_nodes":[15, None], #tree complexity, prob better smaller: binary values,  no need for complexity
                     "warm_start":[True],
                     "min_samples_leaf": [20, 30], #each leaf represents more data, prob better
                     "l2_regularization": [0.0, 1.0] #reg on leaf weights penalize large change per pred 
                     }
    elif MODEL=="LGB":
        param_grid = {"learning_rate": [0.01, 0.05, 0.1],
                      "n_estimators": [100, 200, 300],
                      "max_depth": [-1, 10, 20],
                      "num_leaves": [15, 31, 63],
                      "min_child_samples": [10, 20, 30],
                      "subsample": [0.7, 0.9, 1.0],
                      "colsample_bytree": [0.7, 0.9, 1.0],
                      "is_unbalance": [True],
                      "verbosity":[-1]}
    
    return param_grid

def get_param_grid_multi(MODEL=None):
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
        param_grid= {"estimator__hidden_layer_sizes": [70, 200],
                     "estimator__solver":["lbfgs"], 
                     "estimator__alpha" : [0.05, 0.1],
                     "estimator__learning_rate": ["constant"],
                     "estimator__max_iter": [200, 300],
                     "estimator__early_stopping":[True]
                     }
        #{'estimator__alpha': 0.1, 'estimator__early_stopping': True, 'estimator__hidden_layer_sizes': 100, 'estimator__learning_rate': 'constant', 'estimator__max_iter': 200, 'estimator__solver': 'lbfgs'}
        #89.07632954915364,0.9500657925761088,
        #{'estimator__alpha': 0.1, 'estimator__early_stopping': True, 'estimator__hidden_layer_sizes': 70, 'estimator__learning_rate': 'constant', 'estimator__max_iter': 200, 'estimator__solver': 'lbfgs'}
#best f1 macro 0.9511
    elif MODEL== "HGB":
        param_grid = {"estimator__learning_rate": [0.05],          # Higher = more reactive
                      "estimator__max_iter": [300],                # More boosting rounds
                      "estimator__max_leaf_nodes": [31, 63],             # Larger trees can capture subtler patterns
                      "estimator__min_samples_leaf": [1, 10],             # Smaller leaf = more granular splits
                      "estimator__l2_regularization": [0.0, 0.1],            # Less regularization = more sensitivity
                      "estimator__interaction_cst": [None],                  # Leave open; only use if you want feature limits
                      "estimator__early_stopping": [False],                  # Disable to let all trees grow
                      #"estimator__scoring": ['loss', 'accuracy'],            # Optional: allows internal validation mode change
                      }
    


        """param_grid= {"estimator__learning_rate": [0.01, 0.05],  #smaller prob better for sparse data: step size shrinkage 
                     "estimator__max_iter": [300, 500], #higher to go with smaller lr 
                     "estimator__max_leaf_nodes":[15, 60], #tree complexity, prob better smaller: binary values,  no need for complexity
                     "estimator__warm_start":[True],
                     "estimator__min_samples_leaf": [5, 15], #each leaf represents more data, prob better
                     "estimator__l2_regularization": [0.0, 1.0] #reg on leaf weights penalize large change per pred 
                     }"""
    
    elif MODEL == "LGB":
        param_grid = {"estimator__learning_rate": [0.01, 0.05],
                      "estimator__n_estimators": [100, 200],
                      "estimator__max_depth": [-1, 10],
                      "estimator__num_leaves": [31, 64],
                      "estimator__min_child_samples": [20, 50],
                      'estimator__is_unbalance': [True],
                      "estimator__verbosity":[-1]}

    
    return param_grid


def do_grid_search(model_name, X_train, y_train, model, file_name):
    print(f"gridSearchCV for model:{model_name}")
    
    base_model = model 
    multi_model = MultiOutputClassifier(base_model)
    param_grid = get_param_grid_multi(model_name)
 
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
        cv=skf,         #n of cross validation folds
        n_jobs=-1,    #all CPU cores
        verbose=2    #log progress 
    )
    print("NaNs in X_train", X_train.isnull().any().any())
    print("NaNs in y_train", y_train.isnull().any().any())
    
    valid_cols = [col for col in y_train.columns if y_train[col].notna().sum() > 0 and y_train[col].nunique() > 1]
    y_train = y_train[valid_cols]
    if model_name=="HGB":
        y_train = y_train.fillna(0).astype(bool)
    grid.fit(X_train, y_train)

    print(f"best params for {model_name}: {grid.best_params_}")
    print(f"best f1 macro {grid.best_score_:.4f}")
    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(PARAM_PATH/ f"{model_name}_{file_name}_gridsearch_log.csv", index=False)

    return grid.best_estimator_, grid.best_params_, grid.best_score_

def get_overall_best_params(results, top_n=4):
    param_set= [frozenset(params.items()) for params, _ in results.values()]
    count=Counter(param_set)
    top_n_param_sets=count.most_common(top_n)
    #overall_best= dict(count.most_common(1)[0][0])
    top_n_dicts = [dict(param_set) for param_set, _ in top_n_param_sets]
    return top_n_dicts

def do_halving_search(model_name, X_train, y_train, model, file_name):
    x_t=X_train.copy()
    y_tr=y_train.copy()
    #does not support multioutput, loop feature and find best params for each 
    #get per feat best, save most common best
    gb_columns = [col for col in y_train.columns if col.startswith("GB")]
    results=OrderedDict()
    
    print(f"halvingSearchCV for model:{model_name}")
    
    param_grid = get_param_grid_indiv(model_name)
    skf=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro", zero_division=0) #for class imbalance, levels each feature 
    
    if model_name== "KN":
        metrics= param_grid.get("metric", [])
        if "jaccard" in metrics:
            print("using jaccard= bool conversion")
            X_train = X_train.replace(-1, 0).astype(bool)
            y_train = y_train.replace(-1, 0).astype(bool)
    skipped_cols=[]
    
    for col in gb_columns:
        print(f"\n search for feature {col}")
        y=y_train[col]
        if y.isna().all() or y.nunique() < 2:
            skipped_cols.append(col)
            continue
        search = HalvingGridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,   #which score it is optimizing for
            cv=skf,         #n of cross validation folds
            n_jobs=-1,    #all CPU cores
            verbose=2    #log progress 
        )
        
    
        #y_train = y_train.astype('bool')
        search.fit(x_t, y)
        
        best_params= search.best_params_
        best_score= search.best_score_
        results[col]=(best_params, best_score)
        
    best_params= get_overall_best_params(results)
    pd.DataFrame([best_params]).to_csv(PARAM_PATH/ f"{model_name}_{file_name}_halving_search_log.csv", index=False)
    print(skipped_cols)
    
def sensitivity_scorer_func(estimator, X, y):
    import numpy as np
    import pandas as pd

    X_perturbed = X.copy()
    noise = np.random.normal(scale=0.01, size=X.shape)
    X_perturbed = pd.DataFrame(X_perturbed + noise, columns=X.columns)

    y_pred_orig = estimator.predict(X)
    y_pred_pert = estimator.predict(X_perturbed)

    # For multi-output
    if isinstance(y_pred_orig, np.ndarray) and y_pred_orig.ndim == 2:
        diffs = (y_pred_orig != y_pred_pert).sum(axis=1)
    else:
        diffs = y_pred_orig != y_pred_pert

    return np.mean(diffs)



def hgb_grid_search(model_name, X_train, y_train, model, file_name):
    print(f"gridSearchCV for model:{model_name}")
    
    base_model = model 
    multi_model = MultiOutputClassifier(base_model)
    param_grid = get_param_grid_multi(model_name)

    scoring = {
        'f1_macro': make_scorer(f1_score, average="macro", zero_division=0),
        'sensitivity': sensitivity_scorer_func
    }

    if model_name == "KN":
        metrics = param_grid.get("estimator__metric", [])
        if "jaccard" in metrics:
            print("using jaccard= bool conversion")
            X_train = X_train.replace(-1, 0).astype(bool)
            y_train = y_train.replace(-1, 0).astype(bool)

    valid_cols = [col for col in y_train.columns if y_train[col].notna().sum() > 0 and y_train[col].nunique() > 1]
    y_train = y_train[valid_cols]

    if model_name == "HGB":
        y_train = y_train.fillna(0).astype(bool)

    grid = GridSearchCV(
        estimator=multi_model,
        param_grid=param_grid,
        scoring=scoring,
        refit='f1_macro',   # this decides what best_estimator_ is selected by
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    print("NaNs in X_train", X_train.isnull().any().any())
    print("NaNs in y_train", y_train.isnull().any().any())

    grid.fit(X_train, y_train)

    print(f"best params for {model_name}: {grid.best_params_}")
    print(f"best f1 macro: {grid.cv_results_['mean_test_f1_macro'][grid.best_index_]:.4f}")
    print(f"sensitivity at best: {grid.cv_results_['mean_test_sensitivity'][grid.best_index_]:.4f}")

    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(PARAM_PATH / f"{model_name}_{file_name}_gridsearch_log.csv", index=False)

    return grid.best_estimator_, grid.best_params_, grid.best_score_

def plot_feature_importance(model, X_train, y_train):
    # Get the inner estimator (first tree of multi-output)
    estimator = model.estimators_[0]
    result = permutation_importance(estimator, X_train, y_train.iloc[:, 0], n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=False)

    print("\nTop 10 most important features (by permutation):")
    print(importance_df.head(10))

    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['feature'].head(10), importance_df['importance'].head(10))
    plt.gca().invert_yaxis()
    plt.title('Top 10 Permutation Importances')
    plt.tight_layout()
    plt.show()

    return importance_df


def summarize_input_feature_influence(model, X_train, y_train):
    """
    For MultiOutputClassifier with HGB base, summarize which input features
    are most influential across all predicted output features.
    """
    input_features = X_train.columns
    output_features = y_train.columns

    total_influence = pd.DataFrame(0, index=input_features, columns=output_features)

    for i, estimator in enumerate(model.estimators_):
        output_name = output_features[i]
        explainer = shap.Explainer(estimator, X_train)
        shap_vals = explainer(X_train, check_additivity=False)

        # Sum absolute SHAP values for each input feature
        mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
        total_influence[output_name] = mean_abs_shap

    return total_influence


def plot_partial_dependence(model, X_train, features):
    estimator = model.estimators_[0]  # Use one of the output estimators
    display = PartialDependenceDisplay.from_estimator(
        estimator, X_train, features,
        kind="average", subsample=50, n_jobs=-1
    )
import pandas as pd
import numpy as np
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def explain_individual_models(models, X, y_dev, gb_columns, method="auto", top_k_input=20):
    influence_matrix = pd.DataFrame(index=X.columns, columns=gb_columns)

    for i, (model, feat_name) in enumerate(zip(models, gb_columns)):
        y = y_dev[feat_name]
        if y.nunique() < 2 or y.isnull().all():
            continue  # skip empty or trivial tasks

        try:
            if method == "auto" or method == "shap":
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                influence_matrix[feat_name] = mean_abs_shap
            elif method == "permutation":
                result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
                influence_matrix[feat_name] = result.importances_mean
        except Exception as e:
            print(f"Skipping {feat_name} due to error: {e}")
            continue

    influence_matrix = influence_matrix.fillna(0)

    # Aggregate importance
    feature_global_importance = influence_matrix.mean(axis=1).sort_values(ascending=False)
    top_features = feature_global_importance.head(top_k_input)

    # Heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(influence_matrix.loc[top_features.index], cmap="viridis")
    plt.title("Input Feature Influence on Output Predictions")
    plt.xlabel("Output Features")
    plt.ylabel("Input Features")
    plt.tight_layout()
    plt.show()

    return influence_matrix


