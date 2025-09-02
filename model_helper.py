import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.base import clone
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold



def apply_var_threshold(X_train, X_dev, threshold=0):
    """apply var threshold but then return the df, instead of array"""
    selector = VarianceThreshold(threshold=threshold)
    original_cols=X_train.columns
    X_train_arr = selector.fit_transform(X_train) #selector returns array but need df
    X_dev_arr = selector.transform(X_dev)

    mask = selector.get_support()
    dropped_features = original_cols[~mask]
    kept_features= original_cols[mask]
    print(f"kept {len(kept_features)} input features after selection")
    
    print("dropped features:", dropped_features.tolist())

    X_train_filtered= pd.DataFrame(X_train_arr, columns=kept_features, index=X_train.index)
    X_dev_filtered= pd.DataFrame(X_dev_arr, columns=kept_features, index=X_dev.index)

    return X_train_filtered, X_dev_filtered

def train_pred_multi(X_train, y_train, X_dev, y_dev, dev, model, model_name):
    
  #  if model_name=="HGB":
#    X_train,  X_dev= apply_var_threshold(X_train, X_dev, threshold=0.1)
        
    

    gb_columns = [col for col in y_train.columns if col.startswith("GB")]

    base_model = model                      #HGB, supports Nan 

    multi_model = MultiOutputClassifier(base_model)
    multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = multi_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape
    

    dev_reset = dev.reset_index(drop=True)
    y_pred_interp = y_pred_df.reset_index(drop=True).copy().astype(bool)

    y_pred_interp.insert(0, "id", dev_reset["id"].values)
    y_pred_interp.insert(1, "language_name", dev_reset["language_name"].values)

    return y_pred_df, y_pred_interp, gb_columns, multi_model

def train_pred_chain(X_train, y_train, X_dev, y_dev, dev, model, model_name):
    
    #does not support nans, use with HGB with encoded data
    gb_columns = [col for col in y_train.columns if col.startswith("GB")]

    base_model =model                   

    chain_model = ClassifierChain(base_model)
    chain_model.fit(X_train, y_train)

#predict, convert the array back to df 
    y_pred = chain_model.predict(X_dev)
    y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)
    assert y_pred_df.shape == y_dev.shape
    dev_reset = dev.reset_index(drop=True)
    y_pred_interp = y_pred_df.reset_index(drop=True).copy().astype(bool)

    y_pred_interp.insert(0, "id", dev_reset["id"].values)
    y_pred_interp.insert(1, "language_name", dev_reset["language_name"].values)
    
    return y_pred_df, y_pred_interp, gb_columns, chain_model

def train_pred_indiv(X_train, y_train, X_dev, y_dev, dev, model, model_name):
    gb_columns = [col for col in y_train.columns if col.startswith("GB")]
    
    
    preds=[]   #list of arrays, 1/feature = (209,95)
    fitted_models = []
    base_model= model

    for col in gb_columns:
        model=clone(base_model)
        model.fit(X_train, y_train[col])
        fitted_models.append(model)
        preds.append(model.predict(X_dev))

    y_pred_df = pd.DataFrame(np.array(preds).T, columns=gb_columns, index=X_dev.index)  #transpose so that (95,209)
    assert y_pred_df.shape == y_dev.shape

    dev_reset = dev.reset_index(drop=True)
    y_pred_interp = y_pred_df.reset_index(drop=True).copy().astype(bool)

    y_pred_interp.insert(0, "id", dev_reset["id"].values)
    y_pred_interp.insert(1, "language_name", dev_reset["language_name"].values)
    return y_pred_df, y_pred_interp, gb_columns, fitted_models

def most_freq_baseline(X_train, y_train, X_dev, dev, save_path=None):
    gb_columns= [col for col in y_train.columns if col.startswith("GB")]

    preds= {
        col: [y_train[col].mode(dropna=True)[0]] * len(X_dev)
        for col in gb_columns
    }
    y_pred_df = pd.DataFrame(preds, index=X_dev.index)
    dev_reset = dev.reset_index(drop=True)
    y_pred_interp = y_pred_df.reset_index(drop=True).copy().astype(bool)
    
    y_pred_interp.insert(0, "id", dev_reset["id"].values) #add cols for interpretation and checking
    y_pred_interp.insert(1, "language_name", dev_reset["language_name"].values)
    if save_path:
        y_pred_interp.to_csv(save_path, index=False)
    return y_pred_df, y_pred_interp, gb_columns

    
