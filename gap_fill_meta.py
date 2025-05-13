import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor




#fill missing values with most common
             #gb, the filled, where mode is taken from 
def mode_gap_fill(cols, df, from_df):
    for col in cols:
        mode=from_df[col].mode(dropna=True)  #find the most common value per col, per GB feature (not NAN)
        if not mode.empty:
            df[col]= df[col].fillna(mode[0]).astype("boolean")
    return df 

def all_f_fill(cols, df, value=False):
    for col in cols:
        df[col]= df[col].fillna(value).astype("boolean")
    return df

def all_t_fill(cols, df, value=True):
    for col in cols:
        df[col]= df[col].fillna(value).astype("boolean")
    return df
#need to adjust model to ignore missing values 
def no_fill(cols, df):
    df[cols]= df[cols].astype("boolean")
    return df

def imputer(cols, df): #modeling each feature with missing values as a function of other features 
    x= df[cols].astype("float")
    imputer= IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
    x_imputed= imputer.fit_transform(x)
    df[cols] = pd.DataFrame(x_imputed, columns=cols).round().astype(bool)
    return df

def fill_strat(strategy, df_train, gb_columns):
    #fills the selected gb columns with specified strategy RETURN entire df 
    if strategy== "mode":
        df_train= mode_gap_fill(gb_columns, df_train, df_train)
    elif strategy== "all False":
        df_train= all_f_fill(gb_columns, df_train, value=False)
    elif strategy== "all True":
        df_train= all_t_fill(gb_columns, df_train, value=True)
    elif strategy== "none":
        df_train= no_fill(gb_columns, df_train)
    elif strategy== "impute":
        df_train= imputer(gb_columns, df_train)
    else:
        print("no such fill strategy")
    return df_train