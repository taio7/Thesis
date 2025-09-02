import pandas as pd
from pathlib import Path 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data" 

df_train= pd.read_csv(DATA_PATH/"train.tsv", sep= "\t")
df_dev= pd.read_csv(DATA_PATH/"dev.tsv", sep= "\t")

gb_columns = [col for col in df_train.columns if col.startswith("GB")]

#make sure they are bool
#print(df_train[gb_columns].dtypes.value_counts()) 
X_train= df_train[gb_columns].astype("boolean").copy()
X_dev= df_dev[gb_columns].astype("boolean").copy()
target_feat= 'GB285_QPartVMorph'

train_sub= df_train.dropna(subset=[target_feat])  #exclude rows with missing target
X = train_sub[gb_columns].drop(columns=[target_feat])  #train on all features except the target
y = train_sub[target_feat].astype("boolean")

model = RandomForestClassifier(random_state=42)
#model.fit(X, y)

X_dev_clean = df_dev.dropna(subset=[target_feat])
X_test = X_dev_clean[gb_columns].drop(columns=[target_feat])
y_test = X_dev_clean[target_feat].astype("boolean")

#y_pred = model.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#assert target_feat not in X.columns

