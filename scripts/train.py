import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"


df_train = pd.read_csv(DATA_PATH / "train.tsv", sep="\t")
df_dev = pd.read_csv(DATA_PATH / "dev.tsv", sep="\t")


gb_columns = [col for col in df_train.columns if col.startswith("GB")]

#ensure boolean types
df_train[gb_columns] = df_train[gb_columns].astype("boolean")
df_dev[gb_columns] = df_dev[gb_columns].astype("boolean")

#fill missing values with most common
for col in gb_columns:
    mode=df_train[col].mode(dropna=True)  #find the most common value per col (not NAN)
    #print(mode[0])
    fill= mode[0]
    df_train[col]= df_train[col].fillna(fill).astype("boolean")
    df_dev[col]= df_dev[col].fillna(fill).astype("boolean")      #fill missing values with most common per col

"""train_clean = df_train.dropna(subset=gb_columns)
dev_clean = df_dev.dropna(subset=gb_columns)
print(train_clean)
print(dev_clean)


X_train = train_clean[gb_columns]
y_train = train_clean[gb_columns]

X_dev = dev_clean[gb_columns]
y_dev = dev_clean[gb_columns]"""

X_train = df_train[gb_columns] #train on whole and targets are all features 
y_train = df_train[gb_columns]  

X_dev = df_dev[gb_columns]
y_dev = df_dev[gb_columns]

#Multi_model=train one classifier per feature 

base_model = RandomForestClassifier(random_state=42)  #extend: separate decision tree per family/area
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, y_train)

#predict, convert the array back to df 
y_pred = multi_model.predict(X_dev)
y_pred_df = pd.DataFrame(y_pred, columns=gb_columns, index=X_dev.index)

# evaluation 
print("Evaluation per feature:\n")
for feature in gb_columns:
    print(feature)
    print(classification_report(y_dev[feature], y_pred_df[feature], zero_division=0))
    print("-" * 40)

#convert to int to calculate accuracy 
y_dev_int = y_dev.astype(int)
y_pred_int = y_pred_df.astype(int)


print("Subset accuracy", accuracy_score(y_dev_int, y_pred_int)) #all features predicted correctly per lg 
print("Macro-average precision:", precision_score(y_dev_int, y_pred_int, average="macro", zero_division=0)) #avg equally across features
print("Macro-average recall:", recall_score(y_dev_int, y_pred_int, average="macro", zero_division=0))
print("Weighted F1 score:", f1_score(y_dev_int, y_pred_int, average="weighted", zero_division=0)) # account for lable imbalance 
print("Macro-average F1 score:", f1_score(y_dev_int, y_pred_int, average="macro", zero_division=0))
print("Micro-average F1 score:", f1_score(y_dev_int, y_pred_int, average="micro", zero_division=0)) #global performance 

"""Subset accuracy 0.11578947368421053
Macro-average precision: 0.9215662290416539
Macro-average recall: 0.789148386775071
Weighted F1 score: 0.9654978871211904
Macro-average F1 score: 0.8297660066324019
Micro-average F1 score: 0.9736165349080865"""