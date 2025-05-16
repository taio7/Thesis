import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import time 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""makes mask_df= meta+gb(f/t/masked as <NA> in selected indx)
    and masked_rec_for_eval= gb(all Fasle except masked positions=True)"""
                                            #masked_positions=masked_rec_for_eval
def masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file):
    results=[]
# evaluation 
    masked_positions = masked_positions.reindex(y_pred_df.index)
    for feature in gb_columns:
        mask = masked_positions[feature] #the masked matrix, column with True if masked
        assert mask.shape[0] == y_pred_df.shape[0], f"Mask and y_pred_df row count mismatch for feature {feature}"
        if mask.sum()== 0:
            #print(f"[SKIPPED] {feature} had 0 masked entries.")
            continue #skip features per lang with nan vlaues 
    
        y_true = y_dev.loc[mask, feature].astype(int)
        y_pred = y_pred_df.loc[mask, feature].astype(int)

        re= classification_report(y_true, y_pred, zero_division=0)
        results.append(f"feature: {feature}\n")
        results.append(re+"\n")
        results.append("-" * 40 + "\n")

    #print("masked_positions dtype:", masked_positions.dtypes.unique())
    #print("Sample values:\n", masked_positions.iloc[:5, :5])
    #print("masked_positions.values type:", masked_positions.values.dtype)
    #print("masked_positions masked_positions:", mask_m.shape)
    #print("True values count (via masked_positions.sum().sum()):", masked_positions.sum().sum())

#convert to int to calculate accuracy after removing nans otherwise error
    mask_flat= masked_positions.values.flatten().astype(bool)

    masked_indices = np.where(mask_flat)[0]# where masked feats are 
    print(f"Number of masked: {len(masked_indices)}")

    y_true_flat = y_dev.values.flatten()[mask_flat].astype(int)# use only masked positions for eval
    y_pred_flat = y_pred_df.values.flatten()[mask_flat].astype(int)
    

    sub_acc= accuracy_score(y_true_flat, y_pred_flat)
    macro_prec= precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    macro_recall= recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    weighted_f1= f1_score(y_true_flat, y_pred_flat, average="weighted", zero_division=0)
    macro_f1= f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
    micro_f1=f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)

    res= (
        f"Subset accuracy:{sub_acc:.3f}\n"
        f"Macro-average precision::{macro_prec:.3f}\n"
        f"Macro-average recall:{macro_recall:.3f}\n"
        f"Weighted F1 score:{weighted_f1:.3f}\n"
        f"Macro-average F1 score:{macro_f1:.3f}\n"
        f"Micro-average F1 score:{micro_f1:.3f}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(res)
        f.writelines(results)
    print(f"eval file saved to {output_file}")

def see_feature_importance(multi_model, X_dev, y_dev, gb_columns, masked_positions):
    """feature importance looped for each gb feature=column"""
    start_time = time.time()
    for i, col in enumerate(gb_columns):
        print(f"for feature {col}")

        #need to remove nan from ytrue for it to compute
        mask = masked_positions[col]
        X_dev_masked = X_dev[mask]
        y_dev_masked = y_dev[col][mask]

        #see if too many values are skipped 
        if len(y_dev_masked) < 10:
            print("skipping: too few non-nan values")
            continue

        result= permutation_importance(
            multi_model.estimators_[i], X_dev_masked, y_dev_masked, 
            n_repeats=5, random_state=42, n_jobs=1,scoring="accuracy"
        )
        importances = pd.Series(result.importances_mean, index=X_dev.columns)
        importances = importances.sort_values(ascending=False)
        print(importances.head(10).to_string())

        fam_feat= [c for c in X_dev.columns if "fam_" in c or "top_fams_" in c or c == "language_family"]
        for fcol in fam_feat:
            if fcol in importances:
                print(f"{fcol}: {importances[fcol]:.6f}")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


def eval(gb_columns, y_dev, y_pred_df, output_file):
    results=[]
# evaluation 
   
    for feature in gb_columns:
        re= classification_report(y_dev[feature], y_pred_df[feature], zero_division=0)
        results.append(f"feature: {feature}\n")
        results.append(re+"\n")
        results.append("-" * 40 + "\n")
        

#convert to int to calculate accuracy 
    y_dev_int = y_dev.astype(int)
    y_pred_int = y_pred_df.astype(int)

    sub_acc= accuracy_score(y_dev_int, y_pred_int)
    macro_prec= precision_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    macro_recall= recall_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    weighted_f1= f1_score(y_dev_int, y_pred_int, average="weighted", zero_division=0)
    macro_f1= f1_score(y_dev_int, y_pred_int, average="macro", zero_division=0)
    micro_f1=f1_score(y_dev_int, y_pred_int, average="micro", zero_division=0)

    res= (
        f"Subset accuracy:{sub_acc:.3f}\n"
        f"Macro-average precision::{macro_prec:.3f}\n"
        f"Macro-average recall:{macro_recall:.3f}\n"
        f"Weighted F1 score:{weighted_f1:.3f}\n"
        f"Macro-average F1 score:{macro_f1:.3f}\n"
        f"Micro-average F1 score:{micro_f1:.3f}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(res)
        f.writelines(results)
    print(f"eval file saved to {output_file}")


