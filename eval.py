import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import time 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from pathlib import Path



def family_eval(gb_columns, y_dev, y_pred_df, masked_positions, language_families, encoder_path=Path("final_dfs") / "no_nans" / "label_encoders.pkl", output_dir="evaluation"):
    """per family metrics and blanking ratios for test evaluation"""
    target_families = ['Mayan', 'Tucanoan', 'Sepik', 'Worrorran', 'Tungusic', 'Nilotic']

    #decode since all tests no nans
    with open(encoder_path, "rb") as f:
        label_encoders = pickle.load(f)
        

        language_families = language_families.map(lambda x: label_encoders["language_family"].inverse_transform([x])[0])
        
    #if not in targets, name OTHER
    family_series = language_families.reindex(y_pred_df.index).fillna("Other")
    family_series = family_series.where(family_series.isin(target_families), "Other")

    fam_results = ["\n Per Family Evaluation\n"]
    fam_data= []
    blanking_ratios= {}

    #see counts of all families and other
    print("family counts in test run:\n", family_series.value_counts())
    masked_counts = (masked_positions.sum(axis=1)).groupby(family_series).sum()
    print("masked counts per family:", masked_counts.to_dict())

    for fam in target_families + ["Other"]:
        fam_mask = family_series == fam
        fam_y_true = y_dev.loc[fam_mask, gb_columns] #ground truth vals
        fam_y_pred = y_pred_df.loc[fam_mask, gb_columns] #predicted
        fam_masked = masked_positions.loc[fam_mask, gb_columns] #masked matrix for that family

        mask_flat = fam_masked.values.flatten().astype(bool) #flatten to bool array to mark masked pos
        if mask_flat.sum() == 0:
            fam_results.append(f"Family: {fam} has 0 masked positions")
            fam_data.append({"Family": fam, "Accuracy": None, "Macro F1": None, "Blanking Ratio": None})
            continue

        y_true_flat = fam_y_true.values.flatten()[mask_flat].astype(int)
        y_pred_flat = fam_y_pred.values.flatten()[mask_flat].astype(int)

        fam_acc = accuracy_score(y_true_flat, y_pred_flat)
        fam_f1 = f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
        #ratio of not nan values to masked encoded as 2
        non_unknown_count = (fam_y_true.values != 2).sum()

        blank_ratio = mask_flat.sum() / non_unknown_count if non_unknown_count > 0 else 0
        blanking_ratios[fam] = blank_ratio


        fam_results.append(
            f"Family: {fam}\n"
            f"  Accuracy: {fam_acc:.3f}\n"
            f"  Macro F1: {fam_f1:.3f}\n"
            f"  Blanking ratio: {blank_ratio:.3f}\n"
            + "-" * 40 + "\n"
        )
        fam_data.append({
            "Family": fam,
            "Accuracy": fam_acc,
            "Macro F1": fam_f1,
            "Blanking Ratio": blank_ratio
        })
    print("\n".join(fam_results))
    print("Blanking Ratios:", blanking_ratios)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "family_metrics.csv"
    pd.DataFrame(fam_data).to_csv(metrics_path, index=False)
    print(f"✅ Family metrics saved to: {metrics_path}")
    

def masked_eval(gb_columns, y_dev, y_pred_df, masked_positions, output_file, test_run=True, language_families=None):
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
    if test_run:
        assert language_families is not None, "language_families must be provided for test_run"
        # Ensure alignment before calling family_eval
        if not language_families.index.equals(y_pred_df.index):
            language_families = language_families.reindex(y_pred_df.index)
        family_eval(gb_columns, y_dev, y_pred_df, masked_positions, language_families, output_dir="evaluation")



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

