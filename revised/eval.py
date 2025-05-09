import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.inspection import permutation_importance

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
MODEL= "HGB"
BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / "new_results"
output_file = OUTPUT_PATH / f"{MODEL}_output.txt"

def masked_eval(gb_columns, y_dev, y_pred_df, mask_m, output_file):
    results=[]
# evaluation 
   
    for feature in gb_columns:
        mask = mask_m[feature] #the masked matrix 
        if mask.sum()== 0:
            continue #skip features per lang with nan vlaues 

        y_true = y_dev.loc[mask, feature].astype(int)
        y_pred = y_pred_df.loc[mask, feature].astype(int)

        re= classification_report(y_true, y_pred, zero_division=0)
        results.append(f"feature: {feature}\n")
        results.append(re+"\n")
        results.append("-" * 40 + "\n")
        

#convert to int to calculate accuracy after removing nans otherwise error
    mask_flat= mask_m.values.flatten()

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


# only maks where values are not nan 





    y_true= df_dev[gb_columns]
    #predictions are made on the masked dev set and then evaluated against the original dev set
    masked_eval(gb_columns, y_true, df_pred, masked_positions, RES_PATH/ f"HGB_fam_m3_results{strategy}.txt")
    y_pred_interp.to_csv(OUTPUT_PATH/ f"HGB_fam_m3_output{strategy}.csv", sep="\t", index=False)

if __name__ == "__main__":
    main()

