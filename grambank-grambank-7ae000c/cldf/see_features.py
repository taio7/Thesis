import pandas as pd
from pathlib import Path
BASE_DIR = Path("grambank-grambank-7ae000c")


FEATURE_CSV = BASE_DIR / "docs" / "feature_groupings" / "feature_grouping_for_coding.csv"


feat_groups_df = pd.read_csv(FEATURE_CSV)


feature_codes = [
    "GB149", "GB320", "GB105", "GB309",
    "GB192", "GB059", "GB091", "GB312",
    "GB022", "GB084", "GB314", "GB039",
    "GB408", "GB028", "GB093", "GB521",
    "GB252", "GB329", "GB121"
]

# Drop duplicates if any
feature_codes = list(set(feature_codes))
#filtered_params = params_df[params_df["ID"].isin(feature_codes)]
filtered_feats= feat_groups_df[feat_groups_df["Feature_ID"].isin(feature_codes)]

# Show the relevant columns
#print(filtered_params[["ID", "Name", "Boundness", "Flexivity", "Gender_or_Noun_Class", "Locus_of_Marking", "Word_Order", "Informativity"]])
#filtered_params.to_csv("feature_deltas.csv")
print(filtered_feats[["Feature_ID", "Relevant units","Function","Form"]])
filtered_feats.to_csv("feature_deltas_grouped.csv")