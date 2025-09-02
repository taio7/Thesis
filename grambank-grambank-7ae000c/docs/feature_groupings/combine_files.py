import pandas as pd

descriptions= pd.read_csv("feature_grouping_for_coding.csv")
groupings= pd.read_csv("feature_grouping_for_analysis.csv")


groupings["feat"] = (
    groupings["Feature_ID"].astype(str).str.strip().str.replace(r"[a-z]$", "", regex=True)
)
descriptions["feat"] = descriptions["Feature_ID"]
collapsed = groupings.groupby("feat", as_index=False)[["Main_domain", "Finer_grouping"]].first()
print(len(collapsed))
df = (
    descriptions
    .merge(collapsed, on="feat", how="left")
    [["Feature_ID", "Main_domain", "Finer_grouping", "Feature"]]
)
df = df.sort_values(by=["Main_domain", "Finer_grouping", "Feature_ID"]).reset_index(drop=True)
print(len(df))
df.to_csv("GBfeatures_groups.csv", index=False)