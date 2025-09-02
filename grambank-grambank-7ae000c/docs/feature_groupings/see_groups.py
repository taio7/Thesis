import pandas as pd 
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq 



featdf= pd.read_csv("feature_grouping_for_analysis.csv")
# Group by both columns and list Feature_IDs
grouped = featdf.groupby(["Main_domain", "Finer_grouping"])["Feature_ID"].apply(list)
# Count features for each (Main_domain, Finer_grouping) pair
group_counts = featdf.groupby(["Main_domain", "Finer_grouping"]).size().reset_index(name="Feature_Count")

print(group_counts)

# View the result
print(grouped)
