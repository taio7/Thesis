import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 
import seaborn as sns
"""SEE stats for enc version of dataset, without filling missing values"""

BASE_PATH = Path(__file__).parent
ENTIRE_DATASET_PATH= BASE_PATH/"make_data_scripts"/"data"


entire_dataset= pd.read_csv(ENTIRE_DATASET_PATH/"grambank.condensed.tsv", sep="\t")


gb_columns = [col for col in entire_dataset.columns if col.startswith("GB")]


def print_train_value_distribution(df, gb_columns, lang_column="language_name", macroarea_column="language_macroarea"):
    df_gb_raw = df[gb_columns]
    # 1, 2, 3 :True
    # 0 :False
    # NaN : NaN
    df_gb = df_gb_raw.copy()
    df_gb = df_gb.replace({1: True, 2: True, 3: True, 0: False})

    flat = df_gb.values.flatten()
    total = flat.size

    true_count = pd.Series(flat).apply(lambda x: x is True).sum()
    false_count = pd.Series(flat).apply(lambda x: x is False).sum()
    nan_count = pd.isna(flat).sum()
    

    counts = pd.Series(flat).value_counts(dropna=False)

    print("\nVALUE DISTRIBUTION")
    print(f"Total cells: {total}")
    print(f"True: {true_count} ({true_count / total:.2%})")
    print(f"False: {false_count} ({false_count / total:.2%})")
    print(f"NaN: {nan_count} ({nan_count / total:.2%})")
    print("Full value breakdown:")
    print(counts)

    print("\nSTATS FOR FEATURES")
    c_per_lg=df_gb.notna().sum(axis=1) #non empty vals per 
    total_feats= len(gb_columns)
    total_langs= df.shape[0]

        #TRUE count, RARITY
    true_per_feat = (df_gb == True).sum(axis=0)          # how many languages have TRUE
    true_pct_feat = (true_per_feat / total_langs * 100).round(2)

    top10_true = true_per_feat.sort_values(ascending=False).head(25)
    bottom10_true = true_per_feat.sort_values(ascending=True).head(25)

    print("\nTop 10 most PREVALENT TRUE features:")
    for feat in top10_true.index:
        print(f" - {feat}: {true_per_feat[feat]} languages TRUE ({true_pct_feat[feat]:.2f}%)")

    print("\nBottom 10 RAREST TRUE features:")
    for feat in bottom10_true.index:
        print(f" - {feat}: {true_per_feat[feat]} languages TRUE ({true_pct_feat[feat]:.2f}%)")


    percent_per_lg= (c_per_lg / total_feats * 100).round(2)
    
    
    max_c= c_per_lg.max()
    min_c=c_per_lg.min()
    lg_max = df.loc[c_per_lg == max_c, [lang_column, macroarea_column]]
    lg_min = df.loc[c_per_lg == min_c, [lang_column, macroarea_column]]

    median_c= c_per_lg.median()
    c_per_feat = df_gb.notna().sum(axis=0)
    percent_per_feat = (c_per_feat / total_langs * 100).round(2)

    max_feat_count = c_per_feat.max()
    feat_max = c_per_feat[c_per_feat == max_feat_count].index.tolist()


    print(f"Languages: {total_langs} and Features: {total_feats}")
    print(f"Most feature complete language: [{max_c}/{total_feats} features, {max_c/total_feats:.2%}]:")
    print(f" - {lg_max.head(5).to_string(index=False)}{' ...' if len(lg_max) > 5 else ''}")

    print(f"Least complete language [{min_c}/{total_feats} features, {min_c/total_feats:.2%}]:")
    print(f" - {lg_min.head(5).to_string(index=False)}{' ...' if len(lg_min) > 5 else ''}")
    print(f"Median filled GB features per language: {median_c:.1f} ({median_c/total_feats:.2%})")
    print("\n STATS PER FEATURE")
    c_per_feat= df_gb.notna().sum(axis=0)
    percent_per_feat = (c_per_feat / total_langs * 100).round(2)
    top10 = c_per_feat.sort_values(ascending=False).head(10)
    bottom10 = c_per_feat.sort_values().head(10)
    
    print("Top 10 most documented features:")
    for feat in top10.index:
        print(f" - {feat}: {c_per_feat[feat]} languages ({percent_per_feat[feat]:.2f}%)")

    print("\nBottom 10 least documented features:")
    for feat in bottom10.index:
        print(f" - {feat}: {c_per_feat[feat]} languages ({percent_per_feat[feat]:.2f}%)")

    print("\n=== MACROAREA ANALYSIS ===")
    df['Filled_Feature_Count'] = c_per_lg
    macro_avg = df.groupby(macroarea_column)['Filled_Feature_Count'].mean().sort_values(ascending=False)
    macro_avg_percent = (macro_avg / total_feats * 100).round(2)

    for macro, count in macro_avg.items():
        print(f" - {macro}: {count:.1f} features ({macro_avg_percent[macro]:.2f}%)")




print_train_value_distribution(entire_dataset, gb_columns, lang_column="language_name", macroarea_column="language_macroarea")


def plot_filled_feature_displot(df, gb_columns):

    counts = df[gb_columns].replace(2, np.nan).notna().sum(axis=1).sort_values().reset_index(drop=True)

    x = np.arange(len(counts))
    y = counts.values
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, y, color="k", alpha=0.9)
    plt.plot(x, y, color='k', linewidth=2)
    plt.scatter(x, y, color='k', s=10)
    plt.xlabel("Languages")
    plt.ylabel("Grambank Features")
    plt.title("Feature Coverage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_filled_feature_displot(entire_dataset, gb_columns)

#Most improved features to inspect 
target_features = [
    "GB320", "GB149", "GB192", "GB314", "GB091", "GB252",
    "GB059", "GB312", "GB105", "GB028", "GB329"
]

def report_specific_feature_stats_prefix(df, features, lang_column="language_name"):
    print("\n=== SPECIFIC FEATURE COVERAGE & RARITY ===")
    total_langs = df.shape[0]
    
    for feat in features:
        matching_cols = [col for col in df.columns if col.startswith(feat)]
        colname = matching_cols[0]
        
        true_count = (df[colname] == 1).sum() + (df[colname] == 2).sum() + (df[colname] == 3).sum()
        false_count = (df[colname] == 0).sum()
        nan_count = df[colname].isna().sum()
        
        coverage = total_langs - nan_count
        coverage_pct = coverage / total_langs * 100
        rarity_pct = true_count / total_langs * 100
        
        print(f"{feat} ({colname}): {true_count} TRUE ({rarity_pct:.2f}%), coverage {coverage} langs ({coverage_pct:.2f}%)")
    gb_cols = [c for c in df.columns if c.startswith("GB")]
    df_gb = df[gb_cols].replace({1: True, 2: True, 3: True, 0: False})
    rarity_per_feat = (df_gb == True).sum(axis=0) / df_gb.notna().sum(axis=0)
    avg_rarity = rarity_per_feat.mean() * 100
    print(f"\nAverage rarity across ALL features: {avg_rarity:.2f}% TRUE")

report_specific_feature_stats_prefix(entire_dataset, target_features)

