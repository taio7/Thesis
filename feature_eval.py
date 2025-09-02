import os
import re
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
CLUSTERS_PATH=RES_OUTPUT_PATH/ "clusters"
LOC_PATH= RES_OUTPUT_PATH/"loc"
CHAIN_PATH= RES_OUTPUT_PATH/ "chain"
BASELINE_PATH= RES_OUTPUT_PATH/ "baseline"
LAST_PATH= RES_OUTPUT_PATH/ "last"

EVAL_PATH=BASE_PATH/"evaluation"

def parse_reports(path_pattern):
    #read all results reports, read blocks, return dict of DFs per model
    files = glob(str(path_pattern))
    results = {}

    for path in files:
        model_name = os.path.basename(path).split("_")[0]
        if model_name== "base":
            model_name = "freq_baseline"
        with open(path, encoding="utf-8") as f:
            report = f.read()

        records = []
        feature_blocks = re.split(r'-{10,}', report)  #split each feature block 
        for block in feature_blocks:
            feature_match = re.search(r'feature: (\S+)', block)
            if not feature_match:
                continue
            feature = feature_match.group(1)
            f1_match = re.search(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', block)
            acc_match = re.search(r'accuracy\s+([\d.]+)', block)
            if f1_match and acc_match:
                records.append({
                    'model': model_name,
                    'feature': feature,
                    'f1_macro': float(f1_match.group(1)),
                    'accuracy': float(acc_match.group(1))
                })
        results[model_name] = pd.DataFrame(records)  #dict of dfs: model, feat, f1,
    return results


def merge_all_conditions(chain_reports, freq_baseline_reports, loc_reports, 
                                              base_reports, cluster_reports_dict, 
                                              best_base_reports, best_loc_reports, best_fam_reports, best_famloc_reports):
    #merge all scores, align by feature and model, rename columns, return combined DF with aligned metrics
    all_merged = []

    all_models = set(base_reports) & set(loc_reports)
    freq_baseline_df= freq_baseline_reports.get("freq_baseline")
    freq_baseline = freq_baseline_df.set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"baseline_{x}")
    for model in all_models:
        loc= loc_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"loc_{x}")
        chain=chain_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"chain_{x}")
        base= base_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"base_{x}") #align with features and name a col name 
        
        dfs = [chain, freq_baseline, base, loc]
        if model in best_base_reports:
            best_base = best_base_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"best_base_{x}")
            dfs.append(best_base)
            best_loc = best_loc_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"best_loc_{x}")
            dfs.append(best_loc)
            best_fam = best_fam_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"best_fam_{x}")
            dfs.append(best_fam)
            best_famloc= best_famloc_reports[model].set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"best_famloc_{x}")
            dfs.append(best_famloc)
        else:
            print(f"[!] Skipping best_base for model '{model}' (not found)")
        for cname, creport in cluster_reports_dict.items():
            dfc = creport.get(model)
            if dfc is None:
                print(f"[!] Skipping cluster '{cname}' for model '{model}' (not found)")
                continue
            dfc["feature"] = dfc["feature"].str.strip().str.upper()  # normalize
            cluster_df = dfc.set_index("feature")[["f1_macro", "accuracy"]].rename(columns=lambda x: f"{cname}_{x}")

            num_non_null = cluster_df.notna().sum().to_dict()
            print(f"[+] Cluster '{cname}' for model '{model}': {len(cluster_df)} rows, non-null counts: {num_non_null}")
            dfs.append(cluster_df)

        merged = pd.concat(dfs, axis=1)
        print(f"[=] Merged dataframe for model '{model}' shape: {merged.shape}, non-null values:")
        print(merged.notna().sum().to_string())
        merged["model"] = model
        all_merged.append(merged.reset_index())

    return pd.concat(all_merged, ignore_index=True)

def merge_one_hot_feature_scores(df_scores, prefix_pattern=r"(GB\d{3}|GB[A-Z]+)"):
    #merge 1hot encoded feats, returns df with averaged scores across models and merged feature group
    df = df_scores.copy()

    df['group'] = df['feature'].apply(
        lambda x: re.match(prefix_pattern, x).group(1) if re.match(prefix_pattern, x) else x
    )

    one_hot_df = df[df['feature'] != df['group']]
    unique_groups = one_hot_df['group'].unique()

    print(f"\nOne-Hot feats")
    print(sorted(unique_groups))


    metric_cols = [col for col in df.columns if any(suffix in col for suffix in ['f1_macro', 'accuracy'])]
    merged_df = (
        df.groupby(['model', 'group'])[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={'group': 'feature'})
    )

    merged_one_hot = merged_df[merged_df['feature'].isin(unique_groups)]
    if 'base_f1_macro' in merged_one_hot.columns:
        top10 = merged_one_hot.sort_values('base_f1_macro', ascending=False).head(10)
    else:
        top10 = merged_one_hot.head(10)
    print("\n merged 10 one hot feats")
    print(top10.sort_values(['feature', 'model']).to_string(index=False))

    return merged_df

def main():
    cluster_reports = {
    "k": parse_reports(CLUSTERS_PATH / "*_k_results.txt"),
    "k12": parse_reports(CLUSTERS_PATH / "*_12k_results.txt"),
    "k24": parse_reports(CLUSTERS_PATH / "*_24k_results.txt"),
    "db": parse_reports(CLUSTERS_PATH / "*_db_results.txt"),
    "db12": parse_reports(CLUSTERS_PATH / "*_12db_results.txt"),
    "db24": parse_reports(CLUSTERS_PATH / "*_24db_results.txt"),
    "hdb": parse_reports(CLUSTERS_PATH / "*_hdb_results.txt"),
    "hdb12": parse_reports(CLUSTERS_PATH / "*_12hdb_results.txt"),
    "hdb24": parse_reports(CLUSTERS_PATH / "*_24hdb_results.txt"),
}
    for cname, model_dict in cluster_reports.items():
        print(f"\nParsed cluster: {cname} → {len(model_dict)} models")
        for model_name, df in model_dict.items():
            print(f"  {model_name}: {df.shape[0]} rows, columns: {df.columns.tolist()}")



 
    chain_reports= parse_reports(CHAIN_PATH / "*_mr3_results.txt")
    freq_baseline_reports= parse_reports(BASELINE_PATH / "*_mr3_results.txt")
    base_reports= parse_reports(RES_OUTPUT_PATH / "*_mr3_results.txt")
    loc_reports= parse_reports(LOC_PATH / "*_loc_mr3_results.txt")
    best_base_reports= parse_reports( LAST_PATH/ "*_best_mr3_results.txt")
    best_loc_reports= parse_reports( LAST_PATH/ "loc"/ "*_mr3_results.txt")
    best_fam_reports= parse_reports( LAST_PATH/ "fam"/ "*_mr3_results.txt")
    best_famloc_reports= parse_reports( LAST_PATH/ "famloc"/ "*_mr3_results.txt")

    df_combined_scores = merge_all_conditions(chain_reports, freq_baseline_reports, loc_reports,
                                              base_reports, cluster_reports, 
                                              best_base_reports, best_loc_reports, best_fam_reports, best_famloc_reports)



    merged_df= merge_one_hot_feature_scores(df_combined_scores, prefix_pattern=r"(GB\d{3}|GB[A-Z]+)")
    
    #merged_df.to_csv(TEMP_PATH / "merged_one_hot.csv", index=False)
    merged_df.to_parquet(EVAL_PATH / "merged_df.parquet")
    
    
    

if __name__ == "__main__":
    main()

