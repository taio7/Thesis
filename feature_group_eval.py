import os
import re
import pandas as pd
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon


BASE_PATH = Path(__file__).parent
EVAL_PATH=BASE_PATH/"evaluation"


def compute_feature_stats(df):
           
       
    score_cols= ["chain_f1_macro","baseline_f1_macro", 'base_f1_macro',
                 "k_f1_macro", "k12_f1_macro", "k24_f1_macro", "db_f1_macro", "db12_f1_macro", "db24_f1_macro", "hdb_f1_macro", "hdb12_f1_macro", "hdb24_f1_macro",
                 "best_base_f1_macro", "best_loc_f1_macro", "best_fam_f1_macro", "best_famloc_f1_macro"]
    acc_cols= ["chain_accuracy", "baseline_accuracy", 'base_accuracy',
               "k_accuracy", "k12_accuracy", "k24_accuracy", "db_accuracy", "db12_accuracy", "db24_accuracy", "hdb_accuracy", "hdb12_accuracy", "hdb24_accuracy",
               "best_base_accuracy", "best_loc_accuracy", "best_fam_accuracy", "best_famloc_accuracy"]

    df['f1_mean']= df[score_cols].mean(axis=1)
    df['accuracy_mean']= df[acc_cols].mean(axis=1)

    return df

def add_groupings(merged_df, feat_groupings):
    feat_groupings = feat_groupings.rename(columns={
        'Feature_ID': 'feature',
        'Main_domain': 'group_main',
        'Finer_grouping': 'group_finer'
    })

    df = merged_df.merge(feat_groupings[['feature', 'group_main', 'group_finer']], on="feature", how="left")
    missing = df['group_main'].isna().sum()

    return df

def melt_inputs(df):
    """Reshape wide input columns to long format, preserving both group levels."""
    input_vars = ["base", "chain", "baseline", "k", "k12", "k24", "db", "db12", "db24", "hdb", "hdb12", "hdb24",
                  "best_base", "best_loc", "best_fam", "best_famloc"]
    melted = []
    for input_type in input_vars:
        part = df[["model", "feature", "group_main", "group_finer",
                   f"{input_type}_f1_macro", f"{input_type}_accuracy"]].copy()
        part["input"] = input_type
        part = part.rename(columns={f"{input_type}_f1_macro": "f1_macro", 
                                    f"{input_type}_accuracy": "accuracy"})
        melted.append(part)
    return pd.concat(melted, ignore_index=True)


def group_metrics(df_long):
    #compute mean f1 and accuracy per main & finer group, input type, and model
    grouped = (
        df_long.groupby(["group_main", "group_finer", "model", "input"])
        .agg(f1_macro_mean=("f1_macro", "mean"),
             accuracy_mean=("accuracy", "mean"),
             n_features=("feature", "count"))
        .reset_index()
    )
    return grouped



def save_results(grouped_df):
    output_path = EVAL_PATH / "grouped_performance.parquet"
    grouped_df.to_parquet(output_path, index=False)
    print(f"Saved grouped metrics to: {output_path}")


def plot_group_deltas_by_input(
    df_long,
    model_name="RF",
    grouping_col="group_main",     # /finer
    metric="f1_macro",             
    top_n=None,                   
    sort_by="fam"):
    

    #keep just the model and the 3 inputs we compare to base
    df_model = df_long[
        (df_long["model"] == model_name) &
        (df_long["input"].isin(["best_base", "best_fam", "best_loc"]))
    ].copy()

    #average
    grouped = (
        df_model
        .groupby([grouping_col, "input"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    #wide to compute deltas
    wide = grouped.pivot(index=grouping_col, columns="input", values=metric).fillna(0.0)


    if "best_base" not in wide.columns:
        raise ValueError("best_base not found; check your melted inputs/columns.")
    wide["delta_fam"] = wide.get("best_fam", 0.0) - wide["best_base"]
    wide["delta_loc"] = wide.get("best_loc", 0.0) - wide["best_base"]

    plot_df = (
        wide[["delta_fam", "delta_loc"]]
        .rename(columns={"delta_fam": "Family", "delta_loc": "Location"})
        .reset_index()
        .melt(id_vars=[grouping_col], var_name="Input", value_name="Delta")
    )

    sort_key = "Family" if sort_by == "fam" else "Location"
    order = (
        plot_df[plot_df["Input"] == sort_key]
        .sort_values("Delta", ascending=False)[grouping_col]
        .tolist()
    )
    if top_n:
        order = order[:top_n]
        plot_df = plot_df[plot_df[grouping_col].isin(order)]


    plt.figure(figsize=(12, max(5, 0.35 * len(order))))
    sns.barplot(
        data=plot_df,
        x="Delta",
        y=grouping_col,
        hue="Input",
        dodge=True
    )
    plt.axvline(0, ls="--", c="gray", lw=1)
    pretty_metric = "F1" if metric == "f1_macro" else "Accuracy"
    plt.title(f"Model: {model_name}")
    plt.xlabel(f"{pretty_metric} Difference: (Input − Base)")
    plt.ylabel(grouping_col.replace("_", " ").title())
    plt.legend(title="Input")
    plt.tight_layout()
    plt.show()



def main():
    merged_df = pd.read_parquet(EVAL_PATH/"merged_df.parquet")
    #print(merged_df.columns)
    feat_groupings = pd.read_parquet(EVAL_PATH/"feat_groupings.parquet")
    merged_with_groups = add_groupings(merged_df, feat_groupings)
   
    feat_groupings= pd.read_parquet(EVAL_PATH/"feat_groupings.parquet")
    #print(feat_groupings.columns)
    merged_with_groups= add_groupings(merged_df,feat_groupings)
    
    long_df = melt_inputs(merged_with_groups)
    grouped_df = group_metrics(long_df)

    for model in ["RF", "KN"]:

        # F1 plots
        plot_group_deltas_by_input(long_df, model_name=model, grouping_col="group_main", metric="f1_macro", top_n=None, sort_by="fam")
        plot_group_deltas_by_input(long_df, model_name=model, grouping_col="group_finer", metric="f1_macro", top_n=20, sort_by="loc")
    #save_results(grouped_df)
    models=["RF", "KN", "MLP", "HGB"]
    #for name in models:
        #plot_model_group_analysis(long_df, model_name=name, grouping_col="group_finer")

     #f1_macro, accuracy"""


if __name__ == "__main__":
    main()