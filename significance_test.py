import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent
EVAL_PATH = BASE_PATH / "evaluation"

df = pd.read_parquet(EVAL_PATH / "merged_df.parquet")


acc_cols = ["best_base_accuracy", "best_fam_accuracy", "best_loc_accuracy", "best_famloc_accuracy"]
f1_cols= f1_cols = ["best_base_f1_macro", "best_loc_f1_macro", "best_fam_f1_macro", "best_famloc_f1_macro"]


def test_significance_per_model(model_name, metric="f1"):
    
    if metric == "accuracy":
        base_col, fam_col, loc_col, famloc_col = acc_cols
    elif metric == "f1":
        base_col, fam_col, loc_col, famloc_col = f1_cols
    else:
        raise ValueError
    model_df = df[df["model"] == model_name]

    results = []
    for comparison in [(fam_col, "Family"), (loc_col, "Location"), (famloc_col, "Fam+Loc")]:
        comp_col, label = comparison

        diffs = model_df[comp_col] - model_df[base_col]

        #Shapiro-Wilk test, if distribution is normal then t-test, if not Wilcox
        stat, p_norm = stats.shapiro(diffs)
        print(f"\n{model_name}: {label} vs Base with Normality p={p_norm:.10f}")

        if p_norm > 0.05:  
            t_stat, p_val = stats.ttest_rel(model_df[comp_col], model_df[base_col])
            test_used = "Paired t-test"
        else:  
            w_stat, p_val = stats.wilcoxon(model_df[comp_col], model_df[base_col])
            test_used = "Wilcoxon signed-rank"

        print(f"Result: {test_used} p={p_val:.7f}")
        results.append({"model": model_name, "comparison": label, "test": test_used, "p_value": p_val})

    return pd.DataFrame(results)

def collect_diffs_for_all_models(models, metric="accuracy"):
    all_diffs = []
    if metric == "accuracy":
        base_col, fam_col, loc_col, _ = acc_cols
    elif metric == "f1":
        base_col, fam_col, loc_col, _= f1_cols

    for model in models:
        model_df = df[df["model"] == model]
        for col, label in [(fam_col, "Family"), (loc_col, "Location")]:
            diffs = model_df[col] - model_df[base_col]
            for diff in diffs:
                all_diffs.append({
                    "model": model,
                    "input_type": label,
                    "difference": diff
                })

    return pd.DataFrame(all_diffs)
def plot_model_differences_stripplot(diff_df, metric):
    plt.figure(figsize=(9, 6))
    sns.stripplot(
        data=diff_df,
        x="difference",
        y="input_type",
        hue="model",
        jitter=0.25,
        alpha=0.7,
        dodge=True,
        orient="h"
    )
    plt.axvline(0, linestyle="--", color="gray", linewidth=1, alpha=0.8)
    
    plt.title(f"{metric.title()} Differences (Input - Base)")
    plt.xlabel(f"{metric.title()} Difference")
    plt.ylabel("Input Type")
    plt.xlim(diff_df["difference"].min() - 0.05, diff_df["difference"].max() + 0.05)
    plt.grid(axis="x", linestyle=":", alpha=0.4)
    
   
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
#metric: "accuracy" or "f1"
metric = "f1"
models = ["RF", "KN"]
diff_df = collect_diffs_for_all_models(models, metric=metric)
plot_model_differences_stripplot(diff_df, metric)

# === Run per model ===
rf_acc_results = test_significance_per_model("RF", metric="accuracy")

kn_acc_results = test_significance_per_model("KN", metric="accuracy")


#all_results = pd.concat([rf_acc_results, rf_f1_results, kn_acc_results, kn_f1_results], ignore_index=True)
#print("\n Test Results")
#print(all_results)

#all_results.to_csv(EVAL_PATH / "significance_test_results.csv", index=False)


metric_choice = "f1" # or "f1"
base_col, fam_col, loc_col = (f1_cols if metric_choice == "f1" else acc_cols)[:3]

deltas = []
for model in models:
    model_df = df[df["model"] == model]
    for _, row in model_df.iterrows():
        deltas.append({
            "model": model,
            "feature": row["feature"],
            "delta_fam": row[fam_col] - row[base_col],
            "delta_loc": row[loc_col] - row[base_col]
        })

df_deltas = pd.DataFrame(deltas)
# === Count of improvements per model ===
print(f"\n{metric_choice.title()} Feature Improvement Summary:")
models = df_deltas["model"].unique()

for model in models:
    model_df = df_deltas[df_deltas["model"] == model]
    total = model_df.shape[0]
    fam_improved = (model_df["delta_fam"] > 0).sum()
    loc_improved = (model_df["delta_loc"] > 0).sum()
    mean_fam = model_df["delta_fam"].mean()
    mean_loc = model_df["delta_loc"].mean()

    print(f"\nModel: {model}")
    print(f" - {fam_improved} / {total} features improved with Family ({fam_improved / total:.2%})")
    print(f" - {loc_improved} / {total} features improved with Location ({loc_improved / total:.2%})")
    print(f"\nModel: {model}")
    print(f" - Mean Δ {metric_choice.title()} (Family - Base): {mean_fam:.4f}")
    print(f" - Mean Δ {metric_choice.title()} (Location - Base): {mean_loc:.4f}")

#Most improved features 
top_fam = df_deltas.sort_values(by="delta_fam", ascending=False).head(10)
top_loc = df_deltas.sort_values(by="delta_loc", ascending=False).head(10)

print(f"\nTop 10 feature improvements ({metric_choice.title()}) from adding Family:")
print(top_fam[["model", "feature", "delta_fam"]])
print(f"\nTop 10 feature improvements ({metric_choice.title()}) from adding Location:")
print(top_loc[["model", "feature", "delta_loc"]])



threshold = 0.20  # 20 perc

rows = []
for model in df_deltas["model"].unique():
    sub = df_deltas[df_deltas["model"] == model]
    for col, label in [("delta_fam", "Family"), ("delta_loc", "Location")]:
        n = len(sub)
        mean_imp = sub[col].mean()

        improved = (sub[col] > 0).sum()
        big_improved = (sub[col] >= threshold).sum()

        rows.append({
            "model": model,
            "input": label,
            "n_features": n,
            "mean_delta": mean_imp,
            "improved_count": improved,
            "improved_pct": improved / n if n else None,
            f"≥{threshold:.0%}_count": big_improved,
            f"≥{threshold:.0%}_pct": big_improved / n if n else None
        })

meaningful_imp_df = pd.DataFrame(rows)
print("\n improvement per model, > 20")
print(meaningful_imp_df.sort_values(["model", "input"]))