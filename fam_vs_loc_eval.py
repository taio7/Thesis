import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent
RES_OUTPUT_PATH = BASE_PATH / "new_results"
LAST_PATH= RES_OUTPUT_PATH/ "last"
EVAL_PATH = BASE_PATH / "evaluation"
from glob import glob

def parse_overall_metrics(path_pattern):
    results = []
    for path_str in glob(str(path_pattern)):
        path = Path(path_str)
        model_name = path.name.split("_")[0]
        if model_name == "base":
            model_name = "freq_baseline"

        with open(path, encoding="utf-8") as f:
            text = f.read()

        # Extract top block (before "feature:")
        top_block = text.split("feature:")[0]

        # Parse metrics using regex
        subset_acc = re.search(r"Subset accuracy:(\d\.\d+)", top_block)
        macro_prec = re.search(r"Macro-average precision::(\d\.\d+)", top_block)
        macro_rec = re.search(r"Macro-average recall:(\d\.\d+)", top_block)
        weighted_f1 = re.search(r"Weighted F1 score:(\d\.\d+)", top_block)
        macro_f1 = re.search(r"Macro-average F1 score:(\d\.\d+)", top_block)
        micro_f1 = re.search(r"Micro-average F1 score:(\d\.\d+)", top_block)

        results.append({
            "model": model_name,
            "subset_accuracy": float(subset_acc.group(1)) if subset_acc else None,
            "macro_precision": float(macro_prec.group(1)) if macro_prec else None,
            "macro_recall": float(macro_rec.group(1)) if macro_rec else None,
            "weighted_f1": float(weighted_f1.group(1)) if weighted_f1 else None,
            "macro_f1": float(macro_f1.group(1)) if macro_f1 else None,
            "micro_f1": float(micro_f1.group(1)) if micro_f1 else None
        })
    return pd.DataFrame(results)


def plot_overall_metrics_combined(df, metric="macro_f1"):
    sns.set(style="whitegrid")

    # Color palette for models
    input_palette = {
        "base": "#4C72B0",  # blue
        "fam": "#55A868",   # green
        "loc": "#7D55A8" 
    }

    # Order of input types
    input_order = ["base", "fam", "loc"]  # must match df['input_type'] values
    input_labels = {
        "base": "GB Features (Base)",
        "fam":  "Base + Family",
        "loc":  "Base + Location"
        #"fam_loc": "Base + Family + Location"
    }
        
    model_order = ["KN", "RF"]

    # Plot grouped bars: x = input type, hue = model
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=df,
        x="model",
        y=metric,
        hue="input_type",
        palette=input_palette,
        dodge=True,
        width=0.6,
        order=model_order,
        hue_order=input_order,
        errorbar=None
    )

    # Annotate bars with exact values
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha='center', va='bottom', fontsize=9
            )

    # Title and labels
    ax.set_title("Performance Comparison Across Input Types")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    labels = [input_labels.get(l, l) for l in labels]
    ax.legend(handles, labels, title="Input Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def main():

    base_df = parse_overall_metrics(LAST_PATH/ "*_best_mr3_results.txt")
    base_df["input_type"] = "base"

    fam_df = parse_overall_metrics(LAST_PATH/ "fam"/ "*_mr3_results.txt")
    fam_df["input_type"] = "fam"

    loc_df = parse_overall_metrics(LAST_PATH/ "loc"/ "*_mr3_results.txt")
    loc_df["input_type"] = "loc"
    famloc_df = parse_overall_metrics(LAST_PATH / "famloc" / "*_mr3_results.txt")
    famloc_df["input_type"] = "fam_loc"

            #famloc_df
    combined = pd.concat([base_df, fam_df, loc_df], ignore_index=True)


    #plot_overall_metrics_per_model(combined, metric="subset_accuracy")
    plot_overall_metrics_combined(combined, metric="subset_accuracy")
if __name__ == "__main__":
    main()
