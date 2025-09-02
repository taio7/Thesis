import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent
EVAL_PATH = BASE_PATH / "evaluation"

def melt_cluster_inputs(df):
    
    cluster_inputs = ['loc', 'k', 'k12', 'k24', 'db', 'db12', 'db24', 'hdb', 'hdb12', 'hdb24']
    melted = []
    for cluster in cluster_inputs:
        part = df[["model", "feature", f"{cluster}_f1_macro", f"{cluster}_accuracy"]].copy()
        part["input"] = cluster
        part = part.rename(columns={f"{cluster}_f1_macro": "f1_macro", f"{cluster}_accuracy": "accuracy"})
        melted.append(part)
    return pd.concat(melted, ignore_index=True)

def plot_selected_inputs(long_df, model=None, inputs=None, metric=None):
    df = long_df.copy()
    if model:
        df = df[df['model'] == model]
    if inputs:
        df = df[df['input'].isin(inputs)]
    df['cluster_type'] = df['input'].apply(lambda x: 
                                           'Latitude_Longitude' if x == "loc" else
                                           "K-Means" if x.startswith("k") else
                                           "DBSCAN" if x.startswith("db") else
                                           "HDBSCAN")
    df['x_label'] = df['input'].apply(lambda x:
                                      'Lot_Lon' if x == 'loc' else
                                      'k3' if x == 'k' else
                                      'db6' if x == 'db' else
                                      'hdb7' if x == 'hdb' else x)
   
    x_order = ['Lot_Lon', 'k3', 'k12', 'k24', 'db6', 'db12', 'db24', 'hdb7', 'hdb12', 'hdb24']
    df['x_label'] = pd.Categorical(df['x_label'], categories=x_order, ordered=True)

    palette = {
        'Latitude_Longitude': "#B04C7B",
        'K-Means': '#4C72B0',
        'DBSCAN': '#DD8452',
        'HDBSCAN': '#55A868'
    }

    sns.set(style='whitegrid')
    plt.figure(figsize=(5, 6))
    #return axis 
    ax= sns.barplot(data=df, x='x_label', y=metric, hue='cluster_type',
                errorbar=None, palette=palette, width=0.5, dodge=False, legend=True)
    for bar in ax.patches:
        if bar.get_height() == 0 or bar.get_width() == 0: #remove 0.00
            continue
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=9
        )
    plt.title(f"Model: {model}")
    plt.xlabel("")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_parquet(EVAL_PATH / "merged_df.parquet")
    df_long = melt_cluster_inputs(df)
    cluster_inputs = ['loc', 'k', 'k12', 'k24', 'db', 'db12', 'db24', 'hdb', 'hdb12', 'hdb24']


    plot_selected_inputs(df_long, model='KN', inputs=cluster_inputs, metric='accuracy')
    
    plot_selected_inputs(df_long, model='RF', inputs=cluster_inputs, metric='accuracy')
if __name__ == "__main__":
    main()
