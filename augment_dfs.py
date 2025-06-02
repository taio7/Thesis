from geocluster import geo_clustering, plot_geo_clusters, find_optimal_eps
import pandas as pd
import numpy as np
from pathlib import Path
from config_dfs import select_dfs
from sklearn.metrics import adjusted_rand_score, silhouette_score


"""augment with clusters"""


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "ready_dfs"
WITH_NAN_PATH= DATA_PATH/ "with_nans"
NO_NAN_PATH= DATA_PATH/ "no_nans"
with_nans=False
METHOD="dbscan"

train, masked_df, dev, masked_positions= select_dfs(with_nans=None)

def kmeans_clusters(train, masked_df, method, **kwargs):
    """train model and reuse on dev"""
    lat_lon_cols= ["language_latitude", "language_longitude"]
    train_clustered, model= geo_clustering(train, method, **kwargs)
    masked_df_coords=masked_df[lat_lon_cols].values
    masked_df_cl=model.predict(masked_df_coords)
    masked_df_clustered=masked_df.copy()
    masked_df_clustered["kmeans_cluster"]= masked_df_cl
    return train_clustered, masked_df_clustered

def other_clusters(train, masked_df, method, **kwargs):
    """combine metadata columns of train and dev only so no data leakage,
    since I cannot use trained model on the dev data"""
    #print(train.index)
    #print(masked_df.index)
    train= train.reset_index(drop=True)
    masked_df= masked_df.reset_index(drop=True)
    metadata_cols=  [col for col in train.columns if not col.startswith("GB")]
    combined= pd.concat([train[metadata_cols], masked_df[metadata_cols]], ignore_index=True)
    
    clustered_df= geo_clustering(combined, method, **kwargs)
    cluster_col= f"{method}_cluster"
    train_clustered=train.copy()
    masked_df_clustered=masked_df.copy()

    train_clustered[cluster_col]= clustered_df[cluster_col].iloc[:len(train)].values
    masked_df_clustered[cluster_col]= clustered_df[cluster_col].iloc[len(train):].values
    return train_clustered, masked_df_clustered

def best_kmeans(train, k=range(2, 17), random_state=42):
    best_k=None
    best_score=-1
    for i in k:
        kwargs={"n_clusters":i, "random_state": 42}
        train_clustered, _= geo_clustering(train, method="kmeans", **kwargs)
        score= calc_metrics(train_clustered, method="kmeans")
        if score>best_score:
            best_score=score
            best_k=i
    return best_k, best_score

def best_hdbscan(train, min_cluster_size= range(5,30,5), min_samples=[None, 5, 10], cluster_selection_method=["eom","leaf"]):
    best_params=None
    best_score=-1
    for mcs in min_cluster_size:
        for ms in min_samples:
            for csm in cluster_selection_method:
                kwargs={"min_cluster_size": mcs,
                "min_samples": ms,
                "cluster_selection_method": csm
                }
                train_clustered= geo_clustering(train, method="hdbscan", **kwargs)
                score=calc_metrics(train_clustered, method="hdbscan")
                if score>best_score:
                    best_score=score
                    best_params=kwargs
    return best_params, best_score

def calc_metrics(result_df, method):
        try:
            silhouette = silhouette_score(
                result_df[["language_latitude", "language_longitude"]], result_df[f"{method}_cluster"]
            )
            print(f"Silhouette Score: {silhouette:.3f}")
        except:
            print(
                "Could not calculate Silhouette Score. This can happen if there's only one cluster or if all points belong to the same cluster."
            )
        return silhouette

def main():
    if METHOD=="kmeans":
        #best_k, best_score= best_kmeans(train)
        #print(best_k)
        kwargs= {"n_clusters":3, "random_state": 42}
        #k_clust_train, k_clust_masked=(kmeans_clusters(train, masked_df, method=METHOD, **kwargs))
        #print(k_clust_train.columns)
        #print(k_clust_masked.columns)
        #calc_metrics(k_clust_train, method=METHOD)
        #plot_geo_clusters(k_clust_train, cluster_col=f"{METHOD}_cluster")
    if METHOD=="dbscan":
        metadata_cols=  [col for col in train.columns if not col.startswith("GB")]
        combined= pd.concat([train[metadata_cols], masked_df[metadata_cols]])
        best_eps=find_optimal_eps(combined, min_samples=5)
        #print(f"suggested eps={best_eps}")
        kwargs= {"eps":9.2, #max neighbor radius distance
                 "min_samples":5, #higher for denser clusters
                 "metric":'euclidean', 
                 "metric_params":None, 
                 "algorithm":'auto', 
                 "leaf_size":30, 
                 "p":None, 
                 "n_jobs":-1}
        
        #db_clust_train, db_clust_masked=(other_clusters(train, masked_df, method=METHOD, **kwargs))
        #print(db_clust_train.columns)
        #print(db_clust_masked.columns)
        plot_geo_clusters(db_clust_train, cluster_col=f"{METHOD}_cluster")
        #calc_metrics(db_clust_train)
        

    elif METHOD=="hdbscan":
        #best_params, best_score= best_hdbscan(train)
        #print(best_params, best_score)
        #kwargs=best_params
        kwargs={"min_cluster_size": 25,
                "min_samples": None,
                #"max_cluster_size": 0.0,
                "cluster_selection_method": "eom"}
        #hdb_clust_train, hdb_clust_masked=(other_clusters(train, masked_df, method=METHOD, **kwargs))
        #print(hdb_clust_train.columns)
        #print(hdb_clust_masked.columns)
        #plot_geo_clusters(hdb_clust_train, cluster_col=f"{METHOD}_cluster")


main()