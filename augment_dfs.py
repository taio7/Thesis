from geocluster import geo_clustering, plot_geo_clusters, find_optimal_eps
import pandas as pd
import numpy as np
from pathlib import Path
from config_dfs import select_dfs
from sklearn.metrics import adjusted_rand_score, silhouette_score


"""augment with clusters"""


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "final_dfs"
WITH_NAN_PATH= DATA_PATH/ "with_nans"
NO_NAN_PATH= DATA_PATH/ "no_nans"
with_nans=False
METHOD=None


#train, masked_df, dev, masked_positions= select_dfs(with_nans=None)


def create_clusters(train, masked_df, method, clusters_target=None, **kwargs):
    """combine metadata columns of train and dev only so no data leakage,
    since I cannot use trained model on the dev data"""
    #print(train.index)
    #print(masked_df.index)
    train= train.reset_index(drop=True)
    masked_df= masked_df.reset_index(drop=True)
    required_columns = ["language_name", "language_latitude", "language_longitude"]
    combined= pd.concat([train[required_columns], masked_df[required_columns]], ignore_index=True)
    
    original_col=f"{method}_cluster"
    clustered_df= geo_clustering(combined, method, **kwargs)
    if clusters_target=="optimal":
        cluster_col= original_col
    else:
        cluster_col= f"{clusters_target}_{method}_cluster"
        clustered_df[cluster_col]=clustered_df[original_col] #copy so both are kept 

    calc_metrics(clustered_df, method=method)
    #plot_geo_clusters(clustered_df, title=f"{method} Geographical Clusters", cluster_col=cluster_col)

    train_clustered=train.copy()
    masked_df_clustered=masked_df.copy()

    train_clustered[cluster_col]= clustered_df[cluster_col].iloc[:len(train)].values
    masked_df_clustered[cluster_col]= clustered_df[cluster_col].iloc[len(train):].values
    return train_clustered, masked_df_clustered


def best_kmeans(train, k=range(2, 20), init= ["k-means++", "random"], algorithm=["lloyd", "elkan"], random_state=42):
    best_params=None
    best_score=-1
    for i in k:
        for y in init:
            for a in algorithm:
                kwargs={"n_clusters":i, "init":y, "algorithm": a, "random_state": 42}
                train_clustered= geo_clustering(train, method="kmeans", **kwargs)
                score= calc_metrics(train_clustered, method="kmeans")
                if score>best_score:
                    best_score=score
                    best_params=kwargs
    return best_params, best_score


def best_hdbscan(train, min_cluster_size= range(2,20), min_samples=[None, 5, 7], max_cluster_size=range(2, 20), cluster_selection_epsilon= 9.2, 
                 target_cluster_n=12, select_cl_num=False):
    best_params=None
    best_score=-1
    if select_cl_num==True:
        eps_values = np.arange(cluster_selection_epsilon * 0.5, cluster_selection_epsilon * 1.5, 0.2)
    else:
        eps_values=[cluster_selection_epsilon] if isinstance(cluster_selection_epsilon, (int, float)) else cluster_selection_epsilon
    for mcs in min_cluster_size:
        for ms in min_samples:
            for maxcs in max_cluster_size:
                for cse in eps_values:
                    kwargs={"min_cluster_size": mcs,
                    "min_samples": ms,
                    "max_cluster_size":maxcs,
                    "cluster_selection_epsilon":cse,
                    }
                    train_clustered= geo_clustering(train, method="hdbscan", **kwargs)
                    if select_cl_num==True:
                        cluster_labels=train_clustered["hdbscan_cluster"]
                        cluster_n=len(set(cluster_labels))
                        if cluster_n!=target_cluster_n:
                            continue
                        score=calc_metrics(train_clustered, method="hdbscan")
                        if score>best_score:
                            best_score=score
                            best_params=kwargs 
                    else:
                        score=calc_metrics(train_clustered, method="hdbscan")
                        if score>best_score:
                            best_score=score
                            best_params=kwargs
    return best_params, best_score

def best_dbscan(train, eps=9.2, min_samples=range(2, 30), metric=["euclidean", "haversine"], algorithm=["auto","brute"],
                target_cluster_n=24, select_cl_num=False):
    best_params=None
    best_score=-1
    if select_cl_num==True:
        eps_values = np.arange(eps * 0.1, eps * 3, 0.5)
    else:
        eps_values=[eps] if isinstance(eps, (int, float)) else eps
    for e in eps_values:
            for ms in min_samples:
                for met in metric:
                    for a in algorithm:
                        if met=="haversine" and a == "brute":
                            continue
                        kwargs={"eps":e, #max neighbor radius distance
                        "min_samples":ms, #higher for denser clusters
                        "metric":met,  
                        "algorithm":a,  
                        "n_jobs":-1
                        }
                        train_clustered= geo_clustering(train, method="dbscan", **kwargs)
                        if select_cl_num==True:
                            cluster_labels=train_clustered["dbscan_cluster"]
                            cluster_n=len(set(cluster_labels))
                            if cluster_n!=target_cluster_n:
                                continue
                            score=calc_metrics(train_clustered, method="dbscan")
                            if score>best_score:
                                best_score=score
                                best_params=kwargs 
                        else:
                            score=calc_metrics(train_clustered, method="dbscan")
                            if score>best_score:
                                best_score=score
                                best_params=kwargs   
    return best_params, best_score

def calc_metrics(result_df, method):
        silhouette=-1
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
    
    #optimize for combined train and dev
    metadata_cols=  [col for col in train.columns if not col.startswith("GB")]
    combined= pd.concat([train[metadata_cols], masked_df[metadata_cols]])


    if METHOD=="kmeans":
        #best_params, best_score= best_kmeans(combined)
        #print(best_params, best_score)
#{'n_clusters': 3, 'init': 'k-means++', 'algorithm': 'lloyd', 'random_state': 42}
        #kwargs= {'n_clusters': 3, 'init': 'k-means++', 'algorithm': 'lloyd', 'random_state': 42}
        kwargs={'n_clusters': 24, 
                'init': 'k-means++', #defaults
                'algorithm': 'lloyd', 
                'random_state': 42}
        k_clust_train, k_clust_masked=(create_clusters(train, masked_df, method=METHOD, clusters_target=None, **kwargs))
        #print(k_clust_train.columns)
        #print(k_clust_masked.columns)
        

    if METHOD=="dbscan":
        
        #best_eps=find_optimal_eps(combined, min_samples=3)
        #print(f"suggested eps={best_eps}")
        #best_params, best_score=best_dbscan(combined, select_cl_num=True, target_cluster_n=24)
        #print(best_params)
        #print(best_score)

        #kwargs={'eps': 7.6, 'min_samples': 7, 'metric': 'euclidean', 'algorithm': 'auto', 'n_jobs': -1} #for 12 cl
        #kwargs={'eps': 9.2, 'min_samples': 5, 'metric': 'euclidean', 'algorithm': 'auto', 'n_jobs': -1} #for optimal cluster n

        kwargs={'eps': 5.42, 'min_samples': 4, 'metric': 'euclidean', 'algorithm': 'auto', 'n_jobs': -1} #for 24 cl, 0.374
        
        
        db_clust_train, db_clust_masked=(create_clusters(train, masked_df, method=METHOD, clusters_target=None, **kwargs))
        #print(db_clust_train.columns)
        #print(db_clust_masked.columns)
        
        

    elif METHOD=="hdbscan":
        #best_params, best_score= best_hdbscan(combined, select_cl_num=True, target_cluster_n=24)
        #print(best_params, best_score)
        #kwargs=best_params
        
        #kwargs={'min_cluster_size': 2, 'min_samples': 5, 'cluster_selection_method': 'eom', 'max_cluster_size': 15, 'cluster_selection_epsilon': 9.2} #Silhouette Score: 0.565
        #kwargs={'min_cluster_size': 2, 'min_samples': 7, 'max_cluster_size': 15, 'cluster_selection_epsilon': 7.8} #12 clusters, 0.463
        kwargs= {'min_cluster_size': 2, 'min_samples': 5, 'max_cluster_size': 11, 'cluster_selection_epsilon': 5.4} #24 clusters, 0.383
        hdb_clust_train, hdb_clust_masked=(create_clusters(train, masked_df, method=METHOD, clusters_target=None, **kwargs))
        #print(hdb_clust_train.columns)
        #print(hdb_clust_masked.columns)
        

if __name__ == "__main__":
    main()