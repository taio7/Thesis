from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

#CODE BY TIAGO

def geo_clustering(df, method="kmeans", **kwargs):
    """
    Perform clustering on geographical data.

    @param df: DataFrame with 'label', 'latitude', and 'longitude' columns
    @param method: Clustering method to use ('kmeans' or 'dbscan')
    @param kwargs: Additional parameters for the clustering method
        For KMeans: n_clusters (int), random_state (int), etc.
        For DBSCAN: eps (float), min_samples (int), etc.
    @return: DataFrame with an additional 'cluster' column
    """

    # Validate input DataFrame
    required_columns = ["language_name", "language_latitude", "language_longitude"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing_columns}"
        )

    # Extract coordinates for clustering
    X = df[["language_latitude", "language_longitude"]].values

    # Perform clustering based on the specified method
    if method.lower() == "kmeans":
        # Default parameters for KMeans
        n_clusters = kwargs.get("n_clusters", 5)
        random_state = kwargs.get("random_state", 1305)

        # Create and fit the KMeans model
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["n_clusters", "random_state"]
            },
        )
        clusters = kmeans.fit_predict(X)

    elif method.lower() == "dbscan":
        # Default parameters for DBSCAN
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)

        # Create and fit the DBSCAN model
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            **{k: v for k, v in kwargs.items() if k not in ["eps", "min_samples"]},
        )
        clusters = dbscan.fit_predict(X)
    
    elif method.lower()== "hdbscan":
        min_cluster_size = kwargs.get("min_cluster_size", 10)
        cluster_selection_method=kwargs.get("cluster_selection_method", "eom")
        
        hdbscan= HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            **{k: v for k, v in kwargs.items() if k not in ["min_cluster_size", "cluster_selection_method"]}
        )
        clusters=hdbscan.fit_predict(X)
    else:
        raise ValueError(
            f"Unsupported clustering method: {method}. Use 'kmeans' or 'dbscan'."
        )

    # Add cluster assignments to the DataFrame
    result_df = df.copy()
    result_df[f"{method}_cluster"] = clusters
    
    if method=="kmeans":
        model=kmeans
        return result_df, model

    return result_df


def generate_random_clusters(
    n_clusters=5, points_per_cluster=20, cluster_radius=5, seed=1305
):
    """
    Generate random geographical data with clear clusters.

    @param n_clusters: Number of clusters to generate
    @param points_per_cluster: Number of points per cluster
    @param cluster_radius: Radius of each cluster in degrees
    @param seed: Random seed for reproducibility
    @return: DataFrame with 'label', 'latitude', and 'longitude' columns
    """

    data = []

    # Define cluster centers across the globe
    centers = [
        (40.7128, -74.0060),  # New York
        (51.5074, -0.1278),  # London
        (35.6762, 139.6503),  # Tokyo
        (-33.8688, 151.2093),  # Sydney
        (1.3521, 103.8198),  # Singapore
        (-22.9068, -43.1729),  # Rio de Janeiro
        (37.7749, -122.4194),  # San Francisco
        (55.7558, 37.6173),  # Moscow
        (28.6139, 77.2090),  # New Delhi
        (30.0444, 31.2357),  # Cairo
    ]

    # Use only the required number of centers
    centers = centers[:n_clusters]

    # Generate points around each center
    random.seed(seed)
    for i, (lat, lon) in enumerate(centers):
        for j in range(points_per_cluster):
            # Add some random variation within the cluster_radius
            random_lat = lat + np.random.uniform(-cluster_radius, cluster_radius)
            random_lon = lon + np.random.uniform(-cluster_radius, cluster_radius)

            # Keep latitude within valid range
            random_lat = max(min(random_lat, 90), -90)

            # Adjust longitude to keep it within valid range (-180 to 180)
            random_lon = (random_lon + 180) % 360 - 180

            label = f"Point_{i}_{j}"
            data.append(
                {
                    "label": label,
                    "latitude": random_lat,
                    "longitude": random_lon,
                    "true_cluster": i,
                }
            )

    return pd.DataFrame(data)


def plot_geo_clusters(df, title="Geographical Clusters", cluster_col="cluster"):
    """
    Plot geographical data on a world map, colored by cluster.

    @param df: DataFrame with 'latitude', 'longitude', and 'cluster' columns
    @param title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    m = Basemap(
        projection="mill", llcrnrlat=-60, urcrnrlat=85, llcrnrlon=-180, urcrnrlon=180
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="aqua")
    m.fillcontinents(color="white", lake_color="aqua")

    # Convert lat/lon to map coordinates
    x, y = m(df["language_longitude"].values, df["language_latitude"].values)

    # Plot points colored by cluster
    scatter = m.scatter(x, y, c=df[cluster_col], cmap="viridis", s=50, alpha=0.7)

    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.show()


def find_optimal_eps(df, min_samples=5):
    """
    Find optimal eps parameter for DBSCAN clustering.

    @param df: DataFrame with 'latitude' and 'longitude' columns
    @param min_samples: min_samples parameter for DBSCAN
    @return: Suggested eps value
    """
    coords = df[["language_latitude", "language_longitude"]].values

    # Calculate distances using NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(coords)
    distances, _ = neighbors_fit.kneighbors(coords)

    # Sort distances to the kth nearest neighbor
    distances = np.sort(distances[:, -1])

    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title("K-distance Graph (k={})".format(min_samples))
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Distance to {}th nearest neighbor".format(min_samples))
    plt.grid(True)
    plt.show()

    # Return the suggested eps value at the "elbow" point -- just an heuristic!
    distances_diff = np.diff(distances)
    elbow_index = np.argmax(distances_diff) + 1
    return distances[elbow_index]


def main():
    parser = argparse.ArgumentParser(description="Geographical data clustering")
    parser.add_argument(
        "--method",
        choices=["kmeans", "dbscan"],
        default="kmeans",
        help="Clustering method",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=5, help="Number of clusters (for KMeans)"
    )
    parser.add_argument(
        "--eps", type=float, default=None, help="Epsilon parameter (for DBSCAN)"
    )
    parser.add_argument(
        "--min_samples", type=int, default=5, help="Min samples parameter (for DBSCAN)"
    )
    parser.add_argument(
        "--points_per_cluster",
        type=int,
        default=20,
        help="Points per cluster in random data",
    )
    parser.add_argument(
        "--cluster_radius",
        type=float,
        default=5.0,
        help="Radius of clusters in random data (degrees)",
    )
    parser.add_argument(
        "--find_eps", action="store_true", help="Find optimal eps parameter for DBSCAN"
    )

    args = parser.parse_args()

    # Generate random data
    df = generate_random_clusters(
        n_clusters=args.n_clusters,
        points_per_cluster=args.points_per_cluster,
        cluster_radius=args.cluster_radius,
    )

    print(f"Generated random data with {len(df)} points in {args.n_clusters} clusters")

    # Plot original data with true clusters
    plt.figure(figsize=(12, 8))
    plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["true_cluster"],
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    plt.colorbar(label="True Cluster")
    plt.title("Original Data with True Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()

    # Find optimal eps for DBSCAN if requested
    if args.find_eps:
        suggested_eps = find_optimal_eps(df, min_samples=args.min_samples)
        print(f"Suggested eps parameter for DBSCAN: {suggested_eps}")
        if args.method == "dbscan" and args.eps is None:
            args.eps = suggested_eps

    # Perform clustering
    kwargs = {}
    if args.method == "kmeans":
        kwargs["n_clusters"] = args.n_clusters
    elif args.method == "dbscan":
        if args.eps is not None:
            kwargs["eps"] = args.eps
        else:
            # If eps not provided, find a reasonable value
            coords = df[["latitude", "longitude"]].values
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(coords)
            distances, _ = nn.kneighbors(coords)
            kwargs["eps"] = np.mean(distances[:, 1]) * 2

        kwargs["min_samples"] = args.min_samples

    result_df = geo_clustering(df, method=args.method, **kwargs)

    # Display clustering results
    print(f"\nClustering results using {args.method.upper()}:")
    print(f"Number of clusters found: {result_df['cluster'].nunique()}")

    # Plot clusters
    plot_geo_clusters(
        result_df, title=f"Geographical Clusters using {args.method.upper()}"
    )

    # Calculate clustering metrics
    if "true_cluster" in result_df.columns:
        from sklearn.metrics import adjusted_rand_score, silhouette_score

        ari = adjusted_rand_score(result_df["true_cluster"], result_df["cluster"])
        print(f"\nAdjusted Rand Index: {ari:.3f}")

        try:
            silhouette = silhouette_score(
                result_df[["latitude", "longitude"]], result_df["cluster"]
            )
            print(f"Silhouette Score: {silhouette:.3f}")
        except:
            print(
                "Could not calculate Silhouette Score. This can happen if there's only one cluster or if all points belong to the same cluster."
            )


if __name__ == "__main__":
    main()