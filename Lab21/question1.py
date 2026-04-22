import numpy as np

# finding the Eucleadian Distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# now the centroid
def initialize_centroids(X, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(len(X), size=k, replace=False)
    return X[indices]


# Assign Clusters
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, c) for c in centroids]
        cluster_id = np.argmin(distances)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) == 0:
            # handle empty cluster by reinitializing randomly
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            new_centroids.append(np.mean(cluster_points, axis=0))
    return np.array(new_centroids)


# Convergence
def has_converged(old_centroids, new_centroids, tol=1e-4):
    shifts = np.linalg.norm(new_centroids - old_centroids, axis=1)
    return np.all(shifts < tol)


#Main K-Means Function
def kmeans(X, k, max_iters=100, tol=1e-4, seed=None):
    centroids = initialize_centroids(X, k, seed)

    for iteration in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        if has_converged(centroids, new_centroids, tol):
            print(f"Converged in {iteration+1} iterations")
            break

        centroids = new_centroids

    return centroids, clusters


# example
if __name__ == "__main__":
    X = np.array([
        [1, 2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])

    k = 2

    centroids, clusters = kmeans(X, k, seed=42)

    print("Final Centroids:", centroids)
    print("Cluster Assignments:", clusters)