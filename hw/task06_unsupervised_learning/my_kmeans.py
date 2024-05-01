import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        m, n = X.shape
        self.centroids = np.array(X.iloc[np.random.choice(m, self.n_clusters, replace=False)])

        for _ in range(self.max_iter):
            self.labels_ = [self._closest_centroid(x) for _, x in X.iterrows()]
            new_centroids = np.array([X[np.array(self.labels_) == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) <= self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return [self._closest_centroid(x) for _, x in X.iterrows()]

    def _closest_centroid(self, x):
        distances = np.linalg.norm(self.centroids - np.array(x), axis=1)
        return np.argmin(distances)
        