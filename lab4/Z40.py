import random

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from sklearn.metrics import DistanceMetric

from sklearn.datasets import make_classification
def normalize(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    return img


class KMeansCustom:
    def __init__(self, n_clusters: int = 5, iterator_limit: int = 500) -> None:
        self.n_clusters = n_clusters
        self.iterator_limit = iterator_limit
        self.centroids: np.ndarray

    def initlialize_centroids(self, X: np.ndarray):
        # self.centroids = np.zeros((self.n_clusters, len(X.shape)))
        # self.centroids[0] = random.choice(X)
        self.centroids = X[np.random.choice(len(X), size=self.n_clusters, replace=False)]
        print(self.centroids)
        # print(self.centroids)
        # for iter in range(1, self.n_clusters):
        #     distances = np.min(
        #         [
        #             self.distance(centroid, X)
        #             for centroid in self.centroids
        #         ],
        #         axis=0,
        #     )
        #     distances /= np.sum(distances)
        #     new_centroid_index = np.random.choice(len(X), size=1, p=distances)
        #     self.centroids[iter] = X[new_centroid_index]
        #     print(self.centroids)

    def fit(self, X: np.ndarray):
        self.initlialize_centroids(X)
        x = self.fit_centroids(X)
        return x


    def fit_centroids(self, X: np.ndarray):
        for iteration in range(self.iterator_limit):
            # Assign each datapoint to nearest centroid
            assigned_points = [[] for _ in range(self.n_clusters)]
            for point in X:
                distances = self.distance(point, self.centroids)
                # res = np.min(distances, axis=1)
                assigned_centroid_idx = np.argmin(distances)
                assigned_points[assigned_centroid_idx].append(point)
            # import pdb; pdb.set_trace()
            new_centroids = np.asarray(
                [np.mean(cluster, axis=0) for cluster in assigned_points]
            )
            if np.isin(new_centroids, self.centroids).all():
                print(iteration, "finished at")

                return assigned_points
            self.centroids = new_centroids
            print(iteration)
            # print(self.centroids)
            # import pdb;pdb.set_trace()

    @staticmethod
    def distance(point: np.ndarray, data: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((data - point) ** 2, axis=1))
        # dist = DistanceMetric.get_metric('euclidean')
        # # distance_array = np.zeros(data.shape)
        # distance_array = dist.pairwise(data, points) 
        # print(distance_array)
        # # print(x:=dist.rpairwise(data, point))
        # return distance_array

def main():
    kmeans = KMeansCustom()
    centers = 3
    # X, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X, y = make_classification(1000, 2, n_informative=2, n_redundant=0)
    print(X)
    # print(X, true_labels)
    assignments = kmeans.fit(X)
    # np.zeros((X.shape[1], X.shape[1]))
    # print(assignments[0])
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["blue", "green", "cyan", "yellow", "black"]
    print(len(assignments))
    for idx, data in enumerate(assignments):
        for x in data:
            ax.scatter(x[0], x[1], c=colors[idx])
    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="+")
    fig.savefig("res.jpg")


if __name__ == "__main__":
    main()
