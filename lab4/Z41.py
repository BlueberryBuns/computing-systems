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


class KmeansMPI:
    def __init__(self, n_clusters: int = 5, iterator_limit: int = 500) -> None:
        self.n_clusters = n_clusters
        self.iterator_limit = iterator_limit
        centroids_: np.ndarray
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def initlialize_centroids(self, X: np.ndarray):
        print("test")
        if self.rank == 0:
            centroids = X[np.random.choice(len(X), size=self.n_clusters, replace=False)]
            # ave_centroids, res_centroids = divmod(centroids.shape[0], self.size)
            ave_dataset, res_dataset = divmod(X.shape[0], self.size)
            # counts_centroids = [ave_centroids + 1 if p < res_centroids else ave_centroids for p in range(self.size)]
            counts_dataset = [ave_dataset + 1 if p < res_dataset else ave_dataset for p in range(self.size)]
            # print(counts_centroids)
            # print(counts_dataset)
            # print(f"{ave_centroids=}")
            # print(f"{ave_dataset=}")
            # print(f"{self.n_clusters=}")
            # print(f"{X=}")
            # print(f"{centroids=}")
            # print(f"{res_centroids=}")
            # print(f"{res_dataset=}")
            # starts_centroids = [sum(counts_centroids[:p]) for p in range(self.size)]
            # ends_centroids = [sum(counts_centroids[: p + 1]) for p in range(self.size)]
            starts_dataset = [sum(counts_dataset[:p]) for p in range(self.size)]
            ends_dataset = [sum(counts_dataset[: p + 1]) for p in range(self.size)]
            X = [X[starts_dataset[p]:ends_dataset[p]] for p in range(self.size)]
        else:
            x_data = None
            centroids_data = None
            
        x_data = self.comm.scatter(X, root=0)
        centroids_data = self.comm.bcast(centroids, root=0)

        print(f"Process {self.rank} received data {x_data}")
        print(f"Process {self.rank} received data {centroids_data}")

    def fit(self, X: np.ndarray):
        self.initlialize_centroids(X)
        # x = self.fit_centroids(X)
        


    def fit_centroids(self, X: np.ndarray, centroids_):
        for iteration in range(self.iterator_limit):
            # Assign each datapoint to nearest centroid
            assigned_points = [[] for _ in range(self.n_clusters)]
            for point in X:
                distances = self.distance(point, centroids_)
                # res = np.min(distances, axis=1)
                assigned_centroid_idx = np.argmin(distances)
                assigned_points[assigned_centroid_idx].append(point)
            # import pdb; pdb.set_trace()
            new_centroids = np.asarray(
                [np.mean(cluster, axis=0) for cluster in assigned_points]
            )
            if np.isin(new_centroids, centroids_).all():
                print(iteration, "finished at")

                return assigned_points
            centroids_ = new_centroids
            print(iteration)
            # print(centroids_)
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
    kmeans = KmeansMPI()
    centers = 3
    # X, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X, y = make_classification(1000, 2, n_informative=2, n_redundant=0)
    # print(X, true_labels)
    assignments = kmeans.fit(X)
    # np.zeros((X.shape[1], X.shape[1]))
    # print(assignments[0])
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["blue", "green", "cyan", "yellow", "black"]

    # for idx, data in enumerate(assignments):
    #     for x in data:
    #         ax.scatter(x[0], x[1], c=colors[idx])
    # ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c="red", marker="+")
    # fig.savefig("res.jpg")


if __name__ == "__main__":
    main()
