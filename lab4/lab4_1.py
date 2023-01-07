import random

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from sklearn.datasets import make_blobs


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("Hello from process {} out of {}".format(rank, size))


def normalize(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    return img


class KmeansMPI:
    def __init__(self, n_clusters: int = 5, iterator_limit: int = 500) -> None:
        self.n_clusters = n_clusters
        self.iterator_limit = iterator_limit
        self.centroids: np.ndarray
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def initlialize_centroids(self, X_train: np.ndarray):
        if self.rank == 0:
            centroids = np.zeros((self.n_clusters, len(X_train.shape)))
            ave_centroids, res_centroids = divmod(centroids.shape[0], self.size)
            # ave_dataset, res_dataset = divmod(X_train.shape[0], self.size)
            counts_centroids = [ave_centroids + 1 if p < res_centroids else ave_centroids for p in range(self.size)]
            # counts_dataset = [ave_centroids + 1 if p < res_centroids else ave_centroids for p in range(self.size)]
            print(counts_centroids)
            # print(counts_dataset)
            print(f"{ave_centroids=}")
            # print(f"{ave_dataset=}")
            print(f"{self.n_clusters=}")
            print(f"{X_train=}")
            print(f"{centroids=}")
            print(f"{res_centroids=}")
            # print(f"{res_dataset=}")
            starts_centroids = [sum(counts_centroids[:p]) for p in range(self.size)]
            ends_centroids = [sum(counts_centroids[: p + 1]) for p in range(self.size)]
            # starts_dataset = [sum(counts_dataset[:p]) for p in range(self.size)]
            # ends_dataset = [sum(counts_dataset[: p + 1]) for p in range(self.size)]
            centroids[0] = random.choice(X_train)

            for iter in range(1, self.n_clusters):
                distances = np.min(
                    [
                        self.euclidean_distance(centroid, X_train)
                        for centroid in self.centroids
                    ],
                    axis=0,
                )
                distances /= np.sum(distances)
                new_centroid_index = np.random.choice(len(X_train), size=1, p=distances)
                centroids[iter] = X_train[new_centroid_index]
            # print(self.centroids) 
            centroids_data_ = [centroids[starts_centroids[p] : ends_centroids[p]] for p in range(self.size)]
            # dataset_split = [X_train[starts_dataset[p] : ends_dataset[p]] for p in range(self.size)]
        else:
            centroids_data_ = None

        centroids_data = self.comm.scatter(centroids_data_, root=0)
        # X_train_data = self.comm.scatter(data, root=0)

        print(f"Process {self.rank} received data {centroids_data}")

    def fit(self, X_train: np.ndarray):
        self.centroids = np.zeros((self.n_clusters, len(X_train.shape)))
        self.initlialize_centroids(X_train)
        self.fit_centroids(X_train)
        return self

    def fit_centroids(self, X_train: np.ndarray):
        for iteration in range(self.iterator_limit):
            # Assign each datapoint to nearest centroid
            assigned_points = [[] for _ in range(self.n_clusters)]
            for point in X_train:
                distances = self.euclidean_distance(point, self.centroids)
                assigned_centroid_idx = np.argmin(distances)
                assigned_points[assigned_centroid_idx].append(point)
                # import pdb; pdb.set_trace()
            new_centroids = np.asarray(
                [np.mean(cluster, axis=0) for cluster in assigned_points]
            )
            if np.isin(new_centroids, self.centroids).all():
                print(iteration, "finished at")
                return
            self.centroids = new_centroids
            print(iteration)

    @staticmethod
    def euclidean_distance(point: np.ndarray, data: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((data - point) ** 2, axis=1))


def main():
    centers = 7
    kmeans = KmeansMPI(n_clusters=centers)
    X_train, true_labels = make_blobs(n_samples=100000, centers=centers, random_state=1234)
    # print(X_train, true_labels)
    kmeans.fit(X_train)
    # print(kmeans.centroids)
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.scatter(X_train[:,0], X_train[:,1], c=true_labels)
    # ax.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c="black", marker="+")

    # fig.savefig("res.jpg")


if __name__ == "__main__":
    main()
