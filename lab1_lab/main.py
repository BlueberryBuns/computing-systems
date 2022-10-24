from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# 10/10
selected_files = (
    "breastcan.csv",
    "wisconsin.csv",
    "bupa.csv",
    "ionosphere.csv",
    "soybean.csv",
    # "phoneme.csv",
    "breastcancoimbra.csv",
    # "sonar.csv",
    "balance.csv",
    "ring.csv",
    "haberman.csv",
    "ecoli4.csv",
    "heart.csv",
    "australian.csv",
    "banknote.csv",
    "spambase.csv",
    "mammographic.csv",
    "liver.csv",
    "waveform.csv",
    "popfailures.csv",
    "monk-2.csv",
    "german.csv",
    "iris.csv",
    "glass2.csv",
    "coil2000.csv",
    "titanic.csv",
    "wine.csv",
    "pima.csv",
    "spectfheart.csv",
    "appendicitis.csv",
    "glass4.csv",
    "glass5.csv"
    # "cryotherapy.csv",
    # "twonorm.csv",
    # "vowel0.csv",
)
base_folder = Path("datasets") # zbiory testowe umieszczone w folderze datasets

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

sequential_results_array = np.zeros((30, 2, 10), dtype=float)


def sequential_benchmark():
    for i, filename in enumerate(selected_files):
        dataset = np.genfromtxt(base_folder / filename, delimiter=",")
        X, y = dataset[:, :-1], dataset[:, -1].astype(int)
        for j, model in enumerate((GaussianNB(), DecisionTreeClassifier())):
            for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                sequential_results_array[i, j, k] = accuracy_score(y_test, y_pred)


start = time.perf_counter()
sequential_benchmark()
end = time.perf_counter()
print(sequential_results_array)
print("Standard approach: ", end - start, "secosnds elapsed")


import threading as thr

threading_results_array = np.zeros((30, 2, 10), dtype=float)
array_lock = thr.Lock()


def threading_benchmark_foo(
    filepath: str, i: int, j: int, model, lock: thr.Lock, results_array: np.ndarray
):
    dataset = np.genfromtxt(filepath, delimiter=",")
    X, y = dataset[:, :-1], dataset[:, -1].astype(int)
    for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        with lock:
            results_array[i, j, k] = accuracy_score(y_test, y_pred)


threads: list[thr.Thread] = []
start_threading = time.perf_counter()
for i, filename in enumerate(selected_files):
    dataset_path = base_folder / filename
    for j, model in enumerate((GaussianNB(), DecisionTreeClassifier())):
        t = thr.Thread(
            target=threading_benchmark_foo,
            kwargs={
                "filepath": dataset_path,
                "i": i,
                "j": j,
                "model": model,
                "lock": array_lock,
                "results_array": threading_results_array,
            },
        )
        t.start()
        threads.append(t)

for t in threads:
    t.join()

end_t = time.perf_counter()
print(threading_results_array.mean(axis=2))

print("Threading approach: ", end_t - start_threading, "secosnds elapsed")
