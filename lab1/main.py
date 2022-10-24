from pathlib import Path
import numpy as np
import time
import threading as t

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm

n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=44)
base_path = Path("datasets")

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
result_array = np.empty((30, 2, 10), dtype=float)


def test_not_threading_approach():
    for i, filename in tqdm(enumerate(selected_files)):
        dataset = np.genfromtxt(base_path / filename, delimiter=",")
        for j, clf in enumerate((GaussianNB(), DecisionTreeClassifier())):
            X, y = dataset[:, :-1], dataset[:, -1].astype(int)
            for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf.fit(X_train, y_train)
                predict = clf.predict(X_test)
                result_array[i, j, k] = accuracy_score(y_test, predict)


start = time.perf_counter()
test_not_threading_approach()
end = time.perf_counter()

print("Normal approach time:", end - start, "seconds")


# # With threading

threads: list[t.Thread] = []
array_lock = t.Lock()


def threading_rskfolds(clf, filepath, i, j):
    dataset = np.genfromtxt(filepath, delimiter=",")
    X, y = dataset[:, :-1], dataset[:, -1].astype(int)
    for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        with array_lock:
            result_array[i, j, k] = accuracy_score(y_test, predict)

def threading_rskfolds_2(clf, dataset, i, j):
    X, y = dataset[:, :-1], dataset[:, -1].astype(int)
    for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        with array_lock:  # aquire and release
            result_array[i, j, k] = accuracy_score(y_test, predict)

        # accuracy_score(y_test, predict)

result_array = np.empty((30, 2, 10), dtype=float)

start = time.perf_counter()
for i, filepath in tqdm(enumerate(selected_files)):
    dataset = np.genfromtxt(base_path/filepath, delimiter=",")
    for j, clf in enumerate((GaussianNB(), DecisionTreeClassifier())):
        thread = t.Thread(target=threading_rskfolds_2 ,kwargs={"clf": clf, "dataset": dataset, "i": i, "j": j})
        # thread = t.Thread(
        #     target=threading_rskfolds,
        #     kwargs={"clf": clf, "filepath": base_path / filepath, "i": i, "j": j},
        # )
        threads.append(thread)
        thread.start()

for thread in threads:
    thread.join()

end = time.perf_counter()

print("Threading approach: ", (end - start), "seconds")




# concurent futures
result_array = np.empty((30, 2, 10), dtype=float)
def threading_rskfolds_process_pool_executor(clf, dataset): # , i, j, lock):
    res = np.zeros(n_repeats * n_splits)
    X, y = dataset[:, :-1], dataset[:, -1].astype(int)
    for k, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        res[k] = accuracy_score(y_test, predict)
    return i ,j, res

import concurrent.futures


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    
    results = []
    start = time.perf_counter()
    for i, filepath in tqdm(enumerate(selected_files)):
        dataset = np.genfromtxt(base_path/filepath, delimiter=",")
        for j, clf in enumerate((GaussianNB(), DecisionTreeClassifier())):
            # thread = t.Thread(target=threading_rskfolds_2 ,kwargs={"clf": clf, "dataset": dataset, "i": i, "j": j})

            results.append(executor.submit(threading_rskfolds_process_pool_executor, clf, dataset))#, i, j, array_lock)
    executor.shutdown(wait=True)

for result in concurrent.futures.as_completed(results):
    (i, jj, res) = result.result()
    with array_lock:
        result_array[i, jj] = res
end = time.perf_counter()
print("Concurent.futures approach: ", (end - start), "seconds")

# print(result_array)
# with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#     future = executor.submit(pow, 323, 1235)
#     print(future.result())
