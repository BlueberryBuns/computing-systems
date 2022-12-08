import multiprocessing as mp

def foo():
    print(1)
    return 1

with mp.Pool(1) as p:
    values = p.map(foo, [])

