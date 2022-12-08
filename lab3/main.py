import time
import convolve


# def say_hello_to(name):
#     print("abc")
#     print("Hello %s!" % name)


# def calculate_score(a, b, N):
#     suma = 0
#     for _ in range(N):
#         suma += (a + b) * N
#     return suma


# start = time.perf_counter()
# calculate_score(10.1, 15.6, 200000)
# stop = time.perf_counter()

# print("py: ", stop - start)

# start = time.perf_counter()
# # hello.calculate_score(10.1, 15.6, 2000000)
# stop = time.perf_counter()

# print("pyx: ", stop - start)

# start = time.perf_counter()
# calculate_score(10.1, 15.6, 200000)
# stop = time.perf_counter()

# print("py: ", stop - start)


# convolve.naive_convolve()
import numpy as np
x = np.asarray([1.0,2.0,3.0])

res = convolve.convolve(x, 2.1)
for i in res:
    print(i)