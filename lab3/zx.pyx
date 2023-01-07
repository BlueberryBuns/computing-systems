from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = int

ctypedef np.int_t DTYPE_t

def sequential_convolve(np.ndarray image, np.ndarray kernel):
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    assert image.dtype == DTYPE and kernel.dtype == DTYPE

    cdef int image_h_max = image.shape[0]
    cdef int image_w_max = image.shape[1]
    cdef int kernel_h_max = kernel.shape[0]
    cdef int kernel_w_max = kernel.shape[1]
    cdef int kernel_mid_h_idx = kernel_h_max // 2
    cdef int kernel_mid_w_idx = kernel_w_max // 2
    cdef int i_max = image_h_max + 2 * kernel_mid_h_idx
    cdef int j_max = image_w_max + 2 * kernel_mid_w_idx
    cdef np.ndarray result = np.zeros([i_max, j_max], dtype=DTYPE)
    cdef int x, y, s, t, v, w
    cdef int kernel_h_iterator_from, kernel_h_iterator_to, kernel_w_iterator_from, kernel_w_iterator_to
    cdef DTYPE_t res_value

    for x in range(i_max):
        for y in range(j_max):
            kernel_h_iterator_from = max(kernel_mid_h_idx - x, -kernel_mid_h_idx)
            kernel_h_iterator_to = min((i_max - x) - kernel_mid_h_idx, kernel_mid_h_idx + 1)
            kernel_w_iterator_from = max(kernel_mid_w_idx - y, -kernel_mid_w_idx)
            kernel_w_iterator_to = min((j_max - y) - kernel_mid_w_idx, kernel_mid_w_idx + 1)
            res_value = 0
            for s in range(kernel_h_iterator_from, kernel_h_iterator_to):
                for t in range(kernel_w_iterator_from, kernel_w_iterator_to):
                    v = x - kernel_mid_h_idx + s
                    w = y - kernel_mid_w_idx + t
                    res_value += kernel[kernel_mid_h_idx - s, kernel_mid_w_idx - t] * image[v, w]
            result[x, y] = res_value
    return result

def parallel_convolve(np.ndarray image, np.ndarray kernel):
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    assert image.dtype == DTYPE and kernel.dtype == DTYPE
    cdef int image_h_max = image.shape[0]
    cdef int image_w_max = image.shape[1]
    cdef int kernel_h_max = kernel.shape[0]
    cdef int kernel_w_max = kernel.shape[1]
    cdef int kernel_mid_h_idx = kernel_h_max // 2
    cdef int kernel_mid_w_idx = kernel_w_max // 2
    cdef int i_max = image_h_max + 2 * kernel_mid_h_idx
    cdef int j_max = image_w_max + 2 * kernel_mid_w_idx
    cdef np.ndarray result = np.zeros([i_max, j_max], dtype=DTYPE)
    cdef long [:, :] result_view = result
    cdef long [:, :] image_view = image
    cdef long [:, :] kernel_view = kernel
    cdef int x, y, s, t, v, w
    cdef int kernel_h_iterator_from, kernel_h_iterator_to, kernel_w_iterator_from, kernel_w_iterator_to
    cdef long res_value

    for x in prange(i_max, nogil=True):
        for y in prange(j_max):
            kernel_h_iterator_from = max(kernel_mid_h_idx - x, -kernel_mid_h_idx)
            kernel_h_iterator_to = min((i_max - x) - kernel_mid_h_idx, kernel_mid_h_idx + 1)
            kernel_w_iterator_from = max(kernel_mid_w_idx - y, -kernel_mid_w_idx)
            kernel_w_iterator_to = min((j_max - y) - kernel_mid_w_idx, kernel_mid_w_idx + 1)
            res_value = 0
            for s in prange(kernel_h_iterator_from, kernel_h_iterator_to):
                for t in prange(kernel_w_iterator_from, kernel_w_iterator_to):
                    v = x - kernel_mid_h_idx + s
                    w = y - kernel_mid_w_idx + t
                    res_value += kernel_view[kernel_mid_h_idx - s, kernel_mid_w_idx - t] * image_view[v, w]
            result_view[x, y] = res_value
    return result_view