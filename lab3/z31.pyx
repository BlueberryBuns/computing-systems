from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = int

ctypedef np.int_t DTYPE_t

def parallel_convolve(np.ndarray image, np.ndarray kernel):
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    assert image.dtype == DTYPE and kernel.dtype == DTYPE
    cdef int vmax = image.shape[0]
    cdef int wmax = image.shape[1]
    cdef int smax = kernel.shape[0]
    cdef int tmax = kernel.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2 * smid
    cdef int ymax = wmax + 2 * tmid
    cdef np.ndarray result = np.zeros([xmax, ymax], dtype=DTYPE)
    cdef long [:, :] result_view = result
    cdef long [:, :] image_view = image
    cdef long [:, :] kernel_view = kernel 
    cdef int x, y, s, t, v, w
    cdef int s_from, s_to, t_from, t_to
    cdef long value

    for x in prange(xmax, nogil=True):
        for y in prange(ymax):
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in prange(s_from, s_to):
                for t in prange(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += kernel_view[smid - s, tmid - t] * image_view[v, w]
            result_view[x, y] = value
    return result_view
