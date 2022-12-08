import time
import matplotlib.pyplot as plt
import numpy as np
import convolve


# # ==================================
# def time_count(func: callable) -> callable:
#     def wrapped(*args, **kwargs):
#         start = time.perf_counter_ns()
#         res = func(*args, **kwargs)
#         stop = time.perf_counter_ns()
#         print(f"Total elapsed time: {(stop - start) / 1_000_000_000} seconds")
#         return res

#     return wrapped

# @time_count
# def conv2d_simple(
#     image: np.ndarray, padded_image: np.ndarray, kernel: np.ndarray
# ) -> np.ndarray:
#     new_image = np.zeros(image.shape, dtype=int)
#     for idx_row, row in enumerate(image):
#         for idx_col, _ in enumerate(row):
#             new_image[idx_row][idx_col] = np.sum(
#                 padded_image[
#                     idx_row : idx_row + kernel.shape[0],
#                     idx_col : idx_col + kernel.shape[1],
#                 ]
#                 * kernel
#             )
#     return new_image

# # ==================================


def conv_to_int(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    img *= 255
    print(img.shape)
    return img.astype(int)

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
img = plt.imread("lenna.png")
img = conv_to_int(img)

mask = np.asarray([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
], dtype=int)

def main():
    start = time.perf_counter()
    seqiential_conv_img = convolve.sequential_convolve(img, mask)
    stop = time.perf_counter()
    print("sequential exec time: ", stop - start)
    start = time.perf_counter()
    parallel_conv_img = convolve.parallel_convolve(img, mask)
    stop = time.perf_counter()
    print("parallel exec time: ", stop - start)
    print(parallel_conv_img.shape)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(seqiential_conv_img, cmap="gray")
    ax[2].imshow(parallel_conv_img, cmap="gray")
    fig.savefig("result.png")

if __name__ == "__main__":
    main()