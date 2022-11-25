import numpy as np
import matplotlib.pyplot as plt
import time


def conv_to_int(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(int)


def time_count(func: callable) -> callable:
    def wrapped(*args, **kwargs):
        start = time.perf_counter_ns()
        res = func(*args, **kwargs)
        stop = time.perf_counter_ns()
        print(f"Total elapsed time: {(stop - start) / 1_000_000_000} seconds")
        return res

    return wrapped


@time_count
def conv2d_simple(
    image: np.ndarray, padded_image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    new_image = np.zeros(image.shape, dtype=int)
    for idx_row, row in enumerate(image):
        for idx_col, _ in enumerate(row):
            new_image[idx_row][idx_col] = np.sum(
                padded_image[
                    idx_row : idx_row + kernel.shape[0],
                    idx_col : idx_col + kernel.shape[1],
                ]
                * kernel
            )
    return new_image


def main():
    kernel = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    lenna = plt.imread("lenna.png")
    padding_shape = tuple((i // 2 for i in kernel.shape))
    lenna = conv_to_int(lenna)
    padded_image = np.pad(lenna, padding_shape, mode="edge")
    new_image = conv2d_simple(lenna, padded_image, kernel)
    ax[0].imshow(lenna, cmap="gray")
    ax[1].imshow(new_image, cmap="gray")

    fig.savefig("Zadanie_20.png")


if __name__ == "__main__":
    main()
