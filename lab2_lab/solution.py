from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import get_context
import time


def conv_to_int(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(int)


kernel = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=int)
fig, ax = plt.subplots(1, 3, figsize=(10, 10))
lenna = plt.imread("lenna.png")
padding_shape = tuple((i // 2 for i in kernel.shape))
lenna = conv_to_int(lenna)
padded_image = np.pad(lenna, padding_shape, mode="edge")


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


@time_count
def conv2d_multi(
    image: np.ndarray, padded_image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    new_image = np.zeros(image.shape, dtype=int)

    coordinates = [(x, y) for x in range(image.shape[0]) for y in range(image.shape[1])]

    with get_context("fork").Pool(5) as p:
        values = p.map(
            partial(process_pixel, kernel=kernel, padded_image=padded_image),
            coordinates,
        )

    for idx, (x, y) in enumerate(coordinates):
        new_image[x, y] = values[idx]

    return new_image


def process_pixel(
    pixel_coords: tuple[int, int], kernel: np.ndarray, padded_image: np.ndarray
) -> np.ndarray:
    x, y = pixel_coords
    return np.sum(
        padded_image[x : x + kernel.shape[0], y : y + kernel.shape[1]] * kernel
    )


def main():

    new_image = conv2d_simple(lenna, padded_image, kernel)
    ax[0].imshow(lenna, cmap="gray")
    ax[1].imshow(new_image, cmap="gray")
    ax[0].set_title("Original Lenna.png")
    ax[1].set_title("Lenna conv, singleprocess")
    ax[2].set_title("Lenna conv, multiprocess")
    fig.tight_layout()
    fig.savefig("output.png")


if __name__ == "__main__":
    main()
