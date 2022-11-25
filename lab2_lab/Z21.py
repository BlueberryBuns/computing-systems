from functools import partial
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import time
import concurrent.futures
from multiprocessing import get_context


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


def process_pixel(
    pixel_coords: tuple[int, int], kernel: np.ndarray, padded_image: np.ndarray
) -> np.ndarray:
    x, y = pixel_coords
    return np.sum(
        padded_image[x : x + kernel.shape[0], y : y + kernel.shape[1]] * kernel
    )


def process_row(row: np.ndarray, kernel: np.ndarray, row_id: int, result: np.ndarray):
    for idx_col, _ in enumerate(result[row_id]):
        result[row_id][idx_col] = np.sum(
            row[
                0 : kernel.shape[0],
                idx_col : idx_col + kernel.shape[1],
            ]
            * kernel
        )
    return result[row_id]


@time_count
def conv2d_multiprocess(
    image: np.ndarray, padded_image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    new_image = np.zeros(image.shape, dtype=int)
    coordinates = [(x, y) for x in range(image.shape[0]) for y in range(image.shape[1])]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=5, mp_context=get_context("fork")) as executor:
    #     values = executor.map(
    #         partial(process_pixel, kernel=kernel, padded_image=padded_image),
    #         coordinates,
    #     )
    # for v in values:
    #     print(v)
    with get_context("fork").Pool(5) as p:
        values = p.map(
            partial(process_pixel, kernel=kernel, padded_image=padded_image),
            coordinates,
        )
    for idx, (x, y) in enumerate(coordinates):
        new_image[x, y] = values[idx]
    return new_image


def main():
    kernel = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=int)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    lenna = plt.imread("lenna.png")
    padding_shape = tuple((i // 2 for i in kernel.shape))
    lenna = conv_to_int(lenna)
    padded_image = np.pad(lenna, padding_shape, mode="edge")
    new_image = conv2d_multiprocess(lenna, padded_image, kernel)
    ax[0].imshow(lenna, cmap="gray")
    ax[1].imshow(new_image, cmap="gray")

    fig.savefig("Zadanie_21.png")


if __name__ == "__main__":
    main()
