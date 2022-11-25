import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import concurrent.futures


def conv_to_int(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(int)


def multiply(a: np.ndarray, b: np.ndarray):
    result = 0
    for x, row in enumerate(a):
        for y, _ in enumerate(row):
            result += a[x, y] * b[x, y]
    return result


def convolve_image_simple(
    img: np.ndarray, padded_image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:

    new_image = np.zeros(img.shape, dtype=int)

    for idx_row, row in enumerate(img):
        for idx_col, _ in enumerate(row):
            new_image[idx_row][idx_col] = np.sum(
                padded_image[
                    idx_row : idx_row + kernel.shape[0],
                    idx_col : idx_col + kernel.shape[1],
                ]
                * kernel
            )
    return new_image


def convolve_image_multi(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ...


def main():

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    lenna = plt.imread("lenna.png")
    lenna = conv_to_int(lenna)

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    padding_shape = tuple((dim // 2 for dim in kernel.shape))
    padded_image = np.pad(lenna, padding_shape, mode="edge")

    img = conv_to_int(lenna)
    new_image_1 = convolve_image_simple(img, padded_image, kernel)
    new_image_2 = convolve_image_multi(img, kernel)

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    ax[0].imshow(lenna, cmap="gray")
    ax[1].imshow(padded_image, cmap="gray")
    ax[2].imshow(new_image_1, cmap="gray")
    # ax[3].imshow(new_image_2, cmap="gray")
    ax[0].set_title("Original Lenna.png")
    ax[1].set_title("Padded Lenna.png")
    ax[2].set_title("Lenna conv, singleprocess")
    ax[3].set_title("Lenna conv, multiprocess")
    fig.tight_layout()
    fig.savefig("output.png")


processes = []

if __name__ == "__main__":
    main()
