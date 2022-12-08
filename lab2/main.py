from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import concurrent.futures

# @njit
def multiply(a: np.ndarray, b: np.ndarray) -> Union[int, float]:
    result = 0
    for x, row in enumerate(a):
        for y, _ in enumerate(row):
            result += a[x, y] * b[x, y]
    return result


def process_kernel_multi(
    image_part: np.ndarray, kernel: np.ndarray, padding_shape: tuple[int, int]
) -> np.ndarray:
    result = np.zeros(image_part.shape[1])
    for column_index, _ in enumerate(image_part):
        if (
            column_index - padding_shape[1] < 0
            or column_index + padding_shape[1] > image_part.shape[1] - 1
        ):
            continue
        sub_array = image_part[
            :, column_index - padding_shape[1] : column_index + padding_shape[1] + 1
        ]
        result[column_index - padding_shape[1]] = multiply(sub_array, kernel)
    return result


# @njit
def convolve_image_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding_shape = tuple((dim // 2 for dim in kernel.shape))
    padded_image = np.pad(image, padding_shape, "edge")
    result = np.zeros(image.shape, dtype=int)

    s = time.perf_counter_ns()
    for row_index, image_row in enumerate(padded_image):
        if (
            row_index - padding_shape[0] < 0
            or row_index + padding_shape[0] > padded_image.shape[0] - 1
        ):
            continue
        for column_index, _ in enumerate(image_row):
            if (
                column_index - padding_shape[1] < 0
                or column_index + padding_shape[1] > padded_image.shape[1] - 1
            ):
                continue
            sub_array = padded_image[
                row_index - padding_shape[0] : row_index + padding_shape[0] + 1,
                column_index - padding_shape[1] : column_index + padding_shape[1] + 1,
            ]
            try:
                result[
                    row_index - padding_shape[0], column_index - padding_shape[1]
                ] = multiply(sub_array, kernel)
            except Exception:
                import ipdb

                ipdb.set_trace()
    # print(result)
    print((time.perf_counter_ns() - s) / 1000000000)
    return result


# @njit
def convolve_image_multi(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padding_shape = tuple((dim // 2 for dim in kernel.shape))
    padded_image = np.pad(image, padding_shape, "edge")
    result = np.zeros(image.shape, dtype=int)
    total_time = 0

    for row_index, image_row in enumerate(image):
        if (
            row_index - padding_shape[0] < 0
            or row_index + padding_shape[0] > padded_image.shape[0] - 1
        ):
            continue
        image_part = image[
            row_index - padding_shape[0] : row_index + padding_shape[0] + 1
        ]
        result[row_index] = process_kernel_multi(image_part, kernel, padding_shape)
    #     for column_index, _ in enumerate(image_row):
    #         if (
    #             column_index - padding_shape[1] < 0
    #             or column_index + padding_shape[1] > padded_image.shape[1] - 1
    #         ):
    #             continue
    #         sub_array = padded_image[
    #             row_index - padding_shape[0] : row_index + padding_shape[0] + 1,
    #             column_index - padding_shape[1] : column_index + padding_shape[1] + 1,
    #         ]
    #         try:
    #             stime = time.perf_counter_ns()
    #             result[row_index, column_index] = multiply(sub_array, kernel)
    #             t = time.perf_counter_ns() - stime
    #             total_time += t
    #         except Exception:
    #             import ipdb

    #             ipdb.set_trace()
    # print(total_time / 1_000_000_000)
    # print(result)
    import ipdb

    ipdb.set_trace
    return result

    # def process_single_row(row_index: int) -> np.ndarray:
    #     # for
    #     print(row_index)


def foo():
    time.sleep(1)
    print("hello")
    return "xd"


def main():

    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    lenna = plt.imread("lenna.png")
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    padding_shape = tuple((dim // 2 for dim in kernel.shape))
    padding = np.pad(lenna, padding_shape, "edge")

    def conv_to_int(img: np.ndarray) -> np.ndarray:
        img = img - np.min(img)
        img /= np.max(img)
        img *= 255
        print(img.shape)
        return img.astype(int)

    lenna = conv_to_int(lenna)
    # print(lenna.dtype)

    new_image_1 = convolve_image_simple(lenna, kernel)
    new_image_2 = convolve_image_multi(lenna, kernel)

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    ax[0].imshow(lenna, cmap="gray")
    ax[1].imshow(padding, cmap="gray")
    ax[2].imshow(new_image_1, cmap="gray")
    ax[3].imshow(new_image_2, cmap="gray")
    ax[0].set_title("Original Lenna.png")
    ax[1].set_title("Padded Lenna.png")
    ax[2].set_title("Lenna conv, singleprocess")
    ax[3].set_title("Lenna conv, multiprocess")
    fig.tight_layout()
    fig.savefig("output.png")


processes = []

if __name__ == "__main__":
    # p1 = mp.Process(target=foo)
    # p2 = mp.Process(target=foo)
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()

    main()
    with concurrent.futures.ProcessPoolExecutor() as excecutor:
        futures = [excecutor.submit(foo) for _ in range(10)]

        for f in concurrent.futures.as_completed(futures):
            print(f.result())
