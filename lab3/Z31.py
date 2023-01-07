import time
import matplotlib.pyplot as plt
import numpy as np
import zx

def conv_to_int(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
img = plt.imread("lenna.png")
img = conv_to_int(img)

mask = np.asarray([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
], dtype=int)

def main():
    start = time.perf_counter()
    parallel_conv_img = zx.parallel_convolve(img, mask)
    stop = time.perf_counter()
    print(parallel_conv_img[:, :])
    print("parallel exec time: ", stop - start)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(parallel_conv_img, cmap="gray")
    fig.savefig("result_z31.png")

if __name__ == "__main__":
    main()