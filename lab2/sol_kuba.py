import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import typing as t

with open("lenna.png", mode="rb") as image_file:
    image = plt.imread(image_file)

kernel = np.asarray([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
m, n = kernel.shape

fig, ax = plt.subplots()

start = time.time()

x, y = image.shape
new_image = np.zeros((x, y))
padded_image = np.pad(image, [(0, m), (0, n)], mode='edge')

def convolution_per_pixel(pixel_coords: t.Tuple[int, int]):
    i = pixel_coords[0]
    j = pixel_coords[1]
    return np.sum(padded_image[i:i+m, j:j+m]*kernel)


pixels_coords = [(a,b) for a in range(0, x) for b in range(0, y)]

with Pool(5) as p:
    values = p.map(convolution_per_pixel, pixels_coords)


for idx, pixel_coord in enumerate(pixels_coords):
    new_image[pixel_coord[0]][pixel_coord[1]] = values[idx]

end = time.time()

print(f"Time taken (with multiprocessing) {end - start}")

# * limit values of <0, 1> e.g.: 1.342 --> 1.0
img_after_convolution = np.clip(new_image, 0, 1)

# * binarize image e.g.: 0.3 -> 0, 0.7 -> 1
img_after_convolution = np.rint(img_after_convolution)
# print(img_after_convolution)

im = ax.imshow(img_after_convolution, cmap='gray') # , cmap='gray'
plt.savefig("output_2.png")