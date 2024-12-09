# %%
# to run in google colab
import sys

if "google.colab" in sys.modules:

    def download_from_web(url):
        import requests

        response = requests.get(url)
        if response.status_code == 200:
            with open(url.split("/")[-1], "wb") as file:
                file.write(response.content)
        else:
            raise Exception(
                f"Failed to download the image. Status code: {response.status_code}"
            )

    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_03_edge_detection/ex3/butterfly_noisy.jpg"
    )

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)
import math
def calculte_pixel_weight(source,x,y,sigma_r,sigma_s,i,j):
    spatial_distance_sq = (x - i) ** 2 + (y - j) ** 2
    intensity_distance_sq = (source[x][y] - source[i][j]) ** 2
    spatial_weight = math.exp(-spatial_distance_sq / (2 * sigma_s ** 2))
    range_weight = math.exp(-intensity_distance_sq / (2 * sigma_r ** 2))
    weight = spatial_weight * range_weight
    # print(weight)D
    return weight

# %%
def bilateral_one_pixel(source, x, y, d, sigma_r, sigma_s):
    filtered_pix = 0.0
    Wp = 0.0
    half_d = d // 2
    height, width = source.shape
    for i in range(-half_d, half_d + 1):
        for j in range(-half_d, half_d + 1):
            neighbor_x = x + i
            neighbor_y = y + j
            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= height or neighbor_y >= width:
                continue
            cur_weight = calculte_pixel_weight(source, x, y, sigma_r, sigma_s, neighbor_x, neighbor_y)
            Wp += cur_weight
            filtered_pix += cur_weight * source[neighbor_x][neighbor_y]
    if Wp == 0:
        return source[x][y]
    else:
        return filtered_pix / Wp


# %%
def bilateral_filter(source, d, sigma_r, sigma_s):
    filtered_image = np.zeros(source.shape, np.uint8)
    source = source.astype(float)
    height, width = source.shape
    for x in range(height):
        for y in range(width):
            filtered_pixel = bilateral_one_pixel(source, x, y, d, sigma_r, sigma_s)
            filtered_image[x][y] = np.clip(filtered_pixel, 0, 255).astype(np.uint8)
    return filtered_image

# %%
# upload noisy image
src = cv2.imread("ex3/butterfly_noisy.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10, 10))
plt.imshow(src)
plt.colorbar()
plt.savefig('noisy')
plt.show()

# %%
# ======== run
d = 5  # edge size of neighborhood perimeter
sigma_r = 12  # sigma range
sigma_s = 16  # sigma spatial

my_bilateral_filtered_image = bilateral_filter(src, d, sigma_r, sigma_s)
plt.figure(figsize=(10, 10))
plt.imshow(my_bilateral_filtered_image)
plt.colorbar()
plt.savefig('My_bilateral_filter')
plt.show()


# %%
# compare to opencv
cv2_bilateral_filtered_image = cv2.bilateralFilter(src, d, sigma_r, sigma_s)

plt.figure(figsize=(10, 10))
plt.imshow(cv2_bilateral_filtered_image)
plt.colorbar()
plt.savefig('Bilateral_filter')
plt.show()


# %%
# compare to regular gaussian blur
gaussian_filtered_image = cv2.GaussianBlur(src, (d, d), sigma_s)
plt.figure(figsize=(10, 10))
plt.imshow(gaussian_filtered_image)
plt.colorbar()
plt.savefig('Regular')
plt.show()

# %%
# copare canny results between regular  two images
th_low = 100
th_high = 200
res = cv2.Canny(my_bilateral_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.savefig('my_canny')
plt.show()

th_low = 100
th_high = 200
res = cv2.Canny(cv2_bilateral_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.savefig('A_good_canny')
plt.show()

res = cv2.Canny(gaussian_filtered_image, th_low, th_high)
plt.figure(figsize=(10, 10))
plt.imshow(res)
plt.colorbar()
plt.savefig('Just_Canny')
plt.show()
