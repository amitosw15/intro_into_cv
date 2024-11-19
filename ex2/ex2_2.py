# %% [markdown]
# # EX2_2
# Find different words in newspaper article
# We'll do this using morphology operators and connected components.
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
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_02a_basic_image_processing/ex2/news.jpg"
    )
# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

figsize = (10, 10)

def filter_large_components(binary_image, min_area=100):
    num_labels, labels_im = cv2.connectedComponents(binary_image)
    filtered_im = np.zeros_like(binary_image)

    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        area = cv2.countNonZero(mask)
        if area >= min_area:  # Keep only components larger than min_area
            filtered_im += mask * 255

    return filtered_im



def find_words(dilated_im, im, min_area = 10):
    num_labels, labels_im = cv2.connectedComponents(dilated_im)
    res = im.copy()

    for label in range(1, num_labels):  # Skip background label 0
        mask = (labels_im == label).astype(np.uint8) * 255
        res = plot_rec(mask, res)

    return res


def plot_rec(mask, res_im):
    # plot a rectengle around each word in res image using mask image of the word
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()
    right = x.max()
    up = y.min()
    down = y.max()

    res_im = cv2.rectangle(res_im, (left, up), (right, down), (0, 20, 200), 2)
    return res_im


# %%
im = cv2.imread("news.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=figsize)
plt.imshow(im_gray, cmap="gray", vmin=0, vmax=255)
plt.show()

# %%
# TODO: let's start with turning the image to a binary one
# Use THRESH_BINARY_INV to invert the binary image
_, im_th = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


plt.figure(figsize=(20, 20))
plt.imshow(im_th, cmap='grey')
plt.title("Binary Image")
plt.show()



kernel = np.ones((1, 4), np.uint8) / 4
dilated_im = cv2.dilate(im_th, kernel)
# %%


# # %%
plt.figure(figsize=(20, 20))
plt.imshow(find_words(dilated_im, im))
plt.show()


# %%
# TODO: now we want to mark only the big title words, and do this ONLY using morphological operators)
kernel = np.ones((4, 4), np.uint8)  
eroded_im = cv2.erode(im_th,kernel)
kernel = np.ones((5, 15), np.uint8)  # Wider kernel for dilation
binary_only_title_cc_img = cv2.dilate(eroded_im, kernel)



plt.figure(figsize=(20, 20))
plt.imshow(find_words(binary_only_title_cc_img, im, min_area=1350))
plt.title("Detected Large Title Words")
plt.show()
