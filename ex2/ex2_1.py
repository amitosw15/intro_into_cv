# %% [markdown]
# # EX2_1
# build dilate and erode functions
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

figsize = (10, 10)

# %%

img = np.zeros((50, 50))
img[20:30, 20:30] = 1

plt.figure(figsize=figsize)
plt.imshow(img,cmap="gray")
plt.show()

# %%
kernel = np.zeros((5,5),dtype=np.uint8)
kernel[2,:] = 1
kernel[:,2] = 1


plt.figure(figsize=figsize)
plt.imshow(kernel,cmap="gray")
plt.show()

# %%
def theta(x,t):
    return x>=t

def cross_correlation(f, g):
    """
    Calculate the cross-correlation of a matrix f with a kernel g.

    Parameters:
    f (np.ndarray): Input 2D matrix.
    g (np.ndarray): Kernel 2D matrix.

    Returns:
    np.ndarray: Resulting matrix after cross-correlation.
    """
    # Get dimensions of f and g
    f_height, f_width = f.shape
    g_height, g_width = g.shape
    
    # Calculate the padding size for the kernel
    pad_h = g_height // 2
    pad_w = g_width // 2
    
    # Pad the input matrix f with zeros
    f_padded = np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize the output matrix
    output = np.zeros((f_height, f_width))
    
    # Perform the cross-correlation
    for i in range(f_height):
        for j in range(f_width):
            # Extract the sub-matrix of the same size as g
            sub_matrix = f_padded[i:i+g_height, j:j+g_width]
            # Compute the sum of element-wise multiplication
            output[i, j] = np.sum(sub_matrix * g)
    
    return output
# %%
matrix = cross_correlation(img,kernel)
cv_dilated = cv2.dilate(img, kernel)
cv_eroded = cv2.erode(img, kernel)

def my_dilate(img, kernel):
    # Compute the cross-correlation of the image and kernel
    matrix = cross_correlation(img, kernel)
    # Apply the threshold function
    dilated_img = theta(matrix, 1)
    # Convert boolean array to integer array (0 and 1)
    return dilated_img.astype(np.uint8)
    
    
plt.figure(figsize=figsize)
plt.imshow(my_dilate(img,kernel),cmap="gray")
plt.title("dilate")
plt.show()
# Perform dilation and erosion with OpenCV

# Perform dilation and erosion with custom functions
custom_dilated = my_dilate(img, kernel)

# Compare results using absolute difference
dilate_diff = np.abs(cv_dilated - custom_dilated).sum()
# %%
# TODO: show that cv2.dilate and my_dilate are the same using absolute difference
if dilate_diff == 0:
    print("cv2.dilate & my_dilate are the same!")
else: 
    print("try again...")

# %%
def my_erode(img, kernel):
    # Compute the cross-correlation of the image and kernel
    matrix = cross_correlation(img, kernel)
    # Compute the sum of the kernel elements (number of ones in the kernel)
    kernel_sum = np.sum(kernel)
    # Apply the threshold function
    eroded_img = theta(matrix, kernel_sum)
    # Convert boolean array to integer array (0 and 1)
    return eroded_img.astype(np.uint8)

custom_eroded = my_erode(img, kernel)
erode_diff = np.abs(cv_eroded - custom_eroded).sum()
plt.figure(figsize=figsize)
plt.imshow(my_erode(img,kernel),cmap="gray")
plt.title("eridate")
plt.show()

# %%
# TODO: show that cv2.erode and my_erode are the same using absolute difference
if erode_diff == 0:
    print("cv2.erode & my_erode are the same!")
else: 
    print("try again...")

# %%
