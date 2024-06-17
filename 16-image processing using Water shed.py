import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(r"C:\Users\Dell\OneDrive\Pictures\Screenshots\Screenshot 2024-05-03 211011.png")
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((2, 2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
sure_bg = cv2.dilate(closing, kernel, iterations=1)
plt.subplot(211), plt.imshow(closing, 'gray')
plt.title("morphologyEx: Closing: 2x2"), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.imsave(r'dilation.png', sure_bg)
plt.show()
