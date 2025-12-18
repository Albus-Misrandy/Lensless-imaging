import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1）读入灰度图
img = cv2.imread("light_diff_out/frame_000006_t3.04_diff.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("图片路径不对 or 文件不存在")

# 2）直方图均衡化
img_eq = cv2.equalizeHist(img)

# 3）可视化：原图 vs 均衡化后的图
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Equalized Image")
plt.imshow(img_eq, cmap="gray")
plt.axis("off")

# 4）画直方图对比（0~255 每个灰度级的像素数量）
plt.subplot(2, 2, 3)
plt.title("Original Histogram")
plt.hist(img.ravel(), 256, [0, 256])
plt.xlabel("Gray level")
plt.ylabel("Pixel count")

plt.subplot(2, 2, 4)
plt.title("Equalized Histogram")
plt.hist(img_eq.ravel(), 256, [0, 256])
plt.xlabel("Gray level")
plt.ylabel("Pixel count")

plt.tight_layout()
plt.show()
