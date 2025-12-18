import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("6.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("图片路径不对 or 文件不存在")

img_float = np.float32(img)
dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2   # 频谱中心位置

# 1）做一个低通滤波器掩膜：中心一个圆形是 1，其余都是 0
mask = np.zeros((rows, cols, 2), np.float32)
radius = 30   # 半径可以自己改，大一点就更模糊
y, x = np.ogrid[:rows, :cols]
center_mask = (x - ccol)**2 + (y - crow)**2 <= radius**2
mask[center_mask] = 1

# 2）频域相乘：相当于把高频都砍掉
fshift_filtered = dft_shift * mask

# 3）逆移位 + 逆 DFT
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back_complex = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back_complex[:,:,0], img_back_complex[:,:,1])

# 4）归一化到 0–255
img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back_uint8 = np.uint8(img_back_norm)

# 5）再顺手算一下滤波后的频谱，看看对比
mag_filtered = cv2.magnitude(fshift_filtered[:,:,0], fshift_filtered[:,:,1])
mag_filtered = 20 * np.log(1 + mag_filtered)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Original Spectrum")
orig_mag = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
orig_mag = 20 * np.log(1 + orig_mag)
plt.imshow(orig_mag, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Low-pass Filtered Image")
plt.imshow(img_back_uint8, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Filtered Spectrum")
plt.imshow(mag_filtered, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
