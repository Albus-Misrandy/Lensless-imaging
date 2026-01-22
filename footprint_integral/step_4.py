import numpy as np
import cv2

# ===== 选一个你觉得最“对”的低分辨率图 =====
# 推荐先试 ratio crop
IN_XLR = "./xlr_ratio_crop.npy"    # 或 xlr_ratio.npy / xlr_diff_crop.npy
OUT_PREFIX = "sr6"

UPSCALE = 10  # 8/10/12

# 去栅格参数
# 方法：对 XLR 做一个“轻微低通” + “方向无关的抗混叠”
# sigma_xlr：在 XLR 域的平滑强度（单位：XLR 像素）
SIGMA_XLR = 0.6    # 推荐 0.6~1.2，越大越去纹理但越糊
MEDIAN_K = 3       # 3 or 5，小窗口中值滤波压孤立噪点

# 上采样后轻微保边（可选）
DO_BILATERAL = True
BIL_D = 7          # 邻域直径：5~9
BIL_SIGMA_C = 18   # 颜色域：12~30
BIL_SIGMA_S = 18   # 空间域：12~30

def save_vis01(arr01, path):
    arr01 = np.clip(arr01, 0, 1)
    cv2.imwrite(path, (arr01*255.0 + 0.5).astype(np.uint8))

def norm01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x.reshape(-1), [2, 98])
    y = (x - lo) / (hi - lo + 1e-8)
    return np.clip(y, 0, 1)

X = np.load(IN_XLR).astype(np.float32)

# 有些 npy 里 invalid 位置可能是 0 或 nan；先处理一下
X = np.nan_to_num(X, nan=0.0)

X01 = norm01(X)

# 1) baseline：直接 bicubic
up0 = cv2.resize(X01, (X01.shape[1]*UPSCALE, X01.shape[0]*UPSCALE),
                 interpolation=cv2.INTER_CUBIC)
save_vis01(up0, f"{OUT_PREFIX}_up.png")

# 2) degrid：先在 XLR 域做“抗混叠”
Xf = X01.copy()

# 2.1 中值压掉离散噪点（不伤大结构）
if MEDIAN_K and MEDIAN_K >= 3:
    # 替换掉 XLR 域 GaussianBlur 那段：
    Xf_u8 = (Xf*255.0+0.5).astype(np.uint8)
    Xf_u8 = cv2.bilateralFilter(Xf_u8, d=5, sigmaColor=18, sigmaSpace=18)
    Xf = Xf_u8.astype(np.float32)/255.0


# 2.2 轻微高斯低通（关键：抑制格点带来的高频）
if SIGMA_XLR and SIGMA_XLR > 0:
    Xf = cv2.GaussianBlur(Xf, (0,0), SIGMA_XLR)

# 上采样
up1 = cv2.resize(Xf, (Xf.shape[1]*UPSCALE, Xf.shape[0]*UPSCALE),
                 interpolation=cv2.INTER_CUBIC)
save_vis01(up1, f"{OUT_PREFIX}_degrid.png")

# 3) 可选：上采样后轻微保边（比 TV 更不容易出格纹）
if DO_BILATERAL:
    up1_u8 = (up1*255.0+0.5).astype(np.uint8)
    up2_u8 = cv2.bilateralFilter(up1_u8, BIL_D, BIL_SIGMA_C, BIL_SIGMA_S)
    up2 = up2_u8.astype(np.float32) / 255.0
    save_vis01(up2, f"{OUT_PREFIX}_degrid_smooth.png")

print("[DONE] wrote:",
      f"{OUT_PREFIX}_up.png",
      f"{OUT_PREFIX}_degrid.png",
      (f"{OUT_PREFIX}_degrid_smooth.png" if DO_BILATERAL else ""))
