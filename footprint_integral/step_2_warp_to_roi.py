import numpy as np
import cv2

# ========= 你改这里 =========
CALIB_NPZ   = "calib_fft.npz"
IN_XLR_NPY  = "xlr_diff.npy"   # 或 "xlr_diff.npy"
OUT_PREFIX  = "roiwarp"
R_INT       = 8                 # 要和 step_1 里积分半径一致
STEP        = 1                 # 你现在 m,n 步长就是 1
# ===========================

# 1) 读标定
cal = np.load(CALIB_NPZ)

v1 = cal["v1"].astype(np.float32)   # (2,)
v2 = cal["v2"].astype(np.float32)   # (2,)
p0 = cal["p0"].astype(np.float32)   # (2,)  [x,y] in ROI

roi_w = int(cal["roi_w"])
roi_h = int(cal["roi_h"])

print("Loaded calib:")
print(" v1 =", v1, "|v1| =", float(np.linalg.norm(v1)))
print(" v2 =", v2, "|v2| =", float(np.linalg.norm(v2)))
print(" p0 =", p0)
print(" ROI (h,w) =", (roi_h, roi_w))

# 2) 读 XLR（未 crop 版本）
XLR = np.load(IN_XLR_NPY).astype(np.float32)
rows, cols = XLR.shape
print(" XLR shape =", XLR.shape)

# 3) 根据 R_INT 重新计算有效 m/n 范围（和 step_1 一样）
V2 = np.stack([v1, v2], axis=1).astype(np.float32)   # 2x2, columns are v1,v2
V2inv = np.linalg.inv(V2).astype(np.float32)

# “内缩 ROI” 角点，保证以 R_INT 为半径的 patch 在 ROI 内
corners_inner = np.array([
    [R_INT,               R_INT],
    [roi_w - 1 - R_INT,   R_INT],
    [R_INT,               roi_h - 1 - R_INT],
    [roi_w - 1 - R_INT,   roi_h - 1 - R_INT],
], dtype=np.float32)

mn_inner = (V2inv @ (corners_inner - p0).T).T   # 4x2

m_min = int(np.ceil (mn_inner[:, 0].min()))
m_max = int(np.floor(mn_inner[:, 0].max()))
n_min = int(np.ceil (mn_inner[:, 1].min()))
n_max = int(np.floor(mn_inner[:, 1].max()))

rows_calc = (n_max - n_min) // STEP + 1
cols_calc = (m_max - m_min) // STEP + 1

print("Effective m range:", (m_min, m_max), "n range:", (n_min, n_max))
print("Grid size from calib+R_INT:", (rows_calc, cols_calc))

if (rows, cols) != (rows_calc, cols_calc):
    print("!!! 警告：XLR shape 和重新算的范围不一致，还是继续按 XLR 为准。")
    # 简单修正：用 XLR 的形状反推 m,n 范围（只会引入整体平移，不影响“摆正”方向）
    rows_calc, cols_calc = rows, cols
    # 居中假设
    m_center = (m_min + m_max) / 2.0
    n_center = (n_min + n_max) / 2.0
    m_min = int(round(m_center - (cols_calc - 1) / 2.0))
    m_max = m_min + cols_calc - 1
    n_min = int(round(n_center - (rows_calc - 1) / 2.0))
    n_max = n_min + rows_calc - 1
    print("Adjusted m range:", (m_min, m_max), "n range:", (n_min, n_max))

# 4) 生成 ROI 像素到 XLR 索引的映射
yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)

dx = xx - float(p0[0])
dy = yy - float(p0[1])

# 连续 (m,n)
m_cont = V2inv[0, 0] * dx + V2inv[0, 1] * dy
n_cont = V2inv[1, 0] * dx + V2inv[1, 1] * dy

# 连续 XLR 索引 (map_x, map_y)
# cc = (m - m_min)/STEP  rr = (n - n_min)/STEP
map_x = ((m_cont - float(m_min)) / float(STEP)).astype(np.float32)
map_y = ((n_cont - float(n_min)) / float(STEP)).astype(np.float32)

# 5) remap 得到“摆正后”的 ROI 尺寸结果
S_roi = cv2.remap(
    XLR,
    map_x, map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
).astype(np.float32)

np.save(f"{OUT_PREFIX}_S_roi.npy", S_roi)

# 6) 可视化：百分位拉伸
vals = S_roi[S_roi != 0]
if vals.size > 10:
    lo, hi = np.percentile(vals, [2, 98])
else:
    lo, hi = float(S_roi.min()), float(S_roi.max())

Vis = (S_roi - lo) / (hi - lo + 1e-8)
Vis = np.clip(Vis, 0, 1)

cv2.imwrite(f"{OUT_PREFIX}_S_roi.png", (Vis * 255.0 + 0.5).astype(np.uint8))

print("[DONE] wrote:",
      f"{OUT_PREFIX}_S_roi.npy",
      f"{OUT_PREFIX}_S_roi.png")
