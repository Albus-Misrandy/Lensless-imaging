import cv2
import numpy as np

# =========================
# 0) 路径配置：你改这里
# =========================
CALIB_PATH = "./calib_fft.npz"

MEAS_PATH = "../snapshots/snapshot_20260116_162823.jpg"   # 有按压
FLAT_PATH = "../snapshots/flat.jpg"                       # 无按压/平场
DARK_PATH = "../snapshots/dark.jpg"                       # 盖住镜头/全黑

# 输出名前缀
OUT_PREFIX = "xlr"

# =========================
# 1) 读标定
# =========================
cal = np.load(CALIB_PATH)

v1 = cal["v1"].astype(np.float32)     # (2,)
v2 = cal["v2"].astype(np.float32)
p0 = cal["p0"].astype(np.float32)

roi_x0 = int(cal["roi_x0"])
roi_y0 = int(cal["roi_y0"])
roi_w  = int(cal["roi_w"])
roi_h  = int(cal["roi_h"])

m_min = int(cal["m_min"]); m_max = int(cal["m_max"])
n_min = int(cal["n_min"]); n_max = int(cal["n_max"])

print("Loaded calib:")
print(" v1=", v1, " |v1|=", float(np.linalg.norm(v1)))
print(" v2=", v2, " |v2|=", float(np.linalg.norm(v2)))
print(" p0=", p0)
print(" ROI (x0,y0,w,h) =", (roi_x0, roi_y0, roi_w, roi_h))
print(" m range:", (m_min, m_max), " n range:", (n_min, n_max))

# =========================
# 2) 读图 + 裁 ROI
# =========================
def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im

def crop_roi(im):
    H, W = im.shape
    x1 = min(W, roi_x0 + roi_w)
    y1 = min(H, roi_y0 + roi_h)
    x0 = max(0, roi_x0)
    y0 = max(0, roi_y0)
    roi = im[y0:y1, x0:x1].copy()
    if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
        # 尺寸对不上直接报错，避免默默错位
        raise RuntimeError(f"ROI size mismatch: got {roi.shape}, expected {(roi_h, roi_w)}. "
                           f"Check image size or calib ROI.")
    return roi

meas = crop_roi(read_gray(MEAS_PATH))
flat = crop_roi(read_gray(FLAT_PATH))
dark = crop_roi(read_gray(DARK_PATH))

cv2.imwrite(f"{OUT_PREFIX}_roi_meas.png", meas)
cv2.imwrite(f"{OUT_PREFIX}_roi_flat.png", flat)
cv2.imwrite(f"{OUT_PREFIX}_roi_dark.png", dark)
print("[DONE] wrote roi previews")

# =========================
# 3) 网格坐标 (rr,cc) -> (x,y)
# =========================
# 注意：这里使用 step0 的 n/m 步长=1（你现在就是 1）

# ===== 根据积分半径，重新计算“可积分”的 m/n 范围（避免越界导致 valid 太少）=====
V2 = np.stack([v1, v2], axis=1).astype(np.float32)
V2inv = np.linalg.inv(V2)

R_INT = 8  # 你原来就是 8；如果想更多 valid，可以改成 6

# “内缩后的 ROI”四个角点（x,y），保证以 R_INT 为半径的 patch 一定完整落在 ROI 内
corners_inner = np.array([
    [R_INT, R_INT],
    [roi_w - 1 - R_INT, R_INT],
    [R_INT, roi_h - 1 - R_INT],
    [roi_w - 1 - R_INT, roi_h - 1 - R_INT],
], dtype=np.float32)

mn_inner = (V2inv @ (corners_inner - p0).T).T  # 4x2
m_min_eff = int(np.ceil(mn_inner[:, 0].min()))
m_max_eff = int(np.floor(mn_inner[:, 0].max()))
n_min_eff = int(np.ceil(mn_inner[:, 1].min()))
n_max_eff = int(np.floor(mn_inner[:, 1].max()))

print("Effective m range:", (m_min_eff, m_max_eff), "Effective n range:", (n_min_eff, n_max_eff))

# 用有效范围覆盖掉原范围
m_min, m_max = m_min_eff, m_max_eff
n_min, n_max = n_min_eff, n_max_eff

rows = (n_max - n_min) + 1
cols = (m_max - m_min) + 1
print("Grid size (rows, cols):", (rows, cols))

# (rr,cc) 对应的 m,n
# rr = n - n_min
# cc = m - m_min

# =========================
# 4) 每孔积分（高斯软窗口）
# =========================
def make_weights(r):
    r = int(r)
    yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    rr2 = xx*xx + yy*yy
    mask = (rr2 <= r*r).astype(np.float32)
    sigma = max(1.0, r/2.0)
    w = np.exp(-0.5 * rr2 / (sigma*sigma)) * mask
    w = w / (w.sum() + 1e-8)
    return w

def integrate_grid(im_u8, r_int=8):
    H2, W2 = im_u8.shape
    w = make_weights(r_int)
    X = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.uint8)

    for n in range(n_min, n_max + 1):
        rr = n - n_min
        for m in range(m_min, m_max + 1):
            cc = m - m_min

            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])

            xi, yi = int(round(x)), int(round(y))
            x0 = xi - r_int; x1 = xi + r_int
            y0 = yi - r_int; y1 = yi + r_int

            if x0 < 0 or y0 < 0 or x1 >= W2 or y1 >= H2:
                continue

            patch = im_u8[y0:y1+1, x0:x1+1].astype(np.float32)
            X[rr, cc] = float(np.sum(patch * w))
            valid[rr, cc] = 1

    return X, valid

# 你可以调这个半径：6/8/10
R_INT = 8
X_meas, V_meas = integrate_grid(meas, r_int=R_INT)
X_flat, V_flat = integrate_grid(flat, r_int=R_INT)
X_dark, V_dark = integrate_grid(dark, r_int=R_INT)

valid = (V_meas & V_flat & V_dark).astype(np.uint8)
print("valid count:", int(valid.sum()), "/", valid.size)

np.save(f"{OUT_PREFIX}_meas.npy", X_meas)
np.save(f"{OUT_PREFIX}_flat.npy", X_flat)
np.save(f"{OUT_PREFIX}_dark.npy", X_dark)
np.save(f"{OUT_PREFIX}_valid.npy", valid)
print("[DONE] wrote xlr npy")

# =========================
# 5) 生成差分/比值（推荐）
# =========================
# 差分：接触变化（最直观）
X_diff = (X_meas - X_flat)

# 比值：光照归一化（flat-dark 做分母）
den = (X_flat - X_dark)
eps = 1e-3
X_ratio = (X_meas - X_dark) / np.maximum(den, eps)

np.save(f"{OUT_PREFIX}_diff.npy", X_diff)
np.save(f"{OUT_PREFIX}_ratio.npy", X_ratio)

# =========================
# 6) 可视化（百分位拉伸 + 放大）
# =========================
def vis_save(X, valid_mask, out_png, scale=18, p_lo=5, p_hi=95):
    Xv = X.copy().astype(np.float32)
    vals = Xv[valid_mask > 0]
    if vals.size < 10:
        lo, hi = float(Xv.min()), float(Xv.max())
    else:
        lo, hi = np.percentile(vals, [p_lo, p_hi])
    Y = (Xv - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)
    Y[valid_mask == 0] = 0.0   # 关键：无效点直接涂黑
    u8 = (Y * 255.0 + 0.5).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1]*scale, u8.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, big)

vis_save(X_meas, valid, f"{OUT_PREFIX}_meas_vis.png")
vis_save(X_flat, valid, f"{OUT_PREFIX}_flat_vis.png")
vis_save(X_dark, valid, f"{OUT_PREFIX}_dark_vis.png")

# diff 用更宽的拉伸（对称显示更好，这里用 5/95 先凑合）
vis_save(X_diff, valid, f"{OUT_PREFIX}_diff_vis.png", p_lo=5, p_hi=95)

# ratio 通常范围更集中，用 5/95
vis_save(X_ratio, valid, f"{OUT_PREFIX}_ratio_vis.png", p_lo=5, p_hi=95)

print("[DONE] wrote vis pngs:")
print(" ", f"{OUT_PREFIX}_meas_vis.png")
print(" ", f"{OUT_PREFIX}_flat_vis.png")
print(" ", f"{OUT_PREFIX}_dark_vis.png")
print(" ", f"{OUT_PREFIX}_diff_vis.png")
print(" ", f"{OUT_PREFIX}_ratio_vis.png")
