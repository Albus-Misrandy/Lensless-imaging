import cv2
import numpy as np
import math

IMG_PATH = "../snapshots/snapshot_20260115_160108.jpg"

img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(IMG_PATH)

# ===== 建议：用中间区域做 ROI（避免边缘暗角/畸变）=====
H0, W0 = img.shape
cy0, cx0 = H0 // 2, W0 // 2
roi_h, roi_w = 420, 640
y0 = max(0, cy0 - roi_h // 2)
x0 = max(0, cx0 - roi_w // 2)
roi = img[y0:y0 + roi_h, x0:x0 + roi_w].copy()

cv2.imwrite("roi.png", roi)

roi_f = roi.astype(np.float32)
roi_f = roi_f - roi_f.mean()

H, W = roi_f.shape
print("ROI shape:", (H, W))

# ===== FFT 幅度谱 =====
wy = np.hanning(H).astype(np.float32)
wx = np.hanning(W).astype(np.float32)
win = wy[:, None] * wx[None, :]

F = np.fft.fftshift(np.fft.fft2(roi_f * win))
mag = np.log1p(np.abs(F)).astype(np.float32)
mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

# ===== 抑制 DC，找峰 =====
cy, cx = H // 2, W // 2
Y, X = np.indices((H, W))
R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

rmin = 0.04 * min(H, W)
rmax = 0.45 * min(H, W)
mask = (R >= rmin) & (R <= rmax)

mag2 = mag.copy()
mag2[~mask] = 0.0

K = 80
flat_idx = np.argpartition(mag2.ravel(), -K)[-K:]
peaks = np.column_stack(np.unravel_index(flat_idx, (H, W)))
peaks = peaks[np.argsort(-mag2[peaks[:, 0], peaks[:, 1]])]

def to_k(py, px):
    fy = (py - cy) / float(H)
    fx = (px - cx) / float(W)
    return np.array([fx, fy], dtype=np.float32)

ks = [to_k(int(py), int(px)) for py, px in peaks]

# ===== 选两条“基本峰”：频率尽量低 + 同半径 + 不共线（接近正交更好）=====
best = None
best_cost = 1e9
topN = min(len(ks), 40)

for i in range(topN):
    for j in range(i + 1, topN):
        k1, k2 = ks[i], ks[j]
        det = abs(k1[0] * k2[1] - k1[1] * k2[0])
        if det < 1e-6:
            continue
        r1 = float(np.linalg.norm(k1))
        r2 = float(np.linalg.norm(k2))
        if r1 < 0.006 or r2 < 0.006:
            continue
        ortho = det / (np.linalg.norm(k1) * np.linalg.norm(k2) + 1e-8)
        cost = (r1 + r2) + 2.0 * abs(r1 - r2) - 0.6 * ortho
        if cost < best_cost:
            best_cost = cost
            best = (k1, k2, peaks[i], peaks[j])

if best is None:
    raise RuntimeError("没找到合适的基本峰，试试换 ROI 或换更清晰的图。")

k1, k2, p1, p2 = best
print("Chosen reciprocal peaks (py,px):", tuple(p1), tuple(p2))
print("k1 (fx,fy)=", k1, "  k2 (fx,fy)=", k2)

# ===== 倒易基 -> 空间基（像素向量）=====
Kmat = np.stack([k1, k2], axis=0).astype(np.float32)  # 2x2
V = np.linalg.inv(Kmat)
v1 = V[:, 0]
v2 = V[:, 1]

# 规范化方向：v1 x>0，右手系
if v1[0] < 0:
    v1 = -v1
if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
    v2 = -v2

pitch1 = float(np.linalg.norm(v1))
pitch2 = float(np.linalg.norm(v2))
angle_deg = float(math.degrees(math.atan2(v1[1], v1[0])))

print(f"v1={v1}, |v1|={pitch1:.2f} px")
print(f"v2={v2}, |v2|={pitch2:.2f} px")
print(f"angle(v1) = {angle_deg:.2f} deg (relative to +x)")

# ===== 画频谱图 + 标记两个峰 =====
spec_vis = (mag * 255).astype(np.uint8)
spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)

def draw_peak(vis, py, px, color):
    cv2.circle(vis, (int(px), int(py)), 7, color, 2, lineType=cv2.LINE_AA)

draw_peak(spec_vis, int(p1[0]), int(p1[1]), (0, 255, 0))
draw_peak(spec_vis, int(p2[0]), int(p2[1]), (0, 0, 255))
cv2.circle(spec_vis, (cx, cy), 4, (255, 255, 0), 2, lineType=cv2.LINE_AA)
cv2.imwrite("fft_peaks.png", spec_vis)

# ===== 网格参数 =====
roi_u8 = roi
H2, W2 = roi_u8.shape

# p0：中心附近找一个亮斑（仅用于初始化）
cx2, cy2 = W2 // 2, H2 // 2
r_search = int(0.25 * min(H2, W2))
x0s, x1s = max(0, cx2 - r_search), min(W2, cx2 + r_search)
y0s, y1s = max(0, cy2 - r_search), min(H2, cy2 + r_search)
patch = roi_u8[y0s:y1s, x0s:x1s]
py0, px0 = np.unravel_index(np.argmax(patch), patch.shape)
p0 = np.array([x0s + px0, y0s + py0], dtype=np.float32)
print("p0 guess:", p0)

# 反推 m,n 范围，让网格覆盖 ROI
V2 = np.stack([v1, v2], axis=1)
V2inv = np.linalg.inv(V2)
corners = np.array([[0, 0], [W2 - 1, 0], [0, H2 - 1], [W2 - 1, H2 - 1]], dtype=np.float32)
mn = (V2inv @ (corners - p0).T).T
m_min = int(np.floor(mn[:, 0].min())) - 2
m_max = int(np.ceil(mn[:, 0].max())) + 2
n_min = int(np.floor(mn[:, 1].min())) - 2
n_max = int(np.ceil(mn[:, 1].max())) + 2
print("m range:", m_min, m_max, "n range:", n_min, n_max)

# ===== refine p0 (B版：中心区域 + 阻尼) =====
roi_f32 = roi_u8.astype(np.float32)

# DoG：增强孔斑，抑制大尺度纹理
g1 = cv2.GaussianBlur(roi_f32, (0, 0), 1.2)
g2 = cv2.GaussianBlur(roi_f32, (0, 0), 4.0)
dog = np.maximum(g1 - g2, 0)

def snap_to_centroid(im_pos, x, y, r=7, thr_rel=0.35):
    Hh, Ww = im_pos.shape
    xi, yi = int(round(x)), int(round(y))
    x0 = max(0, xi - r); x1 = min(Ww - 1, xi + r)
    y0 = max(0, yi - r); y1 = min(Hh - 1, yi + r)
    win = im_pos[y0:y1+1, x0:x1+1]
    if win.size == 0:
        return x, y, 0.0
    mx = float(win.max())
    if mx <= 1e-6:
        return x, y, 0.0

    mask = win >= (thr_rel * mx)
    w = win * mask.astype(np.float32)
    s = float(w.sum())
    if s <= 1e-6:
        return x, y, 0.0

    yy, xx = np.indices(w.shape, dtype=np.float32)
    cx = float((w * xx).sum() / s)
    cy = float((w * yy).sum() / s)
    return float(x0 + cx), float(y0 + cy), s

# 只用中心网格点来 refine（减少边缘亮度斜坡影响）
m0 = (m_min + m_max) // 2
n0 = (n_min + n_max) // 2
mR = 8
nR = 8

alpha = 0.3  # 阻尼
p0_ref = p0.copy()

for it in range(5):
    shifts = []
    for n in range(n0 - nR, n0 + nR + 1):
        for m in range(m0 - mR, m0 + mR + 1):
            p = p0_ref + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            if 0 <= x < W2 and 0 <= y < H2:
                xs, ys, conf = snap_to_centroid(dog, x, y, r=7, thr_rel=0.35)
                if conf > 1e-3:
                    shifts.append([xs - x, ys - y])

    if len(shifts) < 30:
        print("refine: too few valid snaps:", len(shifts))
        break

    shifts = np.array(shifts, dtype=np.float32)

    # 去离群：保留 90% 内
    d = np.linalg.norm(shifts, axis=1)
    keep = d <= np.percentile(d, 90)
    shifts = shifts[keep]

    delta = np.median(shifts, axis=0)
    p0_ref = p0_ref + alpha * delta
    print(f"refine it={it}: raw_delta={delta}, damped={alpha*delta}, kept={int(keep.sum())} p0_ref={p0_ref}")

    if float(np.linalg.norm(alpha * delta)) < 0.12:
        break

p0 = p0_ref
print("p0 refined:", p0)

# ===== 用 refined p0 画 overlay =====
overlay = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
count = 0
for n in range(n_min, n_max + 1, 2):
    for m in range(m_min, m_max + 1, 2):
        p = p0 + m * v1 + n * v2
        x, y = float(p[0]), float(p[1])
        if 0 <= x < W2 and 0 <= y < H2:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            count += 1

cv2.circle(overlay, (int(round(p0[0])), int(round(p0[1]))), 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)
cv2.imwrite("grid_overlay_pred.png", overlay)
print("[DONE] wrote fft_peaks.png and grid_overlay_pred.png with", count, "grid pts")

# ===== 保存标定（下一步用）=====
np.savez(
    "calib_fft.npz",
    v1=v1.astype(np.float32),
    v2=v2.astype(np.float32),
    p0=p0.astype(np.float32),
    roi_x0=int(x0), roi_y0=int(y0), roi_w=int(roi_w), roi_h=int(roi_h),
    m_min=int(m_min), m_max=int(m_max), n_min=int(n_min), n_max=int(n_max),
    img_path=IMG_PATH
)
print("[DONE] wrote calib_fft.npz")
