import os
import time
import math
import cv2
import numpy as np

# ============================================================
# 配置区：你主要改这里
# ============================================================

# --- 标定图：孔很清晰、最好无按压（只用来做 calib） ---
IMG_CALIB_PATH = "../snapshots/snapshot_20260115_160108.jpg"

# --- DARK（全黑）仍然建议保留（可以是拍一张盖住镜头/关灯） ---
DARK_PATH = "../snapshots/dark.jpg"

OUT_DIR = "./out_realtime"
CALIB_NAME = "calib_fft.npz"

ROI_H, ROI_W = 420, 640

# 积分半径（像素）
R_INT = 8

# 摄像头设置
CAM_ID = 2
CAM_WARMUP_SEC = 0.5         # 打开摄像头后预热/采集 flat 的时间
FLAT_AVG_SEC = 0.5           # flat 取这段时间内多帧平均
TARGET_FPS = 30              # 仅用于显示节奏（不强制）

# ---------- im_pos（给中心检测/误差评估用） ----------
BG_SIGMA = 18.0
SMOOTH_SIGMA = 2.5
CLAMP_POS = True

# ---------- 亮斑中心检测（全图） ----------
THR_FRAC = 0.18
MIN_AREA = 12
MAX_AREA = 5000
MAX_PTS_KEEP = 4000

# ---------- FFT 找 v1/v2 的辅助（给晶格拟合做初值） ----------
PEAK_TOPK = 80
PEAK_TRY_TOPN = 40
FFT_RMIN_FRAC = 0.04
FFT_RMAX_FRAC = 0.45

# ---------- 晶格拟合（鲁棒迭代） ----------
LATTICE_ITERS = 6
OUTLIER_PCT = 95

# ---------- 自动选 step=1 or 2 的评估参数 ----------
EVAL_ARROW_SCALE = 8.0
SNAP_R = 9
SNAP_THR_REL = 0.25

# ratio 防止除零
RATIO_EPS = 1e-3

# warp 显示动态范围平滑（防止闪烁）
VIS_SMOOTH_ALPHA = 0.12

# ============================================================
# 工具函数
# ============================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im

def center_crop_roi(img_u8, roi_h, roi_w):
    H0, W0 = img_u8.shape
    cy0, cx0 = H0 // 2, W0 // 2
    y0 = max(0, cy0 - roi_h // 2)
    x0 = max(0, cx0 - roi_w // 2)
    roi = img_u8[y0:y0 + roi_h, x0:x0 + roi_w].copy()
    if roi.shape != (roi_h, roi_w):
        raise RuntimeError(f"ROI crop mismatch got {roi.shape}, expected {(roi_h, roi_w)}.")
    return roi, x0, y0

def crop_roi_by_xywh(img_u8, x0, y0, w, h):
    roi = img_u8[y0:y0 + h, x0:x0 + w].copy()
    if roi.shape != (h, w):
        raise RuntimeError(f"ROI size mismatch got {roi.shape}, expected {(h, w)}.")
    return roi

def make_hanning_window(H, W):
    wy = np.hanning(H).astype(np.float32)
    wx = np.hanning(W).astype(np.float32)
    return wy[:, None] * wx[None, :]

def make_weights(r):
    r = int(r)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rr2 = xx * xx + yy * yy
    mask = (rr2 <= r * r).astype(np.float32)
    sigma = max(1.0, r / 2.0)
    w = np.exp(-0.5 * rr2 / (sigma * sigma)) * mask
    w = w / (w.sum() + 1e-8)
    return w

def make_im_pos_for_snap(roi_u8):
    roi_f = roi_u8.astype(np.float32)
    bg = cv2.GaussianBlur(roi_f, (0, 0), BG_SIGMA)
    hp = roi_f - bg
    hp2 = cv2.GaussianBlur(hp, (0, 0), SMOOTH_SIGMA)
    if CLAMP_POS:
        return np.maximum(hp2, 0)
    return hp2

def normalize_u8(im):
    im = im.astype(np.float32)
    mx = float(im.max())
    if mx <= 1e-6:
        return np.zeros_like(im, dtype=np.uint8)
    out = np.clip(im / (mx + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return out

def align_range_to_step(mn_min, mn_max, step, ref=0):
    mn_min_a = mn_min + ((ref - mn_min) % step)
    mn_max_a = mn_max - ((mn_max - ref) % step)
    if mn_max_a < mn_min_a:
        raise RuntimeError("Aligned range became empty; check ROI/step.")
    return int(mn_min_a), int(mn_max_a)

# ============================================================
# 1) FFT 找晶格方向（给拟合做初值）
# ============================================================

def fft_find_v1v2(roi_u8, out_dir):
    roi_f = roi_u8.astype(np.float32) - float(roi_u8.mean())
    H, W = roi_f.shape
    win = make_hanning_window(H, W)

    F = np.fft.fftshift(np.fft.fft2(roi_f * win))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    cy, cx = H // 2, W // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    rmin = FFT_RMIN_FRAC * min(H, W)
    rmax = FFT_RMAX_FRAC * min(H, W)
    mask = (R >= rmin) & (R <= rmax)
    mag2 = mag.copy()
    mag2[~mask] = 0.0

    K = int(PEAK_TOPK)
    flat_idx = np.argpartition(mag2.ravel(), -K)[-K:]
    peaks = np.column_stack(np.unravel_index(flat_idx, (H, W)))
    peaks = peaks[np.argsort(-mag2[peaks[:, 0], peaks[:, 1]])]

    def to_k(py, px):
        fy = (py - cy) / float(H)
        fx = (px - cx) / float(W)
        return np.array([fx, fy], dtype=np.float32)

    ks = [to_k(int(py), int(px)) for py, px in peaks]

    best = None
    best_cost = 1e9
    topN = min(len(ks), int(PEAK_TRY_TOPN))

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
        raise RuntimeError("FFT 没找到合适的基本峰：换 ROI 或换更清晰的图。")

    k1, k2, p1, p2 = best
    Kmat = np.stack([k1, k2], axis=0).astype(np.float32)
    V = np.linalg.inv(Kmat)
    v1 = V[:, 0].astype(np.float32)
    v2 = V[:, 1].astype(np.float32)

    if v1[0] < 0:
        v1 = -v1
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2

    spec_vis = (mag * 255).astype(np.uint8)
    spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)
    cv2.circle(spec_vis, (int(p1[1]), int(p1[0])), 7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (int(p2[1]), int(p2[0])), 7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (cx, cy), 4, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "fft_peaks.png"), spec_vis)

    return v1, v2

# ============================================================
# 2) 全图检测孔中心点（calib 图干净时最合适）
# ============================================================

def detect_blob_centers(im_pos, out_dir):
    im_u8 = normalize_u8(im_pos)
    mx = int(im_u8.max())
    thr = max(1, int(THR_FRAC * mx))
    _, bw = cv2.threshold(im_u8, thr, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    num, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)

    pts = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        cx, cy = cent[i]
        pts.append([float(cx), float(cy)])

    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] < 30:
        raise RuntimeError(f"检测到的孔中心太少：{pts.shape[0]}。调 THR_FRAC/MIN_AREA 或检查 calib 图。")

    if pts.shape[0] > MAX_PTS_KEEP:
        idx = np.linspace(0, pts.shape[0] - 1, MAX_PTS_KEEP).astype(int)
        pts = pts[idx]

    vis = cv2.cvtColor(im_u8, cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "blob_centers.png"), vis)

    return pts

# ============================================================
# 3) 晶格最小二乘拟合（迭代重分配整数索引 + 去离群）
# ============================================================

def lattice_fit_ls(points_xy, v1_init, v2_init, p0_init):
    p0 = p0_init.astype(np.float32).copy()
    v1 = v1_init.astype(np.float32).copy()
    v2 = v2_init.astype(np.float32).copy()
    pts = points_xy.astype(np.float32)

    for _ in range(int(LATTICE_ITERS)):
        V = np.stack([v1, v2], axis=1).astype(np.float32)
        Vinv = np.linalg.inv(V).astype(np.float32)

        mn_cont = (Vinv @ (pts - p0).T).T
        mn_int = np.round(mn_cont).astype(np.int32)
        m = mn_int[:, 0].astype(np.float32)
        n = mn_int[:, 1].astype(np.float32)

        N = pts.shape[0]
        A = np.zeros((2 * N, 6), dtype=np.float32)
        b = np.zeros((2 * N,), dtype=np.float32)

        A[0::2, 0] = 1.0
        A[0::2, 2] = m
        A[0::2, 4] = n
        b[0::2] = pts[:, 0]

        A[1::2, 1] = 1.0
        A[1::2, 3] = m
        A[1::2, 5] = n
        b[1::2] = pts[:, 1]

        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        p0 = np.array([x[0], x[1]], dtype=np.float32)
        v1 = np.array([x[2], x[3]], dtype=np.float32)
        v2 = np.array([x[4], x[5]], dtype=np.float32)

        if v1[0] < 0:
            v1 = -v1
        if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
            v2 = -v2

        pred = p0[None, :] + m[:, None] * v1[None, :] + n[:, None] * v2[None, :]
        err = np.linalg.norm(pts - pred, axis=1)
        thr = np.percentile(err, OUTLIER_PCT)
        keep = err <= thr
        pts = pts[keep]
        if pts.shape[0] < 50:
            break

    # 再稳定一次
    V = np.stack([v1, v2], axis=1).astype(np.float32)
    Vinv = np.linalg.inv(V).astype(np.float32)
    mn_cont = (Vinv @ (pts - p0).T).T
    mn_int = np.round(mn_cont).astype(np.int32)
    m = mn_int[:, 0].astype(np.float32)
    n = mn_int[:, 1].astype(np.float32)

    N = pts.shape[0]
    A = np.zeros((2 * N, 6), dtype=np.float32)
    b = np.zeros((2 * N,), dtype=np.float32)

    A[0::2, 0] = 1.0
    A[0::2, 2] = m
    A[0::2, 4] = n
    b[0::2] = pts[:, 0]

    A[1::2, 1] = 1.0
    A[1::2, 3] = m
    A[1::2, 5] = n
    b[1::2] = pts[:, 1]

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p0 = np.array([x[0], x[1]], dtype=np.float32)
    v1 = np.array([x[2], x[3]], dtype=np.float32)
    v2 = np.array([x[4], x[5]], dtype=np.float32)

    if v1[0] < 0:
        v1 = -v1
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2

    return p0, v1, v2

# ============================================================
# 4) 用“预测格点 -> im_pos附近质心”评估 step=1/2
# ============================================================

def eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step, max_pts=3000):
    H, W = roi_u8.shape
    V = np.stack([v1, v2], axis=1).astype(np.float32)
    Vinv = np.linalg.inv(V).astype(np.float32)

    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float32)
    mn = (Vinv @ (corners - p0).T).T
    m_min = int(np.floor(mn[:, 0].min())) - 2
    m_max = int(np.ceil(mn[:, 0].max())) + 2
    n_min = int(np.floor(mn[:, 1].min())) - 2
    n_max = int(np.ceil(mn[:, 1].max())) + 2

    cand = []
    for n in range(n_min, n_max + 1, step):
        for m in range(m_min, m_max + 1, step):
            pp = p0 + m * v1 + n * v2
            x, y = float(pp[0]), float(pp[1])
            if 0 <= x < W and 0 <= y < H:
                cand.append((x, y))
    if len(cand) == 0:
        return None, None, None

    if len(cand) > max_pts:
        idx = np.linspace(0, len(cand) - 1, max_pts).astype(int)
        cand = [cand[i] for i in idx]

    def snap(x, y, r=SNAP_R, thr_rel=SNAP_THR_REL):
        xi, yi = int(round(x)), int(round(y))
        x0 = max(0, xi - r); x1 = min(W - 1, xi + r)
        y0 = max(0, yi - r); y1 = min(H - 1, yi + r)
        win = im_pos[y0:y1 + 1, x0:x1 + 1]
        if win.size == 0:
            return x, y, 0.0
        mx = float(win.max())
        if mx <= 1e-6:
            return x, y, 0.0
        mask = win >= (thr_rel * mx)
        ww = win * mask.astype(np.float32)
        s = float(ww.sum())
        if s <= 1e-6:
            return x, y, 0.0
        yy, xx = np.indices(ww.shape, dtype=np.float32)
        cx = float((ww * xx).sum() / s)
        cy = float((ww * yy).sum() / s)
        return float(x0 + cx), float(y0 + cy), s

    shifts = []
    pts_pred = []
    pts_snap = []
    for (x, y) in cand:
        xs, ys, conf = snap(x, y)
        if conf > 1e-3:
            shifts.append([xs - x, ys - y])
            pts_pred.append([x, y])
            pts_snap.append([xs, ys])

    if len(shifts) < 20:
        return None, None, None

    shifts = np.array(shifts, dtype=np.float32)
    mag = np.linalg.norm(shifts, axis=1)
    thr = np.percentile(mag, OUTLIER_PCT)
    keep = mag <= thr
    mag_k = mag[keep]

    stats = {
        "count_raw": int(mag.shape[0]),
        "count_keep": int(mag_k.shape[0]),
        "mean": float(np.mean(mag_k)),
        "median": float(np.median(mag_k)),
        "p90": float(np.percentile(mag_k, 90)),
        "p95": float(np.percentile(mag_k, 95)),
    }

    # 简单向量图（只做 sanity check）
    vis = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    pts_pred = np.array(pts_pred, dtype=np.float32)[keep]
    pts_snap = np.array(pts_snap, dtype=np.float32)[keep]
    for p, s, e in zip(pts_pred, pts_snap, mag_k):
        x0, y0 = int(round(p[0])), int(round(p[1]))
        x1 = int(round(p[0] + (s[0] - p[0]) * EVAL_ARROW_SCALE))
        y1 = int(round(p[1] + (s[1] - p[1]) * EVAL_ARROW_SCALE))
        color = (0, 255, 0) if e < 0.8 else ((0, 255, 255) if e < 1.6 else (0, 0, 255))
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, 1, tipLength=0.25)

    return stats, vis, (m_min, m_max, n_min, n_max)

# ============================================================
# 5) 标定主流程：检测中心 + LS 拟合 + 自动 step，并保存 m/n 范围
# ============================================================

def calibrate_from_image(img_path, roi_h, roi_w, out_dir):
    img = read_gray(img_path)
    roi_u8, roi_x0, roi_y0 = center_crop_roi(img, roi_h, roi_w)
    cv2.imwrite(os.path.join(out_dir, "roi.png"), roi_u8)

    im_pos = make_im_pos_for_snap(roi_u8)
    cv2.imwrite(os.path.join(out_dir, "im_pos_for_snap.png"), normalize_u8(im_pos))

    v1_init, v2_init = fft_find_v1v2(roi_u8, out_dir)
    pts = detect_blob_centers(im_pos, out_dir)

    H, W = roi_u8.shape
    center = np.array([W * 0.5, H * 0.5], dtype=np.float32)
    d = np.linalg.norm(pts - center[None, :], axis=1)
    p0_init = pts[int(np.argmin(d))].astype(np.float32)

    p0, v1, v2 = lattice_fit_ls(pts, v1_init, v2_init, p0_init)

    stats1, vis1, rng1 = eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step=1)
    stats2, vis2, rng2 = eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step=2)

    # 选择更好的 step（如果 step2 明显更好就用 step2）
    best_step = 1
    best_stats, best_vis, best_rng = stats1, vis1, rng1
    if stats1 is None and stats2 is not None:
        best_step, best_stats, best_vis, best_rng = 2, stats2, vis2, rng2
    elif stats1 is not None and stats2 is not None:
        if float(stats2["median"]) < 0.7 * float(stats1["median"]):
            best_step, best_stats, best_vis, best_rng = 2, stats2, vis2, rng2

    if best_vis is not None:
        cv2.imwrite(os.path.join(out_dir, "grid_fit_error_vectors.png"), best_vis)

    m_min, m_max, n_min, n_max = best_rng
    m_min, m_max = align_range_to_step(m_min, m_max, best_step, ref=0)
    n_min, n_max = align_range_to_step(n_min, n_max, best_step, ref=0)

    overlay = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    count = 0
    for n in range(n_min, n_max + 1, best_step):
        for m in range(m_min, m_max + 1, best_step):
            pp = p0 + m * v1 + n * v2
            x, y = float(pp[0]), float(pp[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                count += 1
    cv2.circle(overlay, (int(round(p0[0])), int(round(p0[1]))), 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "grid_overlay_pred.png"), overlay)

    calib_path = os.path.join(out_dir, CALIB_NAME)
    np.savez(
        calib_path,
        v1=v1.astype(np.float32),
        v2=v2.astype(np.float32),
        p0=p0.astype(np.float32),
        roi_x0=int(roi_x0), roi_y0=int(roi_y0),
        roi_w=int(roi_w), roi_h=int(roi_h),
        img_path=img_path,
        grid_step=int(best_step),
        m_min=int(m_min), m_max=int(m_max),
        n_min=int(n_min), n_max=int(n_max),
        r_int=int(R_INT),
    )

    pitch1 = float(np.linalg.norm(v1))
    pitch2 = float(np.linalg.norm(v2))
    angle_deg = float(math.degrees(math.atan2(v1[1], v1[0])))

    print("=== CALIB DONE (BLOB + LS FIT) ===")
    print("calib saved:", calib_path)
    print("GRID_STEP =", best_step)
    print("v1 =", v1, " |v1| =", pitch1, "px")
    print("v2 =", v2, " |v2| =", pitch2, "px")
    print("angle(v1) =", angle_deg, "deg")
    print("p0 =", p0)
    print("overlay pts:", count)
    if best_stats is not None:
        print("=== GRID FIT ERROR STATS (px) ===")
        for k, v in best_stats.items():
            print(f"  {k}: {v}")

    return calib_path

# ============================================================
# 实时部分：预计算网格中心 + remap map
# ============================================================

def precompute_grid_centers(cal):
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_w = int(cal["roi_w"]); roi_h = int(cal["roi_h"])
    step = int(cal["grid_step"])
    m_min = int(cal["m_min"]); m_max = int(cal["m_max"])
    n_min = int(cal["n_min"]); n_max = int(cal["n_max"])
    r_int = int(cal.get("r_int", R_INT))

    centers = []
    idx_mn = []
    for n in range(n_min, n_max + 1, step):
        rr = (n - n_min) // step
        for m in range(m_min, m_max + 1, step):
            cc = (m - m_min) // step
            pp = p0 + m * v1 + n * v2
            x, y = float(pp[0]), float(pp[1])
            xi, yi = int(round(x)), int(round(y))
            if (xi - r_int) < 0 or (yi - r_int) < 0 or (xi + r_int) >= roi_w or (yi + r_int) >= roi_h:
                continue
            centers.append((xi, yi))
            idx_mn.append((rr, cc))

    rows = (n_max - n_min) // step + 1
    cols = (m_max - m_min) // step + 1
    return centers, idx_mn, rows, cols

def precompute_warp_map(cal):
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_w = int(cal["roi_w"]); roi_h = int(cal["roi_h"])
    step = int(cal["grid_step"])
    m_min = int(cal["m_min"])
    n_min = int(cal["n_min"])

    V = np.stack([v1, v2], axis=1).astype(np.float32)
    Vinv = np.linalg.inv(V).astype(np.float32)

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
    dx = xx - float(p0[0])
    dy = yy - float(p0[1])

    m_cont = Vinv[0, 0] * dx + Vinv[0, 1] * dy
    n_cont = Vinv[1, 0] * dx + Vinv[1, 1] * dy

    map_x = (m_cont - float(m_min)) / float(step)
    map_y = (n_cont - float(n_min)) / float(step)
    return map_x.astype(np.float32), map_y.astype(np.float32)

def integrate_grid_fast(roi_u8, centers, idx_mn, rows, cols, r_int, w):
    X = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.uint8)

    for (xi, yi), (rr, cc) in zip(centers, idx_mn):
        patch = roi_u8[yi - r_int: yi + r_int + 1, xi - r_int: xi + r_int + 1].astype(np.float32)
        X[rr, cc] = float(np.sum(patch * w))
        valid[rr, cc] = 1

    return X, valid

def capture_flat_from_camera(cap, roi_x0, roi_y0, roi_w, roi_h, sec=0.5):
    t0 = time.time()
    acc = None
    cnt = 0
    while time.time() - t0 < sec:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = crop_roi_by_xywh(gray, roi_x0, roi_y0, roi_w, roi_h).astype(np.float32)
        if acc is None:
            acc = roi
        else:
            acc += roi
        cnt += 1

    if acc is None or cnt < 1:
        raise RuntimeError("没抓到 flat 帧：检查摄像头。")
    flat = (acc / float(cnt)).clip(0, 255).astype(np.uint8)
    return flat, cnt

def warp_grid_to_roi(XLR, map_x, map_y):
    return cv2.remap(
        XLR, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(np.float32)

def render_float_to_u8(im_f, valid_mask=None, lo_hi_state=None, p_lo=2, p_hi=98, smooth_alpha=0.12):
    x = im_f
    if valid_mask is not None:
        vals = x[valid_mask > 0]
    else:
        vals = x[x != 0]

    if vals.size < 10:
        lo = float(np.min(x))
        hi = float(np.max(x))
    else:
        lo, hi = np.percentile(vals, [p_lo, p_hi])
        lo = float(lo); hi = float(hi)

    if lo_hi_state is None:
        lo_s, hi_s = lo, hi
    else:
        lo_s, hi_s = lo_hi_state
        lo_s = (1 - smooth_alpha) * lo_s + smooth_alpha * lo
        hi_s = (1 - smooth_alpha) * hi_s + smooth_alpha * hi

    y = (x - lo_s) / (hi_s - lo_s + 1e-8)
    y = np.clip(y, 0, 1)
    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    return u8, (lo_s, hi_s)

# ============================================================
# main：标定一次 + 实时 ratio warp
# ============================================================

def main():
    ensure_dir(OUT_DIR)
    calib_path = os.path.join(OUT_DIR, CALIB_NAME)

    # 1) calib：没有就跑一次
    if not os.path.exists(calib_path):
        calibrate_from_image(IMG_CALIB_PATH, ROI_H, ROI_W, OUT_DIR)

    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_x0 = int(cal["roi_x0"]); roi_y0 = int(cal["roi_y0"])
    roi_w = int(cal["roi_w"]); roi_h = int(cal["roi_h"])
    step = int(cal["grid_step"])
    m_min = int(cal["m_min"]); m_max = int(cal["m_max"])
    n_min = int(cal["n_min"]); n_max = int(cal["n_max"])

    print("\n=== REALTIME START ===")
    print(f"ROI = ({roi_w}x{roi_h}) at (x0,y0)=({roi_x0},{roi_y0})")
    print(f"grid step={step}, m=[{m_min},{m_max}], n=[{n_min},{n_max}], R_INT={R_INT}")

    # 2) 预计算：grid centers + weights + warp map
    centers, idx_mn, rows, cols = precompute_grid_centers(cal)
    w = make_weights(R_INT)
    map_x, map_y = precompute_warp_map(cal)

    print(f"valid centers (precomputed) = {len(centers)} / total grid {rows*cols}")

    # 3) DARK：读文件并 crop
    try:
        dark_full = read_gray(DARK_PATH)
        dark_roi = crop_roi_by_xywh(dark_full, roi_x0, roi_y0, roi_w, roi_h)
        print("[OK] loaded DARK from", DARK_PATH)
    except Exception as e:
        dark_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        print("[WARN] DARK not found or failed, use zeros. err =", str(e))

    # 4) 打开摄像头
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: CAM_ID={CAM_ID}")

    # 可选：尝试设置分辨率（不保证生效）
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 5) 预热 + 抓 flat（0.5s 多帧平均）
    print(f"[INFO] warming up & capturing FLAT for {FLAT_AVG_SEC:.2f}s ...")
    time.sleep(max(0.0, CAM_WARMUP_SEC - FLAT_AVG_SEC))
    flat_roi, cnt = capture_flat_from_camera(cap, roi_x0, roi_y0, roi_w, roi_h, sec=FLAT_AVG_SEC)
    cv2.imwrite(os.path.join(OUT_DIR, "flat_from_cam.png"), flat_roi)
    print(f"[OK] FLAT captured: averaged {cnt} frames, saved flat_from_cam.png")

    # 6) 预计算 flat/dark 的网格积分（这样实时只算 meas）
    X_dark, V_dark = integrate_grid_fast(dark_roi, centers, idx_mn, rows, cols, R_INT, w)
    X_flat, V_flat = integrate_grid_fast(flat_roi, centers, idx_mn, rows, cols, R_INT, w)
    valid = (V_dark & V_flat).astype(np.uint8)

    # 7) 实时循环
    lo_hi_state = None
    last_time = time.time()
    save_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # roi = crop_roi_by_xywh(gray, roi_x0, roi_y0, roi_w, roi_h)

        # 先裁剪彩色 ROI（用于显示）
        roi_bgr = frame[roi_y0:roi_y0 + roi_h, roi_x0:roi_x0 + roi_w].copy()
        if roi_bgr.shape[0] != roi_h or roi_bgr.shape[1] != roi_w:
            continue

        # 再转灰度 ROI（用于积分/计算）
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)


        # meas grid
        X_meas, V_meas = integrate_grid_fast(roi, centers, idx_mn, rows, cols, R_INT, w)
        valid_now = (valid & V_meas).astype(np.uint8)

        # ratio
        den = (X_flat - X_dark)
        X_ratio = (X_meas - X_dark) / np.maximum(den, RATIO_EPS)
        X_ratio[valid_now == 0] = 0.0

        # warp to ROI
        S_roi = warp_grid_to_roi(X_ratio, map_x, map_y)

        # show
        vis_u8, lo_hi_state = render_float_to_u8(S_roi, valid_mask=None, lo_hi_state=lo_hi_state,
                                                 p_lo=2, p_hi=98, smooth_alpha=VIS_SMOOTH_ALPHA)
        roi_show = roi_bgr.copy()

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - last_time))
        last_time = now

        # cv2.putText(roi_show, f"FPS {fps:.1f}  step={step}  R={R_INT}", (10, 22),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2, cv2.LINE_AA)

        cv2.putText(roi_show, f"FPS {fps:.1f}  step={step}  R={R_INT}", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("ROI (camera)", roi_show)
        cv2.imshow("Warp Ratio (for force field)", vis_u8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 重新抓 flat（0.5s 平均）
        if key == ord('f'):
            print(f"[INFO] recapturing FLAT for {FLAT_AVG_SEC:.2f}s ...")
            flat_roi, cnt = capture_flat_from_camera(cap, roi_x0, roi_y0, roi_w, roi_h, sec=FLAT_AVG_SEC)
            cv2.imwrite(os.path.join(OUT_DIR, "flat_from_cam.png"), flat_roi)
            X_flat, V_flat = integrate_grid_fast(flat_roi, centers, idx_mn, rows, cols, R_INT, w)
            valid = (V_dark & V_flat).astype(np.uint8)
            print(f"[OK] FLAT updated: averaged {cnt} frames")

        # 保存当前结果
        if key == ord('s'):
            fn_npy = os.path.join(OUT_DIR, f"realtime_ratiowarp_{save_id:03d}.npy")
            fn_png = os.path.join(OUT_DIR, f"realtime_ratiowarp_{save_id:03d}.png")
            np.save(fn_npy, S_roi.astype(np.float32))
            cv2.imwrite(fn_png, vis_u8)
            print("[SAVED]", fn_npy, fn_png)
            save_id += 1

        # 简单限速（可选）
        if TARGET_FPS is not None and TARGET_FPS > 0:
            # 如果太快就稍微 sleep 一下
            dt = time.time() - now
            target_dt = 1.0 / float(TARGET_FPS)
            if dt < target_dt:
                time.sleep(target_dt - dt)

    cap.release()
    cv2.destroyAllWindows()
    print("=== REALTIME STOP ===")

if __name__ == "__main__":
    main()
