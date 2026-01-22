import os
import math
import cv2
import numpy as np


# ============================================================
# 配置区：你主要改这里
# ============================================================

IMG_CALIB_PATH = "../snapshots/snapshot_20260115_160108.jpg"  # 用于标定（孔阵列清晰、最好无按压）
MEAS_PATH      = "../snapshots/snapshot_20260116_162823.jpg"  # 有按压
FLAT_PATH      = "../snapshots/flat.jpg"                      # 无按压平场
DARK_PATH      = "../snapshots/dark.jpg"                      # 全黑

OUT_DIR        = "./out_allinone"
OUT_PREFIX     = "xlr"

# ROI（标定时用同样的 ROI 用于后续处理）
ROI_H, ROI_W   = 420, 640

# 积分半径（像素）
R_INT          = 8

# p0 refine 相关
REFINE_ITERS   = 5
REFINE_ALPHA   = 0.3
REFINE_MR      = 8
REFINE_NR      = 8

# FFT 选峰相关
PEAK_TOPK      = 80
PEAK_TRY_TOPN  = 40

# 是否做 crop（按 valid bbox 裁剪网格域）
DO_CROP        = True

# 是否做 warp（摆正成 ROI 尺寸）
DO_WARP        = True

# ===== 改动B：误差量化参数 =====
EVAL_MAX_PTS   = 2000   # 最多评估多少个孔点（太多会慢）
EVAL_OUTLIER_PCT = 95   # 去掉误差最大的 5% 点（避免离群影响统计）
EVAL_R_SNAP    = 7
EVAL_THR_REL   = 0.35
EVAL_ARROW_SCALE = 8.0  # 只是画箭头用（统计不受影响）


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
        raise RuntimeError(f"ROI crop mismatch got {roi.shape}, expected {(roi_h, roi_w)}. "
                           f"Image maybe smaller than ROI?")
    return roi, x0, y0

def crop_roi_by_xywh(img_u8, x0, y0, w, h):
    H, W = img_u8.shape
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)
    roi = img_u8[y0:y1, x0:x1].copy()
    if roi.shape != (h, w):
        raise RuntimeError(f"ROI size mismatch: got {roi.shape}, expected {(h, w)}. "
                           f"Check image size or calib ROI.")
    return roi

def make_hanning_window(H, W):
    wy = np.hanning(H).astype(np.float32)
    wx = np.hanning(W).astype(np.float32)
    return wy[:, None] * wx[None, :]

def make_weights(r):
    r = int(r)
    yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    rr2 = xx*xx + yy*yy
    mask = (rr2 <= r*r).astype(np.float32)
    sigma = max(1.0, r/2.0)
    w = np.exp(-0.5 * rr2 / (sigma*sigma)) * mask
    w = w / (w.sum() + 1e-8)
    return w

def snap_to_centroid(im_pos, x, y, r=7, thr_rel=0.35):
    """在 (x,y) 附近找亮斑质心，用于把预测孔中心吸附到真实孔中心。"""
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


# ============================================================
# 改动B：网格拟合误差量化 + 向量场可视化
# ============================================================

def eval_grid_fit(dog, roi_u8, p0, v1, v2, m_min, m_max, n_min, n_max,
                  r_snap=7, thr_rel=0.35, max_pts=2000, outlier_pct=95,
                  arrow_scale=8.0):
    """
    用 snap_to_centroid 的位移来量化网格拟合误差。
    返回 stats dict 和 可视化图（BGR）。
    """
    H, W = roi_u8.shape

    # 收集候选点（预测孔中心落在 ROI 内）
    cand = []
    for n in range(n_min, n_max + 1):
        for m in range(m_min, m_max + 1):
            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            if 0 <= x < W and 0 <= y < H:
                cand.append((m, n, x, y))

    if len(cand) == 0:
        return None, None

    # 太多就均匀抽样
    if len(cand) > max_pts:
        idx = np.linspace(0, len(cand) - 1, max_pts).astype(int)
        cand = [cand[i] for i in idx]

    shifts = []
    pts_pred = []
    pts_snap = []
    confs = []

    for (_, _, x, y) in cand:
        xs, ys, conf = snap_to_centroid(dog, x, y, r=r_snap, thr_rel=thr_rel)
        if conf > 1e-3:
            shifts.append([xs - x, ys - y])
            pts_pred.append([x, y])
            pts_snap.append([xs, ys])
            confs.append(conf)

    if len(shifts) < 30:
        return None, None

    shifts = np.array(shifts, dtype=np.float32)
    pts_pred = np.array(pts_pred, dtype=np.float32)
    pts_snap = np.array(pts_snap, dtype=np.float32)

    mag = np.linalg.norm(shifts, axis=1)

    # 去离群：丢掉误差最大的 (100-outlier_pct)%
    thr = np.percentile(mag, outlier_pct)
    keep = mag <= thr

    shifts_k = shifts[keep]
    mag_k = mag[keep]
    pts_pred_k = pts_pred[keep]
    pts_snap_k = pts_snap[keep]

    # 统计量
    stats = {
        "count_raw": int(len(mag)),
        "count_keep": int(len(mag_k)),
        "mean": float(np.mean(mag_k)),
        "median": float(np.median(mag_k)),
        "rms": float(np.sqrt(np.mean(mag_k ** 2))),
        "p90": float(np.percentile(mag_k, 90)),
        "p95": float(np.percentile(mag_k, 95)),
        "max_keep": float(np.max(mag_k)),
        "outlier_thr": float(thr),
        "mean_dx": float(np.mean(shifts_k[:, 0])),
        "mean_dy": float(np.mean(shifts_k[:, 1])),
    }

    # 可视化：预测点 -> snap 点的箭头（只是显示用，箭头可放大）
    vis = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)

    for p, s, e in zip(pts_pred_k, pts_snap_k, mag_k):
        x0, y0 = int(round(p[0])), int(round(p[1]))
        x1 = int(round(p[0] + (s[0] - p[0]) * arrow_scale))
        y1 = int(round(p[1] + (s[1] - p[1]) * arrow_scale))

        # 颜色按误差大小分级
        if e < 0.5:
            color = (0, 255, 0)
        elif e < 1.0:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.circle(vis, (x0, y0), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, 1, tipLength=0.25)

    txt = (f"fit err(px): median={stats['median']:.3f} mean={stats['mean']:.3f} "
           f"rms={stats['rms']:.3f} p90={stats['p90']:.3f} p95={stats['p95']:.3f} "
           f"kept={stats['count_keep']}/{stats['count_raw']}")
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return stats, vis


# ============================================================
# Step A: 标定（FFT 找 v1/v2 + refine p0）
# ============================================================

def calibrate_from_image(img_path, roi_h, roi_w, out_dir):
    img = read_gray(img_path)
    roi, roi_x0, roi_y0 = center_crop_roi(img, roi_h, roi_w)
    cv2.imwrite(os.path.join(out_dir, "roi.png"), roi)

    roi_f = roi.astype(np.float32)
    roi_f = roi_f - roi_f.mean()

    H, W = roi_f.shape
    win = make_hanning_window(H, W)

    F = np.fft.fftshift(np.fft.fft2(roi_f * win))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    cy, cx = H // 2, W // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # 掐掉 DC 和超高频
    rmin = 0.04 * min(H, W)
    rmax = 0.45 * min(H, W)
    mask = (R >= rmin) & (R <= rmax)
    mag2 = mag.copy()
    mag2[~mask] = 0.0

    # 取 topK 峰
    K = int(PEAK_TOPK)
    flat_idx = np.argpartition(mag2.ravel(), -K)[-K:]
    peaks = np.column_stack(np.unravel_index(flat_idx, (H, W)))
    peaks = peaks[np.argsort(-mag2[peaks[:, 0], peaks[:, 1]])]

    def to_k(py, px):
        fy = (py - cy) / float(H)
        fx = (px - cx) / float(W)
        return np.array([fx, fy], dtype=np.float32)

    ks = [to_k(int(py), int(px)) for py, px in peaks]

    # 选两条“基本峰”：低频 + 半径相近 + 不共线（最好接近正交）
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
        raise RuntimeError("没找到合适的基本峰：试试换 ROI 或换更清晰的图。")

    k1, k2, p1, p2 = best

    # 倒易基 -> 空间基
    Kmat = np.stack([k1, k2], axis=0).astype(np.float32)  # 2x2
    V = np.linalg.inv(Kmat)
    v1 = V[:, 0]
    v2 = V[:, 1]

    # 规范化方向：v1 x>0 + 右手系
    if v1[0] < 0:
        v1 = -v1
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2

    # 画频谱 + 标峰
    spec_vis = (mag * 255).astype(np.uint8)
    spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)

    def draw_peak(vis, py, px, color):
        cv2.circle(vis, (int(px), int(py)), 7, color, 2, lineType=cv2.LINE_AA)

    draw_peak(spec_vis, int(p1[0]), int(p1[1]), (0, 255, 0))
    draw_peak(spec_vis, int(p2[0]), int(p2[1]), (0, 0, 255))
    cv2.circle(spec_vis, (cx, cy), 4, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "fft_peaks.png"), spec_vis)

    # 初始 p0：中心附近最大亮点
    roi_u8 = roi
    H2, W2 = roi_u8.shape
    cx2, cy2 = W2 // 2, H2 // 2
    r_search = int(0.25 * min(H2, W2))
    x0s, x1s = max(0, cx2 - r_search), min(W2, cx2 + r_search)
    y0s, y1s = max(0, cy2 - r_search), min(H2, cy2 + r_search)
    patch = roi_u8[y0s:y1s, x0s:x1s]
    py0, px0 = np.unravel_index(np.argmax(patch), patch.shape)
    p0 = np.array([x0s + px0, y0s + py0], dtype=np.float32)

    # 反推覆盖 ROI 的 m,n 范围（为了画 overlay、以及 refine）
    V2 = np.stack([v1, v2], axis=1).astype(np.float32)  # 2x2 columns
    V2inv = np.linalg.inv(V2)
    corners = np.array([[0, 0], [W2 - 1, 0], [0, H2 - 1], [W2 - 1, H2 - 1]], dtype=np.float32)
    mn = (V2inv @ (corners - p0).T).T
    m_min = int(np.floor(mn[:, 0].min())) - 2
    m_max = int(np.ceil(mn[:, 0].max())) + 2
    n_min = int(np.floor(mn[:, 1].min())) - 2
    n_max = int(np.ceil(mn[:, 1].max())) + 2

    # refine p0：DoG + centroid snapping（只用中心区域孔点）
    roi_f32 = roi_u8.astype(np.float32)
    g1 = cv2.GaussianBlur(roi_f32, (0, 0), 1.2)
    g2 = cv2.GaussianBlur(roi_f32, (0, 0), 4.0)
    dog = np.maximum(g1 - g2, 0)

    m0 = (m_min + m_max) // 2
    n0 = (n_min + n_max) // 2
    alpha = float(REFINE_ALPHA)
    p0_ref = p0.copy()

    for it in range(int(REFINE_ITERS)):
        shifts = []
        for n in range(n0 - int(REFINE_NR), n0 + int(REFINE_NR) + 1):
            for m in range(m0 - int(REFINE_MR), m0 + int(REFINE_MR) + 1):
                p = p0_ref + m * v1 + n * v2
                x, y = float(p[0]), float(p[1])
                if 0 <= x < W2 and 0 <= y < H2:
                    xs, ys, conf = snap_to_centroid(dog, x, y, r=7, thr_rel=0.35)
                    if conf > 1e-3:
                        shifts.append([xs - x, ys - y])

        if len(shifts) < 30:
            break

        shifts = np.array(shifts, dtype=np.float32)
        d = np.linalg.norm(shifts, axis=1)
        keep = d <= np.percentile(d, 90)
        shifts = shifts[keep]

        delta = np.median(shifts, axis=0)
        p0_ref = p0_ref + alpha * delta

        if float(np.linalg.norm(alpha * delta)) < 0.12:
            break

    p0 = p0_ref

    # ===== 改动B：误差量化 + 向量场 =====
    stats, err_vis = eval_grid_fit(
        dog, roi_u8, p0, v1, v2, m_min, m_max, n_min, n_max,
        r_snap=EVAL_R_SNAP,
        thr_rel=EVAL_THR_REL,
        max_pts=EVAL_MAX_PTS,
        outlier_pct=EVAL_OUTLIER_PCT,
        arrow_scale=EVAL_ARROW_SCALE
    )

    if stats is None:
        print("WARN: fit eval got too few valid points, skip error stats.")
    else:
        print("=== GRID FIT ERROR STATS (px) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        cv2.imwrite(os.path.join(out_dir, "grid_fit_error_vectors.png"), err_vis)
        print("[DONE] wrote grid_fit_error_vectors.png")

    # overlay 检查
    overlay = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    count = 0
    for n in range(n_min, n_max + 1):
        for m in range(m_min, m_max + 1):
            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            if 0 <= x < W2 and 0 <= y < H2:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                count += 1
    cv2.circle(overlay, (int(round(p0[0])), int(round(p0[1]))), 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "grid_overlay_pred.png"), overlay)

    # 保存 calib
    calib_path = os.path.join(out_dir, "calib_fft.npz")
    np.savez(
        calib_path,
        v1=v1.astype(np.float32),
        v2=v2.astype(np.float32),
        p0=p0.astype(np.float32),
        roi_x0=int(roi_x0), roi_y0=int(roi_y0),
        roi_w=int(roi_w), roi_h=int(roi_h),
        img_path=img_path
    )

    # 打印一些信息方便你 sanity check
    pitch1 = float(np.linalg.norm(v1))
    pitch2 = float(np.linalg.norm(v2))
    angle_deg = float(math.degrees(math.atan2(v1[1], v1[0])))

    print("=== CALIB DONE ===")
    print("calib saved:", calib_path)
    print("v1 =", v1, " |v1| =", pitch1, "px")
    print("v2 =", v2, " |v2| =", pitch2, "px")
    print("angle(v1) =", angle_deg, "deg")
    print("p0 =", p0)
    print("overlay pts:", count)

    return calib_path


# ============================================================
# Step B: 网格积分 + diff/ratio + 可视化
# ============================================================

def integrate_grid(im_u8, p0, v1, v2, m_min, m_max, n_min, n_max, r_int):
    rows = (n_max - n_min) + 1
    cols = (m_max - m_min) + 1
    w = make_weights(r_int)

    X = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.uint8)

    H2, W2 = im_u8.shape

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

def vis_save(X, valid_mask, out_png, scale=18, p_lo=5, p_hi=95):
    Xv = X.copy().astype(np.float32)
    vals = Xv[valid_mask > 0]
    if vals.size < 10:
        lo, hi = float(Xv.min()), float(Xv.max())
    else:
        lo, hi = np.percentile(vals, [p_lo, p_hi])

    Y = (Xv - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)
    Y[valid_mask == 0] = 0.0

    u8 = (Y * 255.0 + 0.5).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1]*scale, u8.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, big)

def run_integration(calib_path, meas_path, flat_path, dark_path, out_dir, out_prefix, r_int):
    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_x0 = int(cal["roi_x0"]); roi_y0 = int(cal["roi_y0"])
    roi_w  = int(cal["roi_w"]);  roi_h  = int(cal["roi_h"])

    meas = crop_roi_by_xywh(read_gray(meas_path), roi_x0, roi_y0, roi_w, roi_h)
    flat = crop_roi_by_xywh(read_gray(flat_path), roi_x0, roi_y0, roi_w, roi_h)
    dark = crop_roi_by_xywh(read_gray(dark_path), roi_x0, roi_y0, roi_w, roi_h)

    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_meas.png"), meas)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_flat.png"), flat)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_dark.png"), dark)

    # 计算“有效 m/n 范围”：保证积分 patch 不越界
    V2 = np.stack([v1, v2], axis=1).astype(np.float32)
    V2inv = np.linalg.inv(V2).astype(np.float32)

    corners_inner = np.array([
        [r_int,             r_int],
        [roi_w - 1 - r_int, r_int],
        [r_int,             roi_h - 1 - r_int],
        [roi_w - 1 - r_int, roi_h - 1 - r_int],
    ], dtype=np.float32)

    mn_inner = (V2inv @ (corners_inner - p0).T).T

    m_min = int(np.ceil (mn_inner[:, 0].min()))
    m_max = int(np.floor(mn_inner[:, 0].max()))
    n_min = int(np.ceil (mn_inner[:, 1].min()))
    n_max = int(np.floor(mn_inner[:, 1].max()))

    rows = (n_max - n_min) + 1
    cols = (m_max - m_min) + 1

    print("=== INTEGRATION ===")
    print("Effective m range:", (m_min, m_max), "n range:", (n_min, n_max))
    print("Grid size (rows, cols):", (rows, cols))

    X_meas, V_meas = integrate_grid(meas, p0, v1, v2, m_min, m_max, n_min, n_max, r_int)
    X_flat, V_flat = integrate_grid(flat, p0, v1, v2, m_min, m_max, n_min, n_max, r_int)
    X_dark, V_dark = integrate_grid(dark, p0, v1, v2, m_min, m_max, n_min, n_max, r_int)

    valid = (V_meas & V_flat & V_dark).astype(np.uint8)
    print("valid count:", int(valid.sum()), "/", valid.size)

    # 保存
    np.save(os.path.join(out_dir, f"{out_prefix}_meas.npy"), X_meas)
    np.save(os.path.join(out_dir, f"{out_prefix}_flat.npy"), X_flat)
    np.save(os.path.join(out_dir, f"{out_prefix}_dark.npy"), X_dark)
    np.save(os.path.join(out_dir, f"{out_prefix}_valid.npy"), valid)

    # diff/ratio
    X_diff = (X_meas - X_flat)
    den = (X_flat - X_dark)
    eps = 1e-3
    X_ratio = (X_meas - X_dark) / np.maximum(den, eps)

    np.save(os.path.join(out_dir, f"{out_prefix}_diff.npy"), X_diff)
    np.save(os.path.join(out_dir, f"{out_prefix}_ratio.npy"), X_ratio)

    # 可视化
    vis_save(X_meas,  valid, os.path.join(out_dir, f"{out_prefix}_meas_vis.png"))
    vis_save(X_flat,  valid, os.path.join(out_dir, f"{out_prefix}_flat_vis.png"))
    vis_save(X_dark,  valid, os.path.join(out_dir, f"{out_prefix}_dark_vis.png"))
    vis_save(X_diff,  valid, os.path.join(out_dir, f"{out_prefix}_diff_vis.png"))
    vis_save(X_ratio, valid, os.path.join(out_dir, f"{out_prefix}_ratio_vis.png"))

    meta = {
        "v1": v1, "v2": v2, "p0": p0,
        "roi_w": roi_w, "roi_h": roi_h,
        "roi_x0": roi_x0, "roi_y0": roi_y0,
        "m_min": m_min, "m_max": m_max,
        "n_min": n_min, "n_max": n_max,
        "rows": rows, "cols": cols
    }
    return meta


# ============================================================
# Step C: 网格域 crop（可选）
# ============================================================

def crop_by_valid(X, valid):
    ys, xs = np.where(valid > 0)
    if ys.size < 10:
        raise RuntimeError("valid 太少，没法 crop。先检查标定/积分。")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (X[y0:y1, x0:x1].copy(), valid[y0:y1, x0:x1].copy(), (y0, y1, x0, x1))

def run_crop(out_dir, out_prefix):
    Xdiff  = np.load(os.path.join(out_dir, f"{out_prefix}_diff.npy")).astype(np.float32)
    Xratio = np.load(os.path.join(out_dir, f"{out_prefix}_ratio.npy")).astype(np.float32)
    valid  = np.load(os.path.join(out_dir, f"{out_prefix}_valid.npy")).astype(np.uint8)

    diff_c,  valid_c, bbox = crop_by_valid(Xdiff, valid)
    ratio_c, _,      _     = crop_by_valid(Xratio, valid)

    np.save(os.path.join(out_dir, f"{out_prefix}_diff_crop.npy"), diff_c)
    np.save(os.path.join(out_dir, f"{out_prefix}_ratio_crop.npy"), ratio_c)
    np.save(os.path.join(out_dir, f"{out_prefix}_valid_crop.npy"), valid_c)

    vis_save(diff_c,  valid_c, os.path.join(out_dir, f"{out_prefix}_diff_crop_vis.png"))
    vis_save(ratio_c, valid_c, os.path.join(out_dir, f"{out_prefix}_ratio_crop_vis.png"))

    print("=== CROP DONE ===")
    print("valid bbox (y0,y1,x0,x1) =", bbox)
    print("crop size =", diff_c.shape)


# ============================================================
# Step D: warp 摆正成 ROI 尺寸（可选）
# ============================================================

def warp_to_roi(calib_path, in_xlr_npy, out_dir, out_name, r_int):
    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_w  = int(cal["roi_w"])
    roi_h  = int(cal["roi_h"])

    XLR = np.load(in_xlr_npy).astype(np.float32)
    rows, cols = XLR.shape

    V2 = np.stack([v1, v2], axis=1).astype(np.float32)
    V2inv = np.linalg.inv(V2).astype(np.float32)

    # 用 r_int 算有效 m/n 范围（与积分时一致）
    corners_inner = np.array([
        [r_int,             r_int],
        [roi_w - 1 - r_int, r_int],
        [r_int,             roi_h - 1 - r_int],
        [roi_w - 1 - r_int, roi_h - 1 - r_int],
    ], dtype=np.float32)

    mn_inner = (V2inv @ (corners_inner - p0).T).T
    m_min = int(np.ceil (mn_inner[:, 0].min()))
    m_max = int(np.floor(mn_inner[:, 0].max()))
    n_min = int(np.ceil (mn_inner[:, 1].min()))
    n_max = int(np.floor(mn_inner[:, 1].max()))

    rows_calc = (n_max - n_min) + 1
    cols_calc = (m_max - m_min) + 1

    if (rows, cols) != (rows_calc, cols_calc):
        # 容错：如果 shape 不一致，就按 XLR shape 反推一个范围（主要影响平移，不影响摆正角度）
        m_center = (m_min + m_max) / 2.0
        n_center = (n_min + n_max) / 2.0
        m_min = int(round(m_center - (cols - 1) / 2.0))
        m_max = m_min + cols - 1
        n_min = int(round(n_center - (rows - 1) / 2.0))
        n_max = n_min + rows - 1

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
    dx = xx - float(p0[0])
    dy = yy - float(p0[1])

    m_cont = V2inv[0, 0] * dx + V2inv[0, 1] * dy
    n_cont = V2inv[1, 0] * dx + V2inv[1, 1] * dy

    map_x = (m_cont - float(m_min)).astype(np.float32)  # cc
    map_y = (n_cont - float(n_min)).astype(np.float32)  # rr

    S_roi = cv2.remap(
        XLR, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(np.float32)

    out_npy = os.path.join(out_dir, f"{out_name}.npy")
    out_png = os.path.join(out_dir, f"{out_name}.png")
    np.save(out_npy, S_roi)

    vals = S_roi[S_roi != 0]
    if vals.size > 10:
        lo, hi = np.percentile(vals, [2, 98])
    else:
        lo, hi = float(S_roi.min()), float(S_roi.max())

    Vis = (S_roi - lo) / (hi - lo + 1e-8)
    Vis = np.clip(Vis, 0, 1)
    cv2.imwrite(out_png, (Vis * 255.0 + 0.5).astype(np.uint8))

    print("=== WARP DONE ===")
    print("wrote:", out_npy, out_png)


# ============================================================
# main：一口气跑完
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    # A) 标定
    calib_path = calibrate_from_image(
        IMG_CALIB_PATH, ROI_H, ROI_W, OUT_DIR
    )

    # B) 积分 + diff/ratio
    _ = run_integration(
        calib_path, MEAS_PATH, FLAT_PATH, DARK_PATH,
        OUT_DIR, OUT_PREFIX, R_INT
    )

    # C) crop（可选）
    if DO_CROP:
        run_crop(OUT_DIR, OUT_PREFIX)

    # D) warp（可选）
    if DO_WARP:
        in_diff = os.path.join(OUT_DIR, f"{OUT_PREFIX}_diff.npy")
        in_ratio = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ratio.npy")

        warp_to_roi(calib_path, in_diff,  OUT_DIR, "roiwarp_diff_S_roi",  R_INT)
        warp_to_roi(calib_path, in_ratio, OUT_DIR, "roiwarp_ratio_S_roi", R_INT)

    print("\nALL DONE. Output dir =", OUT_DIR)


if __name__ == "__main__":
    main()
