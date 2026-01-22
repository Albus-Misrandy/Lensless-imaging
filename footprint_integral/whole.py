import os
import math
import cv2
import numpy as np

# ============================================================
# 配置区：你主要改这里
# ============================================================

IMG_CALIB_PATH = "../snapshots/snapshot_20260115_160108.jpg"
MEAS_PATH = "../snapshots/snapshot_20260116_162823.jpg"
FLAT_PATH = "../snapshots/flat.jpg"
DARK_PATH = "../snapshots/dark.jpg"

OUT_DIR = "./out_allinone"
OUT_PREFIX = "xlr"

ROI_H, ROI_W = 420, 640

# 积分半径（像素）
R_INT = 8  # 你要 6 就改成 6

# ====== 固定 step=2（只取一套子晶格）======
GRID_STEP = 2

# p0 refine 相关
REFINE_ITERS = 7
REFINE_ALPHA = 0.35
REFINE_MR = 9
REFINE_NR = 9

# FFT 选峰相关
PEAK_TOPK = 80
PEAK_TRY_TOPN = 40

DO_CROP = True
DO_WARP = True

# ======= 方案二：生成“可吸附”的孔斑响应图 im_pos 参数 =======
BG_SIGMA = 18.0
SMOOTH_SIGMA = 2.5
CLAMP_POS = True

# ======= 吸附参数（refine + eval 共用）=======
SNAP_R = 9
SNAP_THR_REL = 0.25

# ===== 误差评估参数 =====
EVAL_OUTLIER_PCT = 95
EVAL_ARROW_SCALE = 8.0


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
    H, W = img_u8.shape
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)
    roi = img_u8[y0:y1, x0:x1].copy()
    if roi.shape != (h, w):
        raise RuntimeError(f"ROI size mismatch: got {roi.shape}, expected {(h, w)}.")
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


def snap_to_centroid(im_pos, x, y, r=7, thr_rel=0.35):
    """在 (x,y) 附近找亮斑质心，把预测孔中心吸附到真实孔中心。"""
    Hh, Ww = im_pos.shape
    xi, yi = int(round(x)), int(round(y))
    x0 = max(0, xi - r);
    x1 = min(Ww - 1, xi + r)
    y0 = max(0, yi - r);
    y1 = min(Hh - 1, yi + r)

    win = im_pos[y0:y1 + 1, x0:x1 + 1]
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


def make_im_pos_for_snap(roi_u8):
    """方案二：去背景 + 平滑 + 截断正值，得到适合“找质心”的响应图。"""
    roi_f = roi_u8.astype(np.float32)
    bg = cv2.GaussianBlur(roi_f, (0, 0), BG_SIGMA)
    hp = roi_f - bg
    hp2 = cv2.GaussianBlur(hp, (0, 0), SMOOTH_SIGMA)
    if CLAMP_POS:
        return np.maximum(hp2, 0)
    return hp2


def align_range_to_step(mn_min, mn_max, step, ref=0):
    """让 mn_min ≡ ref (mod step), mn_max ≡ ref (mod step)，取固定相位的子晶格。"""
    mn_min_a = mn_min + ((ref - mn_min) % step)
    mn_max_a = mn_max - ((mn_max - ref) % step)
    if mn_max_a < mn_min_a:
        raise RuntimeError("Aligned range became empty; check ROI/r_int/step.")
    return int(mn_min_a), int(mn_max_a)


# ============================================================
# 评估：网格拟合误差 + 向量场可视化（只输出 best 一张）
# ============================================================

def eval_grid_fit(im_pos, roi_u8, p0, v1, v2, m_min, m_max, n_min, n_max,
                  step=2, outlier_pct=95, arrow_scale=8.0):
    H, W = roi_u8.shape

    pts_pred = []
    pts_snap = []
    mags = []

    for n in range(n_min, n_max + 1, step):
        for m in range(m_min, m_max + 1, step):
            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            if not (0 <= x < W and 0 <= y < H):
                continue

            xs, ys, conf = snap_to_centroid(im_pos, x, y, r=SNAP_R, thr_rel=SNAP_THR_REL)
            if conf <= 1e-3:
                continue

            dx, dy = xs - x, ys - y
            e = math.sqrt(dx * dx + dy * dy)
            pts_pred.append([x, y])
            pts_snap.append([xs, ys])
            mags.append(e)

    if len(mags) < 20:
        return None, None

    pts_pred = np.asarray(pts_pred, np.float32)
    pts_snap = np.asarray(pts_snap, np.float32)
    mags = np.asarray(mags, np.float32)

    thr = np.percentile(mags, outlier_pct)
    keep = mags <= thr

    mags_k = mags[keep]
    pred_k = pts_pred[keep]
    snap_k = pts_snap[keep]

    stats = {
        "count_raw": int(len(mags)),
        "count_keep": int(len(mags_k)),
        "mean": float(np.mean(mags_k)),
        "median": float(np.median(mags_k)),
        "rms": float(np.sqrt(np.mean(mags_k ** 2))),
        "p90": float(np.percentile(mags_k, 90)),
        "p95": float(np.percentile(mags_k, 95)),
        "max_keep": float(np.max(mags_k)),
        "outlier_thr": float(thr),
        "mean_dx": float(np.mean((snap_k[:, 0] - pred_k[:, 0]))),
        "mean_dy": float(np.mean((snap_k[:, 1] - pred_k[:, 1]))),
    }

    vis = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    for p, s, e in zip(pred_k, snap_k, mags_k):
        x0, y0 = int(round(p[0])), int(round(p[1]))
        x1 = int(round(p[0] + (s[0] - p[0]) * arrow_scale))
        y1 = int(round(p[1] + (s[1] - p[1]) * arrow_scale))

        if e < 0.5:
            color = (0, 255, 0)
        elif e < 1.0:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.circle(vis, (x0, y0), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, 1, tipLength=0.25)

    txt = (f"step={step}  med={stats['median']:.3f} mean={stats['mean']:.3f} "
           f"p90={stats['p90']:.3f} p95={stats['p95']:.3f} kept={stats['count_keep']}/{stats['count_raw']}")
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return stats, vis


# ============================================================
# refine p0
# ============================================================

def refine_p0(im_pos, roi_u8, p0_in, v1, v2, m_min, m_max, n_min, n_max, step=2):
    H2, W2 = roi_u8.shape
    m0 = (m_min + m_max) // 2
    n0 = (n_min + n_max) // 2
    p0 = p0_in.copy()

    for _ in range(int(REFINE_ITERS)):
        shifts = []
        for n in range(n0 - int(REFINE_NR), n0 + int(REFINE_NR) + 1, step):
            for m in range(m0 - int(REFINE_MR), m0 + int(REFINE_MR) + 1, step):
                p = p0 + m * v1 + n * v2
                x, y = float(p[0]), float(p[1])
                if 0 <= x < W2 and 0 <= y < H2:
                    xs, ys, conf = snap_to_centroid(im_pos, x, y, r=SNAP_R, thr_rel=SNAP_THR_REL)
                    if conf > 1e-3:
                        shifts.append([xs - x, ys - y])

        if len(shifts) < 20:
            break

        shifts = np.asarray(shifts, np.float32)
        d = np.linalg.norm(shifts, axis=1)
        keep = d <= np.percentile(d, 90)
        shifts = shifts[keep]

        delta = np.median(shifts, axis=0)
        p0 = p0 + float(REFINE_ALPHA) * delta

        if float(np.linalg.norm(float(REFINE_ALPHA) * delta)) < 0.10:
            break

    return p0


# ============================================================
# Step A: 标定（FFT 找 v1/v2 + 相位搜索 + refine p0）
# ============================================================

def calibrate_from_image(img_path, roi_h, roi_w, out_dir):
    img = read_gray(img_path)
    roi, roi_x0, roi_y0 = center_crop_roi(img, roi_h, roi_w)
    cv2.imwrite(os.path.join(out_dir, "roi.png"), roi)

    roi_f = roi.astype(np.float32) - float(roi.mean())
    H, W = roi_f.shape
    win = make_hanning_window(H, W)

    F = np.fft.fftshift(np.fft.fft2(roi_f * win))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    cy, cx = H // 2, W // 2
    Y, X = np.indices((H, W))
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    # 抑制 DC & 超高频
    rmin = 0.04 * min(H, W)
    rmax = 0.45 * min(H, W)
    mask = (R >= rmin) & (R <= rmax)
    mag2 = mag.copy()
    mag2[~mask] = 0.0

    # topK 峰
    K = int(PEAK_TOPK)
    flat_idx = np.argpartition(mag2.ravel(), -K)[-K:]
    peaks = np.column_stack(np.unravel_index(flat_idx, (H, W)))
    peaks = peaks[np.argsort(-mag2[peaks[:, 0], peaks[:, 1]])]

    def to_k(py, px):
        fy = (py - cy) / float(H)
        fx = (px - cx) / float(W)
        return np.array([fx, fy], dtype=np.float32)

    ks = [to_k(int(py), int(px)) for py, px in peaks]

    # 选两条基本峰（低频 + 半径相近 + 不共线）
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

    # 方向规范化
    if v1[0] < 0:
        v1 = -v1
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2

    # 保存 fft 峰图
    spec_vis = (mag * 255).astype(np.uint8)
    spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)
    cv2.circle(spec_vis, (int(p1[1]), int(p1[0])), 7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (int(p2[1]), int(p2[0])), 7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (cx, cy), 4, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "fft_peaks.png"), spec_vis)

    roi_u8 = roi
    H2, W2 = roi_u8.shape

    # 初始 p0：中心附近最大亮点
    cx2, cy2 = W2 // 2, H2 // 2
    r_search = int(0.25 * min(H2, W2))
    x0s, x1s = max(0, cx2 - r_search), min(W2, cx2 + r_search)
    y0s, y1s = max(0, cy2 - r_search), min(H2, cy2 + r_search)
    patch = roi_u8[y0s:y1s, x0s:x1s]
    py0, px0 = np.unravel_index(np.argmax(patch), patch.shape)
    p0_init = np.array([x0s + px0, y0s + py0], dtype=np.float32)

    # 粗范围（为了 phase search 和 overlay）
    V2 = np.stack([v1, v2], axis=1).astype(np.float32)
    V2inv = np.linalg.inv(V2)
    corners = np.array([[0, 0], [W2 - 1, 0], [0, H2 - 1], [W2 - 1, H2 - 1]], dtype=np.float32)
    mn = (V2inv @ (corners - p0_init).T).T
    m_min = int(np.floor(mn[:, 0].min())) - 2
    m_max = int(np.ceil(mn[:, 0].max())) + 2
    n_min = int(np.floor(mn[:, 1].min())) - 2
    n_max = int(np.ceil(mn[:, 1].max())) + 2

    # 对齐到 step=2 子晶格相位（ref=0）
    m_min_a, m_max_a = align_range_to_step(m_min, m_max, GRID_STEP, ref=0)
    n_min_a, n_max_a = align_range_to_step(n_min, n_max, GRID_STEP, ref=0)

    # 生成 im_pos
    im_pos = make_im_pos_for_snap(roi_u8)
    im_vis = im_pos.copy()
    im_vis = (im_vis / (im_vis.max() + 1e-8) * 255.0).astype(np.uint8) if im_vis.max() > 1e-6 else np.zeros_like(roi_u8)
    cv2.imwrite(os.path.join(out_dir, "im_pos_for_snap.png"), im_vis)

    # ========= 相位搜索（只保留 best）=========
    best = None
    for a in [0, 1]:
        for b in [0, 1]:
            p0_try = p0_init + float(a) * v1 + float(b) * v2
            p0_ref = refine_p0(im_pos, roi_u8, p0_try, v1, v2, m_min_a, m_max_a, n_min_a, n_max_a, step=GRID_STEP)
            stats, vis = eval_grid_fit(im_pos, roi_u8, p0_ref, v1, v2, m_min_a, m_max_a, n_min_a, n_max_a,
                                       step=GRID_STEP, outlier_pct=EVAL_OUTLIER_PCT, arrow_scale=EVAL_ARROW_SCALE)

            med = 1e9 if stats is None else stats["median"]
            if best is None or med < best["median"]:
                best = {"a": a, "b": b, "p0": p0_ref, "median": float(med), "stats": stats, "vis": vis}

    if best is None or best["stats"] is None:
        raise RuntimeError("Phase search failed: no valid stats. Check im_pos/snap params.")

    p0 = best["p0"]

    # 只保存 best 的误差向量图
    cv2.imwrite(os.path.join(out_dir, "grid_fit_error_vectors.png"), best["vis"])

    print("=== PHASE SEARCH DONE ===")
    print(f" best (a,b)=({best['a']},{best['b']})  median={best['median']:.3f} mean={best['stats']['mean']:.3f}")

    # overlay（step=2 + 对齐后的范围）
    overlay = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    count = 0
    for n in range(n_min_a, n_max_a + 1, GRID_STEP):
        for m in range(m_min_a, m_max_a + 1, GRID_STEP):
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
        img_path=img_path,
        grid_step=int(GRID_STEP)
    )

    pitch1 = float(np.linalg.norm(v1))
    pitch2 = float(np.linalg.norm(v2))
    angle_deg = float(math.degrees(math.atan2(v1[1], v1[0])))

    print("=== CALIB DONE ===")
    print("calib saved:", calib_path)
    print("GRID_STEP =", GRID_STEP)
    print("v1 =", v1, " |v1| =", pitch1, "px")
    print("v2 =", v2, " |v2| =", pitch2, "px")
    print("angle(v1) =", angle_deg, "deg")
    print("p0 =", p0)
    print("overlay pts:", count)

    return calib_path


# ============================================================
# Step B: 网格积分 + diff/ratio + 可视化
# ============================================================

def integrate_grid(im_u8, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step=2):
    rows = (n_max - n_min) // step + 1
    cols = (m_max - m_min) // step + 1
    w = make_weights(r_int)

    X = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=np.uint8)

    H2, W2 = im_u8.shape
    for n in range(n_min, n_max + 1, step):
        rr = (n - n_min) // step
        for m in range(m_min, m_max + 1, step):
            cc = (m - m_min) // step

            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            xi, yi = int(round(x)), int(round(y))

            x0 = xi - r_int;
            x1 = xi + r_int
            y0 = yi - r_int;
            y1 = yi + r_int
            if x0 < 0 or y0 < 0 or x1 >= W2 or y1 >= H2:
                continue

            patch = im_u8[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
            X[rr, cc] = float(np.sum(patch * w))
            valid[rr, cc] = 1

    return X, valid


def vis_save(X, valid_mask, out_png, scale=18, p_lo=5, p_hi=95):
    Xv = X.astype(np.float32).copy()
    vals = Xv[valid_mask > 0]
    if vals.size < 10:
        lo, hi = float(Xv.min()), float(Xv.max())
    else:
        lo, hi = np.percentile(vals, [p_lo, p_hi])

    Y = (Xv - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)
    Y[valid_mask == 0] = 0.0

    u8 = (Y * 255.0 + 0.5).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1] * scale, u8.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, big)


def run_integration(calib_path, meas_path, flat_path, dark_path, out_dir, out_prefix, r_int):
    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)

    roi_x0 = int(cal["roi_x0"]);
    roi_y0 = int(cal["roi_y0"])
    roi_w = int(cal["roi_w"]);
    roi_h = int(cal["roi_h"])
    step = int(cal.get("grid_step", GRID_STEP))

    meas = crop_roi_by_xywh(read_gray(meas_path), roi_x0, roi_y0, roi_w, roi_h)
    flat = crop_roi_by_xywh(read_gray(flat_path), roi_x0, roi_y0, roi_w, roi_h)
    dark = crop_roi_by_xywh(read_gray(dark_path), roi_x0, roi_y0, roi_w, roi_h)

    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_meas.png"), meas)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_flat.png"), flat)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_dark.png"), dark)

    V2 = np.stack([v1, v2], axis=1).astype(np.float32)
    V2inv = np.linalg.inv(V2).astype(np.float32)

    corners_inner = np.array([
        [r_int, r_int],
        [roi_w - 1 - r_int, r_int],
        [r_int, roi_h - 1 - r_int],
        [roi_w - 1 - r_int, roi_h - 1 - r_int],
    ], dtype=np.float32)

    mn_inner = (V2inv @ (corners_inner - p0).T).T
    m_min = int(np.ceil(mn_inner[:, 0].min()))
    m_max = int(np.floor(mn_inner[:, 0].max()))
    n_min = int(np.ceil(mn_inner[:, 1].min()))
    n_max = int(np.floor(mn_inner[:, 1].max()))

    # 对齐到 step=2 相位（ref=0）
    m_min, m_max = align_range_to_step(m_min, m_max, step, ref=0)
    n_min, n_max = align_range_to_step(n_min, n_max, step, ref=0)

    rows = (n_max - n_min) // step + 1
    cols = (m_max - m_min) // step + 1

    print("=== INTEGRATION ===")
    print("Using step =", step)
    print("Aligned m range:", (m_min, m_max), "n range:", (n_min, n_max))
    print("Grid size (rows, cols):", (rows, cols))

    X_meas, V_meas = integrate_grid(meas, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step=step)
    X_flat, V_flat = integrate_grid(flat, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step=step)
    X_dark, V_dark = integrate_grid(dark, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step=step)

    valid = (V_meas & V_flat & V_dark).astype(np.uint8)
    print("valid count:", int(valid.sum()), "/", valid.size)

    np.save(os.path.join(out_dir, f"{out_prefix}_meas.npy"), X_meas)
    np.save(os.path.join(out_dir, f"{out_prefix}_flat.npy"), X_flat)
    np.save(os.path.join(out_dir, f"{out_prefix}_dark.npy"), X_dark)
    np.save(os.path.join(out_dir, f"{out_prefix}_valid.npy"), valid)

    X_diff = (X_meas - X_flat)
    den = (X_flat - X_dark)
    X_ratio = (X_meas - X_dark) / np.maximum(den, 1e-3)

    np.save(os.path.join(out_dir, f"{out_prefix}_diff.npy"), X_diff)
    np.save(os.path.join(out_dir, f"{out_prefix}_ratio.npy"), X_ratio)

    vis_save(X_diff, valid, os.path.join(out_dir, f"{out_prefix}_diff_vis.png"))
    vis_save(X_ratio, valid, os.path.join(out_dir, f"{out_prefix}_ratio_vis.png"))

    return step


# ============================================================
# Step C: crop（可选）
# ============================================================

def crop_by_valid(X, valid):
    ys, xs = np.where(valid > 0)
    if ys.size < 5:
        raise RuntimeError("valid 太少，没法 crop。")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return X[y0:y1, x0:x1].copy(), valid[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


def run_crop(out_dir, out_prefix):
    Xdiff = np.load(os.path.join(out_dir, f"{out_prefix}_diff.npy")).astype(np.float32)
    Xratio = np.load(os.path.join(out_dir, f"{out_prefix}_ratio.npy")).astype(np.float32)
    valid = np.load(os.path.join(out_dir, f"{out_prefix}_valid.npy")).astype(np.uint8)

    diff_c, valid_c, bbox = crop_by_valid(Xdiff, valid)
    ratio_c, _, _ = crop_by_valid(Xratio, valid)

    np.save(os.path.join(out_dir, f"{out_prefix}_diff_crop.npy"), diff_c)
    np.save(os.path.join(out_dir, f"{out_prefix}_ratio_crop.npy"), ratio_c)
    np.save(os.path.join(out_dir, f"{out_prefix}_valid_crop.npy"), valid_c)

    vis_save(diff_c, valid_c, os.path.join(out_dir, f"{out_prefix}_diff_crop_vis.png"))
    vis_save(ratio_c, valid_c, os.path.join(out_dir, f"{out_prefix}_ratio_crop_vis.png"))

    print("=== CROP DONE ===")
    print("valid bbox (y0,y1,x0,x1) =", bbox)
    print("crop size =", diff_c.shape)


# ============================================================
# Step D: warp（可选）
# ============================================================

def warp_to_roi(calib_path, in_xlr_npy, out_dir, out_name, r_int):
    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_w = int(cal["roi_w"])
    roi_h = int(cal["roi_h"])
    step = int(cal.get("grid_step", GRID_STEP))

    XLR = np.load(in_xlr_npy).astype(np.float32)
    rows, cols = XLR.shape

    V2 = np.stack([v1, v2], axis=1).astype(np.float32)
    V2inv = np.linalg.inv(V2).astype(np.float32)

    corners_inner = np.array([
        [r_int, r_int],
        [roi_w - 1 - r_int, r_int],
        [r_int, roi_h - 1 - r_int],
        [roi_w - 1 - r_int, roi_h - 1 - r_int],
    ], dtype=np.float32)

    mn_inner = (V2inv @ (corners_inner - p0).T).T
    m_min = int(np.ceil(mn_inner[:, 0].min()))
    m_max = int(np.floor(mn_inner[:, 0].max()))
    n_min = int(np.ceil(mn_inner[:, 1].min()))
    n_max = int(np.floor(mn_inner[:, 1].max()))

    m_min, m_max = align_range_to_step(m_min, m_max, step, ref=0)
    n_min, n_max = align_range_to_step(n_min, n_max, step, ref=0)

    # 若 shape 不一致，做轻量容错（平移为主）
    rows_calc = (n_max - n_min) // step + 1
    cols_calc = (m_max - m_min) // step + 1
    if (rows, cols) != (rows_calc, cols_calc):
        m_center = (m_min + m_max) / 2.0
        n_center = (n_min + n_max) / 2.0
        m_min = int(round(m_center - (cols - 1) * step / 2.0))
        n_min = int(round(n_center - (rows - 1) * step / 2.0))
        m_min, m_max = align_range_to_step(m_min, m_min + (cols - 1) * step, step, ref=0)
        n_min, n_max = align_range_to_step(n_min, n_min + (rows - 1) * step, step, ref=0)

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
    dx = xx - float(p0[0])
    dy = yy - float(p0[1])

    m_cont = V2inv[0, 0] * dx + V2inv[0, 1] * dy
    n_cont = V2inv[1, 0] * dx + V2inv[1, 1] * dy

    map_x = (m_cont - float(m_min)) / float(step)
    map_y = (n_cont - float(n_min)) / float(step)

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
    lo, hi = (np.percentile(vals, [2, 98]) if vals.size > 10 else (float(S_roi.min()), float(S_roi.max())))
    Vis = np.clip((S_roi - lo) / (hi - lo + 1e-8), 0, 1)
    cv2.imwrite(out_png, (Vis * 255.0 + 0.5).astype(np.uint8))

    print("=== WARP DONE ===")
    print("wrote:", out_npy, out_png)


# ============================================================
# main
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    calib_path = calibrate_from_image(IMG_CALIB_PATH, ROI_H, ROI_W, OUT_DIR)

    _ = run_integration(calib_path, MEAS_PATH, FLAT_PATH, DARK_PATH, OUT_DIR, OUT_PREFIX, R_INT)

    if DO_CROP:
        run_crop(OUT_DIR, OUT_PREFIX)

    if DO_WARP:
        in_diff = os.path.join(OUT_DIR, f"{OUT_PREFIX}_diff.npy")
        in_ratio = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ratio.npy")
        warp_to_roi(calib_path, in_diff, OUT_DIR, "roiwarp_diff_S_roi", R_INT)
        warp_to_roi(calib_path, in_ratio, OUT_DIR, "roiwarp_ratio_S_roi", R_INT)

    print("\nALL DONE. Output dir =", OUT_DIR)


if __name__ == "__main__":
    main()
