import os
import math
import cv2
import numpy as np

# ============================================================
# 配置区：你主要改这里
# ============================================================

IMG_CALIB_PATH = "../snapshots/snapshot_20260115_160108.jpg"  # 标定用：孔很清晰、最好无按压
MEAS_PATH = "../snapshots/snapshot_20260116_162823.jpg"  # 有按压
FLAT_PATH = "../snapshots/flat.jpg"  # 无按压平场
DARK_PATH = "../snapshots/dark.jpg"  # 全黑

OUT_DIR = "./out_allintwo"
OUT_PREFIX = "xlr"

ROI_H, ROI_W = 420, 640

# 积分半径（像素）
R_INT = 8

DO_CROP = True
DO_WARP = True

# ---------- im_pos（给中心检测/误差评估用） ----------
BG_SIGMA = 18.0
SMOOTH_SIGMA = 2.5
CLAMP_POS = True

# ---------- 亮斑中心检测（全图） ----------
# 你现在 calib 图很干净：默认就够用。真要调，优先调 THR_FRAC
THR_FRAC = 0.18  # 阈值 = THR_FRAC * max(im_pos)
MIN_AREA = 12  # 连通域面积下限
MAX_AREA = 5000  # 连通域面积上限（防止把一大片当一个孔）
MAX_PTS_KEEP = 4000  # 检测点最多保留（太多就抽样）

# ---------- FFT 找 v1/v2 的辅助（给晶格拟合做初值） ----------
PEAK_TOPK = 80
PEAK_TRY_TOPN = 40
FFT_RMIN_FRAC = 0.04
FFT_RMAX_FRAC = 0.45

# ---------- 晶格拟合（鲁棒迭代） ----------
LATTICE_ITERS = 6
OUTLIER_PCT = 95  # 去掉误差最大的 5%
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
    """去背景 + 平滑 + 截断正值：适合做“孔亮斑”响应图。"""
    roi_f = roi_u8.astype(np.float32)
    bg = cv2.GaussianBlur(roi_f, (0, 0), BG_SIGMA)
    hp = roi_f - bg
    hp2 = cv2.GaussianBlur(hp, (0, 0), SMOOTH_SIGMA)
    if CLAMP_POS:
        im_pos = np.maximum(hp2, 0)
    else:
        im_pos = hp2
    return im_pos


def normalize_u8(im):
    im = im.astype(np.float32)
    mx = float(im.max())
    if mx <= 1e-6:
        return np.zeros_like(im, dtype=np.uint8)
    out = np.clip(im / (mx + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return out


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
    Kmat = np.stack([k1, k2], axis=0).astype(np.float32)  # 2x2
    V = np.linalg.inv(Kmat)
    v1 = V[:, 0].astype(np.float32)
    v2 = V[:, 1].astype(np.float32)

    # 规范化方向：v1 x>0 + 右手系
    if v1[0] < 0:
        v1 = -v1
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        v2 = -v2

    # 保存 fft_peaks.png
    spec_vis = (mag * 255).astype(np.uint8)
    spec_vis = cv2.cvtColor(spec_vis, cv2.COLOR_GRAY2BGR)
    cv2.circle(spec_vis, (int(p1[1]), int(p1[0])), 7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (int(p2[1]), int(p2[0])), 7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(spec_vis, (cx, cy), 4, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "fft_peaks.png"), spec_vis)

    return v1, v2


# ============================================================
# 2) 全图检测孔中心点（calib 图很干净时最合适）
# ============================================================

def detect_blob_centers(im_pos, out_dir):
    im_u8 = normalize_u8(im_pos)
    mx = int(im_u8.max())
    thr = max(1, int(THR_FRAC * mx))

    _, bw = cv2.threshold(im_u8, thr, 255, cv2.THRESH_BINARY)

    # 轻微形态学：填小洞、去小毛刺（别太狠）
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

    # 太多就抽样
    if pts.shape[0] > MAX_PTS_KEEP:
        idx = np.linspace(0, pts.shape[0] - 1, MAX_PTS_KEEP).astype(int)
        pts = pts[idx]

    # 输出一个可视化（只用于 sanity check）
    vis = cv2.cvtColor(im_u8, cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "blob_centers.png"), vis)

    return pts


# ============================================================
# 3) 晶格最小二乘拟合（带迭代重分配整数索引 + 去离群）
# ============================================================

def lattice_fit_ls(points_xy, v1_init, v2_init, p0_init):
    """
    模型：p_i ≈ p0 + m_i v1 + n_i v2
    给定初值 v1/v2/p0，迭代：
      1) 用 Vinv 把点投影到(m,n)，round 得整数
      2) 最小二乘解 p0,v1,v2
      3) 计算残差，按 OUTLIER_PCT 去离群
    """
    p0 = p0_init.astype(np.float32).copy()
    v1 = v1_init.astype(np.float32).copy()
    v2 = v2_init.astype(np.float32).copy()

    pts = points_xy.astype(np.float32)

    for _ in range(int(LATTICE_ITERS)):
        V = np.stack([v1, v2], axis=1).astype(np.float32)  # 2x2
        Vinv = np.linalg.inv(V).astype(np.float32)

        mn_cont = (Vinv @ (pts - p0).T).T  # Nx2
        mn_int = np.round(mn_cont).astype(np.int32)
        m = mn_int[:, 0].astype(np.float32)
        n = mn_int[:, 1].astype(np.float32)

        # 线性最小二乘：Ax=b, unknown=[p0x,p0y,v1x,v1y,v2x,v2y]
        N = pts.shape[0]
        A = np.zeros((2 * N, 6), dtype=np.float32)
        b = np.zeros((2 * N,), dtype=np.float32)

        # x 方程：x = p0x + m*v1x + n*v2x
        A[0::2, 0] = 1.0
        A[0::2, 2] = m
        A[0::2, 4] = n
        b[0::2] = pts[:, 0]

        # y 方程：y = p0y + m*v1y + n*v2y
        A[1::2, 1] = 1.0
        A[1::2, 3] = m
        A[1::2, 5] = n
        b[1::2] = pts[:, 1]

        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        p0 = np.array([x[0], x[1]], dtype=np.float32)
        v1 = np.array([x[2], x[3]], dtype=np.float32)
        v2 = np.array([x[4], x[5]], dtype=np.float32)

        # 规范化方向
        if v1[0] < 0:
            v1 = -v1
        if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
            v2 = -v2

        # 计算残差并去离群
        pred = p0[None, :] + m[:, None] * v1[None, :] + n[:, None] * v2[None, :]
        err = np.linalg.norm(pts - pred, axis=1)

        thr = np.percentile(err, OUTLIER_PCT)
        keep = err <= thr
        pts = pts[keep]

        # 点太少就停
        if pts.shape[0] < 50:
            break

    # 最后再用全部保留点重新估计一次（稳定一下）
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
# 4) 评估 grid fit，并自动选 step=1 或 step=2（只保留最优）
# ============================================================

def eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step, max_pts=3000):
    """
    用“预测格点 -> 在im_pos附近找局部质心”来评估残差。
    注意：这里只用于评估 step=1/2 哪个更像你看到的孔亮斑。
    """
    H, W = roi_u8.shape
    V = np.stack([v1, v2], axis=1).astype(np.float32)
    Vinv = np.linalg.inv(V).astype(np.float32)

    # 先根据 ROI 四角推一个足够覆盖的 m,n 范围
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float32)
    mn = (Vinv @ (corners - p0).T).T
    m_min = int(np.floor(mn[:, 0].min())) - 2
    m_max = int(np.ceil(mn[:, 0].max())) + 2
    n_min = int(np.floor(mn[:, 1].min())) - 2
    n_max = int(np.ceil(mn[:, 1].max())) + 2

    # 按 step 取子晶格
    cand = []
    for n in range(n_min, n_max + 1, step):
        for m in range(m_min, m_max + 1, step):
            p = p0 + m * v1 + n * v2
            x, y = float(p[0]), float(p[1])
            if 0 <= x < W and 0 <= y < H:
                cand.append((x, y))

    if len(cand) == 0:
        return None, None, None

    if len(cand) > max_pts:
        idx = np.linspace(0, len(cand) - 1, max_pts).astype(int)
        cand = [cand[i] for i in idx]

    def snap_to_centroid_local(x, y, r=9, thr_rel=0.25):
        xi, yi = int(round(x)), int(round(y))
        x0 = max(0, xi - r);
        x1 = min(W - 1, xi + r)
        y0 = max(0, yi - r);
        y1 = min(H - 1, yi + r)
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

    shifts = []
    pts_pred = []
    pts_snap = []

    for (x, y) in cand:
        xs, ys, conf = snap_to_centroid_local(x, y, r=9, thr_rel=0.25)
        if conf > 1e-3:
            shifts.append([xs - x, ys - y])
            pts_pred.append([x, y])
            pts_snap.append([xs, ys])

    if len(shifts) < 20:
        return None, None, None

    shifts = np.array(shifts, dtype=np.float32)
    pts_pred = np.array(pts_pred, dtype=np.float32)
    pts_snap = np.array(pts_snap, dtype=np.float32)
    mag = np.linalg.norm(shifts, axis=1)

    thr = np.percentile(mag, OUTLIER_PCT)
    keep = mag <= thr
    mag_k = mag[keep]
    pts_pred_k = pts_pred[keep]
    pts_snap_k = pts_snap[keep]

    stats = {
        "count_raw": int(mag.shape[0]),
        "count_keep": int(mag_k.shape[0]),
        "mean": float(np.mean(mag_k)),
        "median": float(np.median(mag_k)),
        "rms": float(np.sqrt(np.mean(mag_k ** 2))),
        "p90": float(np.percentile(mag_k, 90)),
        "p95": float(np.percentile(mag_k, 95)),
        "outlier_thr": float(thr),
    }

    vis = cv2.cvtColor(roi_u8, cv2.COLOR_GRAY2BGR)
    for p, s, e in zip(pts_pred_k, pts_snap_k, mag_k):
        x0, y0 = int(round(p[0])), int(round(p[1]))
        x1 = int(round(p[0] + (s[0] - p[0]) * EVAL_ARROW_SCALE))
        y1 = int(round(p[1] + (s[1] - p[1]) * EVAL_ARROW_SCALE))
        color = (0, 255, 0) if e < 0.8 else ((0, 255, 255) if e < 1.6 else (0, 0, 255))
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), color, 1, tipLength=0.25)

    txt = (f"step={step}  med={stats['median']:.3f} mean={stats['mean']:.3f} "
           f"p90={stats['p90']:.3f} p95={stats['p95']:.3f} kept={stats['count_keep']}/{stats['count_raw']}")
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return stats, vis, (m_min, m_max, n_min, n_max)


def align_range_to_step(mn_min, mn_max, step, ref=0):
    mn_min_a = mn_min + ((ref - mn_min) % step)
    mn_max_a = mn_max - ((mn_max - ref) % step)
    if mn_max_a < mn_min_a:
        raise RuntimeError("Aligned range became empty; check ROI/step.")
    return int(mn_min_a), int(mn_max_a)


# ============================================================
# 5) 标定主流程：检测中心 + 最小二乘拟合 + 自动 step
# ============================================================

def calibrate_from_image(img_path, roi_h, roi_w, out_dir):
    img = read_gray(img_path)
    roi_u8, roi_x0, roi_y0 = center_crop_roi(img, roi_h, roi_w)
    cv2.imwrite(os.path.join(out_dir, "roi.png"), roi_u8)

    # 生成 im_pos
    im_pos = make_im_pos_for_snap(roi_u8)
    cv2.imwrite(os.path.join(out_dir, "im_pos_for_snap.png"), normalize_u8(im_pos))

    # FFT 给 v1/v2 初值
    v1_init, v2_init = fft_find_v1v2(roi_u8, out_dir)

    # 全图检测孔中心
    pts = detect_blob_centers(im_pos, out_dir)

    # 选一个 p0 初值：离 ROI 中心最近的检测点
    H, W = roi_u8.shape
    center = np.array([W * 0.5, H * 0.5], dtype=np.float32)
    d = np.linalg.norm(pts - center[None, :], axis=1)
    p0_init = pts[int(np.argmin(d))].astype(np.float32)

    # 晶格最小二乘拟合
    p0, v1, v2 = lattice_fit_ls(pts, v1_init, v2_init, p0_init)

    # 自动选 step=1 或 step=2（只保留更好的那个）
    stats1, vis1, rng1 = eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step=1)
    stats2, vis2, rng2 = eval_grid_fit_using_snap(im_pos, roi_u8, p0, v1, v2, step=2)

    # 默认优先 step=1（因为分辨率更高），除非 step=2 明显更好
    best_step = 1
    best_stats = stats1
    best_vis = vis1
    best_rng = rng1

    if stats1 is None and stats2 is not None:
        best_step = 2;
        best_stats = stats2;
        best_vis = vis2;
        best_rng = rng2
    elif stats1 is not None and stats2 is not None:
        # 如果 step2 的 median 比 step1 小很多（比如 0.7 倍），就选 step2
        if float(stats2["median"]) < 0.7 * float(stats1["median"]):
            best_step = 2;
            best_stats = stats2;
            best_vis = vis2;
            best_rng = rng2

    if best_vis is not None:
        cv2.imwrite(os.path.join(out_dir, "grid_fit_error_vectors.png"), best_vis)

    # overlay（用 best_step + 对齐相位 ref=0）
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
        grid_step=int(best_step),
        # 也把对齐后的范围保存，后续积分/warp直接用（更一致）
        m_min=int(m_min), m_max=int(m_max),
        n_min=int(n_min), n_max=int(n_max),
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
# 6) 积分 + diff/ratio + 可视化（使用保存好的 m/n 范围 + step）
# ============================================================

def integrate_grid(im_u8, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step):
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
    step = int(cal["grid_step"])
    m_min = int(cal["m_min"]);
    m_max = int(cal["m_max"])
    n_min = int(cal["n_min"]);
    n_max = int(cal["n_max"])

    meas = crop_roi_by_xywh(read_gray(meas_path), roi_x0, roi_y0, roi_w, roi_h)
    flat = crop_roi_by_xywh(read_gray(flat_path), roi_x0, roi_y0, roi_w, roi_h)
    dark = crop_roi_by_xywh(read_gray(dark_path), roi_x0, roi_y0, roi_w, roi_h)

    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_meas.png"), meas)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_flat.png"), flat)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_roi_dark.png"), dark)

    rows = (n_max - n_min) // step + 1
    cols = (m_max - m_min) // step + 1

    print("=== INTEGRATION ===")
    print("Using step =", step)
    print("m range:", (m_min, m_max), "n range:", (n_min, n_max))
    print("Grid size (rows, cols):", (rows, cols))

    X_meas, V_meas = integrate_grid(meas, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step)
    X_flat, V_flat = integrate_grid(flat, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step)
    X_dark, V_dark = integrate_grid(dark, p0, v1, v2, m_min, m_max, n_min, n_max, r_int, step)

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

    vis_save(X_meas, valid, os.path.join(out_dir, f"{out_prefix}_meas_vis.png"))
    vis_save(X_flat, valid, os.path.join(out_dir, f"{out_prefix}_flat_vis.png"))
    vis_save(X_dark, valid, os.path.join(out_dir, f"{out_prefix}_dark_vis.png"))
    vis_save(X_diff, valid, os.path.join(out_dir, f"{out_prefix}_diff_vis.png"))
    vis_save(X_ratio, valid, os.path.join(out_dir, f"{out_prefix}_ratio_vis.png"))

    return step


# ============================================================
# 7) crop（可选）
# ============================================================

def crop_by_valid(X, valid):
    ys, xs = np.where(valid > 0)
    if ys.size < 5:
        raise RuntimeError("valid 太少，没法 crop。")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (X[y0:y1, x0:x1].copy(), valid[y0:y1, x0:x1].copy(), (y0, y1, x0, x1))


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
# 8) warp（可选）
# ============================================================

def warp_to_roi(calib_path, in_xlr_npy, out_dir, out_name):
    cal = np.load(calib_path)
    v1 = cal["v1"].astype(np.float32)
    v2 = cal["v2"].astype(np.float32)
    p0 = cal["p0"].astype(np.float32)
    roi_w = int(cal["roi_w"])
    roi_h = int(cal["roi_h"])
    step = int(cal["grid_step"])
    m_min = int(cal["m_min"])
    n_min = int(cal["n_min"])

    XLR = np.load(in_xlr_npy).astype(np.float32)

    V = np.stack([v1, v2], axis=1).astype(np.float32)
    Vinv = np.linalg.inv(V).astype(np.float32)

    yy, xx = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
    dx = xx - float(p0[0])
    dy = yy - float(p0[1])

    m_cont = Vinv[0, 0] * dx + Vinv[0, 1] * dy
    n_cont = Vinv[1, 0] * dx + Vinv[1, 1] * dy

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
        warp_to_roi(calib_path, in_diff, OUT_DIR, "roiwarp_diff_S_roi")
        warp_to_roi(calib_path, in_ratio, OUT_DIR, "roiwarp_ratio_S_roi")

    print("\nALL DONE. Output dir =", OUT_DIR)


if __name__ == "__main__":
    main()
