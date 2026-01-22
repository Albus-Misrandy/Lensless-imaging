import numpy as np
import cv2

IN_NPY = "sr2_hp.npy"
OUT_PREFIX = "sr5"

# ====== 用你 step0 打印的 k1/k2（cycles/pixel）======
k1 = np.array([-0.028125, -0.02619048], dtype=np.float32)  # [fx, fy]
k2 = np.array([-0.028125,  0.02619048], dtype=np.float32)

# 抑制阶数（谐波数）
HARM = 8

# 带状 notch 参数：
# SIG_PERP：对“垂直于方向”的宽度（像素，越大压纹教狠）
# SIG_PAR ：沿方向的长度控制（像素，越大越像一段“线”而不是点）
SIG_PERP = 2.5     # 推荐 2.0~4.0
SIG_PAR  = 10.0    # 推荐 8~20
DEPTH = 0.85       # 0~1，越大抑制越深

# 可选：轻度 TV（建议很轻）
DO_TV = True
TV_WEIGHT = 0.05
TV_ITERS  = 80

def save_vis(arr, out_png, p_lo=2, p_hi=98):
    lo, hi = np.percentile(arr.reshape(-1), [p_lo, p_hi])
    Z = (arr - lo) / (hi - lo + 1e-8)
    Z = np.clip(Z, 0, 1)
    cv2.imwrite(out_png, (Z*255.0+0.5).astype(np.uint8))

def tv_chambolle(u, weight=0.05, n_iter=80, eps=1e-6):
    u = u.astype(np.float32)
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    tau = 0.25
    for _ in range(n_iter):
        div = np.zeros_like(u)
        div[:, :-1] += px[:, :-1]
        div[:, 1:]  -= px[:, :-1]
        div[:-1, :] += py[:-1, :]
        div[1:, :]  -= py[:-1, :]

        x = u - weight * div

        gx = np.zeros_like(u); gy = np.zeros_like(u)
        gx[:, :-1] = x[:, 1:] - x[:, :-1]
        gy[:-1, :] = x[1:, :] - x[:-1, :]

        px_new = px + (tau/weight) * gx
        py_new = py + (tau/weight) * gy
        norm = np.maximum(1.0, np.sqrt(px_new*px_new + py_new*py_new) + eps)
        px = px_new / norm
        py = py_new / norm

    div = np.zeros_like(u)
    div[:, :-1] += px[:, :-1]
    div[:, 1:]  -= px[:, :-1]
    div[:-1, :] += py[:-1, :]
    div[1:, :]  -= py[:-1, :]

    return u - weight * div

Yhp = np.load(IN_NPY).astype(np.float32)
H, W = Yhp.shape
cy, cx = H//2, W//2

save_vis(Yhp, f"{OUT_PREFIX}_hp.png")

F = np.fft.fftshift(np.fft.fft2(Yhp))

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def k_to_pix(k):
    # k=[fx,fy] cycles/pixel -> FFT-shifted pixel coord
    px = cx + k[0] * W
    py = cy + k[1] * H
    return np.array([px, py], dtype=np.float32)

def band_notch_mask(center_px_py, dir_vec, sig_perp, sig_par, depth):
    """
    在频域做一个“沿 dir_vec 的椭圆高斯带状抑制”
    center: (px,py)
    dir_vec: 2D unit vector in pixel coords
    """
    cx0, cy0 = float(center_px_py[0]), float(center_px_py[1])

    dx = xx - cx0
    dy = yy - cy0

    # 平行/垂直分量
    par  = dx * dir_vec[0] + dy * dir_vec[1]
    perp = -dx * dir_vec[1] + dy * dir_vec[0]

    g = np.exp(-0.5*((perp/sig_perp)**2 + (par/sig_par)**2))
    return (1.0 - depth * g).astype(np.float32)

mask = np.ones((H, W), dtype=np.float32)

def add_family(base_k):
    # 方向向量：在频域像素坐标里，沿 base_k 的方向
    # 注意：k->pixel 的缩放不同（W/H），所以要在 pixel 坐标里求方向
    p1 = k_to_pix(base_k)
    p0 = np.array([cx, cy], dtype=np.float32)
    v = p1 - p0
    nv = float(np.linalg.norm(v) + 1e-8)
    v = v / nv  # unit direction (pixel coords)

    for h in range(1, HARM+1):
        kk = h * base_k
        for sgn in (+1, -1):
            c = k_to_pix(sgn * kk)
            # 高阶可稍微加宽一点点
            sp = SIG_PERP * np.sqrt(h)
            # 长度也稍加
            sa = SIG_PAR  * np.sqrt(h)
            mask[:] *= band_notch_mask(c, v, sp, sa, DEPTH)

add_family(k1)
add_family(k2)

F2 = F * mask
y_bn = np.real(np.fft.ifft2(np.fft.ifftshift(F2))).astype(np.float32)

np.save(f"{OUT_PREFIX}_bandnotch.npy", y_bn)
save_vis(y_bn, f"{OUT_PREFIX}_bandnotch.png")

if DO_TV:
    y_tv = tv_chambolle(y_bn, weight=TV_WEIGHT, n_iter=TV_ITERS)
    np.save(f"{OUT_PREFIX}_bandnotch_tv.npy", y_tv)
    save_vis(y_tv, f"{OUT_PREFIX}_bandnotch_tv.png")
    print("[DONE] wrote", f"{OUT_PREFIX}_bandnotch.png and {OUT_PREFIX}_bandnotch_tv.png")
else:
    print("[DONE] wrote", f"{OUT_PREFIX}_bandnotch.png")
