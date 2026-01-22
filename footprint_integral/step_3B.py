import numpy as np
import cv2

IN_NPY    = "xlr_diff.npy"      # or xlr_ratio.npy
VALID_NPY = "xlr_valid.npy"

OUT_PREFIX = "combo2"

# IDW
IDW_POWER = 2.0
IDW_K = 24

# Upscale
UPSCALE = 12

# Strong TV (让效果明显)
TV_ITERS  = 600     # <<< 大幅增加
TV_LAMBDA = 0.30    # <<< 大幅增加（0.25~0.45）
TAU       = 0.15    # <<< 步长稍小，防发散

# Optional: unsharp
DO_UNSHARP = True
UNSHARP_SIGMA = 1.2
UNSHARP_AMOUNT = 0.6

X = np.load(IN_NPY).astype(np.float32)
V = np.load(VALID_NPY).astype(np.uint8)

H, W = X.shape
ys, xs = np.where(V > 0)
vals = X[ys, xs].astype(np.float32)
if len(vals) < 10:
    raise RuntimeError("valid 太少。")

# 1) IDW fill
fill = X.copy()
for r in range(H):
    for c in range(W):
        if V[r, c] > 0:
            continue
        d2 = (ys - r)**2 + (xs - c)**2
        if len(d2) > IDW_K:
            idx = np.argpartition(d2, IDW_K)[:IDW_K]
            dd2 = d2[idx].astype(np.float32)
            vv = vals[idx]
        else:
            dd2 = d2.astype(np.float32)
            vv = vals
        if dd2.min() < 1e-6:
            fill[r, c] = float(vv[int(np.argmin(dd2))])
        else:
            w = 1.0 / (dd2 ** (IDW_POWER/2.0) + 1e-6)
            fill[r, c] = float(np.sum(w * vv) / (np.sum(w) + 1e-8))

# 2) bicubic upsample
Hs, Ws = H * UPSCALE, W * UPSCALE
ref = cv2.resize(fill, (Ws, Hs), interpolation=cv2.INTER_CUBIC).astype(np.float32)

# 3) TV denoise (strong)
def tv_denoise(ref, lam=0.30, n_iter=600, tau=0.15):
    u = ref.copy().astype(np.float32)
    for it in range(n_iter):
        ux = np.zeros_like(u); uy = np.zeros_like(u)
        ux[:, :-1] = u[:, 1:] - u[:, :-1]
        uy[:-1, :] = u[1:, :] - u[:-1, :]

        mag = np.sqrt(ux*ux + uy*uy + 1e-6)
        px = ux / mag
        py = uy / mag

        div = np.zeros_like(u)
        div[:, :-1] += px[:, :-1]
        div[:, 1:]  -= px[:, :-1]
        div[:-1, :] += py[:-1, :]
        div[1:, :]  -= py[:-1, :]

        # u += tau * ((ref - u) + lam * div)
        u += tau * (ref - u + lam * div)

        # 可选：每 50 次输出一下收敛情况
        # if it % 50 == 0:
        #     print("it", it, "mean|ref-u|", float(np.mean(np.abs(ref-u))))
    return u

tv = tv_denoise(ref, lam=TV_LAMBDA, n_iter=TV_ITERS, tau=TAU)

# 4) optional unsharp for edges
out = tv
if DO_UNSHARP:
    blur = cv2.GaussianBlur(out, (0,0), UNSHARP_SIGMA)
    out = out + UNSHARP_AMOUNT * (out - blur)

np.save(f"{OUT_PREFIX}_ref.npy", ref.astype(np.float32))
np.save(f"{OUT_PREFIX}_tv.npy", tv.astype(np.float32))
np.save(f"{OUT_PREFIX}_out.npy", out.astype(np.float32))

def save_vis(arr, out_png, p_lo=2, p_hi=98):
    lo, hi = np.percentile(arr.reshape(-1), [p_lo, p_hi])
    Z = (arr - lo) / (hi - lo + 1e-8)
    Z = np.clip(Z, 0, 1)
    u8 = (Z * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(out_png, u8)

save_vis(ref, f"{OUT_PREFIX}_bicubic.png")
save_vis(tv,  f"{OUT_PREFIX}_tv.png")
save_vis(out, f"{OUT_PREFIX}_tv_unsharp.png")

print("[DONE] wrote:",
      f"{OUT_PREFIX}_bicubic.png",
      f"{OUT_PREFIX}_tv.png",
      f"{OUT_PREFIX}_tv_unsharp.png")
