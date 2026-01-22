import numpy as np
import cv2

# 选择输入：diff 或 ratio（建议先 diff）
IN_NPY    = "./xlr_diff.npy"
VALID_NPY = "./xlr_valid.npy"

OUT_PREFIX = "fill"   # 输出前缀
UPSCALE = 12          # 放大倍数

# IDW 参数
IDW_POWER = 2.0       # 2~3 常用；越大越“贴近邻居”
IDW_K = 24            # 每个空点只用最近K个有效点（越大越慢）

X = np.load(IN_NPY).astype(np.float32)
V = np.load(VALID_NPY).astype(np.uint8)

H, W = X.shape
ys, xs = np.where(V > 0)

if len(xs) < 10:
    raise RuntimeError("valid 太少，无法做补全。")

pts = np.stack([ys, xs], axis=1).astype(np.int32)      # (N,2) in (r,c)
vals = X[ys, xs].astype(np.float32)                    # (N,)

print("X shape:", X.shape, "valid:", int(V.sum()))

# ------------------------
# 可视化工具
# ------------------------
def vis_save(Xarr, out_png, p_lo=5, p_hi=95):
    vals_all = Xarr.reshape(-1)
    lo, hi = np.percentile(vals_all, [p_lo, p_hi])
    Y = (Xarr - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)
    u8 = (Y * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(out_png, u8)

def upsave(Xarr, out_png, scale=UPSCALE):
    u8 = Xarr.astype(np.float32)
    # 先拉伸到 0-255
    lo, hi = np.percentile(u8.reshape(-1), [5,95])
    Y = (u8 - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)
    img = (Y * 255.0 + 0.5).astype(np.uint8)
    big = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_png, big)

# ------------------------
# 1) 最近邻补全 (NN fill)
# ------------------------
# 思路：对每个像素，找离它最近的有效点，用该值填充
# 用 brute force 会慢，但 H,W 都不大（28/35）可以直接做
fill_nn = X.copy()
for r in range(H):
    for c in range(W):
        if V[r, c] > 0:
            continue
        d2 = (ys - r)**2 + (xs - c)**2
        j = int(np.argmin(d2))
        fill_nn[r, c] = vals[j]

np.save(f"{OUT_PREFIX}_nn.npy", fill_nn)
vis_save(fill_nn, f"{OUT_PREFIX}_nn_vis.png")
upsave(fill_nn, f"{OUT_PREFIX}_nn_up.png")
print("[DONE] NN fill")

# ------------------------
# 2) IDW 补全 (Inverse Distance Weighting)
# ------------------------
fill_idw = X.copy()
for r in range(H):
    for c in range(W):
        if V[r, c] > 0:
            continue
        d2 = (ys - r)**2 + (xs - c)**2
        # 取最近 K 个有效点
        if len(d2) > IDW_K:
            idx = np.argpartition(d2, IDW_K)[:IDW_K]
            dd2 = d2[idx].astype(np.float32)
            vv = vals[idx]
        else:
            dd2 = d2.astype(np.float32)
            vv = vals

        # 如果刚好落在有效点上（理论不会），避免除零
        if dd2.min() < 1e-6:
            fill_idw[r, c] = float(vv[int(np.argmin(dd2))])
            continue

        w = 1.0 / (dd2 ** (IDW_POWER/2.0) + 1e-6)  # d = sqrt(d2) => d^p = (d2)^(p/2)
        fill_idw[r, c] = float(np.sum(w * vv) / (np.sum(w) + 1e-8))

np.save(f"{OUT_PREFIX}_idw.npy", fill_idw)
vis_save(fill_idw, f"{OUT_PREFIX}_idw_vis.png")
upsave(fill_idw, f"{OUT_PREFIX}_idw_up.png")
print("[DONE] IDW fill")

print("[DONE] wrote:")
print(" ", f"{OUT_PREFIX}_nn_vis.png", f"{OUT_PREFIX}_nn_up.png")
print(" ", f"{OUT_PREFIX}_idw_vis.png", f"{OUT_PREFIX}_idw_up.png")
