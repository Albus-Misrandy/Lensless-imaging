import numpy as np
import cv2

# 这些文件是 step_1 生成的
DIFF_NPY  = "./xlr_diff.npy"
RATIO_NPY = "./xlr_ratio.npy"
VALID_NPY = "./xlr_valid.npy"

OUT_PREFIX = "xlr"   # 输出前缀
SCALE = 18           # 放大倍数（看得更清楚）
P_LO, P_HI = 5, 95   # 百分位拉伸

Xdiff  = np.load(DIFF_NPY).astype(np.float32)
Xratio = np.load(RATIO_NPY).astype(np.float32)
valid  = np.load(VALID_NPY).astype(np.uint8)

if valid.sum() < 10:
    raise RuntimeError("valid 太少，没法 crop。先检查 step_1 的网格/积分是否成功。")

ys, xs = np.where(valid > 0)
y0, y1 = int(ys.min()), int(ys.max())
x0, x1 = int(xs.min()), int(xs.max())

# bbox 要包含边界，所以 +1
y1 += 1
x1 += 1

print("valid bbox (y0,y1,x0,x1) =", (y0, y1, x0, x1))
print("crop size =", (y1 - y0, x1 - x0))

diff_c  = Xdiff[y0:y1, x0:x1]
ratio_c = Xratio[y0:y1, x0:x1]
valid_c = valid[y0:y1, x0:x1]

np.save(f"{OUT_PREFIX}_diff_crop.npy", diff_c)
np.save(f"{OUT_PREFIX}_ratio_crop.npy", ratio_c)
np.save(f"{OUT_PREFIX}_valid_crop.npy", valid_c)
print("[DONE] wrote crop npy")

def vis_save(X, valid_mask, out_png, scale=SCALE, p_lo=P_LO, p_hi=P_HI):
    Xv = X.astype(np.float32).copy()
    vals = Xv[valid_mask > 0]
    if vals.size < 10:
        lo, hi = float(Xv.min()), float(Xv.max())
    else:
        lo, hi = np.percentile(vals, [p_lo, p_hi])

    Y = (Xv - lo) / (hi - lo + 1e-8)
    Y = np.clip(Y, 0, 1)

    # 无效点黑底
    Y[valid_mask == 0] = 0.0

    u8 = (Y * 255.0 + 0.5).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1]*scale, u8.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_png, big)

vis_save(diff_c,  valid_c, f"{OUT_PREFIX}_diff_crop_vis.png")
vis_save(ratio_c, valid_c, f"{OUT_PREFIX}_ratio_crop_vis.png")
print("[DONE] wrote crop vis pngs:")
print(" ", f"{OUT_PREFIX}_diff_crop_vis.png")
print(" ", f"{OUT_PREFIX}_ratio_crop_vis.png")
