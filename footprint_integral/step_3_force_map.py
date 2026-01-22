import numpy as np
import cv2

# ========= 你改这里 =========
IN_ROI_NPY = "roiwarp_S_roi.npy"   # 上一步生成的 npy，里面是 ratio 场
OUT_GRAY   = "force_gray.png"     # 灰度力场图
OUT_COLOR  = "force_color.png"    # 伪彩色力场图
R_IS_RATIO = True                 # True 表示 S 里存的是 ratio (≈1 无变化)
HP_SIGMA   = 15.0                 # 高斯高通的尺度，10~30 之间都可以试试
P_LO, P_HI = 5, 95                # 百分位拉伸
# ===========================

# 1) 读入“摆正后的场”
S = np.load(IN_ROI_NPY).astype(np.float32)
H, W = S.shape
print("S_roi shape:", S.shape)

# 做一个简单的有效区域 mask（边缘很多是 0 或极小值）
mask = S > 1e-6

# 2) 把 ratio 变成“变化量”
if R_IS_RATIO:
    # ratio=1 表示“跟 flat 一样”，ratio<1 / >1 表示变化
    # 这里先假设按压会让局部变暗 => ratio 变小 => 1 - ratio 变大
    F = 1.0 - S
else:
    # 如果你以后改成直接用 diff 场，这里就 F = S
    F = S.copy()

# 3) 去掉大尺度背景：Gaussian 高通
#    先做一个大 sigma 的平滑，再相减
blur = cv2.GaussianBlur(F, (0, 0), HP_SIGMA)
F_hp = F - blur

# 无效区域直接置 0
F_hp[~mask] = 0.0

# 4) 百分位拉伸到 0~1
vals = F_hp[mask]
if vals.size > 10:
    lo, hi = np.percentile(vals, [P_LO, P_HI])
else:
    lo, hi = float(F_hp.min()), float(F_hp.max())

G = (F_hp - lo) / (hi - lo + 1e-8)
G = np.clip(G, 0.0, 1.0)
G[~mask] = 0.0

# 5) 灰度输出
u8 = (G * 255.0 + 0.5).astype(np.uint8)
cv2.imwrite(OUT_GRAY, u8)

# 6) 伪彩色输出（看起来更像“力场”）
color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
# 把无效区域弄成黑色
color[~mask] = (0, 0, 0)
cv2.imwrite(OUT_COLOR, color)

print("[DONE] wrote:", OUT_GRAY, OUT_COLOR)
