import cv2
import numpy as np

# 1) 改成你的图片路径
IMG_PATH = "../snapshots/snapshot_20260115_160108.jpg"

# 2) 读图（按灰度）
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"读不到图片：{IMG_PATH}")

print("shape(H,W) =", img.shape, "dtype =", img.dtype,
      "min/max =", int(img.min()), int(img.max()))

# 3) 显示（按键关闭窗口）
cv2.imshow("raw_gray", img)
key = cv2.waitKey(0) & 0xFF
if key == ord('q') or key == 27:  # q 或 ESC
    cv2.destroyAllWindows()

# 4) （可选）裁剪 ROI：如果你只想用中间一块更干净的区域，先裁一下
#    你可以先不裁，确认能显示再说
# y0, y1 = 50, 450
# x0, x1 = 50, 600
# roi = img[y0:y1, x0:x1]
# cv2.imshow("roi", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("roi.png", roi)
