import cv2

def crop_center(img, target_height=420):
    """
    居中裁剪图像
    :param img: 输入图像
    :param target_height: 目标高度
    :return: 裁剪后的图像
    """
    height, width = img.shape[:2]
    
    # 计算需要裁剪掉的总高度
    crop_total = height - target_height
    
    # 上下各裁剪一半
    crop_top = crop_total // 2
    crop_bottom = crop_total - crop_top
    
    # 居中裁剪
    roi = img[crop_top:height-crop_bottom, 0:width]
    
    return roi

# 读取图像
img = cv2.imread("../snapshots/flat.jpg")

# 居中裁剪到420高度
roi = crop_center(img, 420)  # 结果640x420

print(f"原始图像尺寸: {img.shape}")
print(f"裁剪后尺寸: {roi.shape}")

# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Centered Crop', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()