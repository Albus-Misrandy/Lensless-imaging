import cv2
import os

def video_diff_then_clahe(
    video_path: str,
    ref_img_path: str,
    out_video_path: str,
    clip_limit: float = 2.0,
    tile_grid_size=(8, 8),
    preview: bool = True
):
    # 参考图（灰度）
    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"参考图读取失败: {ref_img_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # 兜底

    # 以参考图尺寸为准（参考图不变）
    h, w = ref.shape[:2]

    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter 打开失败：可试试 .avi + fourcc='XVID'")

    # CLAHE：局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != ref.shape:
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

        # 1) 灰度差分
        diff = cv2.absdiff(gray, ref)

        # 2) 局部直方图均衡化（CLAHE）
        diff_clahe = clahe.apply(diff)

        # 写出（转3通道更稳）
        out_bgr = cv2.cvtColor(diff_clahe, cv2.COLOR_GRAY2BGR)
        writer.write(out_bgr)

        # 预览
        if preview:
            cv2.imshow("diff", diff)
            cv2.imshow("diff + CLAHE", diff_clahe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成：输出视频 -> {os.path.abspath(out_video_path)}")


if __name__ == "__main__":
    video_diff_then_clahe(
        video_path="2025_12_15_10_28_17.mp4",
        ref_img_path="light_frame/frame_000001_t0.51.jpg",
        out_video_path="diff_clahe_bright.mp4",
        clip_limit=2.0,
        tile_grid_size=(3, 3),
        preview=True
    )
