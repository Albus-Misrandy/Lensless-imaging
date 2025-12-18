import cv2
import os
import time
from datetime import datetime


def camera_diff_then_clahe_with_snapshot(
        out_video_path: str,
        snapshot_folder: str,
        clip_limit: float = 2.0,
        tile_grid_size=(8, 8),
        preview: bool = True,
        wait_time_seconds: float = 0.5  # 等待 0.5秒
):
    # 打开摄像头（0 是默认摄像头，多个摄像头可用 1, 2 等）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    # 获取摄像头的帧率和分辨率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 默认帧率为 30fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频
    os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建输出视频文件")

    # 创建拍照文件夹
    os.makedirs(snapshot_folder, exist_ok=True)

    # CLAHE：局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 1) 捕获第一帧作为参考图
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("无法读取第一帧作为参考图")

    # 保存参考图作为图片
    ref_img_path = "reference_frame.jpg"
    cv2.imwrite(ref_img_path, ref_frame)
    ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)  # 转为灰度参考图
    print(f"参考图已保存: {ref_img_path}")

    # 2) 等待 0.5 秒，确保参考图是在 0.5秒后获取的
    time.sleep(wait_time_seconds)

    # 3) 从摄像头获取实时帧并进行处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 确保视频帧尺寸与参考图一致
        if gray.shape != ref.shape:
            gray = cv2.resize(gray, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

        # 1) 灰度差分
        diff = cv2.absdiff(gray, ref)

        # 2) 局部直方图均衡化（CLAHE）
        diff_clahe = clahe.apply(diff)

        # 写出（转3通道）
        out_bgr = cv2.cvtColor(diff_clahe, cv2.COLOR_GRAY2BGR)
        writer.write(out_bgr)

        # 预览
        if preview:
            cv2.imshow("diff", diff)
            cv2.imshow("diff + CLAHE", diff_clahe)

            # 按下 'c' 键进行拍照保存
            if cv2.waitKey(1) & 0xFF == ord('c'):
                snapshot_filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_path = os.path.join(snapshot_folder, snapshot_filename)
                cv2.imwrite(snapshot_path, frame)
                print(f"拍照并保存: {snapshot_path}")

            # 按下 'q' 键退出
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成：输出视频 -> {os.path.abspath(out_video_path)}")


if __name__ == "__main__":
    camera_diff_then_clahe_with_snapshot(
        out_video_path="camera_diff_clahe.mp4",  # 输出视频
        snapshot_folder="snapshots",  # 指定拍照保存的文件夹
        clip_limit=2.0,
        tile_grid_size=(3, 3),
        preview=True
    )
