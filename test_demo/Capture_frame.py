import cv2
import os
import shutil


def extract_frames_every_half_second(video_path: str, out_dir: str, clear_dir: bool = False):
    if clear_dir and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("读取不到FPS")

    step = max(1, int(round(fps * 0.5)))
    frame_idx, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            filename = f"frame_{saved:06d}_t{t:.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"完成：共保存 {saved} 张到：{os.path.abspath(out_dir)}")


if __name__ == "__main__":
    extract_frames_every_half_second("2025_12_15_10_28_17.mp4", r"./light_frame", clear_dir=False)
