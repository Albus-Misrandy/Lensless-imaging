import cv2
import numpy as np

def _lowpass_single_channel(channel_u8: np.ndarray, mask: np.ndarray):
    """对单通道做低通，返回 (filtered_u8, dft_shift, fshift_filtered)"""
    img_float = np.float32(channel_u8)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    fshift_filtered = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back_complex = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back_complex[:, :, 0], img_back_complex[:, :, 1])

    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back_norm), dft_shift, fshift_filtered


def lowpass_fft_video_luma(
    in_path: str,
    out_path: str,
    radius: int = 30,
    preview: bool = True,
    preview_every: int = 15
):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {out_path}")

    # mask（复用）
    rows, cols = h, w
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    center_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= radius ** 2
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[center_mask] = 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) BGR -> YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # 2) 只对 Y 做频域低通
        Y_f, dftY_shift, fY = _lowpass_single_channel(Y, mask)

        # 3) 合回去
        ycrcb_f = cv2.merge([Y_f, Cr, Cb])
        out_bgr = cv2.cvtColor(ycrcb_f, cv2.COLOR_YCrCb2BGR)

        writer.write(out_bgr)

        # 预览
        if preview:
            cv2.imshow("Original (BGR)", frame)
            cv2.imshow("Low-pass on Luma Y (BGR)", out_bgr)

            # 频谱预览（可选）
            if frame_idx % preview_every == 0:
                mag = cv2.magnitude(dftY_shift[:, :, 0], dftY_shift[:, :, 1])
                mag = 20 * np.log(1 + mag)
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("Spectrum (Y)", mag)

                mag_f = cv2.magnitude(fY[:, :, 0], fY[:, :, 1])
                mag_f = 20 * np.log(1 + mag_f)
                mag_f = cv2.normalize(mag_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("Filtered Spectrum (Y)", mag_f)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成：输出 -> {out_path}")


if __name__ == "__main__":
    lowpass_fft_video_luma(
        in_path="../collect_7.mp4",
        out_path="output_lowpass_luma.mp4",
        radius=30,
        preview=True
    )
