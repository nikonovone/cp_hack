import cv2
import numpy as np
from rich.progress import track


def visualize_optical_flow(
    input_video: str,
    output_video: str = "optical_flow.mp4",
):
    result_hsv = []

    # Открываем видео файл
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Вычисляем длительность видео в секундах
    total_frames / fps

    # Получаем ширину и высоту кадра
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Создаем объект VideoWriter для записи видео в формате mp4 с кодеком mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Инициализируем алгоритм Optical Flow
    prev_frame = None

    for k in track(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Вычисляем Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame,
                current_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=7,
                iterations=1,
                poly_n=3,
                poly_sigma=0.4,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
            )
            # create canvas to paint on
            hsv_canvas = np.zeros_like(frame)
            # set saturation value (position 2 in HSV space) to 255
            hsv_canvas[..., 1] = 255
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # set hue of HSV canvas (position 1)
            hsv_canvas[..., 0] = angle * (180 / (np.pi / 2))
            # set pixel intensity value (position 3
            hsv_canvas[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            frame_rgb = cv2.cvtColor(hsv_canvas, cv2.COLOR_HSV2BGR)
            hsv_canvas = hsv_canvas / 255

            hsv_canvas[..., 2] = hsv_canvas[..., 2] / hsv_canvas[..., 2].max()
            result = hsv_canvas[..., 2].sum() / (width * height)

            # Записываем кадр с визуализацией в выходное видео
            result_hsv.append(result)

            out.write(frame_rgb)

        prev_frame = current_frame

    # Завершаем запись видео
    cap.release()
    out.release()
