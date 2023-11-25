import subprocess
from pathlib import Path

import albumentations as A
import cv2
import fire
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from rich.progress import track

from src.model import CustomModel
from src.utils import calculate_average_brightness, get_background


def load_eval_module(checkpoint_path: str, device: torch.device) -> CustomModel:
    module = CustomModel.load_from_checkpoint(checkpoint_path)
    module.to(device)
    module.eval()

    return module


def cut_part_from_video(input_file, start_time, end_time, output_directory=None):
    if output_directory is None:
        output_directory = "./"
    input_file = Path(input_file)
    output_directory = Path(output_directory)
    group = input_file.parent.name
    save_dir = output_directory / group
    save_dir.mkdir(parents=True, exist_ok=True)
    # Create the FFmpeg command to cut the video
    cut_command = [
        "ffmpeg",
        "-i",
        input_file,
        "-ss",
        start_time,
        "-to",
        end_time,
        "-y",
        "-loglevel",
        "quiet",
        "temp_cut.mp4",
    ]
    subprocess.run(cut_command)


def array2tensor(image: np.array, max_size, device):
    test_transform = A.Compose(
        [
            A.Resize(max_size, max_size),
            ToTensorV2(),
        ]
    )

    img = test_transform(image=image)["image"].float() / 255
    img = (img[np.newaxis, ...]).to(device)
    return img


def predict(
    video_path: str,
    model_original: str,
    model_masked: str,
    max_size=256,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_original = load_eval_module(model_original, device=device)

    model_masked = load_eval_module(model_masked, device=device)

    start_time = "00:02:05"  # HH:MM:SS format
    end_time = "00:02:15"
    cut_part_from_video(
        video_path,
        start_time,
        end_time,
    )

    cap = cv2.VideoCapture("temp_cut.mp4")

    # get the background model
    background = get_background("temp_cut.mp4")

    # convert the background model to grayscale format
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    mean_brightness = None

    #! проверить
    softmax = torch.nn.Softmax(dim=-1)

    predictions_masked = []
    predictions_original = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        orig_frame = frame.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, background)
        # thresholding to convert the frame to binary
        if not mean_brightness:
            mean_brightness = calculate_average_brightness(
                cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            )

        if mean_brightness < 50:
            threshold = 50
        else:
            threshold = 85

        ret, thres = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        contours, hierarchy = cv2.findContours(
            dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000 or area > 50000:
                continue
            # get the xmin, ymin, width, and height coordinates from the contours
            (x, y, w, h) = cv2.boundingRect(contour)
            masks.append((x, y, w, h))
        if len(masks) > 0:
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for mask in masks:
                mask_image = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask_image[
                    mask[1] : mask[1] + mask[3], mask[0] : mask[0] + mask[2]
                ] = 255
                combined_mask = cv2.bitwise_or(combined_mask, mask_image)
            masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

            masked_frame_tensor = array2tensor(masked_frame, max_size, device)
        else:
            masked_frame_tensor = None

        frame_tensor = array2tensor(frame, max_size, device)

        if masked_frame_tensor is not None:
            preds_masked = softmax(model_masked(masked_frame_tensor))
            preds_original = softmax(model_original(frame_tensor))

            predictions_masked.append(preds_masked.cpu().detach().numpy())
            predictions_original.append(preds_original.cpu().detach().numpy())

        else:
            preds_original = softmax(model_original(frame_tensor))
            predictions_original.append(preds_original.cpu().detach().numpy())

    # Release the video and writer objects
    cap.release()

    predictions_original_mean = np.mean(predictions_original, axis=0)
    predictions_masked_mean = np.mean(predictions_masked, axis=0)

    mean_prediction = np.mean(
        [predictions_original_mean, predictions_masked_mean], axis=0
    )

    final_predict = mean_prediction.argmax()

    return final_predict


def main(
    video_folder: str,
    model_original: str,
    model_masked: str,
    max_size=256,
):
    label_map_reverse = {0: "Бетон", 1: "Грунт", 2: "Дерево", 3: "Кирпич"}
    video_folder = Path(video_folder)
    data = []
    for video_path in track(list(video_folder.glob("*.mp4"))):
        prediction = predict(
            video_path,
            model_original,
            model_masked,
            max_size,
        )
        data.append(
            {"Имя файла": video_path.stem, "Категория": label_map_reverse[prediction]}
        )
    pd.DataFrame(data).to_csv("result_v2.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
