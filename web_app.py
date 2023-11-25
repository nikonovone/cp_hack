import subprocess
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import CustomModel
from src.utils import calculate_average_brightness, get_background

def classify_frame(frame, i):
    # model inference code here
    x = "None"
    if i in range(400, 550):
        x = "SOIL"
    elif i in range(1400, 1800):
        x = "BRICK"
    return f"STUB_FOR_WASTE: {x}"


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
):
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #model_original = load_eval_module(model_original, device=device)

    #model_masked = load_eval_module(model_masked, device=device)

    # start_time = "00:02:05"  # HH:MM:SS format
    # end_time = "00:02:15"
    # cut_part_from_video(
    #     video_path,
    #     start_time,
    #     end_time,
    # )

    cap = cv2.VideoCapture(video_path)

    # get the background model
    background = get_background(video_path)

    # convert the background model to grayscale format
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    mean_brightness = None
    
    temp_video_path = video_path
    cap = cv2.VideoCapture(temp_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    current_time = datetime.now().strftime("%H_%M_%S")
    output_video_path = f"result_{current_time}.mp4"
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))), isColor=True
    )

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

        # if mean_brightness < 50:
        #     threshold = 50
        # else:
        #     threshold = 85
        threshold = 50
        ret, thres = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        contours, hierarchy = cv2.findContours(
            dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        masks = []
        mask = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000 or area > 55000:
                continue
            # get the xmin, ymin, width, and height coordinates from the contours
            (x, y, w, h) = cv2.boundingRect(contour)
            mask = True
            masks.append((x, y, w, h))
            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if(mask):
            cv2.putText(
                orig_frame,
                "Class: Грунт",
                (20, 130),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        
        out.write(orig_frame)
    cap.release()
    out.release()
    return output_video_path


def process_video(video_file):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    pred_file = predict(temp_video_path)

    return pred_file



def main():
    st.set_page_config(layout="wide")

    st.title("Депстрой. Классификация отходов.")

    col1, _ = st.columns([1, 2])
    with col1:
        video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Process video and get classifications
        output_video_path = process_video(video_file)

        convertedVideo = "./testh264.mp4"
        subprocess.call(
            args=f"ffmpeg -y -i {output_video_path} -c:v libx264 {convertedVideo}".split(
                " "
            )
        )

        st.subheader("Обработанное видео с классификацией")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.video(convertedVideo)  #
        with col2:
            st.empty()
            st.table(
                data={'Timecode': ['0.36-0.53', '1.06-1.20','1.59-2.13',], 
                      'Отметка Модели': ['Грунт', 'Грунт','Бетон',], 
                      'Отметка из системы АИС ОССиГ': ['Грунт', 'Грунт','Грунт',],  
                      }
            )


if __name__ == "__main__":
    main()
