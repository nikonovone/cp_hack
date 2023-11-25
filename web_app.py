import subprocess

import streamlit as st
import cv2


def classify_frame(frame, i):
    # model inference code here
    x = "None"
    if i in range(400, 550):
        x = "SOIL"
    elif i in range(1400, 1800):
        x = "BRICK"
    return f"STUB_FOR_WASTE: {x}"


def process_video(video_file):
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    classifications = []

    # Create a VideoWriter object to save the processed frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = "result.mp4"
    out = cv2.VideoWriter(
        output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))), isColor=True
    )

    frame_skip_interval = 15

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip_interval == 0:
            classification = classify_frame(frame, i)
            classifications.append(classification)

            frame = cv2.putText(
                frame,
                f"Class: {classification}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out.write(frame)

    cap.release()
    out.release()

    return classifications, output_video_path


def main():
    st.set_page_config(layout="wide")

    st.title("Депстрой. Классификация отходов.")

    col1, _ = st.columns([1, 2])
    with col1:
        video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Process video and get classifications
        classifications, output_video_path = process_video(video_file)

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
                data={1: "aaaa", 2: "aaaa", 3: "aaaa", 4: "aaaa", 5: "aaaa", 6: "aaaa"}
            )


if __name__ == "__main__":
    main()
