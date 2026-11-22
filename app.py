
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import tempfile
import cv2
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Indian  Number Plate Detection",
    page_icon="🚘",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# FRONTEND DESIGN
# -----------------------------
st.title(" Indian Number Plate Detection")
st.write(
    "AI-powered licence plate detection using YOLO26."
)

st.divider()

confidence = st.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

option = st.radio(
    "Choose Detection Type",
    ["Image Detection", "Video Detection"],
    horizontal=True
)

# -----------------------------
# IMAGE DETECTION
# -----------------------------
if option == "Image Detection":

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        if st.button("Detect Number Plate"):

            with st.spinner("Detecting licence plates..."):

                results = model.predict(
                    image,
                    conf=confidence,
                    verbose=False
                )

                result_image = results[0].plot()
                result_image = cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                )

            with col2:
                st.subheader("Detection Result")
                st.image(
                    result_image,
                    use_container_width=True
                )

            detected = len(results[0].boxes)

            st.success(
                f"Detection completed! Plates detected: {detected}"
            )

            result_pil = Image.fromarray(result_image)

            import io
            buffer = io.BytesIO()
            result_pil.save(buffer, format="JPEG")

            st.download_button(
                "Download Result Image",
                data=buffer.getvalue(),
                file_name="detected_plates.jpg",
                mime="image/jpeg"
            )

# -----------------------------
# VIDEO DETECTION
# -----------------------------
elif option == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button("Detect Plates in Video"):

            with tempfile.TemporaryDirectory() as temp_dir:

                input_path = os.path.join(
                    temp_dir,
                    uploaded_video.name
                )

                with open(input_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())

                output_dir = os.path.join(temp_dir, "output")

                with st.spinner(
                    "Processing video. Please wait..."
                ):

                    model.predict(
                        source=input_path,
                        conf=confidence,
                        save=True,
                        project=output_dir,
                        name="prediction",
                        exist_ok=True,
                        verbose=False
                    )

                    output_path = os.path.join(
                        output_dir,
                        "prediction",
                        uploaded_video.name
                    )

                    if os.path.exists(output_path):

                        st.success("Video detection completed!")

                        with open(output_path, "rb") as f:
                            video_bytes = f.read()

                        st.subheader("Processed Video")
                        st.video(video_bytes)

                        st.download_button(
                            "Download Processed Video",
                            data=video_bytes,
                            file_name="detected_video.mp4",
                            mime="video/mp4"
                        )

                    else:
                        st.error(
                            "Output video not found. "
                            "Check the video format and processing logs."
                        )
