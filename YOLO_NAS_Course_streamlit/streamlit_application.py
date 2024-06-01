import streamlit as st
from PIL import Image
from object_detection_image_video_streamlit import load_yolo_nas_process_each_image, load_yolo_nas_process_each_frame
import tempfile
import cv2
import numpy as np
import time

label_audio_mapping = {
    "1feature": "Audio/genuine_money.wav",
    "5feature": "Audio/genuine_money.wav",
    "Otherfeature": "Audio/genuine_money.wav"
}

def main():
    st.title("Counterfeit Money Detection")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:fist-child {
        width: 300px;
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About the App', 'Run on Image', 'Run on Video'])

    if app_mode == 'About the App':
        st.markdown('This project purpose is for SDP 23/24-II')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:fist-child {
            width: 300px;
            }
            </style>
            """,

            unsafe_allow_html=True,
        )

        st.image('https://x1.sdimgs.com/sd_static/u/201501/54b36f9300d9b.png')
        st.markdown('''
                    # About App
                    Genuine (Left) vs Fake (Right) - 
                    User can experince counterfeit money detection in real-time and choose between inference on image or video from left sidebar dropdown menu
                    ''')

    elif app_mode == 'Run on Image':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('Confidence', min_value=0.70, max_value=1.0)
        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:fist-child {
            width: 300px;
            }
            </style>
            """,

            unsafe_allow_html=True,
        )
        img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        DEMO_IMAGE = 'Image/F10.jpg'

        if img_file_buffer is not None:
            img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text('Original Image')
        st.sidebar.image(image)
        detected_labels = []
        load_yolo_nas_process_each_image(image, confidence, st, detected_labels)

        if detected_labels:
            st.subheader('Detected Labels')
            for label in detected_labels:
                st.markdown(
                    f'<div style="font-size:30px; padding:10px; border:2px solid #ccc; border-radius:5px; margin-bottom:10px;">{label}</div>',
                    unsafe_allow_html=True)
        else:
            st.subheader('No objects detected')

    elif app_mode == 'Run on Video':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
            }
            </style>
            """,

            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type = ["mp4", "avi", "mov", "asf"])

        DEMO_VIDEO = 'Video/VR.mp4'

        tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.empty()
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.empty()
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)

        detected_labels = set()
        prev_audio_play_time = time.time()

        load_yolo_nas_process_each_frame(tffile.name, 0.35, kpi1_text, kpi2_text, kpi3_text, stframe, detected_labels)

        if detected_labels:
            st.subheader('Detected Labels')
            for label in detected_labels:
                st.markdown(
                    f'<div style="font-size:30px; padding:10px; border:2px solid #ccc; border-radius:5px; margin-bottom:10px;">{label}</div>',
                    unsafe_allow_html=True)
        else:
            st.subheader('No objects detected')

        current_time = time.time()
        if current_time - prev_audio_play_time > 5:
            for label in detected_labels:
                st.audio(label_audio_mapping.get(label, ""), format='audio/wav')
            prev_audio_play_time = current_time

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
