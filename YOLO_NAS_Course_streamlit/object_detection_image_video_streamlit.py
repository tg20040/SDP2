import cv2
from super_gradients.training import models
import torch
import numpy as np
import math
import streamlit as st
import time
import tempfile

label_audio_mapping = {
    "1feature": "Audio/genuine_money.wav",
    "5feature": "Audio/genuine_money.wav",
    "Otherfeature": "Audio/genuine_money.wav"

}

def play_audio_for_label(label):
    if label in label_audio_mapping:
        audio_file = label_audio_mapping[label]
        st.audio(audio_file, format='audio/wav')

def load_yolo_nas_process_each_image(image, confidence, st, detected_labels):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model with custom checkpoint
    model = models.get('yolo_nas_s', num_classes=9, checkpoint_path='weights/ckpt_best.pth').to(device)

    # Define class names
    classNames = ["1feature", "5feature", "Otherfeature", "RM1", "RM10", "RM100", "RM20", "RM5", "RM50"]

    # Perform prediction
    image_tensor = torch.from_numpy(image).to(device)
    result = model.predict(image_tensor, conf=confidence)

    # Extract predictions
    bbox_xyxys = result.prediction.bboxes_xyxy
    confidences = result.prediction.confidence
    labels = result.prediction.labels

    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        classname = int(cls)
        class_name = classNames[classname]
        conf = math.ceil((confidence * 100)) / 100
        label = f'{class_name} {conf:.2f}'
        detected_labels.append(class_name)

        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.rectangle(image, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    st.subheader('Output Image')
    st.image(image, use_column_width=True)

    current_time = time.time()
    for label in detected_labels:
        play_audio_for_label(label)
        time.sleep(5)

    return detected_labels

def load_yolo_nas_process_each_frame(video_name, confidence, kpi1_text, kpi2_text, kpi3_text, stframe, detected_labels):
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model with custom checkpoint
    model = models.get('yolo_nas_s', num_classes=9, checkpoint_path='weights/ckpt_best.pth').to(device)

    count = 0
    prev_time = 0
    classNames = ["1feature", "5feature", "Otherfeature", "RM1", "RM10", "RM100", "RM20", "RM5", "RM50"]

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        current_time = time.time()
        time_diff = current_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = current_time

        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{frame_width}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{frame_height}</h1>", unsafe_allow_html=True)

        # Perform prediction
        img_tensor = torch.from_numpy(img).to(device)
        result = model.predict(img_tensor, conf=confidence)

        # Extract predictions
        bbox_xyxys = result.prediction.bboxes_xyxy
        confidences = result.prediction.confidence
        labels = result.prediction.labels

        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence * 100)) / 100
            label = f'{class_name} {conf:.2f}'

            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.rectangle(img, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            detected_labels.add(class_name)

        # Display the frame
        stframe.image(img, channels='BGR', use_column_width=True)

        # FPS calculation (optional)
        count += 1

        # Play audio for detected labels periodically
        if count % (fps * 5) == 0:  # Assuming 5 seconds interval
            for label in detected_labels:
                play_audio_for_label(label)
            prev_time = current_time

    cap.release()


# Example usage with Streamlit
def main():
    st.title('YOLO NAS Object Detection')

    option = st.selectbox('Choose Inference Type', ('Image', 'Video'))

    if option == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            load_yolo_nas_process_each_image(image, 0.35, st)

    elif option == 'Video':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            stframe = st.empty()

            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                kpi1_text = st.markdown("0")
            with kpi2:
                kpi2_text = st.markdown("0")
            with kpi3:
                kpi3_text = st.markdown("0")

            load_yolo_nas_process_each_frame(video_path, 0.35, kpi1_text, kpi2_text, kpi3_text, stframe, set())

if __name__ == "__main__":
    main()
