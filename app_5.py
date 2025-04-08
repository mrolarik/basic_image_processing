import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import color, feature, transform
import mediapipe as mp

# ---------------------------
# ฟังก์ชันโหลดภาพจาก URL
# ---------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# ---------------------------
# Face detection ด้วย mediapipe
# ---------------------------
def detect_face_mediapipe(image):
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            return x, y, w, h
        return None

# ---------------------------
# URLs
# ---------------------------
target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
template_url = "https://pbs.twimg.com/media/FyCKKBDWYAwwEZl.jpg"

st.title("🧠 Face Detection (MediaPipe) + Template Matching (scikit-image)")

# โหลดภาพ
target_image = load_image_from_url(target_url)
template_image = load_image_from_url(template_url)

cols = st.columns(2)
with cols[0]:
    st.image(template_image, caption="Template Image", use_container_width=True)
with cols[1]:
    st.image(target_image, caption="Target Image", use_container_width=True)

# 🔍 ตรวจจับใบหน้าใน template_image
st.subheader("🧠 ตรวจจับใบหน้าใน Template Image ด้วย MediaPipe")

bbox = detect_face_mediapipe(template_image)

if not bbox:
    st.error("ไม่พบใบหน้าใน template image")
else:
    x, y, w, h = bbox
    face_crop = template_image[y:y+h, x:x+w]
    st.image(face_crop, caption="ใบหน้าที่ตรวจพบ", width=200)

    # Template Matching
    st.subheader("🔎 ค้นหาใบหน้าใน Target Image")

    target_gray = color.rgb2gray(target_image)
    face_gray = color.rgb2gray(face_crop)

    # Resize template ถ้ากว้างเกิน 100 px
    if face_gray.shape[1] > 100:
        scale = 100 / face_gray.shape[1]
        new_shape = (int(face_gray.shape[0] * scale), 100)
        face_gray = transform.resize(face_gray, new_shape, anti_aliasing=True)

    result = feature.match_template(target_gray, face_gray)

    ij = np.unravel_index(np.argmax(result), result.shape)
    x_match, y_match = ij[::-1]
    h_match, w_match = face_gray.shape

    # วาดกรอบบนภาพ target
    fig, ax = plt.subplots()
    ax.imshow(target_image)
    rect = plt.Rectangle((x_match, y_match), w_match, h_match, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_title("📍 ตำแหน่งใบหน้าที่พบใน Target Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)
