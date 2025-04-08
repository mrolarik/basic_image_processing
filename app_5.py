import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import cv2
from skimage import color, feature, transform

# ---------------------------
# ฟังก์ชันโหลดภาพจาก URL
# ---------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# ---------------------------
# URLs
# ---------------------------
target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
template_url = "https://pbs.twimg.com/media/FyCKKBDWYAwwEZl.jpg"

st.title("🧠 Face Detection (OpenCV) + Template Matching (scikit-image)")

# ---------------------------
# โหลดภาพ
# ---------------------------
st.subheader("📥 โหลดภาพจาก URL")
target_image = load_image_from_url(target_url)
template_image = load_image_from_url(template_url)

cols = st.columns(2)
with cols[0]:
    st.image(template_image, caption="Template Image", use_container_width=True)
with cols[1]:
    st.image(target_image, caption="Target Image", use_container_width=True)

# ---------------------------
# 🔍 ตรวจจับใบหน้าด้วย OpenCV Haar Cascade
# ---------------------------
st.subheader("🧠 ตรวจจับใบหน้าใน Template Image ด้วย OpenCV")

# โหลด haarcascade สำหรับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# แปลงเป็น grayscale สำหรับตรวจจับ
template_gray_cv = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)

# ตรวจจับใบหน้า
faces = face_cascade.detectMultiScale(template_gray_cv, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    st.error("ไม่พบใบหน้าใน template image")
else:
    x, y, w, h = faces[0]
    face_crop = template_image[y:y+h, x:x+w]
    st.image(face_crop, caption="ใบหน้าที่ตรวจพบ", width=200)

    # ---------------------------
    # 🔄 Template Matching
    # ---------------------------
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

    # แสดงภาพพร้อมกรอบ
    fig, ax = plt.subplots()
    ax.imshow(target_image)
    rect = plt.Rectangle((x_match, y_match), w_match, h_match, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_title("📍 ตำแหน่งใบหน้าที่พบใน Target Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)
