import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import face_recognition
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

st.title("🧠 Face Detection + Template Matching with scikit-image")

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
# 🔍 ตรวจจับใบหน้าจาก template_image
# ---------------------------
st.subheader("🧠 ตรวจจับใบหน้าใน Template Image")
face_locations = face_recognition.face_locations(template_image)

if len(face_locations) == 0:
    st.error("ไม่พบใบหน้าใน template image")
else:
    top, right, bottom, left = face_locations[0]
    face_crop = template_image[top:bottom, left:right]

    st.image(face_crop, caption="ใบหน้าที่ตรวจพบ", width=200)

    # ---------------------------
    # 🔄 Template Matching
    # ---------------------------
    st.subheader("🔎 ค้นหาใบหน้าใน Target Image")

    # Convert images to grayscale
    target_gray = color.rgb2gray(target_image)
    face_gray = color.rgb2gray(face_crop)

    # Resize template ถ้ากว้างเกิน 100 px
    if face_gray.shape[1] > 100:
        scale = 100 / face_gray.shape[1]
        new_shape = (int(face_gray.shape[0] * scale), 100)
        face_gray = transform.resize(face_gray, new_shape, anti_aliasing=True)

    result = feature.match_template(target_gray, face_gray)

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    h, w = face_gray.shape

    # วาดกรอบบนภาพ target
    fig, ax = plt.subplots()
    ax.imshow(target_image)
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_title("📍 ตำแหน่งใบหน้าที่พบใน Target Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)
