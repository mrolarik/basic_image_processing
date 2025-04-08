import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import color, feature, transform, data
from skimage.io import imread
from skimage.feature import Cascade
import os

# ---------------------------
# ฟังก์ชันโหลดภาพจาก URL
# ---------------------------
def load_image_from_url(url):
    response = requests.get(url, stream=True)
    
    if "image" not in response.headers.get("content-type", ""):
        raise ValueError("URL does not contain a valid image.")
    
    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


# ---------------------------
# URLs
# ---------------------------
target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
template_url = "https://images.mlssoccer.com/image/private/t_editorial_landscape_12_desktop/f_png/mls-mia-prd/xyfcjysnblxkkprtwect.png"

st.title("🔍 Face Detection (scikit-image) + Template Matching")

# โหลดภาพ
#target_image = load_image_from_url(target_url)
#template_image = load_image_from_url(template_url)

try:
    template_image = load_image_from_url(template_url)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดภาพ template: {e}")
    st.stop()

try:
    target_image = load_image_from_url(target_url)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดภาพ target: {e}")
    st.stop()

# แสดงภาพต้นฉบับ
cols = st.columns(2)
with cols[0]:
    st.image(template_image, caption="Template Image", use_container_width=True)
with cols[1]:
    st.image(target_image, caption="Target Image", use_container_width=True)

# ---------------------------
# Face detection with skimage.feature.Cascade
# ---------------------------
st.subheader("📌 ตรวจจับใบหน้าใน Template ด้วย skimage.feature.Cascade")

# แปลงเป็น grayscale
template_gray = color.rgb2gray(template_image)

# โหลด Haar Cascade classifier ที่มากับ skimage
trained_file = data.lbp_frontal_face_cascade_filename()
#detector = Cascade(filename=trained_file)
detector = feature.Cascade(data.lbp_frontal_face_cascade_filename())

# ตรวจจับใบหน้า
faces = detector.detect_multi_scale(img=template_gray,
                                     scale_factor=1.2,
                                     step_ratio=1,
                                     min_size=(60, 60),
                                     max_size=(300, 300))

# ✅ เพิ่มเงื่อนไขป้องกันเมื่อไม่พบใบหน้า
if faces is None or len(faces) == 0:
    st.warning("ไม่พบใบหน้าในภาพที่ใช้ค้นหา กรุณาใช้ภาพอื่น หรือเลือกใบหน้าแบบแมนนวล")
    st.stop()
else:
    x, y, w, h = faces[0]
    face_crop = template_image[y:y+h, x:x+w]
    st.image(face_crop, caption="✅ ใบหน้าที่ใช้ค้นหา", width=250)

    # ---------------------------
    # Template Matching
    # ---------------------------
    st.subheader("🎯 ค้นหาใบหน้าใน Target Image ด้วย Template Matching")

    target_gray = color.rgb2gray(target_image)
    face_gray = color.rgb2gray(face_crop)

    # Resize template ถ้ากว้างเกิน 100px
    if face_gray.shape[1] > 100:
        scale = 100 / face_gray.shape[1]
        new_shape = (int(face_gray.shape[0] * scale), 100)
        face_gray = transform.resize(face_gray, new_shape, anti_aliasing=True)

    result = feature.match_template(target_gray, face_gray)

    ij = np.unravel_index(np.argmax(result), result.shape)
    x_match, y_match = ij[::-1]
    h_match, w_match = face_gray.shape

    # แสดงตำแหน่งใบหน้าที่ตรวจพบ
    fig, ax = plt.subplots()
    ax.imshow(target_image)
    rect = plt.Rectangle((x_match, y_match), w_match, h_match, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_title("📍 ตำแหน่งใบหน้าที่ตรวจพบในภาพเป้าหมาย")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)

    # แสดงใบหน้าที่ตรวจพบ
    st.subheader("🧑‍🦱 ใบหน้าที่ตรวจพบในภาพเป้าหมาย")
    detected_face = target_image[y_match:y_match+h_match, x_match:x_match+w_match]
    st.image(detected_face, caption="ใบหน้าที่ตรวจพบ", width=250)
