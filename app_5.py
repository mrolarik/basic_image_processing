#target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
#template_url = "https://images.mlssoccer.com/image/private/t_editorial_landscape_12_desktop/f_png/mls-mia-prd/xyfcjysnblxkkprtwect.png"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import color, feature, transform

# ---------------------------
# โหลดภาพจาก URL
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
#template_url = "https://images.mlssoccer.com/image/private/t_editorial_landscape_12_desktop/f_png/mls-mia-prd/xyfcjysnblxkkprtwect.png"
template_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"

st.title("🔍 Template Matching with Manual Face Crop")

# โหลดภาพ
try:
    template_image = load_image_from_url(template_url)
    target_image = load_image_from_url(target_url)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดภาพ: {e}")
    st.stop()

# ---------------------------
# แสดง template image พร้อมแกน X, Y
# ---------------------------
st.subheader("📌 1. เลือกตำแหน่งใบหน้าจาก Template Image")

fig1, ax1 = plt.subplots()
ax1.imshow(template_image)
ax1.set_title("Template Image with X, Y Axes")
ax1.set_xlabel("X (Column)")
ax1.set_ylabel("Y (Row)")
st.pyplot(fig1)

# ขนาดภาพ
max_y, max_x = template_image.shape[0], template_image.shape[1]

# Slider เลือกตำแหน่ง crop
x = st.slider("ตำแหน่ง X (ซ้าย)", 0, max_x - 10, 100)
y = st.slider("ตำแหน่ง Y (บน)", 0, max_y - 10, 100)
w = st.slider("ความกว้าง (Width)", 10, max_x - x, 100)
h = st.slider("ความสูง (Height)", 10, max_y - y, 100)

# Crop ใบหน้าที่เลือก
face_crop = template_image[y:y+h, x:x+w]
st.image(face_crop, caption="✅ ใบหน้าที่คุณเลือกเพื่อใช้ค้นหา", width=250)

# ---------------------------
# Template Matching
# ---------------------------
st.subheader("🎯 2. ค้นหาใบหน้าที่คล้ายกันใน Target Image")

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
fig2, ax2 = plt.subplots()
ax2.imshow(target_image)
rect = plt.Rectangle((x_match, y_match), w_match, h_match, edgecolor='red', facecolor='none', linewidth=2)
ax2.add_patch(rect)
ax2.set_title("Location in the Target image")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
st.pyplot(fig2)

# แสดงใบหน้าที่ตรวจพบ
st.subheader("🧑‍🦱 ใบหน้าที่ตรวจพบใน Target Image")
detected_face = target_image[y_match:y_match+h_match, x_match:x_match+w_match]
st.image(detected_face, caption="Template Matching", width=250)


