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
template_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"

st.title("🔍 Template Matching with Top-5 Matches")

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
st.subheader("📌 1. เลือกตำแหน่งวัตถุ (Object) ที่ต้องการค้นหาจาก Template Image")

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
st.image(face_crop, caption="✅ วัตถุที่คุณเลือกเพื่อใช้ค้นหา", width=250)

# ---------------------------
# Template Matching
# ---------------------------
st.subheader("🎯 2. ค้นหาวัตถุที่คล้ายกันทั้งหมดในภาพเป้าหมาย")

# แปลงเป็น grayscale
target_gray = color.rgb2gray(target_image)
face_gray = color.rgb2gray(face_crop)

# Resize template ถ้ากว้างเกิน 100px
if face_gray.shape[1] > 100:
    scale = 100 / face_gray.shape[1]
    new_shape = (int(face_gray.shape[0] * scale), 100)
    face_gray = transform.resize(face_gray, new_shape, anti_aliasing=True)

# Matching
result = feature.match_template(target_gray, face_gray)

# Threshold + หาขนาด
threshold = st.slider("Threshold สำหรับการ Matching", 0.5, 1.0, 0.85, step=0.01)
match_locations = np.where(result >= threshold)

h_match, w_match = face_gray.shape

# วาดกรอบทั้งหมด
fig2, ax2 = plt.subplots()
ax2.imshow(target_image)

for (y_match, x_match) in zip(*match_locations):
    rect = plt.Rectangle((x_match, y_match), w_match, h_match, edgecolor='red', facecolor='none', linewidth=2)
    ax2.add_patch(rect)

ax2.set_title("All Locations Detected")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
st.pyplot(fig2)

# ---------------------------
# แสดง Top-5 Match
# ---------------------------
st.subheader("🏆 3. วัตถุที่ตรงที่สุด 5 อันดับ")

# หาค่าความตรงกันทั้งหมด แล้วจัดลำดับ
sorted_indices = np.argsort(result.ravel())[::-1]
top_indices = sorted_indices[:5]
top_coords = np.array(np.unravel_index(top_indices, result.shape)).T

cols = st.columns(5)
for i, (y_match, x_match) in enumerate(top_coords):
    top_face = target_image[y_match:y_match+h_match, x_match:x_match+w_match]
    with cols[i]:
        st.image(top_face, caption=f"อันดับ {i+1}", use_container_width=True)

# รายงานจำนวนที่ตรวจพบทั้งหมด
st.success(f"🎯 ตรวจพบทั้งหมด: {len(match_locations[0])} ตำแหน่ง | แสดง Top 5 ที่ตรงที่สุด")

