import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform
from PIL import Image
import requests
from io import BytesIO

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

st.title("🔍 Template Matching (ไม่มี OpenCV)")

# โหลดภาพ
target_image = load_image_from_url(target_url)
template_image = load_image_from_url(template_url)

# แสดงต้นฉบับ
cols = st.columns(2)
with cols[0]:
    st.image(template_image, caption="ใบหน้าที่ใช้ค้นหา", use_container_width=True)
with cols[1]:
    st.image(target_image, caption="ภาพเป้าหมาย", use_container_width=True)

# แปลงเป็น grayscale
target_gray = color.rgb2gray(target_image)
template_gray = color.rgb2gray(template_image)

# ลดขนาด template ถ้าจำเป็น
if template_gray.shape[1] > 100:
    scale = 100 / template_gray.shape[1]
    new_shape = (int(template_gray.shape[0] * scale), 100)
    template_gray = transform.resize(template_gray, new_shape, anti_aliasing=True)

# ทำ template matching
result = feature.match_template(target_gray, template_gray)

ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]
h, w = template_gray.shape

# แสดงผล
fig, ax = plt.subplots()
ax.imshow(target_image)
rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
ax.add_patch(rect)
ax.set_title("📍 ตำแหน่งใบหน้าที่พบ")
ax.set_xlabel("X")
ax.set_ylabel("Y")
st.pyplot(fig)
