import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import io, color, feature

# ฟังก์ชันโหลดภาพจาก URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# URLs
target_url = "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg"
template_url = "https://pbs.twimg.com/media/FyCKKBDWYAwwEZl.jpg"

# โหลดภาพ
st.title("🔍 Face Search Using Template Matching (scikit-image)")

st.subheader("📥 โหลดภาพจาก URL")
target_image = load_image_from_url(target_url)
template_image = load_image_from_url(template_url)

# แสดงภาพทั้งสอง
cols = st.columns(2)
with cols[0]:
    st.image(template_image, caption="ภาพใบหน้าที่ต้องการค้นหา", use_container_width=True)
with cols[1]:
    st.image(target_image, caption="ภาพเป้าหมายที่ต้องค้นหา", use_container_width=True)

# แปลงเป็น grayscale
target_gray = color.rgb2gray(target_image)
template_gray = color.rgb2gray(template_image)

# Template Matching
st.subheader("🔎 ค้นหาบุคคลในภาพโดยใช้ template matching")
result = feature.match_template(target_gray, template_gray)

# หาตำแหน่งที่แมตช์ดีที่สุด
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

# แสดงผลลัพธ์
h, w = template_gray.shape

fig, ax = plt.subplots()
ax.imshow(target_image)
rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
ax.add_patch(rect)
ax.set_title("📍 ตำแหน่งใบหน้าที่พบ")
ax.set_xlabel("X")
ax.set_ylabel("Y")
st.pyplot(fig)
