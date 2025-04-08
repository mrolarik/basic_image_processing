#template_options = {
#    "Template 1": "https://www.shutterstock.com/image-vector/rivne-ukraine-august-10-2023-260nw-2345412515.jpg",
#    "Template 2": "https://kreafolk.com/cdn/shop/articles/new-balance-logo-design-history-and-evolution-kreafolk_6253baa3-d41d-4864-a810-1db6b98997c9.jpg",
#    "Template 3": "https://upload.wikimedia.org/wikipedia/commons/3/32/House_sparrow04.jpg"
#}

#target_options = {
#    "Target 1": "https://image-cdn.essentiallysports.com/wp-content/uploads/2024-02-16T010328Z_1841023319_MT1USATODAY22532030_RTRMADP_3_MLS-PRESEASON-NEWELLS-OLD-BOYS-AT-INTER-MIAMI-CF.jpg",
#    "Target 2": "https://static.vecteezy.com/system/resources/previews/021/066/020/non_2x/set-of-popular-sportswear-logos-free-vector.jpg",
#    "Target 3": "https://upload.wikimedia.org/wikipedia/commons/7/70/Sparrow_on_branch.jpg"
#}

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import color, feature, transform

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
# URLs ของ Template และ Target
# ---------------------------
template_options = {
    "Template 1": "https://www.inspireuplift.com/resizer/?image=https://cdn.inspireuplift.com/uploads/images/seller_products/1686297015_3.jpg",
    "Template 2": "https://www.inspireuplift.com/resizer/?image=https://cdn.inspireuplift.com/uploads/images/seller_products/1686297015_3.jpg",
    "Template 3": "https://www.shutterstock.com/image-vector/rivne-ukraine-august-10-2023-600nw-2345179623.jpg"
}

target_options = {
    "Target 1": "https://www.inspireuplift.com/resizer/?image=https://cdn.inspireuplift.com/uploads/images/seller_products/1686297015_3.jpg",
    "Target 2": "https://emssound.net/wp-content/uploads/2017/12/logowall.jpg",
    "Target 3": "https://www.inspireuplift.com/resizer/?image=https://cdn.inspireuplift.com/uploads/images/seller_products/1686297015_3.jpg"
}

# ---------------------------
# ส่วนหัว
# ---------------------------
st.title("🔍 Multi-Scale Template Matching (Top-5 Results)")

# ---------------------------
# เลือก Template Image
# ---------------------------
st.subheader("🖼️ 1. เลือกรูป Template ที่ต้องการใช้")
if "selected_template_url" not in st.session_state:
    st.session_state.selected_template_url = list(template_options.values())[0]

cols1 = st.columns(3)
for i, (label, url) in enumerate(template_options.items()):
    with cols1[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}", key=f"template_{i}"):
            st.session_state.selected_template_url = url

# ---------------------------
# เลือก Target Image
# ---------------------------
st.subheader("🧭 2. เลือกรูป Target ที่ต้องการค้นหาในภาพ")
if "selected_target_url" not in st.session_state:
    st.session_state.selected_target_url = list(target_options.values())[0]

cols2 = st.columns(3)
for i, (label, url) in enumerate(target_options.items()):
    with cols2[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}", key=f"target_{i}"):
            st.session_state.selected_target_url = url

# ---------------------------
# โหลดภาพ
# ---------------------------
try:
    template_image = load_image_from_url(st.session_state.selected_template_url)
    target_image = load_image_from_url(st.session_state.selected_target_url)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดภาพ: {e}")
    st.stop()

# ---------------------------
# Crop Template ด้วย Slider
# ---------------------------
st.subheader("✂️ 3. เลือกวัตถุจาก Template Image")

fig1, ax1 = plt.subplots()
ax1.imshow(template_image)
ax1.set_title("Template Image with X, Y Axes")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
st.pyplot(fig1)

max_y, max_x = template_image.shape[:2]
x = st.slider("ตำแหน่ง X (ซ้าย)", 0, max_x - 10, 100)
y = st.slider("ตำแหน่ง Y (บน)", 0, max_y - 10, 100)
w = st.slider("ความกว้าง (Width)", 10, max_x - x, 100)
h = st.slider("ความสูง (Height)", 10, max_y - y, 100)

face_crop = template_image[y:y+h, x:x+w]
st.image(face_crop, caption="✅ วัตถุที่คุณเลือก", width=250)

# ---------------------------
# Multi-scale Template Matching
# ---------------------------
st.subheader("🎯 4. ค้นหาในหลายขนาด (Multi-scale Matching)")

target_gray = color.rgb2gray(target_image)
face_gray_original = color.rgb2gray(face_crop)

scale_min = st.slider("🔍 Scale เริ่มต้น", 0.5, 1.0, 0.7, step=0.05)
scale_max = st.slider("🔍 Scale สูงสุด", 1.0, 2.0, 1.3, step=0.05)
scale_step = st.slider("📏 Step ของ scale", 0.05, 0.3, 0.1, step=0.05)
threshold = st.slider("🎚️ Threshold สำหรับ Matching", 0.5, 1.0, 0.85, step=0.01)

scales = np.arange(scale_min, scale_max + scale_step, scale_step)
results = []
positions = []
scales_used = []

for scale in scales:
    try:
        scaled_template = transform.rescale(face_gray_original, scale, anti_aliasing=True, channel_axis=None)
        if scaled_template.shape[0] > target_gray.shape[0] or scaled_template.shape[1] > target_gray.shape[1]:
            continue
        res = feature.match_template(target_gray, scaled_template)
        results.append(res)
        positions.append((scaled_template.shape[0], scaled_template.shape[1]))
        scales_used.append(scale)
    except Exception as e:
        st.warning(f"Scale {scale:.2f}: {e}")

# รวมผลทั้งหมด
all_matches = []
for i, res in enumerate(results):
    for y, x in zip(*np.where(res >= threshold)):
        score = res[y, x]
        h_match, w_match = positions[i]
        all_matches.append((score, x, y, w_match, h_match, scales_used[i]))

# เรียงตาม score สูงสุด
all_matches_sorted = sorted(all_matches, key=lambda x: x[0], reverse=True)
top_matches = all_matches_sorted[:5]

# ---------------------------
# แสดงตำแหน่งทั้งหมดที่ตรวจพบ
# ---------------------------
fig2, ax2 = plt.subplots()
ax2.imshow(target_image)
for (_, x, y, w, h, _) in all_matches:
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
    ax2.add_patch(rect)
ax2.set_title("📍 ตำแหน่งทั้งหมดที่ตรวจพบ (หลายขนาด)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
st.pyplot(fig2)

# ---------------------------
# แสดง Top-5 Match
# ---------------------------
st.subheader("🏆 5. วัตถุที่ตรงที่สุด 5 อันดับ")

cols_top5 = st.columns(5)
for i, (score, x, y, w, h, scale) in enumerate(top_matches):
    matched_crop = target_image[y:y+h, x:x+w]
    with cols_top5[i]:
        st.image(matched_crop, caption=f"อันดับ {i+1}\nScale={scale:.2f}", use_container_width=True)

st.success(f"🎯 รวมตรวจพบทั้งหมด: {len(all_matches)} ตำแหน่ง | แสดง Top 5 จากหลายขนาด")


