#image_options = {
#    "Dog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
#    "Cat": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
#}

import streamlit as st
from skimage import color, filters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# โหลดภาพจาก URL ด้วย PIL
# -------------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(img)

# -------------------------------
# ฟังก์ชันแสดงภาพแยกค่าสี
# -------------------------------
def show_channel_image(channel_data, title, cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(channel_data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

# -------------------------------
# URLs ของภาพตัวอย่าง
# -------------------------------
image_options = {
    "Dog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
    "Cat": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
}

# -------------------------------
# ส่วนแสดง thumbnail
# -------------------------------
st.title("Image Processing: RGB to YCrCb Viewer + Segmentation")

st.subheader("เลือกรูปภาพที่ต้องการประมวลผล")
cols = st.columns(2)
for i, (label, url) in enumerate(image_options.items()):
    with cols[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}"):
            st.session_state.selected_image = load_image_from_url(url)

# -------------------------------
# ประมวลผลเมื่อมีภาพถูกเลือก
# -------------------------------
if 'selected_image' in st.session_state:
    image = st.session_state.selected_image

    st.subheader("1️⃣ แสดงค่าสีในระบบ RGB")

    # แยก R, G, B
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # แสดงภาพแยกแต่ละ channel
    rgb_cols = st.columns(3)
    with rgb_cols[0]:
        show_channel_image(R, "R (Red)")
        st.write("ตารางค่า R")
        st.dataframe(pd.DataFrame(R))

    with rgb_cols[1]:
        show_channel_image(G, "G (Green)")
        st.write("ตารางค่า G")
        st.dataframe(pd.DataFrame(G))

    with rgb_cols[2]:
        show_channel_image(B, "B (Blue)")
        st.write("ตารางค่า B")
        st.dataframe(pd.DataFrame(B))

    # แปลง RGB → YCrCb
    ycrcb = color.rgb2ycbcr(image)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    st.subheader("2️⃣ แสดงค่าสีในระบบ YCrCb")

    ycrcb_cols = st.columns(3)
    with ycrcb_cols[0]:
        show_channel_image(Y, "Y (Luminance)")
        st.write("ตารางค่า Y")
        st.dataframe(pd.DataFrame(Y.astype(int)))

    with ycrcb_cols[1]:
        show_channel_image(Cr, "Cr (Red-difference)")
        st.write("ตารางค่า Cr")
        st.dataframe(pd.DataFrame(Cr.astype(int)))

    with ycrcb_cols[2]:
        show_channel_image(Cb, "Cb (Blue-difference)")
        st.write("ตารางค่า Cb")
        st.dataframe(pd.DataFrame(Cb.astype(int)))

    # -------------------------------
    # 3️⃣ Segmentation จาก Cr Channel
    # -------------------------------
    st.subheader("3️⃣ แยกวัตถุออกจากพื้นหลังด้วย Cr Channel (Segmentation)")

    # Normalize Cr เพื่อใช้ threshold
    cr_normalized = Cr / 255.0
    otsu_threshold = filters.threshold_otsu(cr_normalized)

    # Slider ปรับ threshold
    threshold_val = st.slider(
        "ปรับค่า threshold (เริ่มต้นจาก Otsu)",
        min_value=0.0,
        max_value=1.0,
        value=float(round(otsu_threshold, 3)),
        step=0.01
    )

    st.markdown(f"**Threshold ที่ใช้งาน: {threshold_val:.2f}** (Otsu ≈ {otsu_threshold:.3f})")

    # สร้าง binary mask
    mask = cr_normalized > threshold_val

    # แสดง Binary Mask
    st.markdown("**Binary Mask**")
    fig_mask, ax_mask = plt.subplots()
    ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title("Binary Mask จาก Cr")
    ax_mask.axis("off")
    st.pyplot(fig_mask)

    # สร้าง RGB Mask สีแดง
    rgb_mask = np.zeros_like(image)
    rgb_mask[:, :, 0] = mask * 255
    rgb_mask[:, :, 1] = 0
    rgb_mask[:, :, 2] = 0

    st.subheader("RGB Mask (วัตถุเป็นสีแดง)")
    fig_rgb_mask, ax_rgb_mask = plt.subplots()
    ax_rgb_mask.imshow(rgb_mask)
    ax_rgb_mask.set_title("RGB Mask (Foreground = Red)")
    ax_rgb_mask.axis("off")
    st.pyplot(fig_rgb_mask)

    # ✅ แสดงวัตถุที่แยกออกมาแบบ RGB (พื้นหลังดำ)
    st.subheader("วัตถุที่ถูกแยกออกมา (RGB จริง จากภาพต้นฉบับ)")

    segmented_object = np.zeros_like(image)
    for i in range(3):
        segmented_object[:, :, i] = image[:, :, i] * mask

    fig_result, ax_result = plt.subplots()
    ax_result.imshow(segmented_object)
    ax_result.set_title("Segmented Object (RGB with Black Background)")
    ax_result.axis("off")
    st.pyplot(fig_result)


