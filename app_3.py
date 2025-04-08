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
    "Cat": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"
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
        # 3️⃣ Segmentation จาก Cr หรือ Cb Channel
        # -------------------------------
        st.subheader("3️⃣ แยกวัตถุออกจากพื้นหลัง (Segmentation)")
    
        seg_channel = st.radio("เลือก channel สำหรับ segmentation", ["Cr", "Cb"])
    
        # เลือก channel ที่จะใช้
        channel_map = {"Cr": Cr, "Cb": Cb}
        selected_channel = channel_map[seg_channel]
        selected_norm = selected_channel / 255.0
    
        # คำนวณ threshold
        otsu_threshold = filters.threshold_otsu(selected_norm)
    
        # Slider ให้ปรับ threshold เอง
        threshold_val = st.slider(
            f"ปรับค่า threshold ของ {seg_channel}",
            min_value=0.0,
            max_value=1.0,
            value=float(round(otsu_threshold, 3)),
            step=0.01
        )
    
        st.markdown(f"**Threshold ที่ใช้งานกับ {seg_channel}: {threshold_val:.2f}** (Otsu ≈ {otsu_threshold:.3f})")
    
        # สร้าง binary mask
        mask = selected_norm > threshold_val
    
        st.markdown("**Binary Mask**")
        fig_mask, ax_mask = plt.subplots()
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.set_title(f"Binary Mask from {seg_channel}")
        ax_mask.axis("off")
        st.pyplot(fig_mask)
    
        # แสดง RGB mask (สีแดง)
        rgb_mask = np.zeros_like(image)
        rgb_mask[:, :, 0] = mask * 255
        rgb_mask[:, :, 1] = 0
        rgb_mask[:, :, 2] = 0
    
        st.subheader("RGB Mask (วัตถุเป็นสีแดง)")
        fig_rgb_mask, ax_rgb_mask = plt.subplots()
        ax_rgb_mask.imshow(rgb_mask)
        ax_rgb_mask.set_title(f"RGB Mask จาก {seg_channel}")
        ax_rgb_mask.axis("off")
        st.pyplot(fig_rgb_mask)
    
        # แสดงผลลัพธ์ segmentation แบบ RGB (พื้นหลังดำ)
        st.subheader("วัตถุที่ถูกแยกออกมา (RGB จริง จากภาพต้นฉบับ)")
    
        segmented_object = np.zeros_like(image)
        for i in range(3):
            segmented_object[:, :, i] = image[:, :, i] * mask
    
        fig_result, ax_result = plt.subplots()
        ax_result.imshow(segmented_object)
        ax_result.set_title(f"Segmented Object from {seg_channel} (RGB)")
        ax_result.axis("off")
        st.pyplot(fig_result)



