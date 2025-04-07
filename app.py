import streamlit as st
from skimage import io, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ตั้งชื่อแอป
st.title("Image Processing with scikit-image")

# โหลดภาพตัวอย่าง
image_urls = {
    "ภาพตัวอย่างที่ 1": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
    "ภาพตัวอย่างที่ 2": "https://vetmarlborough.co.nz/wp-content/uploads/old-cats.jpg"
}

# แสดงภาพตัวอย่างให้เลือก
st.subheader("เลือกรูปภาพที่ต้องการแปลง")
cols = st.columns(len(image_urls))
selected_image_url = None

for i, (name, url) in enumerate(image_urls.items()):
    with cols[i]:
        st.image(url, caption=name, use_container_width=True)
        if st.button(f"เลือก {name}"):
            selected_image_url = url

# ถ้ามีการเลือกภาพ
if selected_image_url:
    # โหลดภาพ
    image = io.imread(selected_image_url)
    gray_image = color.rgb2gray(image)

    # สร้างภาพขาวดำโดยใช้ threshold
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh

    # แสดงผลลัพธ์
    st.subheader("ผลลัพธ์ที่ได้จากการแปลงภาพ")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ภาพสีเทา")
        fig1, ax1 = plt.subplots()
        ax1.imshow(gray_image, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)

        # แสดงตารางค่าพิกเซล (0–155)
        st.markdown("ตารางค่าพิกเซล (สีเทา) [0–155]")
        gray_scaled = (gray_image * 155).astype(int)
        gray_df = pd.DataFrame(gray_scaled[:10, :10])
        st.dataframe(gray_df)

    with col2:
        st.markdown("### ภาพขาวดำ")
        fig2, ax2 = plt.subplots()
        ax2.imshow(binary_image, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)

        # แสดงตารางค่าพิกเซล (0 หรือ 1)
        st.markdown("ตารางค่าพิกเซล (ขาวดำ) [0 หรือ 1]")
        binary_int = binary_image.astype(int)
        binary_df = pd.DataFrame(binary_int[:10, :10])
        st.dataframe(binary_df)
