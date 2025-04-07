import streamlit as st
from skimage import io, color
from skimage.filters import threshold_otsu, sobel
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

# ใช้ session_state เพื่อจำภาพที่เลือกไว้
if 'selected_image_url' not in st.session_state:
    st.session_state.selected_image_url = None

for i, (name, url) in enumerate(image_urls.items()):
    with cols[i]:
        st.image(url, caption=name, use_container_width=True)
        if st.button(f"เลือก {name}"):
            st.session_state.selected_image_url = url

selected_image_url = st.session_state.selected_image_url

# ถ้ามีการเลือกภาพ
if selected_image_url:
    # โหลดภาพ
    image = io.imread(selected_image_url)

    # ตรวจสอบว่าภาพมี 3 ช่องสี
    if image.ndim == 3 and image.shape[2] == 3:
        # ดึงช่อง R, G, B
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        st.subheader("ตารางค่าพิกเซลของภาพสี (R, G, B)")

        rgb_cols = st.columns(3)
        with rgb_cols[0]:
            st.markdown("#### ค่า R (แดง)")
            r_df = pd.DataFrame(R[:10, :10])
            st.dataframe(r_df)

        with rgb_cols[1]:
            st.markdown("#### ค่า G (เขียว)")
            g_df = pd.DataFrame(G[:10, :10])
            st.dataframe(g_df)

        with rgb_cols[2]:
            st.markdown("#### ค่า B (น้ำเงิน)")
            b_df = pd.DataFrame(B[:10, :10])
            st.dataframe(b_df)

    # แปลงเป็นภาพสีเทา
    gray_image = color.rgb2gray(image)

    # สร้างภาพขาวดำโดยใช้ threshold
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh

    # สร้างภาพขอบ
    edge_image = sobel(gray_image)

    # แสดงผลลัพธ์
    st.subheader("ผลลัพธ์ที่ได้จากการแปลงภาพ")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ภาพสีเทา (Grayscale)")
        fig1, ax1 = plt.subplots()
        ax1.imshow(gray_image, cmap='gray')
        ax1.axis('off')
        st.pyplot(fig1)

        st.markdown("ตารางค่าพิกเซล (สีเทา) [0–155]")
        gray_scaled = (gray_image * 155).astype(int)
        gray_df = pd.DataFrame(gray_scaled[:10, :10])
        st.dataframe(gray_df)

    with col2:
        st.markdown("### ภาพขาวดำ (Binary)")
        fig2, ax2 = plt.subplots()
        ax2.imshow(binary_image, cmap='gray')
        ax2.axis('off')
        st.pyplot(fig2)

        st.markdown("ตารางค่าพิกเซล (ขาวดำ) [0 หรือ 1]")
        binary_int = binary_image.astype(int)
        binary_df = pd.DataFrame(binary_int[:10, :10])
        st.dataframe(binary_df)

    # แสดงภาพขอบ
    st.subheader("ภาพขอบ (Edge Image)")
    fig3, ax3 = plt.subplots()
    ax3.imshow(edge_image, cmap='gray')
    ax3.axis('off')
    st.pyplot(fig3)

    st.markdown("ตารางค่าพิกเซล (ขอบ) [ค่าความต่างของพิกเซล]")
    edge_df = pd.DataFrame(edge_image[:10, :10])
    st.dataframe(edge_df.style.format("{:.3f}"))

    # ปรับความสว่าง (Brightness Enhancement)
    st.subheader("Image Enhancement: ปรับความสว่างของภาพสีเทา")
    brightness_factor = st.slider("ปรับความสว่าง", -0.20, 0.20, 0.0, step=0.01)
    enhanced_gray = np.clip(gray_image + brightness_factor, 0, 1)
    
    # แสดงภาพสีเทาหลังปรับ brightness
    st.subheader("ภาพสีเทาหลังปรับความสว่าง (Enhanced Gray Image)")
    fig4, ax4 = plt.subplots()
    ax4.imshow(enhanced_gray, cmap='gray')
    ax4.axis('off')
    st.pyplot(fig4)

    st.markdown("ตารางค่าพิกเซล (ภาพสีเทาหลังปรับ) [0–155]")
    enhanced_gray_scaled = (enhanced_gray * 155).astype(int)
    enhanced_gray_df = pd.DataFrame(enhanced_gray_scaled)
    st.dataframe(enhanced_gray_df)
