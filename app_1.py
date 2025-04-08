import streamlit as st
from skimage import io, img_as_float
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import requests
from io import BytesIO

# ------------------------------
# ฟังก์ชันโหลดภาพจาก URL
# ------------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# ------------------------------
# ฟังก์ชัน resize ภาพให้มีขนาดเท่ากัน
# ------------------------------
def resize_images_to_match(img1, img2):
    target_shape = (
        min(img1.shape[0], img2.shape[0]),
        min(img1.shape[1], img2.shape[1])
    )
    resized1 = resize(img1, (*target_shape, 3), anti_aliasing=True)
    resized2 = resize(img2, (*target_shape, 3), anti_aliasing=True)
    return resized1, resized2

# ------------------------------
# Section 1: Image Slice
# ------------------------------
st.title("Image Processing with scikit-image")

# URLs ของภาพตัวอย่าง
image_options = {
    "Bulldog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
    "Cat1": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
    "Cat2": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
}

st.header("Image Slicing with scikit-image")
st.subheader("ภาพตัวอย่าง (เลือก 1 รูป)")

# แสดง thumbnails แบบ 3 คอลัมน์
cols = st.columns(3)
for idx, (label, url) in enumerate(image_options.items()):
    with cols[idx]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}"):
            st.session_state.image = load_image_from_url(url)

# ถ้ามีภาพที่โหลดแล้ว
if 'image' in st.session_state and st.session_state.image is not None:
    image = st.session_state.image

    st.subheader("ภาพต้นฉบับที่โหลดแล้ว (พร้อมแกน X, Y)")
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Original Image")
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)

    st.subheader("เลือกแสดงบางส่วนของภาพ (Slice Image)")

    # รับ input สำหรับ slice
    row_start = st.number_input("Row start", min_value=0, max_value=image.shape[0]-1, value=0, key="row_start")
    row_end = st.number_input("Row end", min_value=row_start+1, max_value=image.shape[0], value=image.shape[0], key="row_end")
    col_start = st.number_input("Column start", min_value=0, max_value=image.shape[1]-1, value=0, key="col_start")
    col_end = st.number_input("Column end", min_value=col_start+1, max_value=image.shape[1], value=image.shape[1], key="col_end")

    # Slice ภาพ
    sliced_image = image[int(row_start):int(row_end), int(col_start):int(col_end)]

    st.subheader("ภาพบางส่วนที่เลือก")
    st.image(sliced_image, caption="ภาพบางส่วน", use_container_width=True)

# ------------------------------
# Section 2: Image Blending
# ------------------------------
st.title("Image Blending with scikit-image")

# URLs ของภาพสำหรับ Blend
image_urls = {
    "Image 1": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
    "Image 2": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
}

# แสดงภาพ thumbnail
st.subheader("Sample Images for Blending")
thumb_cols = st.columns(2)
for i, (name, url) in enumerate(image_urls.items()):
    with thumb_cols[i]:
        st.image(url, caption=name, width=200)

# เตรียม session_state
if 'blend_image1' not in st.session_state:
    st.session_state.blend_image1 = None
    st.session_state.blend_image2 = None

# ปุ่มสำหรับโหลดภาพ
if st.button("Load and Blend Images"):
    st.session_state.blend_image1 = load_image_from_url(image_urls["Image 1"])
    st.session_state.blend_image2 = load_image_from_url(image_urls["Image 2"])

# ถ้ามีภาพแล้ว
if st.session_state.blend_image1 is not None and st.session_state.blend_image2 is not None:
    original_img1 = st.session_state.blend_image1
    original_img2 = st.session_state.blend_image2

    # แสดง Original Images ก่อน resize
    st.subheader("Original Images (Before Resize)")
    ori_cols = st.columns(2)
    with ori_cols[0]:
        fig_ori1, ax_ori1 = plt.subplots()
        ax_ori1.imshow(original_img1)
        ax_ori1.set_title("Original Image 1")
        ax_ori1.set_xlabel("X (Column)")
        ax_ori1.set_ylabel("Y (Row)")
        st.pyplot(fig_ori1)

    with ori_cols[1]:
        fig_ori2, ax_ori2 = plt.subplots()
        ax_ori2.imshow(original_img2)
        ax_ori2.set_title("Original Image 2")
        ax_ori2.set_xlabel("X (Column)")
        ax_ori2.set_ylabel("Y (Row)")
        st.pyplot(fig_ori2)

    # แปลงเป็น float และ resize ให้เท่ากัน
    img1 = img_as_float(original_img1)
    img2 = img_as_float(original_img2)
    img1, img2 = resize_images_to_match(img1, img2)

    # แสดง Images หลัง resize
    st.subheader("Resized Images (Before Blending)")
    resize_cols = st.columns(2)

    with resize_cols[0]:
        fig1, ax1 = plt.subplots()
        ax1.imshow(img1)
        ax1.set_title("Resized Image 1")
        ax1.set_xlabel("X (Column)")
        ax1.set_ylabel("Y (Row)")
        st.pyplot(fig1)

    with resize_cols[1]:
        fig2, ax2 = plt.subplots()
        ax2.imshow(img2)
        ax2.set_title("Resized Image 2")
        ax2.set_xlabel("X (Column)")
        ax2.set_ylabel("Y (Row)")
        st.pyplot(fig2)

    # เลือก blend mode
    st.subheader("Blending Options")
    blend_mode = st.selectbox("Select Blend Mode", ["Simple Average", "Weighted Average", "Difference", "Multiply"])

    if blend_mode == "Simple Average":
        blended = (img1 + img2) / 2

    elif blend_mode == "Weighted Average":
        alpha = st.slider("Alpha (weight for Image 1)", 0.0, 1.0, 0.5)
        blended = alpha * img1 + (1 - alpha) * img2

    elif blend_mode == "Difference":
        blended = np.abs(img1 - img2)

    elif blend_mode == "Multiply":
        blended = img1 * img2

    # ปรับค่าพิกเซลไม่ให้เกินขอบเขต
    blended = np.clip(blended, 0, 1)

    # แสดง Blended Image พร้อมแกน
    st.subheader("Blended Image (with Axes)")
    fig, ax = plt.subplots()
    ax.imshow(blended)
    ax.set_title(f"Blended Image - Mode: {blend_mode}")
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)
