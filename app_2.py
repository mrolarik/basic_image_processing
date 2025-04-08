import streamlit as st
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# ฟังก์ชันโหลดภาพจาก URL
# -------------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

# -------------------------------
# ฟังก์ชัน Flip
# -------------------------------
def flip_image(image, direction):
    if direction == "Horizontal":
        return np.fliplr(image)
    elif direction == "Vertical":
        return np.flipud(image)
    else:
        return image

# -------------------------------
# URLs ของภาพตัวอย่าง
# -------------------------------
image_options = {
    "Cat": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
    "Goat": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Hausziege_04.jpg"
}

# -------------------------------
# ส่วนแสดง UI
# -------------------------------
st.title("Interactive Image Processing with scikit-image")

st.subheader("เลือกภาพตัวอย่าง")
cols = st.columns(2)

for i, (label, url) in enumerate(image_options.items()):
    with cols[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}"):
            st.session_state.selected_image = load_image_from_url(url)

# -------------------------------
# ถ้ามีการเลือกภาพแล้ว
# -------------------------------
if 'selected_image' in st.session_state:
    image = st.session_state.selected_image

    # แสดงภาพต้นฉบับพร้อมแกน
    st.subheader("ภาพต้นฉบับ (Original Image with Axes)")
    fig_orig, ax_orig = plt.subplots()
    ax_orig.imshow(image)
    ax_orig.set_title("Original Image")
    ax_orig.set_xlabel("X (Column)")
    ax_orig.set_ylabel("Y (Row)")
    st.pyplot(fig_orig)

    # ----------------------------
    # Resize
    # ----------------------------
    st.subheader("ปรับขนาด (Resize Image)")
    resize_scale = st.slider("ปรับขนาด (0.1 = เล็กลง, 2.0 = ใหญ่ขึ้น)", 0.1, 2.0, 1.0, step=0.1)
    resized_image = transform.rescale(image, resize_scale, channel_axis=2, anti_aliasing=True)

    # ----------------------------
    # Rotate
    # ----------------------------
    st.subheader("หมุนภาพ (Rotate Image)")
    angle = st.slider("เลือกองศาในการหมุน", -180, 180, 0)
    rotated_image = transform.rotate(resized_image, angle)

    # ----------------------------
    # Flip
    # ----------------------------
    st.subheader("กลับภาพ (Flip Image)")
    flip_option = st.selectbox("เลือกการกลับภาพ", ["None", "Horizontal", "Vertical"])
    final_image = flip_image(rotated_image, flip_option)

    # ----------------------------
    # แสดงภาพผลลัพธ์พร้อมแกน
    # ----------------------------
    st.subheader("ผลลัพธ์ภาพหลังการแปลง (Transformed Image with Axes)")
    fig, ax = plt.subplots()
    ax.imshow(final_image)
    ax.set_title("Transformed Image")
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)
