#mage_options = {
#    "Bulldog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
#    "Cat1": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
#    "Cat2": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
#}


import streamlit as st
from skimage import io, transform, util
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
    "Dog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
    "Cat": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
}

# -------------------------------
# UI เริ่มต้น
# -------------------------------
st.title("Interactive Image Processing with scikit-image")

st.subheader("เลือกภาพตัวอย่าง")
cols = st.columns(2)

selected_image = None
for i, (label, url) in enumerate(image_options.items()):
    with cols[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}"):
            st.session_state.selected_image = load_image_from_url(url)

# -------------------------------
# เมื่อมีการเลือกภาพ
# -------------------------------
if 'selected_image' in st.session_state:
    image = st.session_state.selected_image

    st.subheader("ภาพต้นฉบับ")
    st.image(image, use_container_width=True)

    # ----------------------------
    # Resize
    # ----------------------------
    st.subheader("Resize Image")
    resize_scale = st.slider("ปรับขนาด (0.1 = เล็กลง, 2.0 = ใหญ่ขึ้น)", 0.1, 2.0, 1.0, step=0.1)
    resized_image = transform.rescale(image, resize_scale, channel_axis=2, anti_aliasing=True)

    # ----------------------------
    # Rotate
    # ----------------------------
    st.subheader("Rotate Image")
    angle = st.slider("เลือกองศาในการหมุน", -180, 180, 0)
    rotated_image = transform.rotate(resized_image, angle)

    # ----------------------------
    # Flip
    # ----------------------------
    st.subheader("Flip Image")
    flip_option = st.selectbox("เลือกการกลับภาพ (Flip)", ["None", "Horizontal", "Vertical"])
    final_image = flip_image(rotated_image, flip_option)

    # ----------------------------
    # แสดงภาพผลลัพธ์
    # ----------------------------
    st.subheader("ผลลัพธ์ภาพหลังการแปลง")
    fig, ax = plt.subplots()
    ax.imshow(final_image)
    ax.set_title("Transformed Image")
    ax.set_axis_off()
    st.pyplot(fig)
