#https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg


import streamlit as st
from skimage import io, img_as_float
import numpy as np
import matplotlib.pyplot as plt


# ตั้งชื่อแอป
st.title("Image Processing with scikit-image")

# URL ของภาพตัวอย่าง
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"

# แสดงภาพตัวอย่างแบบ thumbnail
st.subheader("ภาพตัวอย่าง")
thumb_col = st.columns([1, 2, 1])  # จัดตรงกลาง
with thumb_col[1]:
    st.image(image_url, caption="ตัวอย่างรูปภาพ", width=200)

# เตรียม session_state เพื่อเก็บภาพ
if 'image' not in st.session_state:
    st.session_state.image = None

# ปุ่มสำหรับโหลดภาพ
if st.button("คลิกเพื่อแสดงรูปภาพ"):
    st.session_state.image = io.imread(image_url)

# ถ้ามีภาพที่โหลดแล้ว
if st.session_state.image is not None:
    image = st.session_state.image

    st.subheader("ภาพต้นฉบับที่โหลดแล้ว (พร้อมแกน X, Y)")
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Original Image")
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)

    st.subheader("แสดงบางส่วนของภาพ (Slice Image)")

    # รับ input สำหรับ slice
    row_start = st.number_input("Row start", min_value=0, max_value=image.shape[0]-1, value=0, key="row_start")
    row_end = st.number_input("Row end", min_value=row_start+1, max_value=image.shape[0], value=image.shape[0], key="row_end")
    col_start = st.number_input("Column start", min_value=0, max_value=image.shape[1]-1, value=0, key="col_start")
    col_end = st.number_input("Column end", min_value=col_start+1, max_value=image.shape[1], value=image.shape[1], key="col_end")

    # Slice ภาพ
    sliced_image = image[int(row_start):int(row_end), int(col_start):int(col_end)]

    st.subheader("ภาพบางส่วนที่เลือก")
    st.image(sliced_image, caption="ภาพบางส่วน", use_container_width=True)


# ตั้งชื่อแอป
st.title("Image Blending with scikit-image")

# URLs ของภาพ
image_urls = {
    "ภาพที่ 1": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
    "ภาพที่ 2": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
}

# แสดงภาพ thumbnail
st.subheader("ภาพตัวอย่าง")
thumb_cols = st.columns(2)
for i, (name, url) in enumerate(image_urls.items()):
    with thumb_cols[i]:
        st.image(url, caption=name, width=200)

# เตรียม session_state
if 'blend_image1' not in st.session_state:
    st.session_state.blend_image1 = None
    st.session_state.blend_image2 = None

# ปุ่มสำหรับโหลดภาพ
if st.button("โหลดและแสดงภาพเพื่อทำการ Blend"):
    st.session_state.blend_image1 = io.imread(image_urls["ภาพที่ 1"])
    st.session_state.blend_image2 = io.imread(image_urls["ภาพที่ 2"])

# ถ้ามีภาพแล้ว
if st.session_state.blend_image1 is not None and st.session_state.blend_image2 is not None:
    img1 = img_as_float(st.session_state.blend_image1)
    img2 = img_as_float(st.session_state.blend_image2)

    # ปรับขนาดให้เท่ากัน (ใช้แค่ภาพขนาดเท่ากัน)
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_height, :min_width]
    img2 = img2[:min_height, :min_width]

    st.subheader("เลือกวิธีการ Blend")

    blend_mode = st.selectbox("เลือกรูปแบบการ Blend", ["Simple Average", "Weighted Average", "Difference", "Multiply"])

    if blend_mode == "Simple Average":
        blended = (img1 + img2) / 2

    elif blend_mode == "Weighted Average":
        alpha = st.slider("ค่า Weight ของภาพที่ 1 (alpha)", 0.0, 1.0, 0.5)
        blended = alpha * img1 + (1 - alpha) * img2

    elif blend_mode == "Difference":
        blended = np.abs(img1 - img2)

    elif blend_mode == "Multiply":
        blended = img1 * img2

    blended = np.clip(blended, 0, 1)

    # แสดงผล
    st.subheader("ภาพหลังการ Blend")
    fig, ax = plt.subplots()
    ax.imshow(blended)
    ax.set_axis_off()
    st.pyplot(fig)
