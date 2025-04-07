import streamlit as st
from skimage import io

# ตั้งชื่อแอป
st.title("Image Processing with scikit-image")

# URL ของภาพตัวอย่าง
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"

# แสดงภาพตัวอย่างตั้งแต่ต้น (แบบ thumbnail)
st.subheader("ตัวอย่างภาพ")
st.image(image_url, caption="ภาพตัวอย่าง", width=200)

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
