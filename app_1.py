import streamlit as st
from skimage import io
import matplotlib.pyplot as plt

# ตั้งชื่อแอป
st.title("Image Processing with scikit-image")

# โหลดภาพตัวอย่าง
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"
st.subheader("ตัวอย่างรูปภาพ")

# แสดงภาพให้ผู้ใช้คลิกเพื่อแสดง
if st.button("คลิกเพื่อแสดงรูปภาพ"):
    image = io.imread(image_url)
    st.image(image, caption="ภาพต้นฉบับ", use_container_width=True)

    st.subheader("แสดงบางส่วนของภาพ (Slice Image)")

    # รับ input จากผู้ใช้เพื่อ slice ภาพ
    row_start = st.number_input("เริ่มแถว (row start)", min_value=0, max_value=image.shape[0]-1, value=0)
    row_end = st.number_input("สิ้นสุดแถว (row end)", min_value=row_start+1, max_value=image.shape[0], value=image.shape[0])
    col_start = st.number_input("เริ่มคอลัมน์ (col start)", min_value=0, max_value=image.shape[1]-1, value=0)
    col_end = st.number_input("สิ้นสุดคอลัมน์ (col end)", min_value=col_start+1, max_value=image.shape[1], value=image.shape[1])

    # Slice ภาพตามค่าที่ผู้ใช้กำหนด
    sliced_image = image[int(row_start):int(row_end), int(col_start):int(col_end)]

    # แสดงภาพที่ slice แล้ว
    st.subheader("ภาพที่แสดงบางส่วน")
    st.image(sliced_image, caption="ภาพบางส่วน", use_container_width=True)

