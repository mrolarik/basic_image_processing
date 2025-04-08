import streamlit as st
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
# ฟังก์ชันแสดงภาพพร้อมแกน
# -------------------------------
def show_image_with_axes(image, title="Image"):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)

# -------------------------------
# ฟังก์ชัน pad ภาพให้ความสูงเท่ากัน (ไม่ resize)
# -------------------------------
def pad_images_to_same_height(img1, img2):
    h1, h2 = img1.shape[0], img2.shape[0]
    if h1 > h2:
        pad = h1 - h2
        img2 = np.pad(img2, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif h2 > h1:
        pad = h2 - h1
        img1 = np.pad(img1, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
    return img1, img2

# -------------------------------
# URL ของภาพ
# -------------------------------
url_dog = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"
url_cat = "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"

# -------------------------------
# เริ่มต้น Streamlit App
# -------------------------------
st.title("การแสดงภาพด้วย matplotlib และการรวมภาพ")

# 1️⃣ แสดงภาพเดียวจาก URL พร้อมแกน
st.subheader("1️⃣ แสดงภาพสุนัข 1 รูป (พร้อมแกน X, Y)")
image1 = load_image_from_url(url_dog)
show_image_with_axes(image1, "Dog Image")

# 2️⃣ แสดงภาพสุนัขและแมวแบบแยกกัน
st.subheader("2️⃣ แสดงภาพสุนัขและแมว (แยกกัน พร้อมแกน X, Y)")
image2 = load_image_from_url(url_cat)

cols = st.columns(2)
with cols[0]:
    show_image_with_axes(image1, "Dog Image")

with cols[1]:
    show_image_with_axes(image2, "Cat Image")

# 3️⃣ รวมภาพสองภาพในแนวนอนโดยไม่ resize
st.subheader("3️⃣ รวมภาพสุนัขและแมว (ต่อแนวนอน โดยไม่ resize)")

# ทำ padding เพื่อให้ภาพสูงเท่ากัน
image1_pad, image2_pad = pad_images_to_same_height(image1, image2)

# ต่อภาพในแนวนอน
combined_image = np.hstack((image1_pad, image2_pad))

# แสดงภาพรวม
show_image_with_axes(combined_image, "Dog + Cat Combined (Horizontal)")
