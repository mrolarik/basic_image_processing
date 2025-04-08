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
# ฟังก์ชัน pad ความสูง (แนวนอน) หรือความกว้าง (แนวตั้ง)
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

def pad_images_to_same_width(img1, img2):
    w1, w2 = img1.shape[1], img2.shape[1]
    if w1 > w2:
        pad = w1 - w2
        img2 = np.pad(img2, ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)
    elif w2 > w1:
        pad = w2 - w1
        img1 = np.pad(img1, ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)
    return img1, img2

# -------------------------------
# URLs ของภาพ
# -------------------------------
url_dog = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"
url_cat = "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"

# -------------------------------
# เริ่มต้น Streamlit App
# -------------------------------
st.title("การแสดงและรวมภาพแนวนอน + แนวตั้ง")

# โหลดภาพ
image1 = load_image_from_url(url_dog)
image2 = load_image_from_url(url_cat)

# 1️⃣ แสดงภาพเดียว
st.subheader("1️⃣ แสดงภาพสุนัข 1 รูป (พร้อมแกน X, Y)")
show_image_with_axes(image1, "Dog Image")

# 2️⃣ แสดงภาพสุนัขและแมว แยกคอลัมน์
st.subheader("2️⃣ แสดงภาพสุนัขและแมว (แยกกัน พร้อมแกน X, Y)")
cols = st.columns(2)
with cols[0]:
    show_image_with_axes(image1, "Dog Image")
with cols[1]:
    show_image_with_axes(image2, "Cat Image")

# 3️⃣ รวมภาพในแนวนอน
st.subheader("3️⃣ รวมภาพแนวนอน (ไม่ resize)")
img1_h, img2_h = pad_images_to_same_height(image1, image2)
combined_horizontal = np.hstack((img1_h, img2_h))
show_image_with_axes(combined_horizontal, "Dog + Cat Combined (Horizontal)")

# 4️⃣ รวมภาพในแนวตั้ง
st.subheader("4️⃣ รวมภาพแนวตั้ง (ไม่ resize)")
img1_v, img2_v = pad_images_to_same_width(image1, image2)
combined_vertical = np.vstack((img1_v, img2_v))
show_image_with_axes(combined_vertical, "Dog + Cat Combined (Vertical)")
