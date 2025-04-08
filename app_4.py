import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage.transform import resize

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
# ฟังก์ชัน pad ความสูง / ความกว้าง
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
# ฟังก์ชัน resize ภาพให้มีขนาดเท่ากัน
# -------------------------------
def resize_to_same_shape(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    shape = (h, w, 3)
    img1_resized = resize(img1, shape, anti_aliasing=True)
    img2_resized = resize(img2, shape, anti_aliasing=True)
    return img1_resized, img2_resized

# -------------------------------
# URLs ของภาพ
# -------------------------------
url_dog = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"
url_cat = "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"

# -------------------------------
# เริ่มต้นแอป
# -------------------------------
st.title("การแสดงและรวมภาพแนวนอน + แนวตั้ง")

# โหลดภาพ
image1 = load_image_from_url(url_dog)
image2 = load_image_from_url(url_cat)

# 1️⃣ แสดงภาพเดียว
st.subheader("1️⃣ แสดงภาพสุนัข 1 รูป (พร้อมแกน X, Y)")
show_image_with_axes(image1, "Dog Image")

# 2️⃣ แสดงภาพสุนัขและแมว
st.subheader("2️⃣ แสดงภาพสุนัขและแมว (แยกกัน พร้อมแกน X, Y)")
cols = st.columns(2)
with cols[0]:
    show_image_with_axes(image1, "Dog Image")
with cols[1]:
    show_image_with_axes(image2, "Cat Image")

# 3️⃣ รวมภาพแนวนอน (ไม่ resize)
st.subheader("3️⃣ รวมภาพแนวนอน (ไม่ resize)")
img1_pad_h, img2_pad_h = pad_images_to_same_height(image1, image2)
combined_h = np.hstack((img1_pad_h, img2_pad_h))
show_image_with_axes(combined_h, "Dog + Cat Combined (Horizontal, No Resize)")

# 4️⃣ รวมภาพแนวตั้ง (ไม่ resize)
st.subheader("4️⃣ รวมภาพแนวตั้ง (ไม่ resize)")
img1_pad_v, img2_pad_v = pad_images_to_same_width(image1, image2)
combined_v = np.vstack((img1_pad_v, img2_pad_v))
show_image_with_axes(combined_v, "Dog + Cat Combined (Vertical, No Resize)")

# 5️⃣ รวมภาพแนวนอน (resize ให้เท่ากัน)
st.subheader("5️⃣ รวมภาพแนวนอน (resize ให้เท่ากัน)")
img1_resized, img2_resized = resize_to_same_shape(image1, image2)
combined_h_resized = np.hstack((img1_resized, img2_resized))
show_image_with_axes(combined_h_resized, "Dog + Cat Combined (Horizontal, Resized)")

# 6️⃣ รวมภาพแนวตั้ง (resize ให้เท่ากัน)
st.subheader("6️⃣ รวมภาพแนวตั้ง (resize ให้เท่ากัน)")
combined_v_resized = np.vstack((img1_resized, img2_resized))
show_image_with_axes(combined_v_resized, "Dog + Cat Combined (Vertical, Resized)")
