import streamlit as st
from skimage import io
import matplotlib.pyplot as plt

# URLs ของภาพ
url_dog = "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg"
url_cat = "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg"

# ฟังก์ชันแสดงภาพพร้อมแกน x, y
def show_image_with_axes(image, title="Image"):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    ax.set_xlabel("X (Column)")
    ax.set_ylabel("Y (Row)")
    st.pyplot(fig)

# -------------------------------
st.title("แสดงภาพจาก URL ด้วย scikit-image และ matplotlib")

# 1️⃣ แสดงภาพเดียวจาก URL พร้อมแกน
st.subheader("1️⃣ แสดงภาพสุนัข 1 รูป (พร้อมแกน X, Y)")
image1 = io.imread(url_dog)
show_image_with_axes(image1, "Dog Image")

# 2️⃣ แสดงภาพ 2 รูป (สุนัขและแมว)
st.subheader("2️⃣ แสดงภาพสุนัขและแมว (พร้อมแกน X, Y)")

image2 = io.imread(url_cat)

cols = st.columns(2)
with cols[0]:
    show_image_with_axes(image1, "Dog Image")

with cols[1]:
    show_image_with_axes(image2, "Cat Image")
