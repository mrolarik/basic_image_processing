#image_options = {
#    "Dog": "https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg",
#    "Cat": "https://cdn.britannica.com/39/226539-050-D21D7721/Portrait-of-a-cat-with-whiskers-visible.jpg"
#}


import streamlit as st
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# โหลดภาพจาก URL ด้วย PIL
# -------------------------------
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(img)

# -------------------------------
# ฟังก์ชันแสดงภาพแยกค่าสี
# -------------------------------
def show_channel_image(channel_data, title, cmap='gray'):
    fig, ax = plt.subplots()
    ax.imshow(channel_data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

# -------------------------------
# URLs ของภาพตัวอย่าง
# -------------------------------
image_urls = {
    "Cat": "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
    "Goat": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Hausziege_04.jpg"
}

# -------------------------------
# ส่วนแสดง thumbnail
# -------------------------------
st.title("Image Processing: RGB to YCrCb Channel Viewer")

st.subheader("เลือกรูปภาพที่ต้องการประมวลผล")
cols = st.columns(2)
for i, (label, url) in enumerate(image_urls.items()):
    with cols[i]:
        st.image(url, caption=label, width=200)
        if st.button(f"เลือก {label}"):
            st.session_state.selected_image = load_image_from_url(url)

# -------------------------------
# ประมวลผลเมื่อมีภาพถูกเลือก
# -------------------------------
if 'selected_image' in st.session_state:
    image = st.session_state.selected_image

    st.subheader("1️⃣ แสดงค่าสีในระบบ RGB")

    # แยก R, G, B
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # แสดงภาพแยกแต่ละ channel
    rgb_cols = st.columns(3)
    with rgb_cols[0]:
        show_channel_image(R, "R (Red)")
        st.write("ตารางค่า R")
        st.dataframe(pd.DataFrame(R))

    with rgb_cols[1]:
        show_channel_image(G, "G (Green)")
        st.write("ตารางค่า G")
        st.dataframe(pd.DataFrame(G))

    with rgb_cols[2]:
        show_channel_image(B, "B (Blue)")
        st.write("ตารางค่า B")
        st.dataframe(pd.DataFrame(B))

    # แปลง RGB → YCrCb
    ycrcb = color.rgb2ycbcr(image)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    st.subheader("2️⃣ แสดงค่าสีในระบบ YCrCb")

    ycrcb_cols = st.columns(3)
    with ycrcb_cols[0]:
        show_channel_image(Y, "Y (Luminance)")
        st.write("ตารางค่า Y")
        st.dataframe(pd.DataFrame(Y.astype(int)))

    with ycrcb_cols[1]:
        show_channel_image(Cr, "Cr (Red-difference)")
        st.write("ตารางค่า Cr")
        st.dataframe(pd.DataFrame(Cr.astype(int)))

    with ycrcb_cols[2]:
        show_channel_image(Cb, "Cb (Blue-difference)")
        st.write("ตารางค่า Cb")
        st.dataframe(pd.DataFrame(Cb.astype(int)))
