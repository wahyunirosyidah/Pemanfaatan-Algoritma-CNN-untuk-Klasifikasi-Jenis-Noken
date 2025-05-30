import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io

# Load model
model = load_model('mobilenetv2.h5')  # Sesuaikan nama file model
classes = ['Bitu Agia', 'Junum Ese']  # Ganti dengan kelas yang kamu pakai

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Ukuran gambar sesuai input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def resize_display_image(img, max_size=300):
    img = img.copy()
    img.thumbnail((max_size, max_size))  # Resize dengan menjaga rasio
    return img


# Judul aplikasi
st.title("Klasifikasi Gambar Noken")
st.write("Pilih metode input gambar untuk mengklasifikasikan:")

# Pilih metode input
option = st.radio("Pilih metode input:", ["📷 Ambil Foto", "📁 Upload Gambar"])

if option == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # image = Image.open(uploaded_file).convert("RGB")
        # # st.image(image, caption="Gambar yang diupload", use_column_width=True)
        # st.image(image, caption="Gambar yang diupload", use_column_width=True)

        image = Image.open(uploaded_file).convert("RGB")
        display_img = resize_display_image(image)
        st.image(display_img, caption="Gambar yang diupload")
        
        input_array = preprocess_image(image)
        prediction = model.predict(input_array)[0][0]
        label = classes[1] if prediction > 0.5 else classes[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Prediksi: `{label}`")
        st.markdown(f"**Confidence**: {confidence:.2%}")

elif option == "📷 Ambil Foto":
    camera_image = st.camera_input("Ambil foto dengan kamera")
    if camera_image is not None:
        # image = Image.open(camera_image).convert("RGB")
        # st.image(image, caption="Foto dari kamera", use_column_width=True)

        image = Image.open(camera_image).convert("RGB")
        display_img = resize_display_image(image)
        st.image(display_img, caption="Foto dari kamera")

        input_array = preprocess_image(image)
        prediction = model.predict(input_array)[0][0]
        label = classes[1] if prediction > 0.5 else classes[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.markdown(f"### Prediksi: `{label}`")
        st.markdown(f"**Confidence**: {confidence:.2%}")
