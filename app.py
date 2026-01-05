import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Judul
st.title("🌵 Kaktus Classifier - Image Recognition App")
st.write("Upload gambar kaktus untuk mengetahui jenisnya.")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_kaktus.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Daftar nama kelas 
class_names = ["Astrophytum asteria", "Ferocactus", "Gymnocalycium"]

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar kaktus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diupload", use_column_width=True)

    # Preprocess image (sesuaikan ukuran input model kamu)
    img = img.resize((150, 150))  # ganti jika modelmu pakai ukuran lain
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    # Output
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🌵 Jenis kaktus: **{pred_class}**")
    st.write(f"📊 Akurasi prediksi: **{confidence:.2f}%**")
