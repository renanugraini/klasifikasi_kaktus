import streamlit as st
import numpy as np
import json
from PIL import Image
import tflite_runtime.interpreter as tflite

# =========================
# Load Labels
# =========================
with open("labels.json", "r") as f:
    labels = json.load(f)

# =========================
# Load TFLite Models
# =========================
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

cnn_interpreter = load_tflite_model("cnn_kaktus.tflite")
mobilenet_interpreter = load_tflite_model("mobilenetv2_kaktus.tflite")

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# =========================
# Prediction Function
# =========================
def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output)

    return labels[str(class_index)]

# =========================
# Streamlit UI
# =========================
st.title("Klasifikasi Jenis Kaktus 🌵")
st.write("Perbandingan Model CNN dan MobileNetV2")

model_choice = st.selectbox(
    "Pilih Model",
    ("CNN", "MobileNetV2")
)

uploaded_file = st.file_uploader(
    "Upload gambar kaktus",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    img_array = preprocess_image(image)

    if model_choice == "CNN":
        result = predict(img_array, cnn_interpreter)
    else:
        result = predict(img_array, mobilenet_interpreter)

    st.success(f"Hasil Prediksi: **{result}**")
