import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Kaktus Classifier",
    page_icon="🌵",
    layout="centered"
)

# =========================================================
# CUSTOM THEME
# =========================================================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#2e4630,#486c4a,#6f9e72,#8fc79c);
    background-size: 200% 200%;
    animation: gradientMove 12s ease infinite;
}
@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
[data-testid="stSidebar"] {background: rgba(0,0,0,0.25);}
h1,h2,h3,p,label,li,b {color:white;}
.stCard {
    background: rgba(255,255,255,0.18);
    padding:20px;
    border-radius:14px;
    backdrop-filter: blur(6px);
}
.stButton>button {
    background:#2ecc71;
    color:white;
    border-radius:10px;
    font-weight:bold;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD TFLITE MODELS
# =========================================================
@st.cache_resource
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

cnn_interpreter = load_model("modelcnn_kaktus.tflite")
mobilenet_interpreter = load_model("mobilenetv2_kaktus.tflite")

labels = ["Astrophytum Asteria", "Ferocactus", "Gymnocalycium"]

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict(img, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ambil ukuran input dari model TFLite
    _, height, width, _ = input_details[0]["shape"]

    image = img.resize((width, height))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    return preds

# =========================================================
# MENU
# =========================================================
menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Prediksi Kaktus"])

# =========================================================
# PAGE 1
# =========================================================
if menu == "Informasi Kaktus":
    st.markdown("<h1 class='stCard'>🌵 Informasi Tentang Kaktus</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stCard'>
    <p>Kaktus adalah tanaman sukulen yang mampu menyimpan air dan bertahan di lingkungan ekstrem.</p>
    <ul>
        <li>Astrophytum Asteria</li>
        <li>Ferocactus</li>
        <li>Gymnocalycium</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 2
# =========================================================
else:
    st.markdown("<h1 class='stCard'>🔍 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=280)

        # ===== PREDICTION =====
        preds_cnn = predict(img, cnn_interpreter)
        probs_cnn = preds_cnn / np.sum(preds_cnn)
        kelas_cnn = labels[np.argmax(probs_cnn)]
        conf_cnn = np.max(probs_cnn)

        preds_mnet = predict(img, mobilenet_interpreter)
        probs_mnet = preds_mnet / np.sum(preds_mnet)
        kelas_mnet = labels[np.argmax(probs_mnet)]
        conf_mnet = np.max(probs_mnet)

        # ===== RESULT CARD =====
        st.markdown(f"""
        <div class='stCard'>
        <h3>Model CNN</h3>
        <p>{kelas_cnn} ({conf_cnn:.2%})</p>
        <h3>Model MobileNetV2</h3>
        <p>{kelas_mnet} ({conf_mnet:.2%})</p>
        <h3>Kesimpulan</h3>
        <p><b>Model terbaik:</b> {"MobileNetV2" if conf_mnet > conf_cnn else "CNN"}</p>
        </div>
        """, unsafe_allow_html=True)

        # ===== BAR CHART =====
        fig, ax = plt.subplots()
        ax.bar(labels, probs_cnn, alpha=0.7, label="CNN")
        ax.bar(labels, probs_mnet, alpha=0.7, label="MobileNetV2")
        ax.set_ylim(0,1)
        ax.legend()
        st.pyplot(fig)

        # ===== PDF =====
        # ===================== PDF GENERATOR =====================
buffer = io.BytesIO()

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader

c = canvas.Canvas(buffer, pagesize=A4)
width, height = A4

# warna
green_dark  = Color(0/255, 70/255, 32/255)
green_main  = Color(56/255, 142/255, 60/255)
green_light = Color(220/255, 240/255, 220/255)

# background
c.setFillColor(green_light)
c.rect(0, 0, width, height, fill=1)

# header
c.setFillColor(green_main)
c.rect(0, height - 100, width, 100, fill=1)

c.setFillColor(Color(1,1,1))
c.setFont("Helvetica-Bold", 24)
c.drawString(40, height - 60, "🌵 Hasil Prediksi Kaktus")

# card putih
c.setFillColor(Color(1,1,1))
c.roundRect(40, 80, width - 80, height - 220, 20, fill=1)

# gambar
img_buf = io.BytesIO()
img.save(img_buf, format="PNG")
img_buf.seek(0)
c.drawImage(ImageReader(img_buf), 60, height - 420, 220, 220)

# teks hasil
c.setFillColor(green_dark)
c.setFont("Helvetica-Bold", 16)
c.drawString(320, height - 240, "Model CNN")
c.setFont("Helvetica", 13)
c.drawString(320, height - 260, f"Prediksi : {kelas_cnn}")
c.drawString(320, height - 280, f"Confidence : {conf_cnn:.2%}")

c.setFont("Helvetica-Bold", 16)
c.drawString(320, height - 320, "Model MobileNetV2")
c.setFont("Helvetica", 13)
c.drawString(320, height - 340, f"Prediksi : {kelas_mnet}")
c.drawString(320, height - 360, f"Confidence : {conf_mnet:.2%}")

# kesimpulan
c.setFont("Helvetica-Bold", 14)
best = "MobileNetV2" if conf_mnet > conf_cnn else "CNN"
c.drawString(60, 130, f"Kesimpulan: Model terbaik adalah {best}")

# footer
c.setFont("Helvetica-Oblique", 9)
c.drawString(40, 50, "Generated by Kaktus Classifier App")

# SIMPAN PDF (INI WAJIB)
c.save()
buffer.seek(0)

# tombol download
st.download_button(
    "📥 Download Hasil Prediksi (PDF)",
    buffer,
    file_name="hasil_prediksi_kaktus.pdf",
    mime="application/pdf"
)
