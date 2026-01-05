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

    img = img.resize((150,150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")

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
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.colors import Color

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        c.setFillColor(Color(0.9,1,0.9))
        c.rect(0,0,width,height,fill=1)

        c.setFont("Helvetica-Bold",20)
        c.drawString(40,height-60,"Hasil Prediksi Kaktus")

        c.setFont("Helvetica",14)
        y = height-120
        c.drawString(40,y,f"CNN: {kelas_cnn} ({conf_cnn:.2%})")
        y-=25
        c.drawString(40,y,f"MobileNetV2: {kelas_mnet} ({conf_mnet:.2%})")
        y-=30
        c.drawString(40,y,"Model terbaik: " + ("MobileNetV2" if conf_mnet > conf_cnn else "CNN"))

        c.save()
        buffer.seek(0)

        st.download_button(
            "📥 Download PDF",
            data=buffer,
            file_name="hasil_prediksi_kaktus.pdf",
            mime="application/pdf"
        )
