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

/* DOWNLOAD PDF BUTTON */
.stDownloadButton > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: bold;
    border-radius: 10px;
    border: 1px solid #000000 !important;
}
.stDownloadButton > button:hover {
    background-color: #222222 !important;
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

    _, h, w, _ = input_details[0]["shape"]

    image = img.resize((w, h))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]["index"])[0]

# =========================================================
# MENU
# =========================================================
menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Prediksi Kaktus"])

# =========================================================
# PAGE 1: INFORMASI KAKTUS (DISAMAKAN)
# =========================================================
if menu == "Informasi Kaktus":
    st.markdown("<h1 class='stCard'>🌵 Informasi Tentang Kaktus</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='stCard'>
        <h3>Apa itu Kaktus?</h3>
        <p>
        Kaktus merupakan tanaman sukulen unik yang terkenal karena kemampuan menyimpan air dan memiliki duri
        sebagai bentuk adaptasi. Karena kemampuan tersebut, kaktus dapat bertahan hidup di lingkungan ekstrem 
        seperti gurun. Selain tangguh, kaktus juga sering dijadikan tanaman hias karena mudah dirawat dan estetik.
        </p>

        <h3>Fakta Menarik Kaktus:</h3>
        <ul>
            <li>Kaktus dapat hidup hingga ratusan tahun.</li>
            <li>Beberapa kaktus dapat tumbuh lebih dari 20 meter.</li>
            <li>Terdapat lebih dari 2.000 spesies kaktus di dunia.</li>
            <li>Bentuknya sangat beragam.</li>
        </ul>

        <h3>Kegunaan:</h3>
        <ul>
            <li>Tanaman hias</li>
            <li>Konsumsi & kesehatan</li>
            <li>Produk perawatan kulit</li>
        </ul>

        <h3>Jenis Kaktus:</h3>
        <ul>
            <li>Astrophytum Asteria</li>
            <li>Ferocactus</li>
            <li>Gymnocalycium</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 2: PREDIKSI
# =========================================================
else:
    st.markdown("<h1 class='stCard'>🔍 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload gambar kaktus", type=["jpg","jpeg","png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=280)

        preds_cnn = predict(img, cnn_interpreter)
        probs_cnn = preds_cnn / np.sum(preds_cnn)
        kelas_cnn = labels[np.argmax(probs_cnn)]
        conf_cnn = np.max(probs_cnn)

        preds_mnet = predict(img, mobilenet_interpreter)
        probs_mnet = preds_mnet / np.sum(preds_mnet)
        kelas_mnet = labels[np.argmax(probs_mnet)]
        conf_mnet = np.max(probs_mnet)

        best = "MobileNetV2" if conf_mnet > conf_cnn else "CNN"

        st.markdown(f"""
        <div class='stCard'>
        <h3>Model CNN</h3>
        <p>{kelas_cnn} ({conf_cnn:.2%})</p>
        <h3>Model MobileNetV2</h3>
        <p>{kelas_mnet} ({conf_mnet:.2%})</p>
        <h3>Kesimpulan</h3>
        <p><b>Model terbaik:</b> {best}</p>
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
        buffer = io.BytesIO()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, h - 50, "Hasil Prediksi Kaktus")

        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), 40, h - 300, 200, 200)

        c.setFont("Helvetica", 12)
        c.drawString(260, h - 120, f"CNN: {kelas_cnn} ({conf_cnn:.2%})")
        c.drawString(260, h - 150, f"MobileNetV2: {kelas_mnet} ({conf_mnet:.2%})")
        c.drawString(40, 80, f"Kesimpulan: Model terbaik adalah {best}")

        # ===== GRAFIK KE PDF =====
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, probs_cnn, alpha=0.7, label="CNN")
        ax2.bar(labels, probs_mnet, alpha=0.7, label="MobileNetV2")
        ax2.set_ylim(0,1)
        ax2.legend()

        gbuf = io.BytesIO()
        fig2.savefig(gbuf, format="PNG")
        gbuf.seek(0)
        c.drawImage(ImageReader(gbuf), 120, 120, 350, 220)

        c.save()
        buffer.seek(0)

        st.download_button(
            "📥 Download Hasil Prediksi (PDF)",
            buffer,
            file_name="hasil_prediksi_kaktus.pdf",
            mime="application/pdf"
        )
