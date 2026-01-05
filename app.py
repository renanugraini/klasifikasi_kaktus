import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
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
# CUSTOM THEME (GREEN CACTUS + DARK OVERLAY)
# =========================================================

page_bg = """
<style>

/* ===== Premium Green Gradient Background ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #2e4630 0%,
        #486c4a 35%,
        #6f9e72 70%,
        #8fc79c 100%
    ) !important;
    background-size: 200% 200%;
    animation: gradientMove 12s ease infinite;
}

/* Animasi halus (biar keliatan mahal) */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}



/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25) !important;
    backdrop-filter: blur(4px);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ===== ALL TEXT COLOR (biar terlihat) ===== */
h1, h2, h3, h4, h5, h6,
p, label, li, strong, b {
    color: #ffffff !important;
}

/* ===== FILE UPLOADER LABEL ===== */
.stFileUploader > label {
    color: #ffffff !important;
    font-weight: bold;
}

/* ===== Kotak “Card” (semi transparan) ===== */
.stCard {
    background: rgba(255,255,255,0.18) !important;
    padding: 20px;
    border-radius: 14px;
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

/* ===== Input teks dan selectbox ===== */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    color: #ffffff !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background-color: #2ecc71 !important;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: 1px solid #27ae60;
}
.stButton>button:hover {
    background-color: #27ae60 !important;
}

/* ===== Buat ul / li terlihat ===== */
ul li {
    color: #ffffff !important;
    font-size: 16px;
}

/* ===== FORCE VISIBILITY UNTUK FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.25) !important;
    padding: 15px !important;
    border-radius: 12px !important;
}

/* ===== BACKGROUND GELAP UNTUK H3 YANG DI DALAM CARD ===== */
.stCard h3 {
    background: rgba(0,0,0,0.20) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    display: inline-block;
    color: #ffffff !important;
}

/* ===== FIX TANPA KOTAK DI ICON UPLOAD ===== */

/* Hilangkan fill di elemen-elemen icon */
[data-testid="stFileUploaderDropzone"] svg rect,
[data-testid="stFileUploaderDropzone"] svg path,
[data-testid="stFileUploaderDropzone"] svg polygon,
[data-testid="stFileUploaderDropzone"] svg line,
[data-testid="stFileUploaderDropzone"] svg circle {
    fill: none !important;
}

/* Styling icon upload */
[data-testid="stFileUploaderDropzone"] svg {
    stroke: #000000 !important;
    background: transparent !important;
    width: 40px !important;
    height: 40px !important;
    margin-bottom: 10px !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ===== Styling Button Download PDF ===== */
.stDownloadButton > button {
    background-color: #000000 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
    border: 1px solid #000000 !important;
}
.stDownloadButton > button:hover {
    background-color: #fffffff !important;
}


</style>

"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD TFLITE MODEL
# =========================================================
@st.cache_resource
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

cnn_interpreter = load_model("cnn_kaktus.tflite")
mobilenet_interpreter = load_model("mobilenetv2_kaktus.tflite")

# label kelas
labels = ["Astrophytum Asteria", "Ferocactus", "Gymnocalycium"]

# =========================================================
# FUNCTION PREDIKSI
# =========================================================
def predict(img, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = img.resize((150,150))
    arr = np.array(image)/255.0
    arr = np.expand_dims(arr, axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    return preds

# =========================================================
# HALAMAN MENU
# =========================================================

menu = st.sidebar.radio("Navigasi", ["Informasi Kaktus", "Prediksi Kaktus"])

# =========================================================
# PAGE 1: INFORMASI KAKTUS
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
        Tanaman ini termasuk dalam keluarga <i>Cactaceae</i>.
        </p>
        
<h3>Fakta Menarik Kaktus:</h3>    
<ul>
    <li>Kaktus dapat hidup hingga ratusan tahun.</li>
    <li>Beberapa kaktus dapat tumbuh lebih dari 20 meter.</li>
    <li>Terdapat lebih dari 2.000 spesies kaktus di dunia.</li>
    <li>Bentuknya sangat beragam: bulat, pipih, memanjang, hingga bercabang.</li>
</ul>

<h3>Kegunaan:</h3>
<ul> 
    <li>Tanaman hias: sebagai dekorasi rumah, taman, atau kamar tidur karena estetika dan perawatannya mudah.</li>
    <li>Konsumsi & Kesehatan: Buah dan daun muda kaktus (seperti pir berduri) bisa dimakan, kaya serat, vitamin, mineral untuk kesehatan.</li>
    <li>Bisa juga digunakan dalam produk perawatan kulit.</li>
</ul>

<h3>Jenis Kaktus Tanaman Hias:</h3>
<ul>
    <li>Astrophytum Asteria.</li>
    <li>Ferocactus.</li>
    <li>Gymnocalycium.</li>
</ul>

""", unsafe_allow_html=True)

# =========================================================
# PAGE 2: PREDIKSI KAKTUS
# =========================================================
elif menu == "Prediksi Kaktus":
    st.markdown("<h1 class='stCard'>🔍 Prediksi Jenis Kaktus</h1>", unsafe_allow_html=True)
    st.write("Upload gambar kaktus untuk diklasifikasikan menggunakan model CNN.")

    uploaded = st.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        # Prediksi CNN
preds_cnn = predict(img, cnn_interpreter)
probs_cnn = preds_cnn / np.sum(preds_cnn)
kelas_cnn = labels[np.argmax(probs_cnn)]
conf_cnn = np.max(probs_cnn)

# Prediksi MobileNetV2
preds_mnet = predict(img, mobilenet_interpreter)
probs_mnet = preds_mnet / np.sum(preds_mnet)
kelas_mnet = labels[np.argmax(probs_mnet)]
conf_mnet = np.max(probs_mnet)

        st.markdown("<h3 style='text-align:center;'>Gambar yang diupload</h3>", unsafe_allow_html=True)
        st.image(img, width=280, caption="Preview", use_container_width=False)

      st.markdown(
    f"""
    <div class='stCard'>
        <h2>Hasil Prediksi</h2>

        <h3>Model CNN</h3>
        <p><b>Prediksi:</b> {kelas_cnn}</p>
        <p><b>Confidence:</b> {conf_cnn:.2%}</p>

        <h3>Model MobileNetV2</h3>
        <p><b>Prediksi:</b> {kelas_mnet}</p>
        <p><b>Confidence:</b> {conf_mnet:.2%}</p>
    </div>
    """,
    unsafe_allow_html=True
)

        # ===== BAR CHART =====
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(labels, probs)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probabilitas")
        ax.set_title("Probabilitas per Kelas")
        st.pyplot(fig)

        # ===== GENERATE PDF TEMA HIJAU + GRAFIK =====
        buffer = io.BytesIO()
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.colors import Color
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # ===== Warna Tema =====
        green_dark  = Color(0/255, 70/255, 32/255)
        green_main  = Color(56/255, 142/255, 60/255)
        green_light = Color(200/255, 230/255, 201/255)

        # ===== Background =====
        c.setFillColor(green_light)
        c.rect(0, 0, width, height, fill=1)

        # ===== HEADER =====
        c.setFillColor(green_main)
        c.rect(0, height - 120, width, 120, fill=1)

        c.setFillColor(Color(1,1,1))
        c.setFont("Helvetica-Bold", 28)
        c.drawString(40, height - 70, "🌵 Hasil Prediksi Kaktus")

        # ===== Card Utama =====
        c.setFillColor(Color(1,1,1))
        c.roundRect(40, 80, width - 80, height - 220, 20, fill=1)

        # ===== Gambar Kaktus =====
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        cactus_img = ImageReader(img_bytes)

        c.drawImage(cactus_img, 60, height - 450, width=220, height=220)

       # ===== HASIL MODEL CNN =====
c.setFillColor(green_dark)
c.setFont("Helvetica-Bold", 20)
c.drawString(300, height - 250, "Hasil Model CNN")

c.setFont("Helvetica", 14)
y = height - 280
c.drawString(300, y, f"Prediksi: {kelas_cnn}")
y -= 20

for lbl, prob in zip(labels, probs_cnn):
    c.drawString(300, y, f"- {lbl}: {prob:.4f}")
    y -= 18


# ===== HASIL MODEL MobileNetV2 =====
c.setFont("Helvetica-Bold", 20)
y -= 10
c.drawString(300, y, "Hasil Model MobileNetV2")

c.setFont("Helvetica", 14)
y -= 30
c.drawString(300, y, f"Prediksi: {kelas_mnet}")
y -= 20

for lbl, prob in zip(labels, probs_mnet):
    c.drawString(300, y, f"- {lbl}: {prob:.4f}")
    y -= 18

# ===== KESIMPULAN =====
y -= 20
c.setFont("Helvetica-Bold", 18)

if conf_mnet > conf_cnn:
    c.drawString(300, y, "Kesimpulan: Model terbaik adalah MobileNetV2")
else:
    c.drawString(300, y, "Kesimpulan: Model terbaik adalah CNN")

        # ===== GRAFIK PROBABILITAS KE PDF =====
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax.bar(labels, probs_cnn, alpha=0.7, label="CNN")
        ax.bar(labels, probs_mnet, alpha=0.7, label="MobileNetV2")
        ax.legend()
        ax2.set_ylim(0,1)
        ax2.set_ylabel("Probabilitas")
        ax2.set_title("Grafik Probabilitas Kelas")

        graph_buf = io.BytesIO()
        fig2.savefig(graph_buf, format="PNG", bbox_inches="tight")
        graph_buf.seek(0)
        graph_img = ImageReader(graph_buf)

        # Tempel grafik ke PDF
        c.drawImage(graph_img, 140, 120, width=300, height=200)

        # ===== Footer =====
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(green_dark)
        c.drawString(40, 60, "Generated by Kaktus Classifier App")

        c.save()
        buffer.seek(0)

        # ===== BUTTON DOWNLOAD =====
        st.download_button(
            label="📥 Download Hasil Prediksi (PDF)",
            data=buffer,
            file_name="hasil_prediksi_kaktus.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
