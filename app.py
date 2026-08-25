"""
==========================================================
APP.PY
Sistem Klasifikasi Tingkat Kematangan Pepaya
CNN + TensorFlow + Streamlit

Versi:
- Streamlit 1.22.0
- TensorFlow 2.13.0
- NumPy 1.23.5
- Pillow 10.4.0

MODEL:
model_papaya.h5

Semua file:
app.py
model_papaya.h5
requirements.txt
==========================================================
"""

import os
import datetime

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ======================================================
# KONFIGURASI STREAMLIT
# ======================================================

st.set_page_config(
    page_title="Klasifikasi Tingkat Kematangan Pepaya",
    page_icon="🍈",
    layout="wide"
)


# ======================================================
# PATH MODEL
# ======================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_papaya.h5"
)

IMG_SIZE = (128, 128)


# ======================================================
# CLASS NAMES
# ======================================================

# PENTING:
# Urutan ini HARUS sama persis dengan urutan kelas
# ketika model CNN kamu ditraining.

CLASS_NAMES = [
    "Defect",
    "Mature",
    "Pre-mature",
    "Unmature"
]


CLASS_EMOJI = {
    "Defect": "🟥",
    "Mature": "🟢",
    "Pre-mature": "🟡",
    "Unmature": "⚪"
}


CLASS_CARD = {
    "Defect": "danger",
    "Mature": "success",
    "Pre-mature": "warning",
    "Unmature": "info"
}


# ======================================================
# INFORMASI PEPAYA
# ======================================================

INFO = {

    "Defect": {

        "deskripsi": (
            "Pepaya mengalami kerusakan pada kulit "
            "atau daging buah sehingga kualitasnya menurun."
        ),

        "manfaat": [
            "Tidak disarankan dikonsumsi.",
            "Dapat dijadikan kompos.",
            "Pisahkan dari buah sehat."
        ],

        "nutrisi": {
            "Vitamin C": "Menurun",
            "Vitamin A": "Menurun",
            "Serat": "Masih ada"
        },

        "rekomendasi": (
            "Sebaiknya tidak dijual sebagai buah konsumsi."
        )
    },


    "Mature": {

        "deskripsi": (
            "Pepaya telah matang sempurna dan siap dikonsumsi."
        ),

        "manfaat": [
            "Rasa manis maksimal.",
            "Siap dipasarkan.",
            "Tekstur lembut."
        ],

        "nutrisi": {
            "Vitamin C": "Tinggi",
            "Vitamin A": "Tinggi",
            "Serat": "Tinggi"
        },

        "rekomendasi": (
            "Cocok untuk dikonsumsi langsung maupun dijual."
        )
    },


    "Pre-mature": {

        "deskripsi": (
            "Pepaya mulai memasuki fase matang "
            "namun belum optimal."
        ),

        "manfaat": [
            "Masih bisa diperam.",
            "Masih layak dipasarkan.",
            "Rasa mulai manis."
        ],

        "nutrisi": {
            "Vitamin C": "Sedang",
            "Vitamin A": "Sedang",
            "Serat": "Tinggi"
        },

        "rekomendasi": (
            "Simpan 2-3 hari sebelum dikonsumsi."
        )
    },


    "Unmature": {

        "deskripsi": (
            "Pepaya masih mentah dan belum siap dikonsumsi."
        ),

        "manfaat": [
            "Bisa dijadikan sayur.",
            "Masih perlu pemeraman.",
            "Belum memiliki rasa manis."
        ],

        "nutrisi": {
            "Vitamin C": "Sedang",
            "Vitamin A": "Rendah",
            "Serat": "Tinggi"
        },

        "rekomendasi": (
            "Peram terlebih dahulu hingga matang."
        )
    }
}


# ======================================================
# CSS
# ======================================================

st.markdown(
    """
    <style>

    .result-card {
        padding: 25px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-top: 15px;
    }

    .success {
        background: #16a34a;
    }

    .warning {
        background: #ca8a04;
    }

    .info {
        background: #0284c7;
    }

    .danger {
        background: #dc2626;
    }

    .history-header {
        background-color: #f1f5f9;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #334155;
        margin-bottom: 15px;
    }

    .history-row {
        padding: 10px 0;
        border-bottom: 1px solid #e2e8f0;
    }

    .footer {
        text-align: center;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------
    # Cek keberadaan file
    # --------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ MODEL TIDAK DITEMUKAN"
        )

        st.code(
            MODEL_PATH
        )

        st.warning(
            """
Pastikan struktur repository:

app.py
model_papaya.h5
requirements.txt
            """
        )

        st.stop()


    # --------------------------------------------------
    # Cek ukuran file
    # --------------------------------------------------

    file_size = os.path.getsize(
        MODEL_PATH
    )

    file_size_mb = (
        file_size / (1024 * 1024)
    )


    st.sidebar.write(
        f"📦 Ukuran model: "
        f"{file_size_mb:.2f} MB"
    )


    if file_size < 100 * 1024:

        st.error(
            "❌ FILE MODEL TERLALU KECIL"
        )

        st.write(
            f"""
Ukuran model:

{file_size_mb:.3f} MB

File .h5 kemungkinan bukan model asli.
Kemungkinan file yang diupload ke GitHub
merupakan file pointer atau file rusak.
"""
        )

        st.stop()


    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error(
            "❌ GAGAL MEMBUKA MODEL"
        )

        st.code(
            str(e)
        )

        st.warning(
            """
Pastikan model_papaya.h5 memang model
TensorFlow/Keras yang valid dan dibuat
menggunakan arsitektur yang kompatibel
dengan TensorFlow 2.13.
"""
        )

        st.stop()


model = load_model()


# ======================================================
# SESSION STATE
# ======================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "last_label" not in st.session_state:

    st.session_state.last_label = None


# ======================================================
# VALIDASI MODEL
# ======================================================

try:

    OUTPUT_SHAPE = model.output_shape

    st.sidebar.write(
        f"🧠 Output model: {OUTPUT_SHAPE}"
    )

except Exception:

    OUTPUT_SHAPE = None


# ======================================================
# FUNGSI PREDIKSI
# ======================================================

def predict(
    model,
    image_file
):

    """
    Preprocessing:

    1. Buka gambar menggunakan Pillow
    2. RGB
    3. Resize 128x128
    4. Convert float32
    5. Normalisasi /255
    6. Tambahkan batch dimension
    7. Prediksi model
    """

    # --------------------------------------------------
    # Buka gambar
    # --------------------------------------------------

    image_file.seek(0)

    image = Image.open(
        image_file
    ).convert("RGB")


    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )


    # --------------------------------------------------
    # Convert numpy
    # --------------------------------------------------

    img_array = np.asarray(
        image,
        dtype=np.float32
    )


    # --------------------------------------------------
    # Normalisasi
    # --------------------------------------------------

    img_array = (
        img_array / 255.0
    )


    # --------------------------------------------------
    # Batch
    # --------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------
    # Prediksi
    # --------------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )


    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------
    # Pastikan output valid
    # --------------------------------------------------

    if prediction.ndim != 2:

        raise ValueError(
            f"""
Output model tidak sesuai.

Shape output:
{prediction.shape}

Model harus menghasilkan
output berbentuk (1, 4).
"""
        )


    prediction = prediction[0]


    if len(prediction) != len(
        CLASS_NAMES
    ):

        raise ValueError(
            f"""
Jumlah output model:

{len(prediction)}

Jumlah CLASS_NAMES:

{len(CLASS_NAMES)}

Pastikan model memiliki 4 kelas.
"""
        )


    # --------------------------------------------------
    # Cek apakah output logits atau probability
    # --------------------------------------------------

    prediction_sum = float(
        np.sum(prediction)
    )


    # Jika output bukan probability
    # dan nilainya tidak berada sekitar 1,
    # gunakan softmax.

    if (
        np.any(prediction < 0)
        or prediction_sum < 0.90
        or prediction_sum > 1.10
    ):

        prediction_probability = tf.nn.softmax(
            prediction
        ).numpy()

    else:

        prediction_probability = (
            prediction
        )


    # --------------------------------------------------
    # Cari kelas terbesar
    # --------------------------------------------------

    predicted_index = int(
        np.argmax(
            prediction_probability
        )
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # --------------------------------------------------
    # Probability
    # --------------------------------------------------

    probability = {}

    for i, kelas in enumerate(
        CLASS_NAMES
    ):

        probability[kelas] = (
            float(
                prediction_probability[i]
            ) * 100
        )


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    confidence = probability[
        predicted_class
    ]


    return (
        predicted_class,
        confidence,
        probability
    )


# ======================================================
# HEADER
# ======================================================

st.title(
    "🍈 Klasifikasi Tingkat Kematangan Pepaya"
)

st.write(
    "Upload gambar pepaya kemudian tekan tombol "
    "**Analisis Sekarang**."
)

st.divider()


# ======================================================
# UPLOAD
# ======================================================

uploaded = st.file_uploader(
    "Upload Gambar Pepaya",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ======================================================
# PREDIKSI
# ======================================================

if uploaded is not None:

    col1, col2 = st.columns(
        [1, 1]
    )


    # ==================================================
    # GAMBAR
    # ==================================================

    with col1:

        st.subheader(
            "Gambar"
        )

        # Streamlit 1.22.0
        # tidak menggunakan use_container_width
        st.image(
            uploaded,
            width=500
        )


    # ==================================================
    # HASIL
    # ==================================================

    with col2:

        st.subheader(
            "Hasil Prediksi"
        )


        if st.button(
            "🔍 Analisis Sekarang",
            use_container_width=True
        ):

            with st.spinner(
                "Sedang melakukan prediksi..."
            ):

                try:

                    uploaded.seek(0)

                    raw_image_data = (
                        uploaded.read()
                    )

                    uploaded.seek(0)


                    (
                        label,
                        confidence,
                        probability
                    ) = predict(
                        model,
                        uploaded
                    )


                except Exception as e:

                    st.error(
                        "❌ GAGAL MELAKUKAN PREDIKSI"
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()


            # --------------------------------------------------
            # Simpan hasil
            # --------------------------------------------------

            st.session_state.last_label = (
                label
            )


            st.session_state.history.append(
                {
                    "image": raw_image_data,

                    "label": label,

                    "confidence": (
                        f"{confidence:.2f}%"
                    ),

                    "time": (
                        datetime.datetime.now()
                        .strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                    )
                }
            )


            # --------------------------------------------------
            # RESULT CARD
            # --------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card {CLASS_CARD[label]}">

                    <h2>
                        {CLASS_EMOJI[label]} {label}
                    </h2>

                    <h3>
                        Confidence
                    </h3>

                    <h1>
                        {confidence:.2f}%
                    </h1>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write("")


            # --------------------------------------------------
            # PROBABILITAS
            # --------------------------------------------------

            st.subheader(
                "📊 Probabilitas Model CNN"
            )


            probability_sorted = dict(
                sorted(
                    probability.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )


            for kelas, persen in (
                probability_sorted.items()
            ):

                st.write(
                    f"**{CLASS_EMOJI[kelas]} {kelas}**"
                )


                st.progress(
                    min(
                        max(
                            int(persen),
                            0
                        ),
                        100
                    )
                )


                st.caption(
                    f"{persen:.2f}%"
                )


# ======================================================
# INFORMASI PEPAYA
# ======================================================

if (
    st.session_state.last_label
    is not None
):

    st.divider()

    label = (
        st.session_state.last_label
    )

    info = INFO[label]


    st.header(
        "📋 Informasi Pepaya"
    )


    tab1, tab2, tab3 = st.tabs(
        [
            "Deskripsi",
            "Nutrisi",
            "Rekomendasi"
        ]
    )


    # ==================================================
    # DESKRIPSI
    # ==================================================

    with tab1:

        st.markdown(
            info["deskripsi"]
        )

        st.write("")

        st.subheader(
            "Manfaat"
        )

        for item in info["manfaat"]:

            st.write(
                f"✅ {item}"
            )


    # ==================================================
    # NUTRISI
    # ==================================================

    with tab2:

        st.subheader(
            "Kandungan Nutrisi"
        )

        for key, value in (
            info["nutrisi"].items()
        ):

            st.write(
                f"**{key}:** {value}"
            )


    # ==================================================
    # REKOMENDASI
    # ==================================================

    with tab3:

        st.success(
            info["rekomendasi"]
        )


# ======================================================
# RIWAYAT
# ======================================================

if len(
    st.session_state.history
) > 0:

    st.divider()

    st.header(
        "📑 Riwayat Prediksi"
    )


    h_col1, h_col2, h_col3, h_col4 = (
        st.columns(
            [1.5, 2, 2, 2.5]
        )
    )


    with h_col1:

        st.markdown(
            '<div class="history-header">'
            'Gambar'
            '</div>',
            unsafe_allow_html=True
        )


    with h_col2:

        st.markdown(
            '<div class="history-header">'
            'Label'
            '</div>',
            unsafe_allow_html=True
        )


    with h_col3:

        st.markdown(
            '<div class="history-header">'
            'Confidence'
            '</div>',
            unsafe_allow_html=True
        )


    with h_col4:

        st.markdown(
            '<div class="history-header">'
            'Waktu Analisis'
            '</div>',
            unsafe_allow_html=True
        )


    for item in reversed(
        st.session_state.history
    ):

        if (
            not isinstance(item, dict)
            or "image" not in item
            or "label" not in item
        ):

            continue


        r_col1, r_col2, r_col3, r_col4 = (
            st.columns(
                [1.5, 2, 2, 2.5]
            )
        )


        with r_col1:

            st.image(
                item["image"],
                width=90
            )


        with r_col2:

            st.markdown(
                f"""
                <h5 style="
                    text-align:center;
                    margin-top:25px;
                ">

                    {CLASS_EMOJI[item["label"]]}
                    {item["label"]}

                </h5>
                """,
                unsafe_allow_html=True
            )


        with r_col3:

            st.markdown(
                f"""
                <h5 style="
                    text-align:center;
                    margin-top:25px;
                ">

                    {item["confidence"]}

                </h5>
                """,
                unsafe_allow_html=True
            )


        with r_col4:

            st.markdown(
                f"""
                <h5 style="
                    text-align:center;
                    margin-top:25px;
                ">

                    {item["time"]}

                </h5>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            '<div class="history-row"></div>',
            unsafe_allow_html=True
        )


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title(
        "🍈 Papaya Classifier"
    )


    st.metric(
        "Jumlah Prediksi",
        len(
            st.session_state.history
        )
    )


    st.write("---")


    st.write(
        "### Kelas Model"
    )


    for kelas in CLASS_NAMES:

        st.write(
            f"{CLASS_EMOJI[kelas]} {kelas}"
        )


    st.write("---")


    if st.button(
        "♻ Reset Riwayat",
        use_container_width=True
    ):

        st.session_state.history = []

        st.session_state.last_label = None

        # Untuk Streamlit 1.22.0
        st.experimental_rerun()


# ======================================================
# PAPAYA AI ASSISTANT
# ======================================================

st.divider()

st.header(
    "🤖 Papaya AI Assistant"
)


st.info(
    """
Asisten virtual berbasis Knowledge Base yang memberikan
informasi mengenai buah pepaya berdasarkan hasil
klasifikasi CNN.

✅ Tidak memerlukan internet

✅ Seluruh jawaban berasal dari basis pengetahuan
yang telah disiapkan
"""
)


# ======================================================
# CONTOH PERTANYAAN
# ======================================================

contoh = [

    "Bagaimana cara menyimpan pepaya?",

    "Bagaimana mempercepat pematangan pepaya?",

    "Berapa lama pepaya matang dapat disimpan?",

    "Bagaimana memilih pepaya yang bagus?",

    "Bolehkah pepaya dimakan ibu hamil?",

    "Apakah pepaya boleh diberikan kepada bayi?",

    "Kenapa muncul bercak hitam pada pepaya?",

    "Apakah pepaya saya sudah siap dimakan?"
]


with st.expander(
    "💡 Contoh Pertanyaan"
):

    for c in contoh:

        st.write(
            "•",
            c
        )


# ======================================================
# INPUT PERTANYAAN
# ======================================================

pertanyaan = st.text_input(
    "Silakan tuliskan pertanyaan Anda:"
)


# ======================================================
# TANYA AI
# ======================================================

if st.button(
    "🤖 Tanya AI",
    use_container_width=True
):

    tanya = (
        pertanyaan
        .lower()
        .strip()
    )


    if (
        st.session_state.last_label
        is not None
    ):

        label = (
            st.session_state.last_label
        )

    else:

        label = None


    jawaban = ""


    # ==================================================
    # HASIL PREDIKSI
    # ==================================================

    if (
        "siap dimakan" in tanya
        or "hasil" in tanya
        or "prediksi" in tanya
    ):


        if label is None:

            jawaban = """
Silakan lakukan klasifikasi terlebih dahulu
dengan mengunggah gambar pepaya.
"""


        elif label == "Mature":

            jawaban = """
🥭 Berdasarkan hasil klasifikasi CNN,
pepaya Anda termasuk **Mature**.

Pepaya sudah matang sempurna dan siap dikonsumsi.

Jika belum ingin dimakan hari ini, simpan
di lemari pendingin agar kualitasnya tetap terjaga.
"""


        elif label == "Pre-mature":

            jawaban = """
🟡 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Pre-mature**.

Pepaya hampir matang.

Sebaiknya diperam 1–3 hari lagi pada suhu ruang
hingga rasa menjadi lebih manis.
"""


        elif label == "Unmature":

            jawaban = """
⚪ Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Unmature**.

Pepaya masih mentah sehingga belum disarankan
untuk dikonsumsi.

Simpan pada suhu ruang hingga matang.
"""


        else:

            jawaban = """
🔴 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Defect**.

Buah mengalami kerusakan.

Periksa apakah terdapat jamur, bau tidak sedap,
atau tekstur berlendir.

Jika iya, sebaiknya tidak dikonsumsi.
"""


    # ==================================================
    # PENYIMPANAN
    # ==================================================

    elif "menyimpan" in tanya:

        jawaban = """
📦 Cara Penyimpanan Pepaya

• Pepaya mentah disimpan pada suhu ruang.

• Pepaya matang dapat disimpan di lemari pendingin
  agar bertahan lebih lama.

• Jangan mencuci pepaya sebelum disimpan.

• Hindari sinar matahari langsung.
"""


    # ==================================================
    # MEMPERCEPAT PEMATANGAN
    # ==================================================

    elif (
        "mempercepat" in tanya
        or "peram" in tanya
    ):

        jawaban = """
🍌 Untuk mempercepat pematangan:

• Simpan pepaya pada suhu ruang.

• Letakkan bersama buah pisang atau apel.

• Hindari memasukkan pepaya mentah ke kulkas.
"""


    # ==================================================
    # MASA SIMPAN
    # ==================================================

    elif (
        "berapa lama" in tanya
        or "tahan" in tanya
    ):

        jawaban = """
⏳ Umur Simpan Pepaya

• Suhu ruang: sekitar 2–3 hari setelah matang.

• Dalam kulkas: sekitar 5–7 hari.

Lama penyimpanan dapat berbeda tergantung kondisi buah.
"""


    # ==================================================
    # MEMILIH PEPAYA
    # ==================================================

    elif (
        "pilih" in tanya
        or "bagus" in tanya
    ):

        jawaban = """
🥭 Tips Memilih Pepaya

• Kulit berwarna kuning merata.

• Tidak terdapat luka besar.

• Tidak berbau busuk.

• Tekstur sedikit lunak saat ditekan.
"""


    # ==================================================
    # BERCAK HITAM
    # ==================================================

    elif (
        "bercak" in tanya
        or "hitam" in tanya
    ):

        jawaban = """
⚫ Bercak hitam dapat disebabkan oleh benturan,
memar, atau proses pembusukan.

Apabila bercak hanya sedikit, bagian tersebut
dapat dipotong.

Namun jika disertai bau tidak sedap dan berlendir,
sebaiknya pepaya tidak dikonsumsi.
"""


    # ==================================================
    # IBU HAMIL
    # ==================================================

    elif (
        "ibu hamil" in tanya
        or "hamil" in tanya
    ):

        jawaban = """
🤰 Untuk pertanyaan terkait kehamilan,
sebaiknya konsultasikan dengan tenaga kesehatan.

Untuk sistem ini, informasi yang diberikan
bersifat edukatif dan bukan pengganti
nasihat medis profesional.
"""


    # ==================================================
    # BAYI
    # ==================================================

    elif (
        "bayi" in tanya
        or "mpasi" in tanya
    ):

        jawaban = """
👶 Pepaya matang dapat menjadi salah satu
pilihan makanan setelah bayi mulai MPASI.

Untuk usia dan porsi yang sesuai,
ikuti rekomendasi tenaga kesehatan.
"""


    # ==================================================
    # PENCERNAAN
    # ==================================================

    elif "pencernaan" in tanya:

        jawaban = """
💚 Pepaya mengandung serat yang dapat membantu
mendukung kesehatan sistem pencernaan.
"""


    # ==================================================
    # DEFAULT
    # ==================================================

    else:

        jawaban = """
Maaf, saya belum memahami pertanyaan tersebut.

Silakan tanyakan hal seperti:

• Bagaimana cara menyimpan pepaya?

• Bagaimana mempercepat pematangan?

• Berapa lama pepaya dapat disimpan?

• Bagaimana memilih pepaya yang bagus?

• Apakah pepaya saya sudah siap dimakan?

• Kenapa muncul bercak hitam?

• Apakah boleh dimakan ibu hamil?

• Apakah boleh diberikan kepada bayi?
"""


    st.success(
        "🤖 Jawaban Papaya AI"
    )

    st.write(
        jawaban
    )


# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        <h3>
            🍈 Sistem Klasifikasi Tingkat Kematangan Pepaya
        </h3>

        <p>
            Convolutional Neural Network (CNN)
        </p>

        <p>
            TensorFlow • Streamlit
        </p>

        <br>

        <b>
            Developed by Muhammad Ghifari
        </b>

    </div>
    """,
    unsafe_allow_html=True
)
