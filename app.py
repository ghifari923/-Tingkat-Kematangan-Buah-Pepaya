"""
==========================================================
APP.PY - FINAL FIX
Sistem Klasifikasi Tingkat Kematangan Pepaya
CNN + TensorFlow + Streamlit

FIX:
1. Model menggunakan file model_papaya.h5 di folder yang sama
2. Tidak menggunakan use_column_width
3. Tidak menggunakan HTML untuk kartu hasil
4. Preprocessing dibuat konsisten dengan model
5. Urutan kelas dibuat eksplisit
6. Menampilkan probabilitas asli model
7. Riwayat prediksi
8. Papaya AI Assistant offline
==========================================================
"""

import os
import datetime
import numpy as np
import streamlit as st
import tensorflow as tf


# ==========================================================
# KONFIGURASI STREAMLIT
# ==========================================================

st.set_page_config(
    page_title="Klasifikasi Tingkat Kematangan Pepaya",
    page_icon="🍈",
    layout="wide"
)


# ==========================================================
# PATH MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_papaya.h5"
)

IMG_SIZE = (128, 128)


# ==========================================================
# URUTAN KELAS MODEL
# ==========================================================
#
# PENTING:
# Urutan ini HARUS sama dengan urutan kelas ketika model
# dilatih.
#
# Jika saat training menggunakan:
#
# train/
# ├── Defect
# ├── Mature
# ├── Pre-mature
# └── Unmature
#
# maka urutan alfabetisnya adalah:
#
# 0 = Defect
# 1 = Mature
# 2 = Pre-mature
# 3 = Unmature
#
# ==========================================================

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


# ==========================================================
# INFORMASI KELAS
# ==========================================================

INFO = {

    "Defect": {
        "deskripsi":
            "Pepaya mengalami kerusakan pada kulit atau daging "
            "buah sehingga kualitasnya menurun.",

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

        "rekomendasi":
            "Sebaiknya tidak dijual sebagai buah konsumsi."
    },


    "Mature": {
        "deskripsi":
            "Pepaya telah matang sempurna dan siap dikonsumsi.",

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

        "rekomendasi":
            "Cocok untuk dikonsumsi langsung maupun dijual."
    },


    "Pre-mature": {
        "deskripsi":
            "Pepaya mulai memasuki fase matang namun belum optimal.",

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

        "rekomendasi":
            "Simpan 2-3 hari sebelum dikonsumsi."
    },


    "Unmature": {
        "deskripsi":
            "Pepaya masih mentah dan belum siap dikonsumsi.",

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

        "rekomendasi":
            "Peram terlebih dahulu hingga matang."
    }
}


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
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

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ Model tidak ditemukan.\n\n"
            f"Path yang dicari:\n{MODEL_PATH}"
        )

        st.stop()


    # Cek ukuran file
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    if file_size_mb < 0.1:

        st.error(
            f"❌ File model terlalu kecil: "
            f"{file_size_mb:.3f} MB\n\n"
            "Kemungkinan file model tidak ter-upload dengan benar."
        )

        st.stop()


    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error(
            "❌ Gagal membuka model.\n\n"
            f"Error:\n{e}"
        )

        st.stop()


model = load_model()


# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []


if "last_label" not in st.session_state:
    st.session_state.last_label = None


if "last_probability" not in st.session_state:
    st.session_state.last_probability = None


# ==========================================================
# DETEKSI PREPROCESSING MODEL
# ==========================================================

def model_has_rescaling_layer(model):

    """
    Mengecek apakah model mempunyai layer Rescaling.

    Kalau model sudah mempunyai Rescaling, jangan membagi
    gambar dengan 255 lagi karena bisa terjadi double scaling.
    """

    for layer in model.layers:

        layer_name = layer.__class__.__name__.lower()

        if "rescaling" in layer_name:

            return True

    return False


HAS_RESCALING = model_has_rescaling_layer(model)


# ==========================================================
# INFORMASI MODEL DI SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🍈 Papaya Classifier")

    st.write("")

    st.metric(
        "Jumlah Prediksi",
        len(st.session_state.history)
    )

    st.write("---")

    st.write("### Kelas Model")

    for i, kelas in enumerate(CLASS_NAMES):

        st.write(
            f"{i} — {CLASS_EMOJI[kelas]} {kelas}"
        )

    st.write("---")

    st.write("### Informasi Model")

    st.caption(
        f"Input Model: {model.input_shape}"
    )

    st.caption(
        f"Output Model: {model.output_shape}"
    )

    if HAS_RESCALING:

        st.success(
            "Preprocessing: Rescaling di dalam model"
        )

    else:

        st.info(
            "Preprocessing: Input gambar digunakan "
            "tanpa Rescaling layer."
        )

    st.write("---")

    if st.button(
        "♻ Reset Riwayat",
        use_container_width=True
    ):

        st.session_state.history = []

        st.session_state.last_label = None

        st.session_state.last_probability = None

        st.rerun()


# ==========================================================
# FUNGSI PREPROCESSING
# ==========================================================

def preprocess_image(image_file):

    """
    Preprocessing gambar dibuat sedekat mungkin dengan
    penggunaan model di lokal.

    Tidak menggunakan smart threshold.
    Tidak mengubah probabilitas model.
    """

    image_file.seek(0)

    image_bytes = image_file.read()

    # Decode JPG / PNG
    img = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    # Pastikan tipe float32
    img = tf.cast(img, tf.float32)

    # Resize ke 128x128
    img = tf.image.resize(
        img,
        IMG_SIZE,
        method="bilinear"
    )

    # ======================================================
    # PENTING
    # ======================================================
    #
    # Kalau model memiliki Rescaling layer:
    #
    # gambar jangan dibagi 255 di sini.
    #
    # Karena model sendiri yang akan melakukan normalisasi.
    #
    # Kalau model TIDAK mempunyai Rescaling layer:
    #
    # kita pertahankan input raw 0-255.
    #
    # Ini sengaja dibuat demikian karena versi lokal kamu
    # yang menggunakan model .h5 tersebut sudah menghasilkan
    # klasifikasi yang benar.
    #
    # ======================================================

    img = tf.expand_dims(
        img,
        axis=0
    )

    return img


# ==========================================================
# FUNGSI PREDIKSI
# ==========================================================

def predict(model, image_file):

    img_array = preprocess_image(
        image_file
    )

    # Prediksi ASLI model
    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )

    # ======================================================
    # Jika output bukan probabilitas 0-1
    # ======================================================

    # Untuk klasifikasi softmax biasanya jumlahnya sekitar 1.
    # Kita tidak mengubah hasil kalau memang sudah probability.
    #
    # Argmax tetap digunakan untuk menentukan kelas.
    # ======================================================

    predicted_index = int(
        np.argmax(prediction)
    )

    # Validasi jumlah output
    if len(prediction) != len(CLASS_NAMES):

        raise ValueError(
            "Jumlah output model tidak sama dengan jumlah "
            f"CLASS_NAMES.\n\n"
            f"Output model: {len(prediction)}\n"
            f"CLASS_NAMES: {len(CLASS_NAMES)}"
        )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # ======================================================
    # PROBABILITAS
    # ======================================================

    # Kalau output model berupa probability softmax
    # langsung dikali 100.
    #
    # Kalau output ternyata logits, ubah menggunakan softmax.
    # ======================================================

    if (
        np.min(prediction) < 0
        or np.max(prediction) > 1
        or not np.isclose(
            np.sum(prediction),
            1.0,
            atol=0.05
        )
    ):

        prediction_probability = tf.nn.softmax(
            prediction
        ).numpy()

    else:

        prediction_probability = prediction


    probability = {

        CLASS_NAMES[i]:
        float(prediction_probability[i]) * 100

        for i in range(
            len(CLASS_NAMES)
        )
    }


    confidence = probability[
        predicted_class
    ]


    return (
        predicted_class,
        confidence,
        probability
    )


# ==========================================================
# HEADER UTAMA
# ==========================================================

st.title(
    "🍈 Klasifikasi Tingkat Kematangan Pepaya"
)

st.write(
    "Upload gambar pepaya kemudian tekan tombol "
    "**Analisis Sekarang**."
)

st.divider()


# ==========================================================
# UPLOAD GAMBAR
# ==========================================================

uploaded = st.file_uploader(
    "Upload Gambar Pepaya",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================================
# JIKA ADA GAMBAR
# ==========================================================

if uploaded is not None:

    col1, col2 = st.columns(
        [1, 1]
    )


    # ======================================================
    # GAMBAR
    # ======================================================

    with col1:

        st.subheader(
            "🖼️ Gambar"
        )

        # Tidak menggunakan use_column_width
        st.image(
            uploaded,
            width="stretch"
        )


    # ======================================================
    # HASIL PREDIKSI
    # ======================================================

    with col2:

        st.subheader(
            "🔍 Hasil Prediksi"
        )

        analisis = st.button(
            "🔎 Analisis Sekarang",
            use_container_width=True
        )


        if analisis:

            try:

                with st.spinner(
                    "Sedang melakukan prediksi..."
                ):

                    # Simpan gambar untuk riwayat
                    uploaded.seek(0)

                    raw_image_data = uploaded.read()

                    # Prediksi
                    uploaded.seek(0)

                    label, confidence, probability = predict(
                        model,
                        uploaded
                    )


                # Simpan session
                st.session_state.last_label = label

                st.session_state.last_probability = probability


                # Simpan history
                st.session_state.history.append({

                    "image": raw_image_data,

                    "label": label,

                    "confidence":
                        f"{confidence:.2f}%",

                    "time":
                        datetime.datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                })


                # ==================================================
                # HASIL TANPA HTML
                # ==================================================

                if label == "Mature":

                    st.success(
                        f"🟢 {label}"
                    )

                elif label == "Pre-mature":

                    st.warning(
                        f"🟡 {label}"
                    )

                elif label == "Unmature":

                    st.info(
                        f"⚪ {label}"
                    )

                else:

                    st.error(
                        f"🟥 {label}"
                    )


                # Confidence
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


                # ==================================================
                # PROBABILITAS
                # ==================================================

                st.subheader(
                    "📊 Probabilitas Model CNN"
                )


                probability_sorted = sorted(
                    probability.items(),
                    key=lambda x: x[1],
                    reverse=True
                )


                for kelas, persen in probability_sorted:

                    st.write(
                        f"{CLASS_EMOJI[kelas]} "
                        f"**{kelas}**"
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


            except Exception as e:

                st.error(
                    "❌ Terjadi kesalahan saat "
                    "melakukan prediksi."
                )

                st.exception(e)


# ==========================================================
# INFORMASI PEPAYA
# ==========================================================

if st.session_state.last_label is not None:

    st.divider()

    label = st.session_state.last_label

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


    # ======================================================
    # DESKRIPSI
    # ======================================================

    with tab1:

        st.write(
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


    # ======================================================
    # NUTRISI
    # ======================================================

    with tab2:

        st.subheader(
            "Kandungan Nutrisi"
        )

        for key, value in info["nutrisi"].items():

            st.write(
                f"**{key}:** {value}"
            )


    # ======================================================
    # REKOMENDASI
    # ======================================================

    with tab3:

        st.success(
            info["rekomendasi"]
        )


# ==========================================================
# RIWAYAT PREDIKSI
# ==========================================================

if len(st.session_state.history) > 0:

    st.divider()

    st.header(
        "📑 Riwayat Prediksi"
    )


    h_col1, h_col2, h_col3, h_col4 = st.columns(
        [1.5, 2, 2, 2.5]
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
        ):

            continue


        r_col1, r_col2, r_col3, r_col4 = st.columns(
            [1.5, 2, 2, 2.5]
        )


        with r_col1:

            st.image(
                item["image"],
                width=90
            )


        with r_col2:

            st.write("")

            st.write("")

            st.write(
                f"{CLASS_EMOJI[item['label']]} "
                f"**{item['label']}**"
            )


        with r_col3:

            st.write("")

            st.write("")

            st.write(
                f"**{item['confidence']}**"
            )


        with r_col4:

            st.write("")

            st.write("")

            st.write(
                item["time"]
            )


        st.markdown(
            '<div class="history-row"></div>',
            unsafe_allow_html=True
        )


# ==========================================================
# PAPAYA AI ASSISTANT
# ==========================================================

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


# ==========================================================
# CONTOH PERTANYAAN
# ==========================================================

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


pertanyaan = st.text_input(
    "Silakan tuliskan pertanyaan Anda:"
)


# ==========================================================
# TOMBOL AI
# ==========================================================

if st.button(
    "🤖 Tanya AI",
    use_container_width=True
):

    tanya = pertanyaan.lower().strip()


    if st.session_state.last_label is not None:

        label = st.session_state.last_label

    else:

        label = None


    jawaban = ""


    # ======================================================
    # HASIL PREDIKSI
    # ======================================================

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

Pepaya sudah matang sempurna dan siap
dikonsumsi.

Jika belum ingin dimakan hari ini,
simpan di lemari pendingin agar kualitasnya
tetap terjaga.
"""


        elif label == "Pre-mature":

            jawaban = """
🟡 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Pre-mature**.

Pepaya hampir matang.

Sebaiknya diperam 1–3 hari lagi pada suhu
ruang hingga rasa menjadi lebih manis.
"""


        elif label == "Unmature":

            jawaban = """
⚪ Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Unmature**.

Pepaya masih mentah sehingga belum
disarankan untuk dikonsumsi.

Simpan pada suhu ruang hingga matang.
"""


        else:

            jawaban = """
🔴 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk **Defect**.

Buah mengalami kerusakan.

Periksa apakah terdapat jamur, bau tidak
sedap, atau tekstur berlendir.

Jika iya, sebaiknya tidak dikonsumsi.
"""


    # ======================================================
    # PENYIMPANAN
    # ======================================================

    elif "menyimpan" in tanya:

        jawaban = """
📦 **Cara Penyimpanan Pepaya**

• Pepaya mentah disimpan pada suhu ruang.

• Pepaya matang dapat disimpan di lemari
pendingin agar bertahan lebih lama.

• Jangan mencuci pepaya sebelum disimpan.

• Hindari sinar matahari langsung.
"""


    # ======================================================
    # MEMPERCEPAT MATANG
    # ======================================================

    elif (
        "mempercepat" in tanya
        or "peram" in tanya
    ):

        jawaban = """
🍌 **Untuk mempercepat pematangan:**

• Simpan pepaya pada suhu ruang.

• Letakkan bersama buah pisang atau apel
karena menghasilkan gas etilen yang membantu
proses pematangan.

• Hindari memasukkan ke kulkas sebelum matang.
"""


    # ======================================================
    # MASA SIMPAN
    # ======================================================

    elif (
        "berapa lama" in tanya
        or "tahan" in tanya
    ):

        jawaban = """
⏳ **Umur Simpan Pepaya**

• Suhu ruang: sekitar 2–3 hari setelah matang.

• Dalam kulkas: sekitar 5–7 hari.

Lama penyimpanan dapat berbeda tergantung
kondisi buah.
"""


    # ======================================================
    # MEMILIH PEPAYA
    # ======================================================

    elif (
        "pilih" in tanya
        or "bagus" in tanya
    ):

        jawaban = """
🥭 **Tips Memilih Pepaya**

• Kulit berwarna kuning merata.

• Tidak terdapat luka besar.

• Tidak berbau busuk.

• Tekstur sedikit lunak saat ditekan.
"""


    # ======================================================
    # BERCAK HITAM
    # ======================================================

    elif (
        "bercak" in tanya
        or "hitam" in tanya
    ):

        jawaban = """
⚫ Bercak hitam dapat disebabkan oleh
benturan, memar, atau proses pembusukan.

Apabila bercak hanya sedikit, bagian tersebut
dapat dipotong.

Namun jika disertai bau tidak sedap dan
berlendir, sebaiknya pepaya tidak dikonsumsi.
"""


    # ======================================================
    # IBU HAMIL
    # ======================================================

    elif (
        "ibu hamil" in tanya
        or "hamil" in tanya
    ):

        jawaban = """
🤰 Pepaya matang umumnya aman dikonsumsi
dalam jumlah wajar.

Untuk kondisi kehamilan tertentu, sebaiknya
konsultasikan dengan tenaga kesehatan.
"""


    # ======================================================
    # BAYI
    # ======================================================

    elif (
        "bayi" in tanya
        or "mpasi" in tanya
    ):

        jawaban = """
👶 Pepaya matang dapat diberikan sebagai MPASI
setelah bayi berusia sekitar 6 bulan sesuai
anjuran tenaga kesehatan.
"""


    # ======================================================
    # PENCERNAAN
    # ======================================================

    elif "pencernaan" in tanya:

        jawaban = """
💚 Pepaya mengandung serat dan enzim papain
yang dapat membantu mendukung pencernaan.
"""


    # ======================================================
    # DEFAULT
    # ======================================================

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


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    <h3>Sistem Klasifikasi Tingkat Kematangan Pepaya</h3>

    <p>Convolutional Neural Network (CNN)</p>

    <p>TensorFlow • Streamlit</p>

    <br>

    <b>Developed by Muhammad Ghifari</b>

    </div>
    """,
    unsafe_allow_html=True
)
