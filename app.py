"""
==========================================================
APP.PY
Sistem Klasifikasi Tingkat Kematangan Pepaya
CNN + TensorFlow + Streamlit
==========================================================
"""

import os
import datetime
import numpy as np
import streamlit as st
import tensorflow as tf


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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# app.py dan model_papaya.h5 berada dalam satu folder
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_papaya.h5"
)


# ======================================================
# NAMA KELAS
# ======================================================

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
            "Pepaya mengalami kerusakan pada kulit atau "
            "daging buah sehingga kualitasnya menurun."
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
            "Pepaya mulai memasuki fase matang namun "
            "belum optimal."
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

    .main {
        background: #f8fafc;
    }

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

    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    # Cek apakah file model ada
    if not os.path.exists(MODEL_PATH):

        st.error("❌ MODEL TIDAK DITEMUKAN")

        st.write("Path yang dicari:")

        st.code(MODEL_PATH)

        st.warning(
            """
            Pastikan file model_papaya.h5 berada
            satu folder dengan app.py.
            """
        )

        st.stop()

    # Tampilkan informasi file
    file_size = os.path.getsize(MODEL_PATH)

    st.sidebar.write("### Informasi Model")

    st.sidebar.write(
        f"Ukuran model: {file_size / (1024 * 1024):.2f} MB"
    )

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error("❌ GAGAL MEMBUKA MODEL")

        st.code(str(e))

        st.warning(
            """
            File model_papaya.h5 ditemukan,
            tetapi tidak dapat dibaca sebagai
            model TensorFlow/Keras.

            Pastikan file tersebut merupakan file
            model H5 asli yang dihasilkan dari:

            model.save("model_papaya.h5")
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
# FUNGSI PREDIKSI
# ======================================================

def predict(model, image_file):

    # --------------------------------------------------
    # Baca file gambar
    # --------------------------------------------------

    image_file.seek(0)

    img_bytes = image_file.read()


    # --------------------------------------------------
    # Decode gambar
    # --------------------------------------------------

    img_tensor = tf.image.decode_image(
        img_bytes,
        channels=3,
        expand_animations=False
    )


    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    img_resized = tf.image.resize(
        img_tensor,
        IMG_SIZE,
        method="bilinear"
    )


    # --------------------------------------------------
    # Normalisasi
    # --------------------------------------------------

    img_array = tf.cast(
        img_resized,
        tf.float32
    ) / 255.0


    # --------------------------------------------------
    # Tambahkan batch
    # --------------------------------------------------

    img_array = tf.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------
    # Prediksi
    # --------------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]


    # --------------------------------------------------
    # Cek jumlah output
    # --------------------------------------------------

    if len(prediction) != len(CLASS_NAMES):

        raise ValueError(
            f"""
Jumlah output model: {len(prediction)}

Jumlah CLASS_NAMES: {len(CLASS_NAMES)}

CLASS_NAMES:
{CLASS_NAMES}
"""
        )


    # --------------------------------------------------
    # Probabilitas asli
    # --------------------------------------------------

    probability = {}

    for i, kelas in enumerate(CLASS_NAMES):

        probability[kelas] = (
            float(prediction[i]) * 100
        )


    # --------------------------------------------------
    # Cari kelas dengan probabilitas tertinggi
    # --------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    confidence = (
        float(prediction[predicted_index]) * 100
    )


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
# UPLOAD GAMBAR
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
# PROSES PREDIKSI
# ======================================================

if uploaded is not None:

    col1, col2 = st.columns(
        [1, 1]
    )


    # --------------------------------------------------
    # GAMBAR
    # --------------------------------------------------

    with col1:

        st.subheader(
            "Gambar"
        )

        st.image(
            uploaded,
            use_container_width=True
        )


    # --------------------------------------------------
    # HASIL
    # --------------------------------------------------

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

                uploaded.seek(0)

                raw_image_data = uploaded.read()

                uploaded.seek(0)

                try:

                    label, confidence, probability = predict(
                        model,
                        uploaded
                    )

                except Exception as e:

                    st.error(
                        "❌ TERJADI KESALAHAN SAAT PREDIKSI"
                    )

                    st.code(
                        str(e)
                    )

                    st.stop()


            # --------------------------------------------------
            # Simpan label terakhir
            # --------------------------------------------------

            st.session_state.last_label = label


            # --------------------------------------------------
            # Simpan history
            # --------------------------------------------------

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
                        int(persen),
                        100
                    )
                )

                st.caption(
                    f"{persen:.2f}%"
                )


# ======================================================
# INFORMASI PEPAYA
# ======================================================

if st.session_state.last_label is not None:

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


    # --------------------------------------------------
    # DESKRIPSI
    # --------------------------------------------------

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


    # --------------------------------------------------
    # NUTRISI
    # --------------------------------------------------

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


    # --------------------------------------------------
    # REKOMENDASI
    # --------------------------------------------------

    with tab3:

        st.success(
            info["rekomendasi"]
        )


# ======================================================
# RIWAYAT PREDIKSI
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
            '<div class="history-header">Gambar</div>',
            unsafe_allow_html=True
        )


    with h_col2:

        st.markdown(
            '<div class="history-header">Label</div>',
            unsafe_allow_html=True
        )


    with h_col3:

        st.markdown(
            '<div class="history-header">Confidence</div>',
            unsafe_allow_html=True
        )


    with h_col4:

        st.markdown(
            '<div class="history-header">Waktu Analisis</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------
    # Tampilkan history
    # --------------------------------------------------

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

    st.write("")


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

        st.rerun()


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
informasi mengenai buah pepaya berdasarkan hasil klasifikasi CNN.

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
# PERTANYAAN
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


    label = (
        st.session_state.last_label
        if st.session_state.last_label
        is not None
        else None
    )


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
🟢 Berdasarkan hasil klasifikasi CNN,
pepaya Anda termasuk Mature.

Pepaya sudah matang sempurna dan siap dikonsumsi.

Jika belum ingin dimakan hari ini,
simpan di lemari pendingin agar kualitasnya tetap terjaga.
"""


        elif label == "Pre-mature":

            jawaban = """
🟡 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk Pre-mature.

Pepaya hampir matang. Sebaiknya diperam
1–3 hari lagi pada suhu ruang hingga rasa
menjadi lebih manis.
"""


        elif label == "Unmature":

            jawaban = """
⚪ Berdasarkan hasil klasifikasi CNN,
pepaya termasuk Unmature.

Pepaya masih mentah sehingga belum disarankan
untuk dikonsumsi.

Simpan pada suhu ruang hingga matang.
"""


        else:

            jawaban = """
🔴 Berdasarkan hasil klasifikasi CNN,
pepaya termasuk Defect.

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

• Jangan mencuci pepaya sebelum disimpan karena
  dapat mempercepat pembusukan.

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

• Letakkan bersama buah pisang atau apel karena
  menghasilkan gas etilen yang membantu proses
  pematangan.

• Hindari memasukkan ke kulkas sebelum matang.
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
sebaiknya konsultasikan dengan tenaga kesehatan
karena kondisi setiap orang dapat berbeda.

Untuk keamanan, hindari mengandalkan klasifikasi
gambar sebagai dasar keputusan medis.
"""


    # ==================================================
    # BAYI
    # ==================================================

    elif (
        "bayi" in tanya
        or "mpasi" in tanya
    ):

        jawaban = """
👶 Pepaya matang dapat menjadi salah satu pilihan
makanan pendamping setelah bayi mulai MPASI.

Sesuaikan dengan anjuran tenaga kesehatan dan
kondisi bayi.
"""


    # ==================================================
    # PENCERNAAN
    # ==================================================

    elif "pencernaan" in tanya:

        jawaban = """
💚 Pepaya mengandung serat dan enzim papain yang
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

• Apakah boleh diberikan kepada bayi?
"""


    # ==================================================
    # TAMPILKAN JAWABAN
    # ==================================================

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
    <div style="text-align:center">

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
