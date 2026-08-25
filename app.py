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
# KONFIGURASI
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

# File model berada satu folder dengan app.py
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_papaya.h5"
)

IMG_SIZE = (128, 128)


# ======================================================
# CLASS NAMES
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


# ======================================================
# INFORMASI PEPAYA
# ======================================================

INFO = {

    "Defect": {
        "deskripsi":
            "Pepaya mengalami kerusakan pada kulit atau "
            "daging buah sehingga kualitasnya menurun.",

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


# ======================================================
# CSS
# ======================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
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

    if not os.path.exists(MODEL_PATH):

        st.error("❌ MODEL TIDAK DITEMUKAN")

        st.write("Path yang dicari:")

        st.code(MODEL_PATH)

        st.warning(
            "Pastikan file model_papaya.keras berada "
            "satu folder dengan app.py."
        )

        st.stop()

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
            "Pastikan model_papaya.keras merupakan "
            "model TensorFlow/Keras yang valid."
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

    image_file.seek(0)

    img_bytes = image_file.read()

    img_tensor = tf.image.decode_image(
        img_bytes,
        channels=3,
        expand_animations=False
    )

    img_resized = tf.image.resize(
        img_tensor,
        IMG_SIZE,
        method="bilinear"
    )

    img_array = tf.cast(
        img_resized,
        tf.float32
    ) / 255.0

    img_array = tf.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    # Pastikan output model 4 kelas
    if len(prediction) != 4:

        raise ValueError(
            "Jumlah output model tidak sesuai. "
            f"Model menghasilkan {len(prediction)} output, "
            "sedangkan aplikasi membutuhkan 4 kelas."
        )

    # ----------------------------------------------
    # PROBABILITAS
    # ----------------------------------------------

    probability = {}

    for i, kelas in enumerate(CLASS_NAMES):

        probability[kelas] = (
            float(prediction[i]) * 100
        )

    # ----------------------------------------------
    # KELAS TERBESAR
    # ----------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # ----------------------------------------------
    # SMART DEFECT THRESHOLD
    # ----------------------------------------------

    clean_probability = {}

    pangkas_total = 0.0

    for i, kelas in enumerate(CLASS_NAMES):

        persen = float(prediction[i]) * 100

        if (
            kelas == "Defect"
            and predicted_class != "Defect"
            and persen < 40.0
        ):

            pangkas_total += persen

            clean_probability[kelas] = 0.0

        else:

            clean_probability[kelas] = persen

    # Masukkan nilai Defect yang dipangkas
    # ke kelas utama
    clean_probability[predicted_class] += pangkas_total

    confidence = clean_probability[
        predicted_class
    ]

    return (
        predicted_class,
        confidence,
        clean_probability
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
# PROSES UPLOAD
# ======================================================

if uploaded is not None:

    col1, col2 = st.columns(2)

    # ==================================================
    # GAMBAR
    # ==================================================

    with col1:

        st.subheader("🖼️ Gambar Pepaya")

        st.image(
            uploaded,
            use_container_width=True
        )

    # ==================================================
    # HASIL
    # ==================================================

    with col2:

        st.subheader("📊 Hasil Prediksi")

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
                        "❌ Terjadi kesalahan saat prediksi."
                    )

                    st.code(str(e))

                    st.stop()

            # ------------------------------------------
            # SIMPAN LABEL
            # ------------------------------------------

            st.session_state.last_label = label

            # ------------------------------------------
            # SIMPAN HISTORY
            # ------------------------------------------

            st.session_state.history.append(
                {
                    "image": raw_image_data,
                    "label": label,
                    "confidence":
                        f"{confidence:.2f}%",
                    "time":
                        datetime.datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                }
            )

            # ------------------------------------------
            # HASIL UTAMA
            # ------------------------------------------

            st.success(
                f"{CLASS_EMOJI[label]} "
                f"Hasil klasifikasi: {label}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # ------------------------------------------
            # PROBABILITAS
            # ------------------------------------------

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
                    f"{CLASS_EMOJI[kelas]} **{kelas}**"
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

    # ==================================================
    # DESKRIPSI
    # ==================================================

    with tab1:

        st.write(
            info["deskripsi"]
        )

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
        st.write("**Gambar**")

    with h_col2:
        st.write("**Label**")

    with h_col3:
        st.write("**Confidence**")

    with h_col4:
        st.write("**Waktu Analisis**")

    st.divider()

    # ==================================================
    # TAMPILKAN HISTORY
    # ==================================================

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

            st.write(
                f"{CLASS_EMOJI[item['label']]} "
                f"{item['label']}"
            )

        with r_col3:

            st.write(
                item["confidence"]
            )

        with r_col4:

            st.write(
                item["time"]
            )

        st.divider()


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

    st.divider()

    st.write(
        "### Kelas Model"
    )

    for kelas in CLASS_NAMES:

        st.write(
            f"{CLASS_EMOJI[kelas]} {kelas}"
        )

    st.divider()

    if st.button(
        "♻️ Reset Riwayat",
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
informasi mengenai buah pepaya berdasarkan hasil
klasifikasi CNN.

✅ Tidak memerlukan internet

✅ Seluruh jawaban berasal dari basis pengetahuan
yang telah disiapkan.
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
# INPUT AI
# ======================================================

pertanyaan = st.text_input(
    "Silakan tuliskan pertanyaan Anda:"
)


# ======================================================
# AI ASSISTANT
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
        if st.session_state.last_label is not None
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

            jawaban = (
                "Silakan lakukan klasifikasi terlebih dahulu "
                "dengan mengunggah gambar pepaya."
            )

        elif label == "Mature":

            jawaban = (
                "🟢 Berdasarkan hasil klasifikasi CNN, "
                "pepaya Anda termasuk Mature.\n\n"
                "Pepaya sudah matang sempurna dan siap "
                "dikonsumsi. Jika belum ingin dimakan hari ini, "
                "simpan di lemari pendingin."
            )

        elif label == "Pre-mature":

            jawaban = (
                "🟡 Berdasarkan hasil klasifikasi CNN, "
                "pepaya termasuk Pre-mature.\n\n"
                "Pepaya hampir matang. Sebaiknya diperam "
                "1–3 hari lagi pada suhu ruang hingga rasa "
                "menjadi lebih manis."
            )

        elif label == "Unmature":

            jawaban = (
                "⚪ Berdasarkan hasil klasifikasi CNN, "
                "pepaya termasuk Unmature.\n\n"
                "Pepaya masih mentah sehingga belum "
                "disarankan untuk dikonsumsi. Simpan "
                "pada suhu ruang hingga matang."
            )

        else:

            jawaban = (
                "🔴 Berdasarkan hasil klasifikasi CNN, "
                "pepaya termasuk Defect.\n\n"
                "Buah mengalami kerusakan. Periksa apakah "
                "terdapat jamur, bau tidak sedap, atau "
                "tekstur berlendir. Jika iya, sebaiknya "
                "tidak dikonsumsi."
            )


    # ==================================================
    # PENYIMPANAN
    # ==================================================

    elif "menyimpan" in tanya:

        jawaban = (
            "📦 Cara Penyimpanan Pepaya\n\n"
            "• Pepaya mentah disimpan pada suhu ruang.\n\n"
            "• Pepaya matang dapat disimpan di lemari "
            "pendingin agar bertahan lebih lama.\n\n"
            "• Jangan mencuci pepaya sebelum disimpan.\n\n"
            "• Hindari sinar matahari langsung."
        )


    # ==================================================
    # MEMPERCEPAT PEMATANGAN
    # ==================================================

    elif (
        "mempercepat" in tanya
        or "peram" in tanya
    ):

        jawaban = (
            "🍌 Untuk mempercepat pematangan:\n\n"
            "• Simpan pepaya pada suhu ruang.\n\n"
            "• Letakkan bersama buah pisang atau apel "
            "karena menghasilkan gas etilen.\n\n"
            "• Hindari memasukkan pepaya ke kulkas "
            "sebelum matang."
        )


    # ==================================================
    # MASA SIMPAN
    # ==================================================

    elif (
        "berapa lama" in tanya
        or "tahan" in tanya
    ):

        jawaban = (
            "⏳ Umur Simpan Pepaya\n\n"
            "• Suhu ruang: sekitar 2–3 hari setelah matang.\n\n"
            "• Dalam kulkas: sekitar 5–7 hari.\n\n"
            "Lama penyimpanan dapat berbeda tergantung "
            "kondisi buah."
        )


    # ==================================================
    # MEMILIH PEPAYA
    # ==================================================

    elif (
        "pilih" in tanya
        or "bagus" in tanya
    ):

        jawaban = (
            "🥭 Tips Memilih Pepaya\n\n"
            "• Kulit berwarna kuning merata.\n\n"
            "• Tidak terdapat luka besar.\n\n"
            "• Tidak berbau busuk.\n\n"
            "• Tekstur sedikit lunak saat ditekan."
        )


    # ==================================================
    # BERCAK HITAM
    # ==================================================

    elif (
        "bercak" in tanya
        or "hitam" in tanya
    ):

        jawaban = (
            "⚫ Bercak hitam dapat disebabkan oleh "
            "benturan, memar, atau proses pembusukan.\n\n"
            "Apabila bercak hanya sedikit, bagian tersebut "
            "dapat dipotong.\n\n"
            "Namun jika disertai bau tidak sedap dan "
            "berlendir, sebaiknya pepaya tidak dikonsumsi."
        )


    # ==================================================
    # IBU HAMIL
    # ==================================================

    elif (
        "ibu hamil" in tanya
        or "hamil" in tanya
    ):

        jawaban = (
            "🤰 Untuk pertanyaan terkait kehamilan, "
            "sebaiknya konsultasikan konsumsi pepaya "
            "dengan dokter atau tenaga kesehatan, "
            "terutama jika pepaya masih mentah."
        )


    # ==================================================
    # BAYI
    # ==================================================

    elif (
        "bayi" in tanya
        or "mpasi" in tanya
    ):

        jawaban = (
            "👶 Pepaya matang dapat menjadi salah satu "
            "pilihan MPASI setelah bayi mulai mendapatkan "
            "MPASI sesuai anjuran tenaga kesehatan."
        )


    # ==================================================
    # PENCERNAAN
    # ==================================================

    elif "pencernaan" in tanya:

        jawaban = (
            "💚 Pepaya mengandung serat dan enzim papain "
            "yang dapat membantu mendukung sistem pencernaan."
        )


    # ==================================================
    # DEFAULT
    # ==================================================

    else:

        jawaban = (
            "Maaf, saya belum memahami pertanyaan tersebut.\n\n"
            "Silakan tanyakan hal seperti:\n\n"
            "• Bagaimana cara menyimpan pepaya?\n\n"
            "• Bagaimana mempercepat pematangan?\n\n"
            "• Berapa lama pepaya dapat disimpan?\n\n"
            "• Bagaimana memilih pepaya yang bagus?\n\n"
            "• Apakah pepaya saya sudah siap dimakan?\n\n"
            "• Kenapa muncul bercak hitam?"
        )


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

st.title(
    "🍈 Sistem Klasifikasi Tingkat Kematangan Pepaya"
)

st.write(
    "Convolutional Neural Network (CNN)"
)

st.write(
    "TensorFlow • Streamlit"
)

st.write(
    "Developed by Muhammad Ghifari"
)
