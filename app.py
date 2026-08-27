# ======================================================
# FUNGSI PREDIKSI (FIX UNMATURE VS DEFECT)
# ======================================================

def predict(model, image_file):

    # Baca ulang file gambar
    image_file.seek(0)

    img = tf.image.decode_image(
        image_file.read(),
        channels=3,
        expand_animations=False
    )

    # Resize sesuai training
    img = tf.image.resize(img, IMG_SIZE)

    # WAJIB sama seperti training ImageDataGenerator(rescale=1./255)
    img = tf.cast(img, tf.float32) / 255.0

    # Tambahkan batch
    img = tf.expand_dims(img, 0)

    # Prediksi
    prediction = model.predict(img, verbose=0)[0]

    probability = {
        CLASS_NAMES[i]: float(prediction[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    predicted_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probability[predicted_class]

    # =====================================================
    # FIX DEFECT FALSE POSITIVE
    # =====================================================

    # Kalau Defect bukan kelas tertinggi DAN nilainya kecil,
    # jangan tampilkan sebagai probabilitas tinggi.
    if (
        predicted_class != "Defect"
        and probability["Defect"] < 35
    ):
        probability["Defect"] = 0

        total = (
            probability["Mature"]
            + probability["Pre-mature"]
            + probability["Unmature"]
        )

        if total > 0:
            probability["Mature"] = probability["Mature"] / total * 100
            probability["Pre-mature"] = probability["Pre-mature"] / total * 100
            probability["Unmature"] = probability["Unmature"] / total * 100

            confidence = probability[predicted_class]

    return predicted_class, confidence, probability
