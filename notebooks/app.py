import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"

SENSOR_INPUTS = {
    "beban_pct": ("Beban motor", "%", 0.1),
    "arus_A": ("Arus", "A", 0.01),
    "tegangan_V": ("Tegangan", "V", 0.1),
    "suhu_belitan_C": ("Suhu belitan", "°C", 0.1),
    "getaran_rms_mm_s": ("Getaran RMS", "mm/s", 0.01),
    "kecepatan_rpm": ("Kecepatan putar", "rpm", 1.0),
}

# Label yang dilihat pengguna dipetakan ke kategori saat pelatihan.
MOTOR_TYPES = {
    "5.5 kW": "5.5kw",
    "11 kW": "11kw",
    "22 kW": "22kw",
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


st.set_page_config(page_title="Perkiraan umur motor", page_icon="⚙️")
st.title("Perkiraan Sisa Umur Motor")
st.write("Masukkan tipe motor dan pembacaan sensor terbaru.")

try:
    model, metadata = load_model()
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Model atau metadata tidak bisa dibaca: {exc}")
    st.stop()

defaults = metadata["preprocessing"]["numeric_medians"]

with st.form("sensor_motor"):
    motor_label = st.selectbox("Tipe motor", list(MOTOR_TYPES))

    left, right = st.columns(2)
    values = {}

    for index, (name, (label, unit, step)) in enumerate(SENSOR_INPUTS.items()):
        column = left if index % 2 == 0 else right
        minimum = 0.01 if name in {"tegangan_V", "kecepatan_rpm"} else 0.0

        with column:
            values[name] = st.number_input(
                f"{label} ({unit})",
                min_value=minimum,
                value=max(float(defaults[name]), minimum),
                step=step,
                format="%.2f",
            )

    submitted = st.form_submit_button("Hitung perkiraan")


if submitted:
    warnings = []

    for name, value in values.items():
        if not math.isfinite(value):
            st.error(f"Nilai {name} harus berupa angka yang valid.")
            st.stop()

    # Beri tahu pengguna jika pembacaan tampak ekstrem.
    if values["beban_pct"] > 100:
        warnings.append("Beban di atas 100%; periksa kembali pembacaan sensor.")
    if values["suhu_belitan_C"] > 350:
        warnings.append("Suhu di atas 350°C; verifikasi pembacaan sensor.")

    # Opsional: metadata dapat memuat rentang nilai saat pelatihan.
    training_ranges = metadata.get("training_sensor_ranges", {})
    for name, value in values.items():
        limits = training_ranges.get(name)
        if limits and not (limits["min"] <= value <= limits["max"]):
            warnings.append(
                f"{name} di luar rentang data pelatihan; prediksi memperkirakan "
                "dari kondisi yang belum banyak terlihat."
            )

    # Susun input dengan nama dan urutan fitur yang dipakai saat pelatihan.
    input_row = {
        **values,
        "tipe_motor_norm": MOTOR_TYPES[motor_label],
    }
    input_data = pd.DataFrame([input_row], columns=metadata["features"])

    prediction = max(0.0, float(model.predict(input_data)[0]))

    st.metric("Perkiraan sisa umur", f"{prediction:.0f} hari")

    # Status awal mengikuti waktu pengiriman motor 2–4 minggu.
    if prediction <= 14:
        st.error("Status: Segera ganti")
        st.write("Prioritaskan penggantian dan siapkan mitigasi.")
    elif prediction <= 28:
        st.warning("Status: Pesan pengganti sekarang")
        st.write("Pesan motor pengganti agar tersedia sebelum perkiraan kerusakan.")
    else:
        st.success("Status: Aman")
        st.write("Lanjutkan pemantauan rutin.")

    for warning in warnings:
        st.warning(warning)

    st.caption(
        "Ini adalah perkiraan model, bukan jaminan. "
        "Gunakan bersama pemeriksaan dan pertimbangan perencana pemeliharaan."
    )