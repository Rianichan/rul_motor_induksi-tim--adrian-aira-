"""Streamlit interface for the motor remaining-useful-life model."""

import json
from pathlib import Path

import streamlit as st

from src.features import NUMERIC_FEATURES, predict_rul


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "final_model.json"
STATUS_PATH = ROOT / "config" / "status_rules.json"


@st.cache_data
def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Perkiraan Umur Motor", page_icon="⚙️", layout="centered")
st.title("Perkiraan Sisa Umur Motor")
st.write(
    "Masukkan tipe motor dan pembacaan sensor terbaru untuk melihat perkiraan sisa umur "
    "serta tindakan pemeliharaan yang disarankan."
)

try:
    model = load_json(MODEL_PATH)
    status_rules = load_json(STATUS_PATH)
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Model atau aturan status tidak dapat dibaca: {exc}")
    st.stop()

medians = model["median_imputation_train"]
labels = {
    "beban_pct": ("Beban motor", "%", 0.1),
    "arus_A": ("Arus listrik", "A", 0.01),
    "tegangan_V": ("Tegangan", "V", 0.1),
    "suhu_belitan_C": ("Suhu belitan", "°C", 0.1),
    "getaran_rms_mm_s": ("Getaran RMS", "mm/s", 0.01),
    "kecepatan_rpm": ("Kecepatan putar", "rpm", 1.0),
}

with st.form("sensor_form"):
    motor_type = st.selectbox("Tipe motor", ["5.5kW", "11kW", "22kW"])
    left, right = st.columns(2)
    payload = {"tipe_motor": motor_type}
    for i, name in enumerate(NUMERIC_FEATURES):
        label, unit, step = labels[name]
        container = left if i % 2 == 0 else right
        with container:
            payload[name] = st.number_input(
                f"{label} ({unit})",
                value=float(medians[name]),
                step=step,
                format="%.2f",
                help="Nilai contoh awal berasal dari median data train. Ubah dengan pembacaan sensor motor saat ini.",
                key=name,
            )
    submitted = st.form_submit_button("Hitung perkiraan")

if submitted:
    try:
        result = predict_rul(payload, model, status_rules)
    except ValueError as exc:
        st.error(f"Periksa masukan: {exc}")
    else:
        st.metric("Perkiraan sisa umur", f"{result['rul_hari']} hari")
        status = result["status"]
        if status == "aman":
            st.success(f"Status: {result['status_label']}")
        elif status == "pesan_pengganti_sekarang":
            st.warning(f"Status: {result['status_label']}")
        else:
            st.error(f"Status: {result['status_label']}")
        st.write(result["tindakan"])
        if result["peringatan"]:
            with st.expander("Peringatan pembacaan sensor", expanded=True):
                for warning in result["peringatan"]:
                    st.warning(warning)

st.info(
    "Status mengikuti waktu pengiriman motor pengganti 2 sampai 4 minggu. Pada validation, "
    "MAE model untuk motor dengan RUL ≤60 hari sekitar 26 hari; gunakan hasil sebagai "
    "panduan awal dan minta perencana pemeliharaan meninjau keputusan."
)

with st.expander("Tentang model"):
    st.write(
        "Model menggunakan enam sensor dan tipe motor. Sensor kosong diimputasi dengan "
        "median train; pembacaan negatif yang tidak masuk akal ditolak. Nilai di luar "
        "rentang train diberi peringatan. ID motor tidak diperlukan saat prediksi."
    )
