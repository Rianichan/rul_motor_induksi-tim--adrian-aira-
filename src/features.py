"""Input validation and feature preparation for the polynomial RUL model."""

import math
import re

from .status import classify_status


NUMERIC_FEATURES = [
    "beban_pct",
    "arus_A",
    "tegangan_V",
    "suhu_belitan_C",
    "getaran_rms_mm_s",
    "kecepatan_rpm",
]


def normalize_motor_type(value):
    if not isinstance(value, str):
        raise ValueError("Tipe motor harus dipilih")
    normalized = re.sub(r"\s+", "", value.strip().lower()).replace(",", ".")
    normalized = normalized.replace("-", "")
    aliases = {
        "5.5kw": "5.5kW",
        "5.5kilowatt": "5.5kW",
        "11kw": "11kW",
        "11kilowatt": "11kW",
        "22kw": "22kW",
        "22kilowatt": "22kW",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("Tipe motor harus 5.5kW, 11kW, atau 22kW") from exc


def _read_sensor(name, payload, model, warnings):
    value = payload.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        warnings.append(f"{name} kosong; median data train dipakai untuk imputasi.")
        return float(model["median_imputation_train"][name])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} harus berupa angka") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} harus berupa angka terbatas")

    if name in {"beban_pct", "arus_A", "suhu_belitan_C", "getaran_rms_mm_s"} and number < 0:
        raise ValueError(f"{name} tidak boleh negatif")
    if name in {"tegangan_V", "kecepatan_rpm"} and number <= 0:
        raise ValueError(f"{name} harus lebih besar dari nol")

    limits = model.get("training_sensor_ranges", {}).get(name)
    if limits:
        if number < float(limits["min"]) or number > float(limits["max"]):
            warnings.append(f"{name} di luar rentang yang terlihat saat train; prediksi mengekstrapolasi data.")
        elif number < float(limits["p01"]) or number > float(limits["p99"]):
            warnings.append(f"{name} berada di ekor distribusi train; periksa pembacaan sensornya.")
    if name == "suhu_belitan_C" and number > 350:
        warnings.append("Suhu belitan di atas 350°C pernah muncul sebagai anomali pada data; verifikasi sensor.")
    return number


def prepare_features(payload, model):
    """Return the model vector and input warnings without accessing target/test data."""
    if not isinstance(payload, dict):
        raise ValueError("Masukan harus berupa objek dengan tipe motor dan sensor")

    motor_type = normalize_motor_type(payload.get("tipe_motor"))
    warnings = []
    numeric = {name: _read_sensor(name, payload, model, warnings) for name in NUMERIC_FEATURES}

    z = []
    for name in NUMERIC_FEATURES:
        mean = float(model["numeric_means_train"][name])
        std = float(model["numeric_stds_train"][name])
        if std <= 0 or not math.isfinite(std):
            raise ValueError(f"Parameter skala untuk {name} tidak valid")
        z.append((numeric[name] - mean) / std)

    raw_derived = [z[1] ** 2, z[3] ** 2, z[4] ** 2, z[0] * z[1], z[3] * z[4]]
    derived = []
    for name, value in zip(
        ["arus_z2", "suhu_z2", "getaran_z2", "beban_x_arus", "suhu_x_getaran"], raw_derived
    ):
        mean = float(model["nonlinear_term_means_train"][name])
        std = float(model["nonlinear_term_stds_train"][name])
        if std <= 0 or not math.isfinite(std):
            raise ValueError(f"Parameter skala untuk fitur {name} tidak valid")
        derived.append((value - mean) / std)

    values = {
        "intercept": 1.0,
        "beban_z": z[0],
        "arus_z": z[1],
        "tegangan_z": z[2],
        "suhu_z": z[3],
        "getaran_z": z[4],
        "kecepatan_z": z[5],
        "arus_z2": derived[0],
        "suhu_z2": derived[1],
        "getaran_z2": derived[2],
        "beban_x_arus": derived[3],
        "suhu_x_getaran": derived[4],
        "tipe_11kW": 1.0 if motor_type == "11kW" else 0.0,
        "tipe_22kW": 1.0 if motor_type == "22kW" else 0.0,
    }
    return [values[name] for name in model["feature_names"]], warnings, numeric


def predict_rul(payload, model, status_rules):
    vector, warnings, numeric = prepare_features(payload, model)
    coefficients = model["coefficients"]
    if len(vector) != len(coefficients):
        raise ValueError("Jumlah fitur tidak cocok dengan model")
    raw_estimate = sum(float(coef) * value for coef, value in zip(coefficients, vector))
    estimate = max(0.0, raw_estimate)
    if raw_estimate < 0:
        warnings.append("Prediksi linear berada di bawah nol; ditampilkan sebagai 0 hari.")

    target_range = model.get("target_range_train", {})
    if target_range and estimate > float(target_range.get("max", math.inf)):
        warnings.append("Prediksi di atas RUL maksimum yang terlihat saat train; gunakan dengan hati-hati.")

    status = classify_status(estimate, status_rules)
    return {
        "rul_hari": round(estimate),
        "rul_hari_desimal": estimate,
        "status": status["status"],
        "status_label": status["label"],
        "tindakan": status["action"],
        "peringatan": warnings,
        "sensor_dipakai": numeric,
    }
