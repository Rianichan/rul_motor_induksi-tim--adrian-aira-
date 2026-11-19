"""Pure-Python polynomial OLS candidate for the motor RUL project.

Uses only the Python standard library and never evaluates held-out test rows.
"""
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


NUMERIC = [
    "beban_pct",
    "arus_A",
    "tegangan_V",
    "suhu_belitan_C",
    "getaran_rms_mm_s",
    "kecepatan_rpm",
]
DERIVED = [
    "arus_z2",
    "suhu_z2",
    "getaran_z2",
    "beban_x_arus",
    "suhu_x_getaran",
]


def canonical_id(value):
    return re.sub(r"[^A-Za-z0-9]", "", value.strip()).upper()


def canonical_type(value):
    value = value.strip().lower().replace(" ", "").replace(",", ".")
    return {"5.5kw": "5.5kW", "11kw": "11kW", "22kw": "22kW"}.get(value, "UNKNOWN")


def number_or_missing(value, column):
    if value is None or value.strip() == "":
        return None
    value = float(value)
    if column == "getaran_rms_mm_s" and value < 0:
        return None
    return value


def add_mode_type(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["motor_id"]].append(row)
    for motor_rows in grouped.values():
        counts = Counter(row["tipe_motor"] for row in motor_rows)
        mode = sorted(counts, key=lambda value: (-counts[value], value))[0]
        for row in motor_rows:
            row["tipe_motor"] = mode


def base_vector(row, medians, means, stds):
    out = []
    for name in NUMERIC:
        value = number_or_missing(row.get(name), name)
        if value is None:
            value = medians[name]
        out.append((value - means[name]) / stds[name])
    return out


def derived_vector(z):
    return [z[1] ** 2, z[3] ** 2, z[4] ** 2, z[0] * z[1], z[3] * z[4]]


def feature_vector(row, medians, means, stds, derived_means, derived_stds):
    z = base_vector(row, medians, means, stds)
    d = derived_vector(z)
    d = [(value - derived_means[name]) / derived_stds[name] for name, value in zip(DERIVED, d)]
    return [
        1.0,
        *z,
        *d,
        1.0 if row["tipe_motor"] == "11kW" else 0.0,
        1.0 if row["tipe_motor"] == "22kW" else 0.0,
    ]


def solve_linear_system(matrix, vector):
    """Gauss-Jordan elimination with partial pivoting for a small dense system."""
    a = [row[:] for row in matrix]
    b = vector[:]
    n = len(b)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError(f"Singular design matrix near column {col}")
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        divisor = a[col][col]
        for j in range(col, n):
            a[col][j] /= divisor
        b[col] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]
    return b


def main(csv_path, split_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(split_path, newline="", encoding="utf-8-sig") as f:
        split = {row["motor_id"]: row["split"] for row in csv.DictReader(f)}

    rows = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            row["motor_id"] = canonical_id(row["motor_id"])
            row["tipe_motor"] = canonical_type(row["tipe_motor"])
            key = tuple(row[name] for name in reader.fieldnames)
            if key in seen:
                continue
            seen.add(key)
            if row["motor_id"] not in split:
                raise ValueError(f"No split assignment for {row['motor_id']}")
            row["split"] = split[row["motor_id"]]
            rows.append(row)

    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    add_mode_type(train)
    add_mode_type(validation)

    impute = {}
    means = {}
    stds = {}
    training_sensor_ranges = {}
    for name in NUMERIC:
        values = [number_or_missing(row.get(name), name) for row in train]
        observed = [value for value in values if value is not None]
        ordered = sorted(observed)
        impute[name] = statistics.median(observed)
        training_sensor_ranges[name] = {
            "min": ordered[0],
            "p01": ordered[int(0.01 * (len(ordered) - 1))],
            "p99": ordered[int(0.99 * (len(ordered) - 1))],
            "max": ordered[-1],
        }
        filled = [impute[name] if value is None else value for value in values]
        means[name] = statistics.fmean(filled)
        stds[name] = math.sqrt(statistics.fmean((value - means[name]) ** 2 for value in filled)) or 1.0

    derived_values = {name: [] for name in DERIVED}
    for row in train:
        z = base_vector(row, impute, means, stds)
        for name, value in zip(DERIVED, derived_vector(z)):
            derived_values[name].append(value)
    derived_means = {name: statistics.fmean(values) for name, values in derived_values.items()}
    derived_stds = {
        name: math.sqrt(statistics.fmean((value - derived_means[name]) ** 2 for value in values)) or 1.0
        for name, values in derived_values.items()
    }

    p = 14
    gram = [[0.0] * p for _ in range(p)]
    cross = [0.0] * p
    for row in train:
        x = feature_vector(row, impute, means, stds, derived_means, derived_stds)
        y = float(row["rul_hari"])
        for i, xi in enumerate(x):
            cross[i] += xi * y
            for j, xj in enumerate(x):
                gram[i][j] += xi * xj
    coefficients = solve_linear_system(gram, cross)

    errors = []
    critical_errors = []
    raw_predictions = []
    clipped_predictions = 0
    status_counts = {"aman": 0, "pesan_pengganti_sekarang": 0, "segera_ganti": 0}
    for row in validation:
        x = feature_vector(row, impute, means, stds, derived_means, derived_stds)
        raw_prediction = sum(coef * value for coef, value in zip(coefficients, x))
        raw_predictions.append(raw_prediction)
        prediction = max(0.0, raw_prediction)
        if prediction != raw_prediction:
            clipped_predictions += 1
        if prediction <= 14:
            status_counts["segera_ganti"] += 1
        elif prediction <= 28:
            status_counts["pesan_pengganti_sekarang"] += 1
        else:
            status_counts["aman"] += 1
        actual = float(row["rul_hari"])
        error = abs(actual - prediction)
        errors.append(error)
        if actual <= 60:
            critical_errors.append(error)

    metrics = {
        "rows_all": len(errors),
        "rows_rul_le_60": len(critical_errors),
        "mae_all_days": round(statistics.fmean(errors), 4),
        "mae_rul_le_60_days": round(statistics.fmean(critical_errors), 4),
        "negative_raw_predictions_clipped_to_zero": clipped_predictions,
    }
    feature_names = [
        "intercept", "beban_z", "arus_z", "tegangan_z", "suhu_z", "getaran_z", "kecepatan_z",
        *DERIVED, "tipe_11kW", "tipe_22kW",
    ]
    model = {
        "model": "polynomial_regression_ols_degree_2_selected_terms",
        "split_manifest": "artifacts/split_motor_kasus03.csv",
        "feature_names": feature_names,
        "coefficients": coefficients,
        "median_imputation_train": impute,
        "numeric_means_train": means,
        "numeric_stds_train": stds,
        "training_sensor_ranges": training_sensor_ranges,
        "nonlinear_term_means_train": derived_means,
        "nonlinear_term_stds_train": derived_stds,
        "target_range_train": {
            "min": min(float(row["rul_hari"]) for row in train),
            "max": max(float(row["rul_hari"]) for row in train),
        },
        "categorical_reference": "5.5kW",
        "metrics_validation": metrics,
        "validation_status_counts_at_28_14_day_thresholds": status_counts,
    }
    model_path = output_dir / "polynomial_regression_kasus03.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"MODEL={model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python polynomial_regression_kasus03.py raw.csv split_motor_kasus03.csv output_dir")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
