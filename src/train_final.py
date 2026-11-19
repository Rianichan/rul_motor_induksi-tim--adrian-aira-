"""Fit locked final models on train+validation and evaluate the held-out motors once.

Run from the project root:
python src/train_final.py data/raw/raw.csv artifacts/split_motor_kasus03.csv artifacts
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

from train_candidate import (
    DERIVED,
    NUMERIC,
    add_mode_type,
    base_vector,
    canonical_id,
    canonical_type,
    derived_vector,
    feature_vector,
    number_or_missing,
    solve_linear_system,
)


def read_rows(raw_path, split_path):
    with open(split_path, newline="", encoding="utf-8-sig") as handle:
        split_map = {row["motor_id"]: row["split"] for row in csv.DictReader(handle)}

    rows, seen = [], set()
    with open(raw_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            row = dict(source)
            row["motor_id"] = canonical_id(row["motor_id"])
            row["tipe_motor"] = canonical_type(row["tipe_motor"])
            key = tuple(row[name] for name in reader.fieldnames)
            if key in seen:
                continue
            seen.add(key)
            if row["motor_id"] not in split_map:
                raise ValueError(f"No split assignment for {row['motor_id']}")
            row["split"] = split_map[row["motor_id"]]
            rows.append(row)
    development = [r for r in rows if r["split"] in {"train", "validation"}]
    held_out = [r for r in rows if r["split"] == "test"]
    add_mode_type(development)
    add_mode_type(held_out)
    return development, held_out


def fit_polynomial(rows):
    impute, means, stds, ranges = {}, {}, {}, {}
    for name in NUMERIC:
        observed = [number_or_missing(r.get(name), name) for r in rows]
        observed = [value for value in observed if value is not None]
        ordered = sorted(observed)
        impute[name] = statistics.median(observed)
        ranges[name] = {
            "min": ordered[0],
            "p01": ordered[int(0.01 * (len(ordered) - 1))],
            "p99": ordered[int(0.99 * (len(ordered) - 1))],
            "max": ordered[-1],
        }
        filled = [impute[name] if value is None else value for value in
                  [number_or_missing(r.get(name), name) for r in rows]]
        means[name] = statistics.fmean(filled)
        stds[name] = math.sqrt(statistics.fmean((x - means[name]) ** 2 for x in filled)) or 1.0

    derived_values = {name: [] for name in DERIVED}
    for row in rows:
        z = base_vector(row, impute, means, stds)
        for name, value in zip(DERIVED, derived_vector(z)):
            derived_values[name].append(value)
    derived_means = {name: statistics.fmean(vals) for name, vals in derived_values.items()}
    derived_stds = {
        name: math.sqrt(statistics.fmean((x - derived_means[name]) ** 2 for x in vals)) or 1.0
        for name, vals in derived_values.items()
    }

    p = 14
    gram, cross = [[0.0] * p for _ in range(p)], [0.0] * p
    for row in rows:
        vector = feature_vector(row, impute, means, stds, derived_means, derived_stds)
        target = float(row["rul_hari"])
        for i, xi in enumerate(vector):
            cross[i] += xi * target
            for j, xj in enumerate(vector):
                gram[i][j] += xi * xj
    coefficients = solve_linear_system(gram, cross)
    return {
        "model": "polynomial_regression_ols_degree_2_selected_terms",
        "feature_names": ["intercept", "beban_z", "arus_z", "tegangan_z", "suhu_z",
                          "getaran_z", "kecepatan_z", *DERIVED, "tipe_11kW", "tipe_22kW"],
        "coefficients": coefficients,
        "median_imputation_train_validation": impute,
        "median_imputation_train": impute,
        "numeric_means_train": means,
        "numeric_stds_train": stds,
        "training_sensor_ranges": ranges,
        "nonlinear_term_means_train": derived_means,
        "nonlinear_term_stds_train": derived_stds,
        "target_range_train": {"min": min(float(r["rul_hari"]) for r in rows),
                               "max": max(float(r["rul_hari"]) for r in rows)},
        "categorical_reference": "5.5kW",
    }


def fit_linear(rows):
    impute, means, stds = {}, {}, {}
    for name in NUMERIC:
        observed = [number_or_missing(r.get(name), name) for r in rows]
        observed = [x for x in observed if x is not None]
        impute[name] = statistics.median(observed)
        values = [impute[name] if x is None else x for x in
                  [number_or_missing(r.get(name), name) for r in rows]]
        means[name] = statistics.fmean(values)
        stds[name] = math.sqrt(statistics.fmean((x - means[name]) ** 2 for x in values)) or 1.0

    def vector(row):
        z = base_vector(row, impute, means, stds)
        return [1.0, *z, float(row["tipe_motor"] == "11kW"), float(row["tipe_motor"] == "22kW")]

    p = 9
    gram, cross = [[0.0] * p for _ in range(p)], [0.0] * p
    for row in rows:
        x, y = vector(row), float(row["rul_hari"])
        for i, xi in enumerate(x):
            cross[i] += xi * y
            for j, xj in enumerate(x):
                gram[i][j] += xi * xj
    model = {
        "model": "ordinary_least_squares_linear_regression",
        "coefficients": solve_linear_system(gram, cross),
        "feature_names": ["intercept", *[f"{name}_z" for name in NUMERIC], "tipe_11kW", "tipe_22kW"],
        "median_imputation_train": impute,
        "numeric_means_train": means,
        "numeric_stds_train": stds,
        "categorical_reference": "5.5kW",
    }
    return model, vector


def score(predictions, actuals):
    errors = [abs(y - max(0.0, p)) for y, p in zip(actuals, predictions)]
    critical = [abs(y - max(0.0, p)) for y, p in zip(actuals, predictions) if y <= 60]
    return {
        "rows_all": len(errors),
        "rows_rul_le_60": len(critical),
        "mae_all_days": round(statistics.fmean(errors), 4),
        "mae_rul_le_60_days": round(statistics.fmean(critical), 4),
        "negative_raw_predictions_clipped_to_zero": sum(p < 0 for p in predictions),
    }


def main(raw_path, split_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    development, test_rows = read_rows(raw_path, split_path)
    y_test = [float(row["rul_hari"]) for row in test_rows]

    polynomial = fit_polynomial(development)
    def poly_vector(row):
        z = base_vector(row, polynomial["median_imputation_train"],
                        polynomial["numeric_means_train"], polynomial["numeric_stds_train"])
        derived = [(x - polynomial["nonlinear_term_means_train"][name]) /
                   polynomial["nonlinear_term_stds_train"][name]
                   for name, x in zip(DERIVED, derived_vector(z))]
        return [1.0, *z, *derived, float(row["tipe_motor"] == "11kW"),
                float(row["tipe_motor"] == "22kW")]
    poly_preds = [sum(c * x for c, x in zip(polynomial["coefficients"], poly_vector(row)))
                  for row in test_rows]
    polynomial["metrics_test"] = score(poly_preds, y_test)
    polynomial["fit_rows"] = len(development)
    polynomial["split_manifest"] = "artifacts/split_motor_kasus03.csv"
    (output / "final_model.json").write_text(json.dumps(polynomial, ensure_ascii=False, indent=2), encoding="utf-8")

    linear, linear_vector = fit_linear(development)
    linear_preds = [sum(c * x for c, x in zip(linear["coefficients"], linear_vector(row)))
                    for row in test_rows]
    linear["metrics_test"] = score(linear_preds, y_test)
    linear["fit_rows"] = len(development)
    (output / "final_baseline.json").write_text(json.dumps(linear, ensure_ascii=False, indent=2), encoding="utf-8")

    results = {
        "evaluation": "one-time held-out test evaluation after model selection on validation",
        "development_motors": 255,
        "test_motors": 45,
        "baseline_test": linear["metrics_test"],
        "final_model_test": polynomial["metrics_test"],
    }
    (output / "final_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python src/train_final.py data/raw/raw.csv artifacts/split_motor_kasus03.csv artifacts")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
