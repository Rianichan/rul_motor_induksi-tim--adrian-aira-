"""Decision status for the motor replacement lead-time bands."""

from math import isfinite


def classify_status(rul_days, rules):
    """Return the maintenance action for a non-negative RUL estimate."""
    estimate = float(rul_days)
    if not isfinite(estimate) or estimate < 0:
        raise ValueError("RUL estimate must be a finite, non-negative number")

    order_now = float(rules["order_now_threshold_days"])
    replace_soon = float(rules["replace_soon_threshold_days"])
    if estimate <= replace_soon:
        return {
            "status": "segera_ganti",
            "label": "Segera ganti",
            "action": "Prioritaskan penggantian dan siapkan mitigasi bila motor pengganti belum tersedia.",
        }
    if estimate <= order_now:
        return {
            "status": "pesan_pengganti_sekarang",
            "label": "Pesan pengganti sekarang",
            "action": "Pesan motor pengganti sekarang agar tiba sebelum perkiraan kerusakan.",
        }
    return {
        "status": "aman",
        "label": "Aman",
        "action": "Belum masuk jendela waktu pengiriman; lanjutkan pemantauan rutin.",
    }
