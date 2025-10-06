import argparse
import pandas as pd
from typing import Dict, Any, List

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD tuning)
try:
    import os as _os
    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:  # pragma: no cover
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:  # pragma: no cover
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:  # pragma: no cover
    pass


def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5


def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


# ============================================================
# =============== MAIN EVALUATION FUNCTION ===================
# ============================================================

def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa una transacción y devuelve la decisión de riesgo."""
    score = 0
    reasons: List[str] = []
    rep = str(row.get("user_reputation", "new")).lower()

    # 1. Hard block
    hard = _check_hard_block(row, cfg)
    if hard:
        return hard

    # 2. Riesgos categóricos
    score, reasons = _apply_categorical_risks(row, cfg, score, reasons)

    # 3. Reputación
    score, reasons = _apply_reputation(rep, cfg, score, reasons)

    # 4. Factores contextuales
    score, reasons = _apply_contextual_factors(row, cfg, score, reasons, rep)

    # 5. Buffer de frecuencia
    score, reasons = _apply_frequency_buffer(row, score, reasons, rep)

    # 6. Decisión final
    return _make_decision(score, reasons, cfg)


# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def _check_hard_block(row, cfg):
    """Rechaza automáticamente si hay demasiados chargebacks y alto riesgo IP."""
    if int(row.get("chargeback_count", 0)) >= cfg["chargeback_hard_block"] and \
       str(row.get("ip_risk", "low")).lower() == "high":
        return {
            "decision": DECISION_REJECTED,
            "risk_score": 100,
            "reasons": "hard_block:chargebacks>=2+ip_high",
        }
    return None


def _apply_categorical_risks(row, cfg, score, reasons):
    """Evalúa riesgos categóricos (IP, email, fingerprint)."""
    for field in ["ip_risk", "email_risk", "device_fingerprint_risk"]:
        val = str(row.get(field, "low")).lower()
        add = cfg["score_weights"][field].get(val, 0)
        if add:
            score += add
            reasons.append(f"{field}:{val}(+{add})")
    return score, reasons


def _apply_reputation(rep, cfg, score, reasons):
    """Aplica el puntaje de reputación del usuario."""
    add = cfg["score_weights"]["user_reputation"].get(rep, 0)
    if add:
        score += add
        reasons.append(f"user_reputation:{rep}({('+' if add >= 0 else '')}{add})")
    return score, reasons


def _apply_contextual_factors(row, cfg, score, reasons, rep):
    """Evalúa factores de contexto como hora, geolocalización, monto, latencia."""
    hr = int(row.get("hour", 12))
    if is_night(hr):
        _add_reason(reasons, f"night_hour:{hr}", cfg["score_weights"]["night_hour"], score)

    bin_c = str(row.get("bin_country", "")).upper()
    ip_c = str(row.get("ip_country", "")).upper()
    if bin_c and ip_c and bin_c != ip_c:
        _add_reason(reasons, f"geo_mismatch:{bin_c}!={ip_c}", cfg["score_weights"]["geo_mismatch"], score)

    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        _add_reason(reasons, f"high_amount:{ptype}:{amount}", cfg["score_weights"]["high_amount"], score)
        if rep == "new":
            _add_reason(reasons, "new_user_high_amount", cfg["score_weights"]["new_user_high_amount"], score)

    lat = int(row.get("latency_ms", 0))
    if lat >= cfg["latency_ms_extreme"]:
        _add_reason(reasons, f"latency_extreme:{lat}ms", cfg["score_weights"]["latency_extreme"], score)

    return score, reasons


def _apply_frequency_buffer(row, score, reasons, rep):
    """Reduce 1 punto si el usuario confiable tiene muchas transacciones recientes."""
    freq = int(row.get("customer_txn_30d", 0))
    if rep in ("recurrent", "trusted") and freq >= 3 and score > 0:
        score -= 1
        reasons.append("frequency_buffer(-1)")
    return score, reasons


def _make_decision(score, reasons, cfg):
    """Devuelve la decisión final según el puntaje total."""
    if score >= cfg["score_to_decision"]["reject_at"]:
        decision = DECISION_REJECTED
    elif score >= cfg["score_to_decision"]["review_at"]:
        decision = DECISION_IN_REVIEW
    else:
        decision = DECISION_ACCEPTED
    return {
        "decision": decision,
        "risk_score": int(score),
        "reasons": ";".join(reasons)
    }


def _add_reason(reasons, label, add):
    """Agrega una razón al listado."""
    reasons.append(f"{label}(+{add})")


# ============================================================
# ====================== CSV HANDLERS ========================
# ============================================================

def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Ejecuta la evaluación de todas las transacciones en el CSV."""
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = assess_row(row, cfg)
        results.append(res)
    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]
    out.to_csv(output_csv, index=False)
    return out


# pragma: no cover
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv",
                    help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv",
                    help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))


# pragma: no cover
if __name__ == "__main__":
    main()
