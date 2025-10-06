import argparse
import pandas as pd
from typing import Dict, Any, List

# ============================================================
# Constantes y configuración por defecto
# ============================================================

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000,
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
    "score_to_decision": {"reject_at": 10, "review_at": 4},
}

# Permitir override de umbrales mediante variables de entorno
try:
    import os as _os

    _rej = _os.getenv("REJECT_AT")
    _rev = _os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass


# ============================================================
# Funciones utilitarias
# ============================================================

def is_night(hour: int) -> bool:
    """Determina si la hora corresponde a horario nocturno."""
    return hour >= 22 or hour <= 5


def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    """Verifica si el monto supera el umbral para el tipo de producto."""
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


# ============================================================
# Función principal: evaluación de riesgo
# ============================================================

def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa una transacción y retorna decisión, score y razones."""
    score = 0
    reasons: List[str] = []
    rep = str(row.get("user_reputation", "new")).lower()

    # 1️⃣ Hard block
    if _is_hard_block(row, cfg):
        return _reject_hard()

    # 2️⃣ Riesgos categóricos
    score, reasons = _apply_categorical_risks(row, cfg, score, reasons)

    # 3️⃣ Reputación
    score, reasons = _apply_reputation(rep, cfg, score, reasons)

    # 4️⃣ Riesgos adicionales (hora, país, monto, latencia)
    score, reasons = _apply_contextual_risks(row, cfg, score, reasons, rep)

    # 5️⃣ Buffer de frecuencia
    score, reasons = _apply_frequency_buffer(row, cfg, score, reasons, rep)

    # 6️⃣ Decisión final
    return _map_decision(score, reasons, cfg)


# ============================================================
# Funciones auxiliares internas
# ============================================================

def _is_hard_block(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Bloquea si hay múltiples contracargos e IP de alto riesgo."""
    return (
        int(row.get("chargeback_count", 0)) >= cfg["chargeback_hard_block"]
        and str(row.get("ip_risk", "low")).lower() == "high"
    )


def _reject_hard() -> Dict[str, Any]:
    """Respuesta directa para casos de hard block."""
    return {
        "decision": DECISION_REJECTED,
        "risk_score": 100,
        "reasons": "hard_block:chargebacks>=2+ip_high",
    }


def _apply_categorical_risks(row, cfg, score, reasons):
    """Suma puntajes basados en riesgos categóricos (IP, email, dispositivo)."""
    for field in ["ip_risk", "email_risk", "device_fingerprint_risk"]:
        val = str(row.get(field, "low")).lower()
        add = cfg["score_weights"][field].get(val, 0)
        if add:
            score += add
            reasons.append(f"{field}:{val}(+{add})")
    return score, reasons


def _apply_reputation(rep, cfg, score, reasons):
    """Ajusta puntaje según reputación del usuario."""
    add = cfg["score_weights"]["user_reputation"].get(rep, 0)
    if add:
        score += add
        reasons.append(f"user_reputation:{rep}({('+' if add >= 0 else '')}{add})")
    return score, reasons


def _apply_contextual_risks(row, cfg, score, reasons, rep):
    """Evalúa riesgos dependientes de contexto (hora, país, monto, latencia)."""
    hr = int(row.get("hour", 12))
    if is_night(hr):
        add = cfg["score_weights"]["night_hour"]
        score += add
        reasons.append(f"night_hour:{hr}(+{add})")

    bin_c, ip_c = str(row.get("bin_country", "")).upper(), str(row.get("ip_country", "")).upper()
    if bin_c and ip_c and bin_c != ip_c:
        add = cfg["score_weights"]["geo_mismatch"]
        score += add
        reasons.append(f"geo_mismatch:{bin_c}!={ip_c}(+{add})")

    amount = float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        score += add
        reasons.append(f"high_amount:{ptype}:{amount}(+{add})")
        if rep == "new":
            add2 = cfg["score_weights"]["new_user_high_amount"]
            score += add2
            reasons.append(f"new_user_high_amount(+{add2})")

    lat = int(row.get("latency_ms", 0))
    if lat >= cfg["latency_ms_extreme"]:
        add = cfg["score_weights"]["latency_extreme"]
        score += add
        reasons.append(f"latency_extreme:{lat}ms(+{add})")

    return score, reasons


def _apply_frequency_buffer(row, cfg, score, reasons, rep):
    """Aplica un pequeño descuento si el usuario es frecuente."""
    freq = int(row.get("customer_txn_30d", 0))
    if rep in ("recurrent", "trusted") and freq >= 3 and score > 0:
        score -= 1
        reasons.append("frequency_buffer(-1)")
    return score, reasons


def _map_decision(score, reasons, cfg):
    """Mapea el score total a una decisión final."""
    reject_at = cfg["score_to_decision"]["reject_at"]
    review_at = cfg["score_to_decision"]["review_at"]

    if score >= reject_at:
        decision = DECISION_REJECTED
    elif score >= review_at:
        decision = DECISION_IN_REVIEW
    else:
        decision = DECISION_ACCEPTED

    return {"decision": decision, "risk_score": int(score), "reasons": ";".join(reasons)}


# ============================================================
# Ejecución principal
# ============================================================

def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Evalúa todas las filas del CSV y exporta resultados."""
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = [assess_row(row, cfg) for _, row in df.iterrows()]

    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]

    out.to_csv(output_csv, index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    parser.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = parser.parse_args()

    out = run(args.input, args.output)
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
