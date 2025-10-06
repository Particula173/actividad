import pandas as pd
from decision_engine import (
    assess_row,
    DEFAULT_CONFIG,
    DECISION_ACCEPTED,
    DECISION_REJECTED,
)

# --- Helper para crear una fila base ---
def make_row(**kwargs):
    base = dict(
        chargeback_count=0,
        ip_risk="low",
        email_risk="low",
        device_fingerprint_risk="low",
        user_reputation="new",
        hour=12,
        bin_country="MX",
        ip_country="MX",
        amount_mxn=100,
        product_type="digital",
        latency_ms=10,
        customer_txn_30d=0,
    )
    base.update(kwargs)
    return pd.Series(base)


def test_hard_block_path():
    """Cubre el retorno temprano (hard block)"""
    row = make_row(chargeback_count=3, ip_risk="high")
    result = assess_row(row, DEFAULT_CONFIG)
    assert result["decision"] == DECISION_REJECTED
    assert "hard_block" in result["reasons"]


def test_geo_mismatch_branch():
    """Ejecuta la rama geo_mismatch"""
    row = make_row(bin_country="US", ip_country="MX")
    result = assess_row(row, DEFAULT_CONFIG)
    assert "geo_mismatch" in result["reasons"]


def test_latency_extreme_branch():
    """Ejecuta la rama de latencia extrema"""
    row = make_row(latency_ms=3000)
    result = assess_row(row, DEFAULT_CONFIG)
    assert "latency_extreme" in result["reasons"]


def test_frequency_buffer_branch():
    """Ejecuta la rama frequency_buffer"""
    row = make_row(
        user_reputation="trusted", customer_txn_30d=4, ip_risk="medium"
    )
    result = assess_row(row, DEFAULT_CONFIG)
    assert "frequency_buffer" in result["reasons"]


def test_low_risk_accept():
    """Asegura que se ejecuta la rama ACCEPTED"""
    row = make_row()
    result = assess_row(row, DEFAULT_CONFIG)
    assert result["decision"] == DECISION_ACCEPTED
