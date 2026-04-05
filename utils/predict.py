import logging
import numpy as np
import pandas as pd
import pickle
import os

logger = logging.getLogger(__name__)

_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "loan_model.pkl")
    with open(model_path, "rb") as f:
        _MODEL = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return _MODEL

DEFAULT_THRESHOLD = 0.25

def _get_threshold(model_meta) -> float:
    if model_meta and "threshold" in model_meta:
        return model_meta["threshold"]
    return DEFAULT_THRESHOLD

def _prob_to_risk_score(probability: float) -> int:
    clamped = max(0.0, min(1.0, probability))
    return int(round(850 - clamped * (850 - 300)))

def _confidence(probability: float, threshold: float) -> str:
    distance = abs(probability - threshold)
    if distance >= 0.30:
        return "HIGH"
    elif distance >= 0.15:
        return "MEDIUM"
    return "LOW"

_RISK_RULES = [
    ("NumberOfTimes90DaysLate",               1,    ">=", "Has {:.0f} payment(s) 90+ days late"),
    ("NumberOfTime60-89DaysPastDueNotWorse",  2,    ">=", "Has {:.0f} payment(s) 60-89 days late"),
    ("NumberOfTime30-59DaysPastDueNotWorse",  3,    ">=", "Has {:.0f} payment(s) 30-59 days late"),
    ("RevolvingUtilizationOfUnsecuredLines",  0.75, ">=", "Credit utilization {:.0%} (very high)"),
    ("RevolvingUtilizationOfUnsecuredLines",  0.50, ">=", "Credit utilization {:.0%} (elevated)"),
    ("DebtRatio",                             0.43, ">=", "Debt ratio {:.2f} (above safe threshold)"),
    ("MonthlyIncome",                         2000, "<=", "Monthly income ${:,.0f} (low)"),
]

def _get_risk_factors(data: dict, decision: str) -> list:
    factors = []
    for feat, threshold, cmp, tmpl in _RISK_RULES:
        val = data.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if (cmp == ">=" and val >= threshold) or (cmp == "<=" and val <= threshold):
            factors.append(tmpl.format(val))
    if not factors and decision == "APPROVE":
        factors.append("Profile meets all standard credit criteria")
    return factors[:3]


def predict(data: dict) -> dict:
    warnings = data.pop("_warnings", [])

    row_df = pd.DataFrame([data])

    try:
        model_meta     = _load_model()
        pipeline       = model_meta["pipeline"]
        indicator_cols = model_meta.get("indicator_cols", [])

        # Add missingness indicator columns the model was trained with
        for col in indicator_cols:
            source_col = col.replace("_was_missing", "")
            if source_col in row_df.columns:
                row_df[col] = row_df[source_col].isna().astype(int)
            else:
                row_df[col] = 0

        probability = float(pipeline.predict_proba(row_df)[0, 1])

    except Exception as exc:
        logger.exception("Model prediction failed")
        return {
            "probability":  None,
            "risk_score":   None,
            "decision":     "ERROR",
            "confidence":   None,
            "risk_factors": [],
            "warnings":     warnings,
            "error":        str(exc),
        }

    threshold = _get_threshold(model_meta)
    decision  = "REJECT" if probability >= threshold else "APPROVE"

    return {
        "probability":  round(probability, 4),
        "risk_score":   _prob_to_risk_score(probability),
        "decision":     decision,
        "confidence":   _confidence(probability, threshold),
        "risk_factors": _get_risk_factors(data, decision),
        "warnings":     warnings,
        "error":        None,
    }
