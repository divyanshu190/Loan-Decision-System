# =============================================================================
#  predict.py  —  Production prediction engine
#  Use this for single predictions, batch predictions, or API calls
# =============================================================================
"""
Handles every real-world prediction problem:
  - Missing fields         → imputed by the saved pipeline
  - Wrong data types       → coerced or nulled by validator
  - Impossible values      → nulled by validator
  - Too many missing       → hard reject with clear error message
  - Batch predictions      → memory-efficient chunked processing
  - Risk factors           → human-readable explanation of each decision
"""

import pickle
import logging
import numpy as np
import pandas as pd
import os
from typing import Union

from config    import MODEL_SAVE_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ANOMALY_MODEL_PATH
from validator import validate_single_row, ValidationError

logger = logging.getLogger(__name__)

# =============================================================================
#  Risk factor rules — maps feature values to human-readable warnings
#  Used to explain WHY a loan was rejected or approved
# =============================================================================

RISK_RULES = [
    # (feature, threshold, comparison, message_template)
    ("NumberOfTimes90DaysLate",               1,    ">=", "Has {} payment(s) 90+ days late (serious delinquency)"),
    ("NumberOfTime60-89DaysPastDueNotWorse",  2,    ">=", "Has {} payment(s) 60-89 days late"),
    ("NumberOfTime30-59DaysPastDueNotWorse",  3,    ">=", "Has {} payment(s) 30-59 days late"),
    ("RevolvingUtilizationOfUnsecuredLines",  0.75, ">=", "Credit utilization is {:.0%} (very high, ideal <30%)"),
    ("RevolvingUtilizationOfUnsecuredLines",  0.50, ">=", "Credit utilization is {:.0%} (elevated, ideal <30%)"),
    ("DebtRatio",                             0.43, ">=", "Debt-to-income ratio is {:.2f} (above safe threshold of 0.43)"),
    ("MonthlyIncome",                         2000, "<=", "Monthly income is ${:,.0f} (low relative to typical obligations)"),
    ("NumberOfOpenCreditLinesAndLoans",       12,   ">=", "Has {} open credit lines (many active obligations)"),
]


def get_risk_factors(data: dict, decision: str) -> list:
    """
    Generate human-readable risk factors explaining the decision.
    Checks the raw input values against known risk thresholds.
    Returns top 3 most severe risk factors found.
    """
    factors = []
    for feat, threshold, cmp, message in RISK_RULES:
        val = data.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        triggered = (cmp == ">=" and val >= threshold) or \
                    (cmp == "<=" and val <= threshold)
        if triggered:
            factors.append(message.format(val))

    if not factors and decision == "APPROVE":
        factors.append("Profile meets all standard credit criteria")

    return factors[:3]   # top 3 most severe


# =============================================================================
#  Load model (once, at startup)
# =============================================================================

_MODEL_CACHE  = None
_ANOMALY_CACHE = None


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with open(MODEL_SAVE_PATH, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_SAVE_PATH}")
    return _MODEL_CACHE


def load_anomaly_detector():
    """Load IsolationForest anomaly detector. Returns None if not yet trained."""
    global _ANOMALY_CACHE
    if _ANOMALY_CACHE is None:
        if not os.path.exists(ANOMALY_MODEL_PATH):
            logger.warning(
                "Anomaly detector not found. Run train.py first. "
                "Skipping anomaly scoring."
            )
            return None
        with open(ANOMALY_MODEL_PATH, "rb") as f:
            _ANOMALY_CACHE = pickle.load(f)
        logger.info(f"Anomaly detector loaded from {ANOMALY_MODEL_PATH}")
    return _ANOMALY_CACHE


def score_anomaly(cleaned: dict, row_df: pd.DataFrame) -> tuple[float, bool]:
    """
    Score an application with the IsolationForest.

    Returns:
        (anomaly_score, is_suspicious)
        anomaly_score < 0  → outlier / deviates from normal applications
        is_suspicious      → True if below the configured flag threshold
    """
    detector = load_anomaly_detector()
    if detector is None:
        return 0.0, False

    iso         = detector["iso_forest"]
    imputer     = detector["imputer"]
    scaler      = detector["scaler"]
    feat_cols   = detector["feature_cols"]
    flag_thresh = detector["flag_threshold"]

    # Build input vector using only the features the detector was trained on
    X_det = row_df.reindex(columns=feat_cols, fill_value=np.nan)
    X_imp = imputer.transform(X_det)
    X_scl = scaler.transform(X_imp)

    # decision_function returns negative scores for outliers
    anomaly_score = float(iso.decision_function(X_scl)[0])
    is_suspicious = anomaly_score < flag_thresh

    return round(anomaly_score, 4), is_suspicious


# =============================================================================
#  Single prediction
# =============================================================================

def predict_one(data: dict) -> dict:
    """
    Make a prediction for one loan applicant.

    Args:
        data: dict with feature names as keys, raw values as values.
              Missing keys are treated as missing values (NaN).

    Returns:
        dict with:
          - decision      : "APPROVE", "REJECT", or "REVIEW" (suspicious profile)
          - default_prob  : float, probability of default (0-1)
          - confidence    : "HIGH" / "MEDIUM" / "LOW"
          - risk_factors  : list of human-readable reasons for the decision
          - anomaly_score : IsolationForest score (negative = suspicious)
          - suspicious    : bool, True if application deviates from normal patterns
          - warnings      : list of data quality issues found (missing/coerced fields)
          - error         : None or error message
    """
    metadata  = load_model()
    pipeline  = metadata["pipeline"]
    threshold = metadata["threshold"]

    # ── Validate and clean the input ───────────────────────────────────────────────
    try:
        cleaned = validate_single_row(data)
    except ValidationError as e:
        return {
            "decision":     "ERROR",
            "default_prob": None,
            "confidence":   None,
            "risk_factors": [],
            "anomaly_score": None,
            "suspicious":   False,
            "warnings":     [],
            "error":        str(e),
        }

    # Extract and remove internal validator flags from cleaned dict
    suspicious_flags = cleaned.pop("_suspicious_flags", [])
    data_issues      = cleaned.pop("_data_issues", [])

    # Warn about missing/coerced values (data_issues are plain strings already)
    warnings_list = [
        f"{issue} — imputed by model"
        for issue in data_issues
        if "missing" in issue.lower() or "nan" in issue.lower() or "outside" in issue.lower()
    ]

    # ── Build a single-row DataFrame ─────────────────────────────────────────────────
    row_df = pd.DataFrame([cleaned])

    # Add missingness indicator columns (must match training)
    for col in metadata.get("indicator_cols", []):
        source_col = col.replace("_was_missing", "")
        if source_col in row_df.columns:
            row_df[col] = row_df[source_col].isna().astype(int)
        else:
            row_df[col] = 0

    # ── Anomaly / fake-data scoring ───────────────────────────────────────────────
    anomaly_score, is_ml_suspicious = score_anomaly(cleaned, row_df)

    # Combine ML anomaly flag with rule-based validator flags
    is_suspicious = is_ml_suspicious or len(suspicious_flags) > 0

    # ── Predict default probability ─────────────────────────────────────────────
    try:
        default_prob = float(pipeline.predict_proba(row_df)[0, 1])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "decision":     "ERROR",
            "default_prob": None,
            "confidence":   None,
            "risk_factors": [],
            "anomaly_score": anomaly_score,
            "suspicious":   is_suspicious,
            "warnings":     warnings_list,
            "error":        f"Model prediction error: {str(e)}",
        }

    # Decision with suspicious override:
    # If model says APPROVE but application looks suspicious → escalate to REVIEW
    if default_prob >= threshold:
        decision = "REJECT"
    elif is_suspicious:
        decision = "REVIEW"   # human review recommended
    else:
        decision = "APPROVE"

    # Confidence: distance from threshold
    distance = abs(default_prob - threshold)
    if distance >= 0.30:
        confidence = "HIGH"
    elif distance >= 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"   # borderline — flag for human review

    # Risk factors from rule-based engine
    risk_factors = get_risk_factors(data, decision)

    # Append suspicious flags as additional context
    if suspicious_flags:
        risk_factors = suspicious_flags + risk_factors

    return {
        "decision":     decision,
        "default_prob": round(default_prob, 4),
        "confidence":   confidence,
        "risk_factors": risk_factors,
        "warnings":     warnings_list,
        "error":        None,
    }


# =============================================================================
#  Batch prediction
# =============================================================================

def predict_batch(csv_path: str, output_path: str, chunk_size: int = 1000):
    """
    Run predictions on a large CSV file in memory-efficient chunks.

    Args:
        csv_path    : path to CSV with applicant data
        output_path : where to write results CSV
        chunk_size  : rows per chunk (tune based on RAM)
    """
    metadata       = load_model()
    pipeline       = metadata["pipeline"]
    threshold      = metadata["threshold"]
    indicator_cols = metadata.get("indicator_cols", [])

    logger.info(f"Starting batch prediction: {csv_path}")
    results = []

    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        logger.info(f"Processing chunk {chunk_num + 1}...")

        valid_rows  = []
        row_indices = []
        error_rows  = []

        for idx, row in chunk.iterrows():
            try:
                cleaned = validate_single_row(row.to_dict())
                valid_rows.append(cleaned)
                row_indices.append(idx)
            except ValidationError as e:
                error_rows.append({
                    "original_index": idx,
                    "decision":       "ERROR",
                    "default_prob":   None,
                    "confidence":     None,
                    "risk_factors":   [],
                    "error":          str(e),
                })

        if not valid_rows:
            results.extend(error_rows)
            continue

        valid_df = pd.DataFrame(valid_rows, index=row_indices)

        for col in indicator_cols:
            source_col = col.replace("_was_missing", "")
            if source_col in valid_df.columns:
                valid_df[col] = valid_df[source_col].isna().astype(int)
            else:
                valid_df[col] = 0

        probs      = pipeline.predict_proba(valid_df)[:, 1]
        decisions  = ["REJECT" if p >= threshold else "APPROVE" for p in probs]
        distances  = [abs(p - threshold) for p in probs]
        confidences = [
            "HIGH" if d >= 0.30 else ("MEDIUM" if d >= 0.15 else "LOW")
            for d in distances
        ]

        for i, idx in enumerate(row_indices):
            results.append({
                "original_index": idx,
                "decision":       decisions[i],
                "default_prob":   round(float(probs[i]), 4),
                "confidence":     confidences[i],
                "risk_factors":   [],
                "error":          None,
            })

        results.extend(error_rows)

    results_df = pd.DataFrame(results).sort_values("original_index")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Batch complete. Results saved to: {output_path}")

    total    = len(results_df)
    approved = (results_df["decision"] == "APPROVE").sum()
    rejected = (results_df["decision"] == "REJECT").sum()
    errors   = (results_df["decision"] == "ERROR").sum()
    logger.info(f"Summary → Total: {total} | Approved: {approved} | "
                f"Rejected: {rejected} | Errors: {errors}")

    return results_df


# =============================================================================
#  Quick test — using cs-training.csv feature names
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Perfect applicant — young professional, low debt, clean record
    good = {
        "RevolvingUtilizationOfUnsecuredLines":  0.05,
        "age":                                   35,
        "NumberOfTime30-59DaysPastDueNotWorse":  0,
        "DebtRatio":                             0.10,
        "MonthlyIncome":                         8000,
        "NumberOfOpenCreditLinesAndLoans":        6,
        "NumberOfTimes90DaysLate":               0,
        "NumberRealEstateLoansOrLines":          1,
        "NumberOfTime60-89DaysPastDueNotWorse":  0,
        "NumberOfDependents":                    1,
    }

    # Risky applicant — maxed credit, serial late payer, low income
    risky = {
        "RevolvingUtilizationOfUnsecuredLines":  0.98,
        "age":                                   38,
        "NumberOfTime30-59DaysPastDueNotWorse":  8,
        "DebtRatio":                             0.95,
        "MonthlyIncome":                         None,   # missing income
        "NumberOfOpenCreditLinesAndLoans":        12,
        "NumberOfTimes90DaysLate":               5,
        "NumberRealEstateLoansOrLines":          0,
        "NumberOfTime60-89DaysPastDueNotWorse":  4,
        "NumberOfDependents":                    4,
    }

    # Borderline — one past-due, otherwise decent
    borderline = {
        "RevolvingUtilizationOfUnsecuredLines":  0.45,
        "age":                                   33,
        "NumberOfTime30-59DaysPastDueNotWorse":  1,
        "DebtRatio":                             0.35,
        "MonthlyIncome":                         4500,
        "NumberOfOpenCreditLinesAndLoans":        7,
        "NumberOfTimes90DaysLate":               0,
        "NumberRealEstateLoansOrLines":          1,
        "NumberOfTime60-89DaysPastDueNotWorse":  0,
        "NumberOfDependents":                    2,
    }

    for label, applicant in [("Good", good), ("Risky", risky), ("Borderline", borderline)]:
        print(f"\n--- {label} Applicant ---")
        result = predict_one(applicant)
        print(f"  Decision     : {result['decision']}")
        print(f"  Default prob : {result['default_prob']}")
        print(f"  Confidence   : {result['confidence']}")
        print(f"  Risk factors : {result['risk_factors']}")
        if result["warnings"]:
            print(f"  Warnings     : {result['warnings']}")
