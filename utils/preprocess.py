# =============================================================================
#  utils/preprocess.py  —  Input cleaning & validation
#  Keeps all data-cleaning logic away from the model and the UI.
# =============================================================================

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Feature catalogue ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Hard physical limits — values outside these are coerced to NaN
VALID_RANGES = {
    "age":                                  (18, 100),
    "DebtRatio":                            (0, 500),
    "MonthlyIncome":                        (0, 10_000_000),
    "RevolvingUtilizationOfUnsecuredLines": (0, 50),
    "NumberOfDependents":                   (0, 30),
}

# Known dataset artifacts in delinquency count columns
_ARTIFACT_VALUES = {96, 98}
_DELINQUENCY_COLS = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
}

# Reject if more than this fraction of features are missing
MAX_MISSING_FRACTION = 0.50


def preprocess_input(raw: dict) -> dict:
    """
    Validate and clean one applicant dict from the UI.

    Steps:
      1. Type-coerce all numeric fields (str → float, bad string → NaN)
      2. Null out physically impossible values
      3. Remove known dataset artifacts (96 / 98 in delinquency columns)
      4. Hard-reject if >50% of fields are missing
      5. Attach a '_warnings' list for the UI to surface

    Args:
        raw: dict of {feature_name: raw_value} from the Streamlit form

    Returns:
        Cleaned dict, ready for utils/predict.py.
        Raises ValueError if the row is unprocessable.
    """
    cleaned  = {}
    warnings = []

    for feat in NUMERIC_FEATURES:
        raw_val = raw.get(feat, None)

        # ── Missing / None / explicit NaN ────────────────────────────────────
        if raw_val is None or raw_val == "":
            cleaned[feat] = np.nan
            warnings.append(f"'{feat}' is missing — will be imputed")
            continue

        if isinstance(raw_val, float) and np.isnan(raw_val):
            cleaned[feat] = np.nan
            warnings.append(f"'{feat}' is NaN — will be imputed")
            continue

        # ── Coerce to float ───────────────────────────────────────────────────
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            cleaned[feat] = np.nan
            warnings.append(f"'{feat}' = '{raw_val}' is not a number — set to NaN")
            continue

        # ── Range check ───────────────────────────────────────────────────────
        if feat in VALID_RANGES:
            lo, hi = VALID_RANGES[feat]
            if not (lo <= val <= hi):
                cleaned[feat] = np.nan
                warnings.append(
                    f"'{feat}' = {val} is outside [{lo}, {hi}] — set to NaN"
                )
                continue

        # ── Dataset artifact filter ───────────────────────────────────────────
        if feat in _DELINQUENCY_COLS and val in _ARTIFACT_VALUES:
            cleaned[feat] = np.nan
            warnings.append(
                f"'{feat}' = {val:.0f} is a known dataset artifact — set to NaN"
            )
            continue

        cleaned[feat] = val

    # ── Too many missing? ────────────────────────────────────────────────────
    n_missing  = sum(1 for v in cleaned.values() if v is None or (isinstance(v, float) and np.isnan(v)))
    frac       = n_missing / len(NUMERIC_FEATURES) if NUMERIC_FEATURES else 0

    if frac > MAX_MISSING_FRACTION:
        raise ValueError(
            f"{n_missing}/{len(NUMERIC_FEATURES)} fields are missing "
            f"({frac:.0%}). Cannot produce a reliable prediction."
        )

    # Attach warnings for the UI — predict.py will pop them out
    cleaned["_warnings"] = warnings
    return cleaned
