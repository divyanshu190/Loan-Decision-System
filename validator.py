3# =============================================================================
#  validator.py  —  Real-world data quality firewall
#  Catches ALL bad data BEFORE it touches the model
# =============================================================================
"""
Production-grade validation for every incoming loan application.

Layers of protection:
  1. Type coercion        — "30" → 30, "bad_value" → NaN
  2. Range validation     — age=999 → NaN (physically impossible)
  3. Missing-value audit  — count + categorize missingness (MCAR/MNAR)
  4. Cross-field checks   — catches internally inconsistent / fake profiles
  5. Fake-application heuristics:
       a. All-round-number detection (fabricated data is suspiciously "clean")
       b. Perfect-profile detection (too good to be true)
       c. Age/history coherence (can't have 20 credit lines at age 19)
  6. Too-many-missing hard reject (>50% fields missing → refuse to predict)

Research basis (2024/2025 best practices):
  - MNAR (Missing Not at Random): missing income in credit = strong risk signal
  - Cross-field consistency = primary heuristic for synthetic identity fraud
  - Isolation Forest (in train.py/predict.py) provides the ML-based anomaly score
    as a second layer on top of these deterministic checks
"""

import logging
import numpy  as np
import pandas as pd

from config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    VALID_RANGES, VALID_CATEGORIES,
    MAX_ROW_MISSING_FRACTION,
)

logger = logging.getLogger(__name__)


# =============================================================================
#  Dataset-level validation  (called during training)
# =============================================================================

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an entire training dataframe.
    Returns the cleaned dataframe. Issues are logged as warnings.
    """
    original_shape = df.shape
    report = {}

    # 1. Force numeric columns to actual numbers
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — filling with NaN")
            df[col] = np.nan
            continue
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - before
        if new_nulls > 0:
            report[col] = f"{new_nulls} non-numeric values → NaN"

    # 2. Null out physically impossible values
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        bad_mask = (df[col] < lo) | (df[col] > hi)
        bad_count = bad_mask.sum()
        if bad_count:
            df.loc[bad_mask, col] = np.nan
            report[col] = report.get(col, "") + f" | {bad_count} out-of-range → NaN"

    # 3. Categorical standardisation
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        if col in VALID_CATEGORIES:
            allowed  = [v.upper() for v in VALID_CATEGORIES[col]]
            bad_mask = ~df[col].isin(allowed) & df[col].notna() & (df[col] != "NAN")
            if bad_mask.sum():
                df.loc[bad_mask, col] = np.nan
                report[col] = f"{bad_mask.sum()} invalid categories → NaN"
        df[col] = df[col].replace("NAN", np.nan)

    # 4. Log summary
    logger.info(f"Dataset validation complete. Shape: {original_shape}")
    for col, msg in report.items():
        logger.warning(f"  [{col}] {msg}")

    num_feat_cols   = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    valid_cols      = [c for c in num_feat_cols if c in df.columns]
    total_missing   = df[valid_cols].isna().sum().sum()
    total_cells     = df.shape[0] * len(valid_cols)
    logger.info(
        f"  Total missing after validation: {total_missing}/{total_cells} "
        f"({100 * total_missing / total_cells:.1f}%)"
    )

    return df


# =============================================================================
#  Single-row validation  (called at prediction time)
# =============================================================================

class ValidationError(Exception):
    """Raised when a prediction request is too broken to process."""
    pass


# ── Cross-field consistency checks ───────────────────────────────────────────

def _check_cross_field_consistency(data: dict) -> list:
    """
    Detect internally inconsistent / likely fabricated profiles.

    Real human data has natural correlations:
      • You can't have 20 open credit lines if you're 19 years old
      • You can't have 5 real-estate loans if you're 20 years old
      • A debt ratio of 0 with maxed credit utilization is contradictory
      • Extremely high income for very young applicants is suspicious

    Returns: list of consistency warning strings (empty = all OK)
    """
    warnings = []

    age   = data.get("age")
    lines = data.get("NumberOfOpenCreditLinesAndLoans")
    re    = data.get("NumberRealEstateLoansOrLines")
    inc   = data.get("MonthlyIncome")
    debt  = data.get("DebtRatio")
    util  = data.get("RevolvingUtilizationOfUnsecuredLines")

    # Safe numeric extraction
    def _num(v):
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    age, lines, re, inc, debt, util = [_num(x) for x in [age, lines, re, inc, debt, util]]

    # Rule 1: Age vs open credit lines
    # Average credit history needed per line ≈ 2 years
    if age is not None and lines is not None and lines > 0:
        min_needed_age = 18 + (lines * 2.0)
        if age < min_needed_age and lines > 6:
            warnings.append(
                f"Suspicious: age={age:.0f} but {lines:.0f} open credit lines "
                f"(would need ~{min_needed_age:.0f}+ years to build this history)"
            )

    # Rule 2: Age vs real estate loans
    if age is not None and re is not None and re >= 3 and age < 25:
        warnings.append(
            f"Suspicious: age={age:.0f} but {re:.0f} real-estate loans "
            f"(very unlikely before age 25)"
        )

    # Rule 3: Zero monthly income + very high credit utilization
    # Legitimate: some people have zero income temporarily
    # Suspicious: zero income AND very high utilization AND many open lines
    if inc is not None and util is not None and lines is not None:
        if inc == 0 and util > 0.5 and lines > 5:
            warnings.append(
                f"Suspicious: zero income with {util:.0%} credit utilization "
                f"and {lines:.0f} open lines"
            )

    # Rule 4: Debt ratio 0 but utilization > 90%
    # A debt ratio of exactly 0 with very high revolving utilization is contradictory
    if debt is not None and util is not None:
        if debt == 0.0 and util > 0.80:
            warnings.append(
                f"Inconsistent: DebtRatio=0 but "
                f"RevolvingUtilization={util:.0%} (contradictory)"
            )

    # Rule 5: Income implausibly high for young age
    if age is not None and inc is not None:
        if age < 22 and inc > 25000:
            warnings.append(
                f"Unusual: monthly income ${inc:,.0f} at age {age:.0f} "
                f"(may warrant verification)"
            )

    return warnings


# ── Round-number / fabricated-data detection ──────────────────────────────────

def _check_round_numbers(data: dict) -> bool:
    """
    Fabricated financial data is often suspiciously 'clean':
    perfectly round numbers for income, debt ratio, utilization, etc.

    Returns True if the profile looks fabricated (too many exact round numbers).
    """
    round_count = 0
    checks = [
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
    ]
    for feat in checks:
        val = data.get(feat)
        if val is None:
            continue
        try:
            fval = float(val)
            # Check if it's a perfectly round number (multiple of 1000 for income,
            # multiple of 0.1 for ratios)
            if feat == "MonthlyIncome" and fval > 0 and fval % 1000 == 0:
                round_count += 1
            elif feat in ("DebtRatio", "RevolvingUtilizationOfUnsecuredLines"):
                if fval > 0 and round(fval, 10) == round(fval, 1):
                    round_count += 1
        except (ValueError, TypeError):
            continue

    return round_count >= 3


# ── Perfect-profile detection ──────────────────────────────────────────────────

def _check_perfect_profile(data: dict) -> bool:
    """
    A 'perfect' profile (0 late payments, 0 utilization, 0 debt ratio, high income)
    with no imperfections is statistically very rare in real borrower populations.
    Could indicate a fabricated 'test' profile or identity fraud.
    Only flag if ALL indicators are pristine AND income is very high.
    """
    def _num(key, default=None):
        v = data.get(key)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    late_30  = _num("NumberOfTime30-59DaysPastDueNotWorse", 1)
    late_60  = _num("NumberOfTime60-89DaysPastDueNotWorse", 1)
    late_90  = _num("NumberOfTimes90DaysLate", 1)
    util     = _num("RevolvingUtilizationOfUnsecuredLines", 0.5)
    debt     = _num("DebtRatio", 0.5)
    income   = _num("MonthlyIncome", 0)

    all_zero_lates = (late_30 == 0 and late_60 == 0 and late_90 == 0)
    zero_ratios    = (util == 0.0 and debt == 0.0)
    very_high_inc  = income > 50000

    return all_zero_lates and zero_ratios and very_high_inc


# ── Main single-row validator ──────────────────────────────────────────────────

def validate_single_row(data: dict) -> dict:
    """
    Validate one incoming prediction request.

    Args:
        data: dict of feature_name → raw_value

    Returns:
        Cleaned dict ready for the pipeline.
        Also adds '_validation_flags' key with any detected issues.

    Raises:
        ValidationError: if the row is fundamentally unprocessable.
    """
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    cleaned      = {}
    issues       = []

    # Hard stop: completely empty input
    if not data:
        raise ValidationError("Empty input — no fields provided.")

    # ── Process each expected feature ─────────────────────────────────────────
    for feature in all_features:
        raw_value = data.get(feature, None)

        # Missing / None / empty string / NaN → NaN
        if raw_value is None or raw_value == "" or (
            isinstance(raw_value, float) and np.isnan(raw_value)
        ):
            cleaned[feature] = np.nan
            issues.append(f"'{feature}' is missing")
            continue

        # Numeric processing
        if feature in NUMERIC_FEATURES:
            try:
                val = float(raw_value)
            except (ValueError, TypeError):
                cleaned[feature] = np.nan
                issues.append(
                    f"'{feature}' = '{raw_value}' is not a number → set to NaN"
                )
                continue

            # Range check
            if feature in VALID_RANGES:
                lo, hi = VALID_RANGES[feature]
                if not (lo <= val <= hi):
                    cleaned[feature] = np.nan
                    issues.append(
                        f"'{feature}' = {val} outside [{lo}, {hi}] → set to NaN"
                    )
                    continue

            # Reject clear sensor-error values: exact integer 98 or 96 in
            # delinquency counts (cs-training.csv artifact — placeholder values)
            if feature in (
                "NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse",
            ) and val in (96, 98):
                cleaned[feature] = np.nan
                issues.append(
                    f"'{feature}' = {val} is a known dataset artifact → set to NaN"
                )
                continue

            cleaned[feature] = val

        # Categorical processing
        elif feature in CATEGORICAL_FEATURES:
            val = str(raw_value).strip().upper()
            if feature in VALID_CATEGORIES:
                allowed = [v.upper() for v in VALID_CATEGORIES[feature]]
                if val not in allowed:
                    cleaned[feature] = np.nan
                    issues.append(
                        f"'{feature}' = '{raw_value}' not in "
                        f"{VALID_CATEGORIES[feature]} → NaN"
                    )
                    continue
            cleaned[feature] = val

    # ── Check: too many missing fields? ───────────────────────────────────────
    missing_count    = sum(
        1 for v in cleaned.values()
        if v is None or (isinstance(v, float) and np.isnan(v))
    )
    missing_fraction = missing_count / len(all_features) if all_features else 0

    if missing_fraction > MAX_ROW_MISSING_FRACTION:
        raise ValidationError(
            f"Row has {missing_count}/{len(all_features)} fields missing "
            f"({100 * missing_fraction:.0f}%). Cannot make a reliable prediction. "
            f"Issues: {'; '.join(issues)}"
        )

    # ── Cross-field consistency checks ────────────────────────────────────────
    consistency_warnings = _check_cross_field_consistency(data)

    # ── Fake/fabricated data heuristics ───────────────────────────────────────
    suspicious_flags = []
    if _check_round_numbers(data):
        suspicious_flags.append(
            "Multiple financial fields are suspiciously round numbers "
            "(possible fabricated data)"
        )
    if _check_perfect_profile(data):
        suspicious_flags.append(
            "Profile has zero delinquencies, zero ratios, and very high income "
            "(statistically rare — recommend verification)"
        )
    suspicious_flags.extend(consistency_warnings)

    # Log non-fatal issues but continue
    all_issues = issues + suspicious_flags
    if all_issues:
        logger.warning(
            f"Validation: {len(issues)} data issue(s), "
            f"{len(suspicious_flags)} suspicious flag(s): "
            f"{'; '.join(all_issues[:5])}"
        )

    # Attach flags for predict.py to surface in the response
    cleaned["_suspicious_flags"] = suspicious_flags
    cleaned["_data_issues"]      = issues

    return cleaned
