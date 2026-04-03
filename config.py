# =============================================================================
#  config.py  —  All settings in one place
#  Dataset: Give Me Some Credit (cs-training.csv)
#    - 150,000 real credit records from Kaggle competition
#    - 10 numeric features
#    - Target: SeriousDlqin2yrs (1=default, 0=repaid, ~7% default rate)
# =============================================================================

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

CATEGORICAL_FEATURES = []   # cs-training.csv is entirely numeric

TARGET_COLUMN = "SeriousDlqin2yrs"   # 1 = serious delinquency in 2 yrs, 0 = repaid

# ── Hard physical/logical range limits ───────────────────────────────────────
#   Values outside these are physically impossible — nulled by validator
VALID_RANGES = {
    "age":                                  (18, 100),
    "DebtRatio":                            (0, 500),
    "MonthlyIncome":                        (0, 10_000_000),
    "RevolvingUtilizationOfUnsecuredLines": (0, 50),
    "NumberOfDependents":                   (0, 30),
}

VALID_CATEGORIES = {}

# ── Missing-value thresholds ──────────────────────────────────────────────────
MAX_MISSING_FRACTION_TO_KEEP = 0.60   # drop column if >60% missing in training
MAX_ROW_MISSING_FRACTION     = 0.50   # reject prediction if >50% fields missing

# ── Training settings ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# ── Business cost matrix for threshold optimization ───────────────────────────
# Approving a defaulter (FN) costs far more than rejecting a good applicant (FP)
# Bank loses loan principal on FN vs. lost future interest on FP
COST_FN = 5.0   # weight for False Negative (missed default)
COST_FP = 1.0   # weight for False Positive (rejected good applicant)

# Starting default — overridden by cost-matrix optimization in train.py
DECISION_THRESHOLD = 0.25

# ── Anomaly / Fake-Application Detection ──────────────────────────────────────
# IsolationForest is trained on legitimate training data.
# At prediction time, new applications are scored:
#   anomaly_score < 0  → outlier (possible fake/fraudulent application)
#   anomaly_score >= 0 → normal (fits training distribution)
#
# ANOMALY_CONTAMINATION: expected proportion of outliers in training data
#   Real-world credit fraud rates: 0.5–3% → use 0.02 (2%)
ANOMALY_CONTAMINATION = 0.02

# Threshold for flagging as suspicious (IsolationForest decision score)
# Scores below this trigger a REVIEW flag instead of direct APPROVE/REJECT
ANOMALY_FLAG_THRESHOLD = -0.05

# ── Cross-field consistency rules (fake-data detection) ───────────────────────
# These catch profiles that are internally inconsistent — a signal of made-up data
# Format: (field_A, field_B, rule_description, max_allowed_ratio)
CONSISTENCY_RULES = {
    # If someone claims they have 10 open credit lines but age is 19,
    # that's suspicious (would require credit since age ~14)
    "age_vs_credit_lines": {
        "min_age_per_credit_line": 2.5,   # need ~2.5 yrs per open line
        "description": "Age too low for number of open credit lines"
    },
    # Very high income but also claiming 0 monthly income is inconsistent
    # (handled structurally by range validation)

    # Real estate loans require age + stable income
    "age_vs_real_estate": {
        "min_age_for_real_estate": 22,
        "description": "Too young to have multiple real estate loans"
    },
}

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH   = "models/loan_model.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_detector.pkl"
LOG_FILE          = "logs/pipeline.log"