import logging
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late 30-59 Days",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "Late 90+ Days",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late 60-89 Days",
    "NumberOfDependents": "Dependents",
}

FEATURE_RISK_DIRECTION = {
    "RevolvingUtilizationOfUnsecuredLines": +1,
    "age": -1,
    "NumberOfTime30-59DaysPastDueNotWorse": +1,
    "DebtRatio": +1,
    "MonthlyIncome": -1,
    "NumberOfOpenCreditLinesAndLoans": +1,
    "NumberOfTimes90DaysLate": +1,
    "NumberRealEstateLoansOrLines": -1,
    "NumberOfTime60-89DaysPastDueNotWorse": +1,
    "NumberOfDependents": +1,
}

_BASELINES = {
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 40.0,
    "NumberOfTime30-59DaysPastDueNotWorse": 0.0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5000.0,
    "NumberOfOpenCreditLinesAndLoans": 6.0,
    "NumberOfTimes90DaysLate": 0.0,
    "NumberRealEstateLoansOrLines": 1.0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0.0,
    "NumberOfDependents": 1.0,
}

_SCALES = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "age": 30.0,
    "NumberOfTime30-59DaysPastDueNotWorse": 5.0,
    "DebtRatio": 1.0,
    "MonthlyIncome": 8000.0,
    "NumberOfOpenCreditLinesAndLoans": 10.0,
    "NumberOfTimes90DaysLate": 5.0,
    "NumberRealEstateLoansOrLines": 3.0,
    "NumberOfTime60-89DaysPastDueNotWorse": 5.0,
    "NumberOfDependents": 5.0,
}

def explain_prediction(data: dict, model_meta=None) -> list:
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
    impacts = []
    for feat, direction in FEATURE_RISK_DIRECTION.items():
        val = clean_data.get(feat)
        baseline = _BASELINES.get(feat, 0.0)
        scale = _SCALES.get(feat, 1.0)
        label = FEATURE_LABELS.get(feat, feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            deviation = (float(val) - baseline) / (scale or 1.0)
        except (ValueError, TypeError):
            continue
        impact = round(deviation * direction * 0.3, 4)
        impacts.append((label, impact))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts
