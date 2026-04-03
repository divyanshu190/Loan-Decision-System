# Loan Default Decision System 

## Dataset

**Give Me Some Credit** (Kaggle Competition Dataset)  
Real-world credit bureau data — 150,000 records, all numeric features.

| Property | Value |
|---|---|
| Rows | 150,000 |
| Features | 10 numeric |
| Target | `SeriousDlqin2yrs` (1=default, 0=repaid) |
| Default rate | ~7% (imbalanced — handled by SMOTE) |

The dataset (`cs-training.csv`) is already included in this project.

---

## How the Model Works

```
cs-training.csv
      ↓
  Validation & Cleaning (validator.py)
      ↓
  Missingness Indicators (missing income = signal of risk)
      ↓
  Preprocessing (median imputation → standard scaling)
      ↓
  SMOTE (oversample minority class — training only, no leakage)
      ↓
  ┌─────────────────────────────────────┐
  │  XGBoost   LightGBM   Decision Tree │  ← 3 base models
  └────────────────┬────────────────────┘
                   ↓
         Logistic Regression         ← meta-learner (stacking)
                   ↓
      Cost-optimized Threshold       ← FN costs 5× FP
                   ↓
        APPROVE / REJECT + Risk Factors
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 1 — Train the model

```bash
python train.py --data cs-training.csv
```

Expected output (after ~5 min):
```
ROC-AUC Score          : ~0.86
Average Precision Score: ~0.35
Optimal threshold      : ~0.18  (cost-matrix optimized)
```

Training also prints:
- Top 3 levels of the Decision Tree (interpretable rules)
- XGBoost feature importance ranking

---

## Step 2 — Test a single prediction

```bash
python predict.py
```

---

## Step 3 — Run the API

```bash
python api/app.py
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.10,
    "MonthlyIncome": 8000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }'
```

Expected response:
```json
{
  "decision": "APPROVE",
  "default_prob": 0.041,
  "confidence": "HIGH",
  "risk_factors": ["Profile meets all standard credit criteria"],
  "warnings": [],
  "error": null
}
```

---

## File Reference

| File | Purpose |
|---|---|
| `config.py` | All settings (features, thresholds, cost matrix) |
| `validator.py` | Catches bad/missing/corrupt data before it touches the model |
| `train.py` | Full training pipeline |
| `predict.py` | Production prediction engine + risk factor explanations |
| `api/app.py` | REST API for deployment |
| `cs-training.csv` | Dataset (150K real credit records) |

---

## Real-world problems handled

| Problem | How it's handled |
|---|---|
| Missing numeric value | Median imputation |
| Wrong data type (text in number field) | Coerced to NaN then imputed |
| Impossible value (age=999) | Nulled then imputed |
| >50% of row missing | Hard reject with error message |
| Class imbalance (~7% defaults) | SMOTE oversampling (training only) |
| Asymmetric error costs | Cost-matrix threshold optimization |
| Decision explainability | Rule-based risk factor extraction |
| Interpretability (PS-9) | Decision Tree visualization + feature importance |
