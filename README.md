# 🏦 LoanSight — Credit Risk Engine

> AI-powered loan default prediction with real-time risk scoring, explainability, and what-if analysis.

Built for **Data Heist 2026** · UC Irvine Datathon

---

## 🎯 Problem Statement

Banks lose billions approving loans for applicants who default. Traditional credit checks are slow and opaque. LoanSight gives loan officers a fast, explainable APPROVE/REJECT decision with full reasoning — in one click.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ Real-time scoring | Instant APPROVE / REJECT / REVIEW decision |
| 📊 Default probability | Raw model output as percentage |
| 🎯 Risk score | FICO-style 300–850 score |
| 🔍 Explainability | Feature impact bar chart |
| 🔄 What-if analysis | Sweep any feature and see risk change live |
| 🛡️ Data validation | Catches bad/fake inputs before prediction |
| 🚨 Anomaly detection | IsolationForest flags suspicious profiles |
| 🌐 REST API | Flask API for production integration |

---

## 📊 Dataset

**Give Me Some Credit (Kaggle)** — real-world credit bureau dataset.

| Property | Value |
|---|---|
| Rows | 150,000 |
| Features | 10 numeric |
| Target | `SeriousDlqin2yrs` (1 = default, 0 = repaid) |
| Default rate | ~7% (highly imbalanced) |

### Features

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit card utilization ratio |
| `age` | Borrower age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Late payments (30–59 days) |
| `DebtRatio` | Monthly debt / monthly income |
| `MonthlyIncome` | Gross monthly income ($) |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines |
| `NumberOfTimes90DaysLate` | Payments 90+ days late |
| `NumberRealEstateLoansOrLines` | Mortgages / real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Late payments (60–89 days) |
| `NumberOfDependents` | Number of dependents |

---

## 🧠 Model Architecture

```
Raw Input
    ↓
Validation & Cleaning (validator.py)
    ↓
Missingness Indicators (MNAR signal features)
    ↓
Imputation + Scaling
    ↓
SMOTE (class imbalance handling)
    ↓
Stacked Ensemble:
    ├── XGBoost
    ├── LightGBM
    └── Decision Tree
    ↓
Meta-Learner: Logistic Regression
    ↓
Cost-Sensitive Threshold (FN costs 5x FP)
    ↓
Decision: APPROVE / REJECT / REVIEW
    ↓
IsolationForest Anomaly Score (fraud flag)
```

**ROC-AUC ≈ 0.86** · Decision threshold = **0.25** (cost-optimized)

---

## 🏗️ Project Structure

```
loan_decision_system/
├── app.py                  ← Streamlit UI (run this)
├── config.py               ← All settings & thresholds
├── train.py                ← Full training pipeline
├── predict.py              ← Production prediction engine
├── validator.py            ← Data validation firewall
├── validate_model.py       ← Model evaluation & metrics
├── run_all.sh              ← One-command pipeline runner
├── utils/
│   ├── predict.py          ← UI prediction bridge
│   ├── preprocess.py       ← Input cleaning & validation
│   └── explain.py          ← Explainability hooks (SHAP-ready)
├── api/
│   └── app.py              ← Flask REST API
├── models/
│   ├── loan_model.pkl      ← Trained stacked pipeline
│   └── anomaly_detector.pkl← IsolationForest detector
└── cs-training.csv         ← Dataset
```

---

## ⚙️ Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install xgboost imbalanced-learn
```

---

## 🚀 Run

### Streamlit UI
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Full Pipeline (train → validate → API)
```bash
chmod +x run_all.sh
./run_all.sh
```

### Train manually
```bash
python train.py --data cs-training.csv
```

### Flask API only
```bash
python api/app.py
```

---

## 🔌 API Usage

### Single prediction
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "MonthlyIncome": 8000,
    "DebtRatio": 0.10,
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfDependents": 1
  }'
```

### Response
```json
{
  "decision": "APPROVE",
  "default_prob": 0.0312,
  "confidence": "HIGH",
  "risk_factors": ["Profile meets all standard credit criteria"],
  "warnings": [],
  "error": null
}
```

### Batch prediction
```bash
curl -X POST http://127.0.0.1:5001/predict/batch \
  -F "file=@applicants.csv"
```

---

## 🧪 Test Cases

| Profile | Expected Decision |
|---|---|
| Age 35, Income $8000, Util 5%, 0 late payments | ✅ APPROVE |
| Age 38, Income $0, Util 98%, 5x 90-day late | ❌ REJECT |
| Age 33, Income $4500, Util 45%, 1x 30-day late | ⚠️ Borderline |

---

## 🛠 Key Technical Decisions

- **Missing income as a feature** — MNAR: missing income is itself a default risk signal
- **Cost matrix threshold** — FN (missed default) costs 5x FP (rejected good applicant) → threshold = 0.25 not 0.5
- **SMOTE** — Handles 7% minority class without distorting decision boundary
- **Stacking** — XGBoost + LightGBM + DT base learners, LR meta-learner reduces variance
- **IsolationForest** — Detects fake/synthetic applications at inference time

---

## ⚠️ Real-World Safeguards

- Hard range validation (age 18–100, income 0–10M, etc.)
- Cross-field consistency checks (age vs credit lines, income vs utilization)
- Round-number fabrication detection
- Artifact value removal (96/98 in delinquency columns)
- Hard reject if >50% fields missing

---

## 📜 License

MIT License

---

## 👤 Author

**Divyanshu** · Data Heist 2026 · UC Irvine Datathon
