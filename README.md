# Loan Decision System

A production-style machine learning system for loan approval using ensemble modeling, anomaly detection, and cost-sensitive decisioning. Exposed via a Flask API for real-time credit risk evaluation.

---

## 📊 Dataset

**Give Me Some Credit (Kaggle)** — real-world credit bureau dataset.

| Property | Value |
|---|---|
| Rows | 150,000 |
| Features | 10 numeric |
| Target | `SeriousDlqin2yrs` (1 = default, 0 = repaid) |
| Default rate | ~7% (highly imbalanced) |

> Note: Dataset is used for training and evaluation only.

---

## 🧠 Model Architecture

```
Raw Data
   ↓
Validation & Cleaning (validator.py)
   ↓
Feature Engineering (missingness indicators)
   ↓
Preprocessing (imputation + scaling)
   ↓
SMOTE (handle class imbalance)
   ↓
Stacked Ensemble:
   - XGBoost
   - LightGBM
   - Decision Tree
   ↓
Meta-Learner:
   - Logistic Regression
   ↓
Threshold Optimization (cost-sensitive)
   ↓
Final Decision: APPROVE / REJECT / REVIEW
```

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run Full Pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

This executes:
- Model training  
- Validation  
- API startup  

---

## 🧪 Manual Training

```bash
python train.py --data cs-training.csv
```

Expected performance:
- ROC-AUC ≈ 0.85+  
- Cost-optimized threshold  

---

## 🔌 API Usage

### Endpoint
```
POST /predict
```

### Example Request

```bash
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{
  "age": 31,
  "MonthlyIncome": 48765,
  "DebtRatio": 0.27,
  "RevolvingUtilizationOfUnsecuredLines": 0.36,
  "NumberOfOpenCreditLinesAndLoans": 6,
  "NumberOfTimes90DaysLate": 0,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfTime30-59DaysPastDueNotWorse": 1,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfDependents": 2
}'
```

### Example Response

```json
{
  "decision": "APPROVE",
  "default_prob": 0.0526,
  "confidence": "MEDIUM",
  "risk_factors": ["Profile meets all standard credit criteria"],
  "warnings": [],
  "error": null
}
```

---

## 📁 Project Structure

```
api/                  # Flask API
train.py              # Training pipeline
predict.py            # Inference logic
validator.py          # Data validation
validate_model.py     # Model evaluation
config.py             # Configurations
run_all.sh            # Automation script
requirements.txt      # Dependencies
```

---

## 🛠 Key Features

- Ensemble learning (XGBoost + LightGBM + Decision Tree)  
- Stacking with Logistic Regression  
- SMOTE for class imbalance  
- Cost-sensitive threshold optimization  
- Input validation and anomaly handling  
- Risk factor explanations  
- End-to-end automation via bash script  

---

## ⚠️ Real-World Considerations

- Handles missing and corrupt inputs  
- Detects unrealistic financial patterns  
- Flags suspicious profiles  
- Prevents data leakage  
- Provides explainable decisions  

---

## 📜 License

MIT License

---

## 👤 Author

Divyanshu
