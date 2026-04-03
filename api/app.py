# =============================================================================
#  api/app.py  —  REST API for real-world deployment
#  Run: python api/app.py
#  Test: curl -X POST http://localhost:5000/predict -H "Content-Type: application/json"
#        -d '{"person_age": 30, "person_income": 75000, ...}'
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from flask      import Flask, request, jsonify
from predict    import predict_one, predict_batch

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Quick health check — ping this to confirm API is running."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Single applicant prediction.

    Features (cs-training.csv / Give Me Some Credit):
      RevolvingUtilizationOfUnsecuredLines, age, DebtRatio,
      MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
      NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
      NumberOfTime30-59DaysPastDueNotWorse,
      NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents
    Missing fields are OK — model will impute them.

    Response:
    {
        "decision":     "APPROVE" | "REJECT" | "ERROR",
        "default_prob": 0.123,
        "confidence":   "HIGH" | "MEDIUM" | "LOW",
        "risk_factors": ["Has 3 payments 90+ days late", ...],
        "warnings":     ["'MonthlyIncome' was missing — imputed"],
        "error":        null
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    result = predict_one(data)

    status_code = 200 if result["error"] is None else 422
    return jsonify(result), status_code


@app.route("/predict/batch", methods=["POST"])
def batch_endpoint():
    """
    Batch prediction from an uploaded CSV file.
    Use multipart/form-data with field name 'file'.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    input_path  = "/tmp/batch_input.csv"
    output_path = "/tmp/batch_output.csv"
    file.save(input_path)

    try:
        results_df = predict_batch(input_path, output_path)
        return jsonify({
            "total":    len(results_df),
            "approved": int((results_df["decision"] == "APPROVE").sum()),
            "rejected": int((results_df["decision"] == "REJECT").sum()),
            "errors":   int((results_df["decision"] == "ERROR").sum()),
            "results_file": output_path,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
