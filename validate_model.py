# =============================================================================
#  validate_model.py  —  Test the model with known real-world cases
#  Run: python validate_model.py
# =============================================================================
"""
25 hand-crafted test cases with EXPECTED outcomes.
Each case is designed to test a specific scenario.
The script prints a full report showing where the model agrees/disagrees.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from predict import predict_one

# =============================================================================
#  Test cases — (description, input_data, expected_decision)
#  Features match cs-training.csv (Give Me Some Credit dataset)
# =============================================================================

TEST_CASES = [

    # ── OBVIOUS APPROVALS (should clearly be APPROVE) ────────────────────────

    ("Perfect applicant — young, high income, no late payments",
     {"RevolvingUtilizationOfUnsecuredLines": 0.05,
      "age": 35,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.10,
      "MonthlyIncome": 12000,
      "NumberOfOpenCreditLinesAndLoans": 6,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 1},
     "APPROVE"),

    ("Senior professional — stable income, low debt, long credit history",
     {"RevolvingUtilizationOfUnsecuredLines": 0.10,
      "age": 55,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.15,
      "MonthlyIncome": 9000,
      "NumberOfOpenCreditLinesAndLoans": 10,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 2,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 2},
     "APPROVE"),

    ("Middle aged, moderate income, clean record",
     {"RevolvingUtilizationOfUnsecuredLines": 0.20,
      "age": 42,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.25,
      "MonthlyIncome": 6500,
      "NumberOfOpenCreditLinesAndLoans": 7,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 3},
     "APPROVE"),

    ("High earner, minimal credit usage",
     {"RevolvingUtilizationOfUnsecuredLines": 0.03,
      "age": 48,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.08,
      "MonthlyIncome": 20000,
      "NumberOfOpenCreditLinesAndLoans": 5,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 3,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 0},
     "APPROVE"),

    ("Young professional, good habits, no dependents",
     {"RevolvingUtilizationOfUnsecuredLines": 0.15,
      "age": 28,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.20,
      "MonthlyIncome": 5500,
      "NumberOfOpenCreditLinesAndLoans": 4,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 0},
     "APPROVE"),

    # ── OBVIOUS REJECTIONS (should clearly be REJECT) ────────────────────────

    ("Serial defaulter — late on everything, maxed out credit",
     {"RevolvingUtilizationOfUnsecuredLines": 0.98,
      "age": 38,
      "NumberOfTime30-59DaysPastDueNotWorse": 8,
      "DebtRatio": 0.95,
      "MonthlyIncome": 2000,
      "NumberOfOpenCreditLinesAndLoans": 12,
      "NumberOfTimes90DaysLate": 5,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 4,
      "NumberOfDependents": 4},
     "REJECT"),

    ("Completely maxed credit, multiple 90-day lates",
     {"RevolvingUtilizationOfUnsecuredLines": 1.00,
      "age": 45,
      "NumberOfTime30-59DaysPastDueNotWorse": 10,
      "DebtRatio": 1.20,
      "MonthlyIncome": 1500,
      "NumberOfOpenCreditLinesAndLoans": 15,
      "NumberOfTimes90DaysLate": 8,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 6,
      "NumberOfDependents": 5},
     "REJECT"),

    ("Very low income, very high debt, consistent late payer",
     {"RevolvingUtilizationOfUnsecuredLines": 0.90,
      "age": 52,
      "NumberOfTime30-59DaysPastDueNotWorse": 6,
      "DebtRatio": 0.88,
      "MonthlyIncome": 1200,
      "NumberOfOpenCreditLinesAndLoans": 9,
      "NumberOfTimes90DaysLate": 4,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 3,
      "NumberOfDependents": 6},
     "REJECT"),

    ("Young, no income, maxed cards, many dependents",
     {"RevolvingUtilizationOfUnsecuredLines": 0.95,
      "age": 23,
      "NumberOfTime30-59DaysPastDueNotWorse": 5,
      "DebtRatio": 1.50,
      "MonthlyIncome": 800,
      "NumberOfOpenCreditLinesAndLoans": 8,
      "NumberOfTimes90DaysLate": 3,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 2,
      "NumberOfDependents": 3},
     "REJECT"),

    ("Elderly, massive debt ratio, chronic late payments",
     {"RevolvingUtilizationOfUnsecuredLines": 0.85,
      "age": 68,
      "NumberOfTime30-59DaysPastDueNotWorse": 7,
      "DebtRatio": 2.50,
      "MonthlyIncome": 1800,
      "NumberOfOpenCreditLinesAndLoans": 11,
      "NumberOfTimes90DaysLate": 6,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 5,
      "NumberOfDependents": 2},
     "REJECT"),

    # ── BORDERLINE CASES (model judgment tested here) ────────────────────────

    ("One-time late payment, otherwise good",
     {"RevolvingUtilizationOfUnsecuredLines": 0.30,
      "age": 33,
      "NumberOfTime30-59DaysPastDueNotWorse": 1,
      "DebtRatio": 0.30,
      "MonthlyIncome": 5000,
      "NumberOfOpenCreditLinesAndLoans": 6,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 2},
     "APPROVE"),

    ("High utilization but no late payments ever",
     {"RevolvingUtilizationOfUnsecuredLines": 0.75,
      "age": 40,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.40,
      "MonthlyIncome": 7000,
      "NumberOfOpenCreditLinesAndLoans": 8,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 3},
     "APPROVE"),

    ("Average everything — true middle-ground applicant",
     {"RevolvingUtilizationOfUnsecuredLines": 0.50,
      "age": 38,
      "NumberOfTime30-59DaysPastDueNotWorse": 1,
      "DebtRatio": 0.35,
      "MonthlyIncome": 4500,
      "NumberOfOpenCreditLinesAndLoans": 7,
      "NumberOfTimes90DaysLate": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfDependents": 2},
     "APPROVE"),

    # ── MISSING DATA CASES (tests validator + imputation) ────────────────────

    ("Good applicant — income missing",
     {"RevolvingUtilizationOfUnsecuredLines": 0.10,
      "age": 40,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.20,
      "MonthlyIncome": None,           # MISSING
      "NumberOfOpenCreditLinesAndLoans": 6,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 1},
     "APPROVE"),

    ("Risky applicant — age missing",
     {"RevolvingUtilizationOfUnsecuredLines": 0.92,
      "age": None,                      # MISSING
      "NumberOfTime30-59DaysPastDueNotWorse": 5,
      "DebtRatio": 0.90,
      "MonthlyIncome": 1500,
      "NumberOfOpenCreditLinesAndLoans": 10,
      "NumberOfTimes90DaysLate": 4,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 3,
      "NumberOfDependents": 5},
     "REJECT"),

    ("Good applicant — 3 fields missing",
     {"RevolvingUtilizationOfUnsecuredLines": None,  # MISSING
      "age": 45,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": None,                             # MISSING
      "MonthlyIncome": 8000,
      "NumberOfOpenCreditLinesAndLoans": 7,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": None,           # MISSING
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 2},
     "APPROVE"),

    # ── CORRUPT / WRONG DATA CASES (tests validator) ─────────────────────────

    ("Good applicant — age is text (corrupt)",
     {"RevolvingUtilizationOfUnsecuredLines": 0.10,
      "age": "thirty",                  # WRONG TYPE
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.15,
      "MonthlyIncome": 7000,
      "NumberOfOpenCreditLinesAndLoans": 5,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 1},
     "APPROVE"),

    ("Risky applicant — impossible debt ratio (999)",
     {"RevolvingUtilizationOfUnsecuredLines": 0.95,
      "age": 40,
      "NumberOfTime30-59DaysPastDueNotWorse": 6,
      "DebtRatio": 999,                 # IMPOSSIBLE VALUE
      "MonthlyIncome": 1000,
      "NumberOfOpenCreditLinesAndLoans": 12,
      "NumberOfTimes90DaysLate": 5,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 4,
      "NumberOfDependents": 4},
     "REJECT"),

    ("Good applicant — income is negative (corrupt)",
     {"RevolvingUtilizationOfUnsecuredLines": 0.15,
      "age": 35,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.20,
      "MonthlyIncome": -5000,           # IMPOSSIBLE VALUE
      "NumberOfOpenCreditLinesAndLoans": 6,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 1},
     "APPROVE"),

    # ── EDGE CASES ───────────────────────────────────────────────────────────

    ("Elderly retired person — low income but zero debt history",
     {"RevolvingUtilizationOfUnsecuredLines": 0.05,
      "age": 72,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.05,
      "MonthlyIncome": 2500,
      "NumberOfOpenCreditLinesAndLoans": 3,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 1,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 0},
     "APPROVE"),

    ("High income but terrible payment history",
     {"RevolvingUtilizationOfUnsecuredLines": 0.80,
      "age": 44,
      "NumberOfTime30-59DaysPastDueNotWorse": 7,
      "DebtRatio": 0.60,
      "MonthlyIncome": 15000,
      "NumberOfOpenCreditLinesAndLoans": 14,
      "NumberOfTimes90DaysLate": 5,
      "NumberRealEstateLoansOrLines": 2,
      "NumberOfTime60-89DaysPastDueNotWorse": 4,
      "NumberOfDependents": 1},
     "REJECT"),

    ("Just turned 18 — no credit history at all",
     {"RevolvingUtilizationOfUnsecuredLines": 0.0,
      "age": 18,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.0,
      "MonthlyIncome": 1800,
      "NumberOfOpenCreditLinesAndLoans": 0,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 0},
     "APPROVE"),

    ("Single parent, stretched thin but always paid on time",
     {"RevolvingUtilizationOfUnsecuredLines": 0.65,
      "age": 36,
      "NumberOfTime30-59DaysPastDueNotWorse": 0,
      "DebtRatio": 0.55,
      "MonthlyIncome": 3500,
      "NumberOfOpenCreditLinesAndLoans": 5,
      "NumberOfTimes90DaysLate": 0,
      "NumberRealEstateLoansOrLines": 0,
      "NumberOfTime60-89DaysPastDueNotWorse": 0,
      "NumberOfDependents": 3},
     "APPROVE"),

    ("All fields missing except age — should be rejected/error",
     {"RevolvingUtilizationOfUnsecuredLines": None,
      "age": 30,
      "NumberOfTime30-59DaysPastDueNotWorse": None,
      "DebtRatio": None,
      "MonthlyIncome": None,
      "NumberOfOpenCreditLinesAndLoans": None,
      "NumberOfTimes90DaysLate": None,
      "NumberRealEstateLoansOrLines": None,
      "NumberOfTime60-89DaysPastDueNotWorse": None,
      "NumberOfDependents": None},
     "ERROR"),
]


# =============================================================================
#  Run validation
# =============================================================================

def run_validation():
    print("\n" + "="*70)
    print("  MODEL VALIDATION REPORT")
    print("="*70)

    correct   = 0
    incorrect = 0
    errors    = 0
    results   = []

    for i, (description, data, expected) in enumerate(TEST_CASES, 1):
        result   = predict_one(data)
        decision = result["decision"]
        prob     = result["default_prob"]
        conf     = result["confidence"]
        warns    = result["warnings"]
        err      = result["error"]

        # Check if prediction matches expectation
        # REVIEW is treated as a near-APPROVE — model agrees it's low risk
        # but flagged suspicious data for human verification (correct behavior)
        if decision == "ERROR":
            errors += 1
            if expected == "ERROR":
                correct += 1
                status = "PASS  "
                match  = True
            else:
                status = "WRONG "
                match  = False
        elif decision == expected:
            correct += 1
            status = "PASS  "
            match  = True
        elif decision == "REVIEW" and expected == "APPROVE":
            correct += 1
            status = "PASS~ "   # REVIEW is acceptable for borderline APPROVE cases
            match  = True
        else:
            incorrect += 1
            status = "FAIL  "
            match  = False

        results.append((i, status, match, description, expected,
                        decision, prob, conf, warns, err))

        # Print each result
        print(f"\n[{i:02d}] {status} | Expected: {expected:<7} | Got: {decision:<7} "
              f"| Prob: {str(prob):<6} | Confidence: {conf}")
        print(f"      {description}")
        if warns:
            print(f"      Warnings: {' | '.join(warns)}")
        if err:
            print(f"      Error msg: {err}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(TEST_CASES)
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Total test cases : {total}")
    print(f"  Passed           : {correct}  ({100*correct/total:.0f}%)")
    print(f"  Failed           : {incorrect}  ({100*incorrect/total:.0f}%)")
    print(f"  Model accuracy on known cases: {100*correct/total:.0f}%")
    print("="*70)

    # ── Failed cases detail ───────────────────────────────────────────────────
    failed = [(i, desc, exp, dec, prob)
              for i, status, match, desc, exp, dec, prob, *_ in results
              if not match]

    if failed:
        print("\n  FAILED CASES — model disagreed with expectation:")
        for i, desc, exp, dec, prob in failed:
            print(f"  [{i:02d}] Expected {exp}, got {dec} (prob={prob}) — {desc}")
    else:
        print("\n  All cases passed!")

    print("="*70 + "\n")


if __name__ == "__main__":
    run_validation()
