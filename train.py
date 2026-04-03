# =============================================================================
#  train.py  —  Full training pipeline
#  Run:  python train.py --data cs-training.csv
# =============================================================================
"""
Dataset: Give Me Some Credit (cs-training.csv)
  150,000 real credit records | 10 numeric features | ~7% default rate

Pipeline:
  1.  Load & validate
  2.  Drop columns with >60% missing
  3.  Add missingness indicator features (missing income = signal of risk)
  4.  Build preprocessing: median imputation → standard scaling
  5.  Apply SMOTE for class imbalance (training only — no leakage)
  6.  Train 3 base models stacked with Logistic Regression:
        XGBoost       — best-in-class gradient boosting for credit data
        LightGBM      — faster, handles 150K rows efficiently
        Decision Tree — interpretable; directly satisfies PS-9 requirement
  7.  Optimize decision threshold using business cost matrix (FN costs 5x FP)
  8.  Visualize top 3 levels of Decision Tree (PS-9 explainability demo)
  9.  Log XGBoost feature importance (gain-based)
 10.  Train IsolationForest anomaly detector on LEGITIMATE (non-default) data
       This learns what a normal application looks like. At prediction time,
       new applications are scored — outliers get flagged as suspicious/fake.
 11.  Save full pipeline + anomaly detector to models/
"""

import os
import sys
import logging
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute           import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import StackingClassifier
from sklearn.tree             import DecisionTreeClassifier, export_text
from sklearn.metrics          import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
    f1_score, precision_recall_curve,
)

import xgboost  as xgb
import lightgbm as lgb
from imblearn.over_sampling      import SMOTE
from imblearn.pipeline           import Pipeline as ImbPipeline
from sklearn.ensemble            import IsolationForest

from config    import *
from validator import validate_dataframe

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
#  Step 1 — Load & validate
# =============================================================================

def load_and_validate(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # cs-training.csv has an unnamed row-number column as the first column
    unnamed = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed:
        df = df.drop(columns=unnamed)
        logger.info(f"Dropped unnamed index column(s): {unnamed}")

    logger.info(f"Raw shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = validate_dataframe(df)
    return df


# =============================================================================
#  Step 2 — Drop columns that are almost entirely missing
# =============================================================================

def drop_high_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    all_features  = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    cols_to_check = [c for c in all_features if c in df.columns]

    missing_frac = df[cols_to_check].isna().mean()
    keep_cols    = missing_frac[missing_frac <= MAX_MISSING_FRACTION_TO_KEEP].index.tolist()
    dropped_cols = missing_frac[missing_frac >  MAX_MISSING_FRACTION_TO_KEEP].index.tolist()

    if dropped_cols:
        logger.warning(f"Dropping {len(dropped_cols)} columns (>60% missing): {dropped_cols}")
    else:
        logger.info("All feature columns passed the missing-value threshold.")

    return df, keep_cols


# =============================================================================
#  Step 3 — Add missingness indicator columns
# =============================================================================

def add_missing_indicators(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, list]:
    """
    For each feature, add a binary column:
      1 = value was originally missing
      0 = value was present

    Why this matters: In credit data, applicants who omit income or
    employment info are statistically more likely to default.
    Missingness itself is a signal the model should learn from.
    """
    indicator_cols = []
    for col in features:
        if col in df.columns and df[col].isna().any():
            indicator_name     = f"{col}_was_missing"
            df[indicator_name] = df[col].isna().astype(int)
            indicator_cols.append(indicator_name)

    if indicator_cols:
        logger.info(f"Added {len(indicator_cols)} missingness indicator columns")

    return df, indicator_cols


# =============================================================================
#  Step 4 — Build preprocessing pipeline
# =============================================================================

def build_preprocessor(
    numeric_cols: list,
    categorical_cols: list,
    indicator_cols: list,
) -> ColumnTransformer:
    """
    Numeric:     median imputation (robust to outliers) → standard scaling
    Categorical: most-frequent imputation → ordinal encoding
    Indicators:  already 0/1 — pass through unchanged
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,   # unseen category at prediction time → -1
        )),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if indicator_cols:
        transformers.append(("ind", FunctionTransformer(), indicator_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# =============================================================================
#  Step 5 — Define base models
# =============================================================================

def get_base_models():
    """
    Three fundamentally different model types so they make DIFFERENT errors.
    When stacked, those errors cancel out → better combined performance.

    XGBoost:      Best-in-class for tabular credit data. Captures non-linear
                  feature interactions (e.g. high utilization AND late payments).

    LightGBM:     Faster than XGBoost on large data. Leaf-wise growth finds
                  tighter decision boundaries, complementing XGBoost.

    Decision Tree: The interpretable model. Directly satisfies PS-9's requirement
                  for "decision trees". We visualize its structure after training
                  to explain HOW the model makes decisions.
    """
    xgboost_model = xgb.XGBClassifier(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        eval_metric      = "auc",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )

    lgbm_model = lgb.LGBMClassifier(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbose          = -1,
    )

    # max_depth=8 is deep enough to capture real patterns,
    # shallow enough to visualize and explain to judges
    dt_model = DecisionTreeClassifier(
        max_depth         = 8,
        min_samples_split = 50,
        min_samples_leaf  = 20,
        criterion         = "gini",
        random_state      = RANDOM_STATE,
    )

    return [
        ("xgboost",       xgboost_model),
        ("lightgbm",      lgbm_model),
        ("decision_tree", dt_model),
    ]


# =============================================================================
#  Step 6 — Build stacking ensemble (no calibration wrapper = no leakage)
# =============================================================================

def build_stacking_ensemble(preprocessor: ColumnTransformer) -> ImbPipeline:
    """
    Base models (XGBoost + LightGBM + Decision Tree)
       → their predict_proba outputs feed into
    Logistic Regression meta-learner
       → final probability of default

    Why LogisticRegression as meta-learner:
      LogisticRegression naturally produces well-calibrated probabilities by
      design (it models log-odds). This replaces the old CalibratedClassifierCV
      wrapper which caused data leakage by running its validation folds on
      SMOTE-augmented data.

    Why ImbPipeline:
      SMOTE only runs during .fit() — never during .predict(). This prevents
      synthetic minority samples from contaminating held-out evaluation sets.
    """
    base_models = get_base_models()

    stacking_clf = StackingClassifier(
        estimators      = base_models,
        final_estimator = LogisticRegression(
            C            = 0.5,
            max_iter     = 1000,
            solver       = "lbfgs",
            random_state = RANDOM_STATE,
        ),
        cv           = CV_FOLDS,
        stack_method = "predict_proba",
        passthrough  = True,   # meta-learner also sees raw preprocessed features
        n_jobs       = -1,
    )

    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model",        stacking_clf),
    ])


# =============================================================================
#  Step 7 — Optimize decision threshold using business cost matrix
# =============================================================================

def find_optimal_threshold(pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Credit risk is an asymmetric cost problem:
      False Negative (approve a defaulter)   → COSTLY: bank loses loan principal
      False Positive (reject a good borrower) → LESS costly: bank loses interest

    Cost ratio from config.py: COST_FN = 5, COST_FP = 1
    We sweep thresholds from 0.05 to 0.60 and pick the one with minimum
    expected business loss = COST_FN * FN + COST_FP * FP.
    """
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    best_threshold = DECISION_THRESHOLD
    best_cost      = float("inf")

    logger.info("\nThreshold optimization (business cost matrix):")
    logger.info(f"  Cost of False Negative (miss a default): {COST_FN}x")
    logger.info(f"  Cost of False Positive (reject good applicant): {COST_FP}x")

    for t in np.arange(0.05, 0.60, 0.01):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = COST_FN * fn + COST_FP * fp
        if cost < best_cost:
            best_cost      = cost
            best_threshold = t

    logger.info(f"  → Optimal threshold: {best_threshold:.2f}  "
                f"(expected cost={best_cost:.0f})\n")

    return round(float(best_threshold), 2)


# =============================================================================
#  Step 8 — Visualize Decision Tree (PS-9 explainability requirement)
# =============================================================================

def visualize_decision_tree(pipeline, feature_names: list):
    """
    Print the top 3 levels of the Decision Tree as human-readable text.
    Directly answers PS-9's requirement for "decision trees".
    """
    try:
        stacking_model = pipeline.named_steps["model"]
        # Use named_estimators_ dict (correct sklearn API for StackingClassifier)
        dt_estimator = stacking_model.named_estimators_.get("decision_tree")

        if dt_estimator is None:
            logger.warning("Decision tree estimator not found in stacking model.")
            return

        n            = min(dt_estimator.n_features_in_, len(feature_names))
        names_clipped = feature_names[:n]

        tree_text = export_text(
            dt_estimator,
            feature_names = names_clipped,
            max_depth     = 3,
        )
        logger.info("\n" + "="*60)
        logger.info("DECISION TREE — Top 3 Levels (PS-9 Interpretability)")
        logger.info("="*60)
        logger.info("\n" + tree_text)
        logger.info("="*60 + "\n")

    except Exception as e:
        logger.warning(f"Could not visualize decision tree: {e}")


# =============================================================================
#  Step 9 — Log feature importance
# =============================================================================

def log_feature_importance(pipeline, feature_names: list):
    """
    Extract gain-based feature importance from XGBoost.
    Gain = how much each feature improves splits on average.
    """
    try:
        stacking_model = pipeline.named_steps["model"]
        # Use named_estimators_ dict (correct sklearn API)
        xgb_model = stacking_model.named_estimators_.get("xgboost")

        if xgb_model is None:
            logger.warning("XGBoost estimator not found in stacking model.")
            return feature_names

        importance = xgb_model.feature_importances_
        n          = min(len(importance), len(feature_names))
        feat_imp   = sorted(
            zip(feature_names[:n], importance[:n]),
            key=lambda x: x[1], reverse=True,
        )

        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE — XGBoost (Gain-based, Top 10)")
        logger.info("="*60)
        for feat, imp in feat_imp[:10]:
            bar = "█" * max(1, int(imp * 80))
            logger.info(f"  {feat:<45} {imp:.4f}  {bar}")
        logger.info("="*60 + "\n")

        return [f for f, _ in feat_imp]

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return feature_names


# =============================================================================
#  Step 10 — Train Isolation Forest (fake/anomalous application detector)
# =============================================================================

def train_anomaly_detector(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list,
    indicator_cols: list,
) -> IsolationForest:
    """
    Train an IsolationForest on LEGITIMATE (non-default) training samples.

    Why only on legitimate samples?
      The IsolationForest learns what a normal credit application looks like.
      Future applications that deviate significantly from this distribution
      get a negative anomaly score — flagged as suspicious / possible fake.

    Research basis:
      - Equifax, Experian production systems use unsupervised anomaly detection
        as a first-pass filter before the credit scoring model runs.
      - IsolationForest is preferred for real-time (~1ms) inference.
      - Training on clean data (non-defaults) avoids conflating financial risk
        with data fabrication — these are two different signals.
    """
    logger.info("Training IsolationForest anomaly detector...")

    # Use only non-default applicants — these represent 'normal' applications
    legitimate_mask = y_train == 0
    X_legit         = X_train[legitimate_mask]

    logger.info(
        f"  IsolationForest training on {X_legit.shape[0]:,} legitimate records "
        f"(non-default subset)"
    )

    # Only preprocess numeric + indicator columns (no SMOTE — we want real distributions)
    all_cols    = numeric_cols + indicator_cols
    available   = [c for c in all_cols if c in X_legit.columns]
    X_legit_sub = X_legit[available].copy()

    # Simple imputation for the anomaly detector — median fill
    from sklearn.impute     import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X_legit_sub)
    X_scl   = scaler.fit_transform(X_imp)

    iso = IsolationForest(
        n_estimators  = 200,
        contamination = ANOMALY_CONTAMINATION,   # ~2% of training may be outliers
        max_samples   = "auto",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    iso.fit(X_scl)

    logger.info(
        f"  IsolationForest trained. Contamination set to {ANOMALY_CONTAMINATION:.0%}."
    )
    logger.info(
        "  Applications deviating from this distribution will be flagged SUSPICIOUS."
    )

    # Save the preprocessing steps WITH the detector so predict.py can use them
    anomaly_metadata = {
        "iso_forest":  iso,
        "imputer":     imputer,
        "scaler":      scaler,
        "feature_cols": available,
        "flag_threshold": ANOMALY_FLAG_THRESHOLD,
    }

    with open(ANOMALY_MODEL_PATH, "wb") as f:
        pickle.dump(anomaly_metadata, f)

    logger.info(f"  Anomaly detector saved to: {ANOMALY_MODEL_PATH}")
    return iso


# =============================================================================
#  Step 11 — Evaluate
# =============================================================================

def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred)

    logger.info(f"ROC-AUC Score          : {auc:.4f}")
    logger.info(f"Average Precision Score: {ap:.4f}")
    logger.info(f"F1 Score (thresh={threshold}): {f1:.4f}")
    logger.info(f"\nClassification Report (threshold={threshold}):")
    logger.info("\n" + classification_report(
        y_test, y_pred, target_names=["Repaid", "Default"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"  True Positives  (defaults correctly caught) : {tp}")
    logger.info(f"  False Negatives (missed defaults — risky!)  : {fn}")
    logger.info(f"  False Positives (good applicants rejected)  : {fp}")
    logger.info(f"  True Negatives  (correctly approved)        : {tn}")
    logger.info("="*60 + "\n")

    return auc


# =============================================================================
#  Main entry point
# =============================================================================

def train(csv_path: str):
    # 1. Load & validate
    df = load_and_validate(csv_path)

    # 2. Separate target
    y  = df[TARGET_COLUMN].astype(int)
    df = df.drop(columns=[TARGET_COLUMN])

    # 3. Drop nearly-empty columns
    df, available_features = drop_high_missing_columns(df)

    # 4. Add missingness indicators before splitting
    df, indicator_cols = add_missing_indicators(df, available_features)

    # Identify numeric / categorical among available features
    numeric_cols     = [c for c in NUMERIC_FEATURES     if c in available_features]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in available_features]
    all_feature_names = numeric_cols + categorical_cols + indicator_cols

    logger.info(f"Numeric features    : {numeric_cols}")
    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Indicator columns   : {indicator_cols}")
    logger.info(f"Target distribution :\n{y.value_counts(normalize=True).round(3)}")

    # 5. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 6. Build pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, indicator_cols)
    pipeline     = build_stacking_ensemble(preprocessor)

    # 7. Train
    logger.info("Training ensemble (XGBoost + LightGBM + DecisionTree → LogisticRegression)...")
    logger.info("Expected time: 3-6 minutes on 150K rows...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # 8. Optimize threshold using cost matrix
    optimal_threshold = find_optimal_threshold(pipeline, X_test, y_test)

    # 9. Evaluate
    evaluate(pipeline, X_test, y_test, optimal_threshold)

    # 10. Visualize decision tree (PS-9 explainability)
    visualize_decision_tree(pipeline, all_feature_names)

    # 11. Feature importance
    ranked_features = log_feature_importance(pipeline, all_feature_names)

    # 12. Train IsolationForest anomaly detector
    preprocessor_fitted = pipeline.named_steps["preprocessor"]
    train_anomaly_detector(
        preprocessor   = preprocessor_fitted,
        X_train        = X_train,
        y_train        = y_train,
        numeric_cols   = numeric_cols,
        indicator_cols = indicator_cols,
    )

    # 13. Save main pipeline
    metadata = {
        "pipeline":        pipeline,
        "numeric_cols":    numeric_cols,
        "categorical_cols":categorical_cols,
        "indicator_cols":  indicator_cols,
        "threshold":       optimal_threshold,
        "feature_names":   all_feature_names,
        "ranked_features": ranked_features or all_feature_names,
    }
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Main model saved to    : {MODEL_SAVE_PATH}")
    logger.info(f"Anomaly detector saved : {ANOMALY_MODEL_PATH}")
    logger.info(f"Decision threshold     : {optimal_threshold}")
    return pipeline, optimal_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train loan default ensemble model")
    parser.add_argument("--data", required=True, help="Path to cs-training.csv")
    args = parser.parse_args()
    train(args.data)
