"""
=============================================================================
Landslide Susceptibility Model — Training & Export Pipeline
=============================================================================
Author  : Senior ML Engineer
Dataset : sikkim_landslide_cleaned.csv
Target  : CLASS  (0 = Non-landslide, 1 = Landslide)

Features used
─────────────
ASPECT1, ELEVATION1, GEOLOGY1, LULC1, NDVI1,
RAINFALL1, ROAD1, SLOPE1, SOIL1, STREAM1

Excluded
────────
LONGITUDE, LATITUDE  (spatial leakage prevention)

Outputs
───────
landslide_model.pkl   — best trained model (joblib)
model_features.json   — ordered feature list for FastAPI validation
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless backend — safe on servers
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, Any

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn(
        "xgboost not installed — XGBoost will be skipped. "
        "Install with: pip install xgboost",
        stacklevel=2,
    )

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Configuration — edit paths here if needed
# ---------------------------------------------------------------------------
DATA_PATH          = Path("sikkim_landslide_cleaned.csv")
MODEL_OUTPUT       = Path("landslide_model.pkl")
FEATURES_JSON      = Path("model_features.json")
PLOTS_DIR          = Path("training_plots")

TARGET             = "CLASS"
SPATIAL_COLS       = ["LONGITUDE", "LATITUDE"]          # excluded from X
FEATURE_COLS       = [
    "ASPECT1", "ELEVATION1", "GEOLOGY1", "LULC1",
    "NDVI1",   "RAINFALL1",  "ROAD1",   "SLOPE1",
    "SOIL1",   "STREAM1",
]

TEST_SIZE          = 0.20
RANDOM_STATE       = 42
CV_FOLDS           = 5          # stratified cross-validation folds

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
DIVIDER = "=" * 70

def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Return a dict of all evaluation metrics for *model*."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred),                 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0),   4),
        "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0),       4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),                  4),
    }
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    for key, val in metrics.items():
        if key != "Model":
            print(f"  {key:<12}: {val}")


def feature_importance_table(
    model,
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract and return a ranked feature-importance DataFrame."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    df.index += 1                     # 1-based rank
    df["Importance (%)"] = (df["Importance"] * 100).round(2)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# 3. Load & Validate Data
# ---------------------------------------------------------------------------
section("LOADING DATASET")

if not DATA_PATH.exists():
    sys.exit(
        f"[ERROR] Dataset not found at: {DATA_PATH.resolve()}\n"
        "Place 'sikkim_landslide_cleaned.csv' in the same directory as this script."
    )

df = pd.read_csv(DATA_PATH)
print(f"  Loaded  : {DATA_PATH}  ({len(df):,} rows × {df.shape[1]} cols)")

# Basic sanity checks
missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    sys.exit(f"[ERROR] Missing feature columns: {missing_features}")
if TARGET not in df.columns:
    sys.exit(f"[ERROR] Target column '{TARGET}' not found in dataset.")

print(f"  Target  : {TARGET}  |  Class distribution:")
class_counts = df[TARGET].value_counts()
for cls, cnt in class_counts.items():
    label = "Non-landslide" if cls == 0 else "Landslide"
    pct   = cnt / len(df) * 100
    print(f"            Class {cls} ({label}): {cnt:,}  ({pct:.1f} %)")

# Drop rows with any NaN in features or target
before = len(df)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
if len(df) < before:
    print(f"  Dropped  : {before - len(df)} rows with missing values.")

X = df[FEATURE_COLS].copy()
y = df[TARGET].astype(int).copy()

# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------
section("TRAIN / TEST SPLIT  (80 / 20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,
)

print(f"  Train set : {len(X_train):,} samples")
print(f"  Test  set : {len(X_test):,} samples")
print(f"  Features  : {FEATURE_COLS}")

# ---------------------------------------------------------------------------
# 5. Define Classifiers
# ---------------------------------------------------------------------------
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators      = 300,
        max_depth         = None,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        class_weight      = "balanced",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators      = 300,
        learning_rate     = 0.05,
        max_depth         = 5,
        subsample         = 0.8,
        min_samples_split = 5,
        random_state      = RANDOM_STATE,
    ),
}

if XGB_AVAILABLE:
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators      = 300,
        learning_rate     = 0.05,
        max_depth         = 6,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
else:
    print("\n  [!] XGBoost skipped — library not available.")

# ---------------------------------------------------------------------------
# 6. Train, Cross-Validate & Evaluate
# ---------------------------------------------------------------------------
section("TRAINING & EVALUATION")

cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results  = []
trained  = {}

for name, clf in classifiers.items():
    print(f"\n  ── {name} ──")

    # Cross-validation on training set
    cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # Fit on full training split
    clf.fit(X_train, y_train)
    trained[name] = clf

    # Hold-out test metrics
    metrics = evaluate_model(name, clf, X_test, y_test)
    results.append(metrics)
    print_metrics(metrics)

# ---------------------------------------------------------------------------
# 7. Comparison Table & Best Model Selection
# ---------------------------------------------------------------------------
section("MODEL COMPARISON")

results_df = pd.DataFrame(results).set_index("Model")
print("\n", results_df.to_string(), "\n")

best_name  = results_df["ROC-AUC"].idxmax()
best_model = trained[best_name]
best_auc   = results_df.loc[best_name, "ROC-AUC"]

print(f"  ✔  Best model : {best_name}  (ROC-AUC = {best_auc})")

# ---------------------------------------------------------------------------
# 8. Detailed Report for Best Model
# ---------------------------------------------------------------------------
section(f"DETAILED REPORT — {best_name}")

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Non-Landslide", "Landslide"]))

# ---------------------------------------------------------------------------
# 9. Feature Importance Rankings
# ---------------------------------------------------------------------------
section("FEATURE IMPORTANCE RANKINGS")

fi_df = feature_importance_table(best_model, FEATURE_COLS)
print(f"  Top features driving predictions in {best_name}:\n")
print(fi_df[["Feature", "Importance (%)"]].to_string())
print()

# Highlight key spatial factors
for factor in ["SLOPE1", "ELEVATION1", "RAINFALL1"]:
    row = fi_df[fi_df["Feature"] == factor]
    if not row.empty:
        rank = row.index[0]
        imp  = row["Importance (%)"].values[0]
        print(f"  → {factor:<12} : Rank #{rank}  |  {imp:.2f} %")

# ---------------------------------------------------------------------------
# 10. Save Plots
# ---------------------------------------------------------------------------
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 10a. Feature Importance Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
fi_all = pd.DataFrame({
    "Feature"   : FEATURE_COLS,
    "Importance": best_model.feature_importances_,
}).sort_values("Importance", ascending=True)
colors = ["#c0392b" if f in ["SLOPE1", "ELEVATION1", "RAINFALL1"] else "#2980b9"
          for f in fi_all["Feature"]]
ax.barh(fi_all["Feature"], fi_all["Importance"], color=colors)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title(f"Feature Importance — {best_name}", fontweight="bold")
ax.axvline(x=fi_all["Importance"].mean(), color="gray", linestyle="--", label="Mean")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"\n  Plot saved → {PLOTS_DIR / 'feature_importance.png'}")

# 10b. ROC Curves (all models)
fig, ax = plt.subplots(figsize=(7, 6))
for name, clf in trained.items():
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_title("ROC Curves — All Models", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'roc_curves.png'}")

# 10c. Confusion Matrix (best model)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=["Non-Landslide", "Landslide"],
    cmap="Blues", ax=ax,
)
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print(f"  Plot saved → {PLOTS_DIR / 'confusion_matrix.png'}")

# ---------------------------------------------------------------------------
# 11. Export Artifacts
# ---------------------------------------------------------------------------
section("EXPORTING ARTIFACTS")

# 11a. Serialise best model
joblib.dump(best_model, MODEL_OUTPUT, compress=3)
print(f"  ✔  Model   saved → {MODEL_OUTPUT.resolve()}")

# 11b. Feature list for FastAPI input validation
features_payload = {
    "feature_names"   : FEATURE_COLS,
    "feature_count"   : len(FEATURE_COLS),
    "target"          : TARGET,
    "best_model"      : best_name,
    "roc_auc"         : float(best_auc),
    "excluded_columns": SPATIAL_COLS,
    "description"     : (
        "Ordered list of features the model expects. "
        "Pass values in exactly this order to /predict."
    ),
}
with open(FEATURES_JSON, "w") as fp:
    json.dump(features_payload, fp, indent=2)
print(f"  ✔  Features saved → {FEATURES_JSON.resolve()}")

# ---------------------------------------------------------------------------
# 12. Summary
# ---------------------------------------------------------------------------
section("PIPELINE COMPLETE")
print(f"""
  Dataset      : {DATA_PATH}
  Rows used    : {len(X):,}
  Features     : {len(FEATURE_COLS)}  {FEATURE_COLS}
  Excluded     : {SPATIAL_COLS}
  Split        : 80 % train  /  20 % test  (stratified, seed={RANDOM_STATE})

  ┌─────────────────────────────────────────────┐
  │  Best Model : {best_name:<29} │
  │  ROC-AUC    : {best_auc:<29} │
  └─────────────────────────────────────────────┘

  Artifacts
  ─────────
  {MODEL_OUTPUT.name:<30} ← load with joblib.load()
  {FEATURES_JSON.name:<30} ← feature schema for FastAPI
  {str(PLOTS_DIR):<30} ← diagnostic charts
""")
