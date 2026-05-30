"""
RetainIQ · src/models/train_cancellation.py
--------------------------------------------
Benchmarks multiple classifiers on the cancellation task and reports
the REAL metrics each one scores. No metric is hard-coded.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS    = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Naive Bayes":         GaussianNB(),
        "KNN (k=15)":          KNeighborsClassifier(n_neighbors=15),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Linear SVM":          LinearSVC(C=0.5, random_state=RANDOM_STATE, dual="auto"),
        "Random Forest":       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=400, random_state=RANDOM_STATE, verbose=-1),
        "XGBoost":             xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                                                 subsample=0.9, colsample_bytree=0.9,
                                                 eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1),
    }


# Models that take scaled input and/or lack predict_proba
SCALED_MODELS = ("Logistic Regression", "Naive Bayes", "KNN (k=15)", "Linear SVM")


def run(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Scale for distance/linear models; trees are scale-invariant but it's harmless
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rows = []
    best_name, best_auc, best_model = None, -1, None
    for name, model in get_models().items():
        scaled = name in SCALED_MODELS
        Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train)

        # AUC needs a score: use predict_proba if available, else decision_function
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xte)[:, 1]
        else:
            scores = model.decision_function(Xte)
            prob = (scores - scores.min()) / (scores.max() - scores.min())
        pred = model.predict(Xte)

        auc = roc_auc_score(y_test, prob)
        acc = accuracy_score(y_test, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        rows.append({"Model": name, "AUC": round(auc, 4), "Accuracy": round(acc, 4),
                     "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)})
        print(f"  {name:22s}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, model

    results = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    results.to_csv(REPORTS / "cancellation_benchmark.csv", index=False)

    print(f"\nBest model: {best_name}  (AUC={best_auc:.4f})")
    print("\nClassification report (best model):")
    best_scaled = best_name in SCALED_MODELS
    best_pred = best_model.predict(X_test_s if best_scaled else X_test)
    print(classification_report(y_test, best_pred, target_names=["Honoured", "Cancelled"]))

    # Feature importance (if tree-based best model)
    fi = None
    if hasattr(best_model, "feature_importances_"):
        fi = (pd.DataFrame({"feature": feature_names,
                            "importance": best_model.feature_importances_})
              .sort_values("importance", ascending=False).reset_index(drop=True))
        fi.to_csv(REPORTS / "cancellation_feature_importance.csv", index=False)
        print("\nTop 10 cancellation drivers:")
        print(fi.head(10).to_string(index=False))

    summary = {"best_model": best_name, "best_auc": round(best_auc, 4),
               "benchmark": results.to_dict(orient="records")}
    with open(REPORTS / "cancellation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results, summary, fi


if __name__ == "__main__":
    from src.features.build_features import build_cancellation_features
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    X, y, names = build_cancellation_features(df)
    print(f"Cancellation features: {X.shape[1]} cols, {len(X):,} rows, churn rate {y.mean():.2%}\n")
    run(X, y, names)
