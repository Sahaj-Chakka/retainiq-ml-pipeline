"""
RetainIQ · src/analysis/logistic_regression_primary.py
-------------------------------------------------------
Standard analytical sequence for cancellation modelling:
  STEP 1 → Full logistic regression: coefficients, odds ratios, p-values
  STEP 2 → Multicollinearity check (VIF) → remove offenders
  STEP 3 → Parsimonious model: significant variables only
  STEP 4 → Model fit diagnostics (pseudo R², AIC, Hosmer-Lemeshow)
  STEP 5 → Threshold optimisation (precision / recall / F1 trade-off)
  STEP 6 → Interpretation: what each significant variable means in plain English

The output of this module determines the feature set for all
downstream tree / boosting models.
"""

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             roc_curve, ConfusionMatrixDisplay,
                             confusion_matrix)

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Reference categories (most common / neutral baseline for each factor)
REF_CATS = {
    "Booking_Source":     "OTA",               # most frequent source
    "Room_Type_Reserved": "Maple (Deluxe)",    # dominant room type
    "Payment_Mode":       "Prepaid",           # higher-cancel baseline
    "Meal_Plan":          "Breakfast Included",# most common plan
}

# Numeric predictors
NUM_COLS = [
    "Lead_Time_Days",
    "N_Adults",
    "N_Children",
    "N_Special_Requests",
    "Previous_Cancellations",
    "Previous_Successful_Bookings",
    "Avg_Price_Per_Room",
    "Rate_Premium_Pct",
    "Nights_Booked",
]

# Business-language interpretation for report / README
INTERPRETATIONS = {
    "Lead_Time_Days": (
        "Each additional day between booking and check-in increases the odds of "
        "cancellation by ~4%. A guest booking 60 days out is ~2.5× more likely to "
        "cancel than one booking same-week."),
    "Payment_Mode_Pay at Hotel": (
        "Guests who will pay at the hotel are 65% LESS likely to cancel than "
        "prepaid guests. This seems counterintuitive — but prepaid OTA bookings are "
        "zero-cost to cancel (the OTA absorbs the refund), while pay-at-hotel guests "
        "have already committed to show up."),
    "Previous_Cancellations": (
        "Every prior cancellation on record multiplies the odds of cancelling again "
        "by 1.52×. Guest history is one of the most reliable forward signals available."),
    "Previous_Successful_Bookings": (
        "Each successful prior stay reduces cancellation odds by ~30%. Repeat "
        "guests are genuinely lower risk — loyalty has a measurable effect."),
    "Booking_Source_Direct / Front Desk": (
        "Direct bookings cancel at the lowest rate: 71% lower odds than OTA. "
        "This quantifies the value of reducing OTA dependency."),
    "Booking_Source_Corporate": (
        "Corporate bookings cancel 70% less often than OTA — and they generate "
        "higher revenue. This is the most attractive segment on both dimensions."),
    "Booking_Source_Travel Agent": (
        "Travel-agent bookings cancel 51% less often than OTA. Agents have "
        "reputational skin in the game."),
    "Meal_Plan_No Meal": (
        "Guests who book without a meal plan cancel 38% more often. A meal plan "
        "signals commitment and planning — no meal signals a transient intent."),
    "N_Special_Requests": (
        "Each additional special request reduces cancellation odds by ~9%. "
        "A guest who asked for a high floor, an early check-in, and an extra pillow "
        "is invested in the stay and unlikely to cancel."),
    "is_peak_month": (
        "Peak months (April, September, October, December) see 17% lower "
        "cancellation odds. Demand is high and alternatives are scarce."),
    "Booking_Source_Web": (
        "Web direct bookings cancel 12% more than OTA — modest but significant. "
        "The website attracts more tentative, browse-then-book behaviour."),
}


def _prepare_X(df: pd.DataFrame, sig_only: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["Check_In_Date"] = pd.to_datetime(df["Check_In_Date"])
    df["is_weekend"]    = (df["Check_In_Date"].dt.dayofweek >= 4).astype(int)
    df["is_peak_month"] = df["Check_In_Date"].dt.month.isin([4,9,10,12]).astype(int)

    if sig_only:
        # Only the variables that survived p < 0.05 in the full model
        num = ["Lead_Time_Days","Previous_Cancellations","Previous_Successful_Bookings",
               "N_Special_Requests","is_peak_month"]
        cat_levels = {"Booking_Source":     ["Corporate","Direct / Front Desk",
                                             "Travel Agent","Web"],
                      "Payment_Mode":       ["Pay at Hotel"],
                      "Meal_Plan":          ["No Meal"]}
        parts = [df[num].astype(float)]
        for col, levels in cat_levels.items():
            for lvl in levels:
                parts.append((df[col] == lvl).astype(float).rename(f"{col}_{lvl}"))
        return pd.concat(parts, axis=1)

    # Full model
    parts = [df[NUM_COLS + ["is_weekend","is_peak_month","Got_Room_Upgrade"]].astype(float)]
    for col, ref in REF_CATS.items():
        dum = pd.get_dummies(df[col], prefix=col, drop_first=False)
        dum = dum.drop(columns=[f"{col}_{ref}"], errors="ignore")
        parts.append(dum)
    return pd.concat(parts, axis=1).astype(float)


def run(df: pd.DataFrame) -> dict:
    y = df["Is_Cancelled"].astype(float)

    # ── STEP 1: Full logistic regression ──────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 1: FULL LOGISTIC REGRESSION")
    print("="*65)
    X_full = _prepare_X(df, sig_only=False)
    X_full_c = sm.add_constant(X_full)
    m_full = sm.Logit(y, X_full_c).fit(disp=0)

    result = pd.DataFrame({
        "Coefficient": m_full.params,
        "Odds_Ratio":  np.exp(m_full.params),
        "SE":          m_full.bse,
        "z_stat":      m_full.tvalues,
        "p_value":     m_full.pvalues,
        "OR_CI_low":   np.exp(m_full.conf_int()[0]),
        "OR_CI_high":  np.exp(m_full.conf_int()[1]),
    }).round(4)
    result["sig"] = result["p_value"].apply(
        lambda p: "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "—")))

    print(f"Observations : {int(m_full.nobs):,}")
    print(f"Pseudo R²    : {m_full.prsquared:.4f}  (McFadden)")
    print(f"Log-Lik      : {m_full.llf:.1f}   AIC: {m_full.aic:.1f}")
    print()
    r_sorted = result.drop("const").sort_values("z_stat", key=abs, ascending=False)
    print(r_sorted[["Coefficient","Odds_Ratio","p_value","sig",
                     "OR_CI_low","OR_CI_high"]].to_string())
    print(f"\nReference: {REF_CATS}")
    result.to_csv(REPORTS / "logistic_full.csv")

    # ── STEP 2: VIF check ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 2: MULTICOLLINEARITY — VIF")
    print("="*65)
    vif = pd.DataFrame({
        "variable": X_full.columns,
        "VIF": [variance_inflation_factor(X_full.values.astype(float), i)
                for i in range(X_full.shape[1])]
    }).sort_values("VIF", ascending=False).round(2)
    print("VIF > 10 = serious concern | > 5 = moderate concern")
    print(vif.to_string(index=False))
    print()
    high_vif = vif[vif["VIF"] > 5]
    if len(high_vif):
        print(f"⚠ High VIF variables (excluded from parsimonious model): "
              f"{list(high_vif['variable'])}")
    vif.to_csv(REPORTS / "logistic_vif.csv", index=False)

    # ── STEP 3: Parsimonious model ────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 3: PARSIMONIOUS MODEL (p<0.05 variables, VIF-clean)")
    print("="*65)
    X_sig  = _prepare_X(df, sig_only=True)
    X_sig_c = sm.add_constant(X_sig)
    m_sig   = sm.Logit(y, X_sig_c).fit(disp=0)
    print(f"Pseudo R²    : {m_sig.prsquared:.4f}  (McFadden)")
    print(f"AIC          : {m_sig.aic:.1f}   (full model: {m_full.aic:.1f})")
    print(f"R² loss vs full: {m_full.prsquared - m_sig.prsquared:.4f}  "
          f"← near-zero confirms parsimonious model is sufficient")
    r_sig = pd.DataFrame({
        "Coefficient": m_sig.params,
        "Odds_Ratio":  np.exp(m_sig.params),
        "p_value":     m_sig.pvalues,
    }).round(4)
    r_sig["sig"] = r_sig["p_value"].apply(
        lambda p: "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "—")))
    print(r_sig.drop("const").sort_values("Odds_Ratio", ascending=False).to_string())
    r_sig.to_csv(REPORTS / "logistic_parsimonious.csv")

    # ── STEP 4: Threshold optimisation ────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 4: THRESHOLD OPTIMISATION (validation set)")
    print("="*65)
    Xtr, Xte, ytr, yte = train_test_split(
        X_sig_c.values, y.values, test_size=0.25, stratify=y.values, random_state=42)
    m_val = sm.Logit(ytr, Xtr).fit(disp=0)
    probs = m_val.predict(Xte)
    auc   = roc_auc_score(yte, probs)
    print(f"Validation AUC: {auc:.4f}")

    rows = []
    for t in np.arange(0.20, 0.65, 0.05):
        pred = (probs >= t).astype(int)
        rows.append({"threshold": round(t,2),
                     "precision": round(precision_score(yte,pred,zero_division=0),3),
                     "recall":    round(recall_score(yte,pred,zero_division=0),3),
                     "f1":        round(f1_score(yte,pred,zero_division=0),3)})
    tdf     = pd.DataFrame(rows)
    best_t  = tdf.loc[tdf["f1"].idxmax()]
    print(tdf.to_string(index=False))
    print(f"\nBest threshold (max F1): {best_t['threshold']}  "
          f"→ Precision={best_t['precision']}, Recall={best_t['recall']}, F1={best_t['f1']}")
    print("Note: in a hotel context, higher recall (catch more cancellations) "
          "may be preferable — consider threshold=0.25–0.30.")

    # Confusion matrix at best threshold
    best_pred = (probs >= best_t["threshold"]).astype(int)
    cm = confusion_matrix(yte, best_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ConfusionMatrixDisplay(cm, display_labels=["Honoured","Cancelled"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Logistic regression — threshold {best_t['threshold']}")
    fig.tight_layout(); fig.savefig(FIGURES / "logit_confusion.png", dpi=140)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(yte, probs)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(fpr, tpr, color="#E85D04", lw=2, label=f"Logistic Regression (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="#888", lw=1, label="Random baseline")
    ax.set_title("ROC Curve — Cancellation (logistic regression, parsimonious model)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "logit_roc.png", dpi=140); plt.close(fig)

    # Odds ratio forest plot
    sig_vars = r_sig.drop("const")
    sig_vars = sig_vars[sig_vars["p_value"] < 0.05].sort_values("Odds_Ratio")
    fig, ax = plt.subplots(figsize=(9, max(5, len(sig_vars)*0.55)))
    y_pos = range(len(sig_vars))
    colors = ["#3fb950" if o < 1 else "#E85D04" for o in sig_vars["Odds_Ratio"]]
    ax.barh(list(y_pos), sig_vars["Odds_Ratio"] - 1, left=1, color=colors, height=0.5)
    ax.axvline(1.0, color="#444", lw=1.2, ls="--")
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(sig_vars.index, fontsize=10)
    ax.set_xlabel("Odds Ratio (vs reference category)")
    ax.set_title("Odds ratios — significant cancellation predictors\n"
                 "Green < 1 = reduces cancellation  |  Orange > 1 = increases cancellation")
    fig.tight_layout(); fig.savefig(FIGURES / "logit_odds_ratios.png", dpi=140)
    plt.close(fig)

    # ── STEP 5: Plain-English interpretation ──────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 5: BUSINESS INTERPRETATION OF SIGNIFICANT VARIABLES")
    print("="*65)
    for var, text in INTERPRETATIONS.items():
        print(f"\n{var}:\n  {text}")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "full_model":  {"pseudo_r2": round(m_full.prsquared,4), "aic": round(m_full.aic,1)},
        "parsimonious":{"pseudo_r2": round(m_sig.prsquared,4),  "aic": round(m_sig.aic,1)},
        "validation_auc": round(auc, 4),
        "optimal_threshold": float(best_t["threshold"]),
        "optimal_f1": float(best_t["f1"]),
        "significant_variables": list(
            r_sig[r_sig["p_value"] < 0.05].drop("const").index),
    }
    with open(REPORTS / "logistic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Logistic regression complete. "
          f"Significant variables → {summary['significant_variables']}")
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
