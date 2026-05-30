"""
RetainIQ · src/segmentation/cluster.py
---------------------------------------
Builds per-guest value features, segments customers with K-Means,
and quantifies revenue concentration (Pareto). All figures are real.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def build_guest_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Primary_Guest_Name").agg(
        bookings=("Reservation_ID", "count"),
        total_revenue=("Total_Amount", "sum"),
        avg_revenue=("Total_Amount", "mean"),
        cancel_rate=("Is_Cancelled", "mean"),
        nights=("Stayed_Room_Nights", "sum"),
    ).reset_index()
    return g


def segment(g: pd.DataFrame, k: int = 4):
    feats = ["bookings", "total_revenue", "avg_revenue", "cancel_rate", "nights"]
    X = StandardScaler().fit_transform(g[feats])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    g["segment"] = km.fit_predict(X)

    # PCA 2D projection for visualising the clusters (on a sample for clarity)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    samp = np.random.default_rng(RANDOM_STATE).choice(len(g), size=min(4000, len(g)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[samp, 0], coords[samp, 1], c=g["segment"].values[samp],
                    cmap="viridis", s=8, alpha=0.6)
    ax.set_title(f"Guest segments — PCA projection "
                 f"({pca.explained_variance_ratio_.sum():.0%} variance)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.colorbar(sc, label="segment")
    fig.tight_layout(); fig.savefig(FIGURES / "segments_pca.png", dpi=140); plt.close(fig)

    profile = g.groupby("segment")[feats].mean().round(2)
    profile["n_guests"] = g["segment"].value_counts().sort_index().values
    profile["share_of_guests"] = (profile["n_guests"] / len(g)).round(3)
    return g, profile


def revenue_concentration(g: pd.DataFrame) -> dict:
    gg = g.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    total = gg["total_revenue"].sum()
    out = {}
    for q in (0.10, 0.18, 0.20):
        n = int(len(gg) * q)
        out[f"top_{int(q*100)}pct_revenue_share"] = round(
            gg.head(n)["total_revenue"].sum() / total, 4)
    return out


def run(df: pd.DataFrame):
    g = build_guest_features(df)
    g, profile = segment(g)
    conc = revenue_concentration(g)

    print(f"Guests: {len(g):,}")
    print("\nSegment profiles:")
    print(profile.to_string())
    print("\nRevenue concentration:")
    for k, v in conc.items():
        print(f"  {k}: {v:.1%}")

    profile.to_csv(REPORTS / "segment_profiles.csv")
    summary = {"n_guests": int(len(g)), "concentration": conc,
               "segment_profiles": json.loads(profile.reset_index().to_json(orient="records"))}
    with open(REPORTS / "segmentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
