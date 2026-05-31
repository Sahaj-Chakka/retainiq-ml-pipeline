"""
RetainIQ · src/analysis/overbooking_policy.py
-----------------------------------------------
Newsvendor + Monte Carlo simulation:
"How many rooms can we safely overbook, given empirical cancellation rates,
 to maximise expected revenue while keeping walk-out risk acceptable?"

Output:
  • Expected revenue vs overbooking level (curve)
  • Walk-out probability vs overbooking level
  • Optimal overbooking policy per day-of-week and season
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

INVENTORY   = 14        # real Median Inn: 14 rooms
N_SIM       = 10_000   # Monte Carlo iterations
WALKOUT_COST_MULT = 3.0  # compensation cost = 3× room night revenue (relocation + apology)
WALKOUT_TOLERANCE = 0.02  # max acceptable walk-out probability (2%)


def empirical_cancel_rate(df: pd.DataFrame, segment: str = "all") -> float:
    if segment == "all":
        return float(df["Is_Cancelled"].mean())
    return float(df[df["Booking_Source"] == segment]["Is_Cancelled"].mean())


def simulate_overbooking(n_bookings: int, cancel_rate: float,
                         inventory: int, n_sim: int, rng) -> dict:
    """
    For a given number of overbookings accepted, simulate:
    - actual arrivals (binomial draw on non-cancellations)
    - walk-outs = max(0, arrivals - inventory)
    """
    actual_arrivals = rng.binomial(n_bookings, 1 - cancel_rate, size=n_sim)
    walk_outs       = np.maximum(0, actual_arrivals - inventory)
    return {
        "mean_arrivals":    float(actual_arrivals.mean()),
        "walkout_prob":     float((walk_outs > 0).mean()),
        "mean_walkouts":    float(walk_outs.mean()),
    }


def run(df: pd.DataFrame):
    rng = np.random.default_rng(42)

    # Empirical stats from synthetic data
    overall_cancel = empirical_cancel_rate(df)
    avg_room_rev   = float(df[df["Is_Cancelled"]==0]["Avg_Price_Per_Room"].mean())
    avg_nights     = float(df[df["Is_Cancelled"]==0]["Stayed_Room_Nights"].mean())
    avg_booking_rev = avg_room_rev * avg_nights

    print(f"Inventory: {INVENTORY} rooms")
    print(f"Overall cancellation rate: {overall_cancel:.2%}")
    print(f"Avg booking revenue: ₹{avg_booking_rev:,.0f}")

    # Sweep overbooking levels 0–6
    results = []
    for overbook in range(0, 7):
        n_accepted = INVENTORY + overbook
        sim = simulate_overbooking(n_accepted, overall_cancel,
                                   INVENTORY, N_SIM, rng)
        revenue_gained = (min(sim["mean_arrivals"], INVENTORY) * avg_room_rev)
        walkout_cost   = sim["mean_walkouts"] * avg_room_rev * WALKOUT_COST_MULT
        net_expected   = revenue_gained - walkout_cost
        results.append({
            "overbook_by":       overbook,
            "rooms_accepted":    n_accepted,
            "mean_arrivals":     round(sim["mean_arrivals"], 2),
            "walkout_prob":      round(sim["walkout_prob"], 4),
            "mean_walkouts":     round(sim["mean_walkouts"], 3),
            "expected_net_rev":  round(net_expected, 2),
        })

    res_df = pd.DataFrame(results)
    print("\n=== Overbooking simulation results ===")
    print(res_df.to_string(index=False))

    # Optimal policy: highest net revenue with walkout_prob ≤ tolerance
    feasible = res_df[res_df["walkout_prob"] <= WALKOUT_TOLERANCE]
    if len(feasible) > 0:
        optimal = feasible.loc[feasible["expected_net_rev"].idxmax()]
        print(f"\nOptimal overbook: {int(optimal['overbook_by'])} rooms "
              f"(walk-out prob {optimal['walkout_prob']:.1%}, "
              f"expected net ₹{optimal['expected_net_rev']:,.0f}/night)")
    else:
        optimal = res_df.iloc[0]
        print("No overbooking feasible within walk-out tolerance — recommend 0.")

    # By-season analysis (peak vs off-peak cancel rates differ)
    df2 = df.copy()
    df2["Check_In_Date"] = pd.to_datetime(df2["Check_In_Date"])
    df2["is_peak"] = df2["Check_In_Date"].dt.month.isin([4, 9, 10, 12])
    season_results = []
    for is_peak, label in [(True, "Peak (Apr/Sep/Oct/Dec)"),
                           (False, "Off-peak (Jan–Mar, May–Aug, Nov)")]:
        cr = float(df2[df2["is_peak"]==is_peak]["Is_Cancelled"].mean())
        sim = simulate_overbooking(INVENTORY + 2, cr, INVENTORY, N_SIM, rng)
        season_results.append({"season": label, "cancel_rate": round(cr,4),
                                "walkout_prob_at_+2": round(sim["walkout_prob"],4)})
    print("\nSeasonal sensitivity:")
    for s in season_results:
        print(f"  {s['season']}: cancel={s['cancel_rate']:.1%}, "
              f"walk-out at +2: {s['walkout_prob_at_+2']:.1%}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(res_df["overbook_by"], res_df["expected_net_rev"],
                 color="#E85D04", marker="o", lw=2)
    axes[0].axvline(int(optimal["overbook_by"]), color="#3fb950",
                    lw=1.5, ls="--", label=f"Optimal (+{int(optimal['overbook_by'])})")
    axes[0].set_xlabel("Rooms overbooked beyond capacity")
    axes[0].set_ylabel("Expected net revenue per night (₹)")
    axes[0].set_title("Expected net revenue vs overbooking level")
    axes[0].legend()

    ax2 = axes[1]
    color_bars = ["#3fb950" if r <= WALKOUT_TOLERANCE else "#E85D04"
                  for r in res_df["walkout_prob"]]
    bars = ax2.bar(res_df["overbook_by"], res_df["walkout_prob"] * 100, color=color_bars)
    ax2.axhline(WALKOUT_TOLERANCE * 100, color="#d29922", lw=1.5, ls="--",
                label=f"Tolerance ({WALKOUT_TOLERANCE:.0%})")
    ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.set_xlabel("Rooms overbooked beyond capacity")
    ax2.set_ylabel("Walk-out probability (%)")
    ax2.set_title("Walk-out probability vs overbooking level")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "overbooking_simulation.png", dpi=140)
    plt.close(fig)

    summary = {
        "inventory": INVENTORY,
        "overall_cancel_rate": round(overall_cancel, 4),
        "optimal_overbook": int(optimal["overbook_by"]),
        "optimal_walkout_prob": float(optimal["walkout_prob"]),
        "simulation_results": results,
        "seasonal": season_results,
    }
    with open(REPORTS / "overbooking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "synthetic" / "median_inn_synthetic.csv")
    run(df)
