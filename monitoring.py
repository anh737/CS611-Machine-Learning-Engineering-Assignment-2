"""
=============================================================================
monitoring.py  -  Model Performance & Stability Monitoring (Gold table + viz)
=============================================================================
Purpose
    Read the Gold predictions table produced by inference.py and monitor the
    model across time on two axes:
        1. PERFORMANCE  -> AUC and Gini per application month (where the actual
                           label is available).
        2. STABILITY    -> Population Stability Index (PSI) of the predicted
                           score distribution per month vs a baseline month.
    Results are written back as a Gold monitoring table, and a summary chart
    (PNG) is produced for the presentation deck.

Design notes
    * Pure pandas + matplotlib (NO Spark).
    * PSI interpretation (industry rule of thumb):
        PSI < 0.10           -> stable (no significant shift)
        0.10 <= PSI < 0.25   -> moderate shift (investigate)
        PSI >= 0.25          -> major shift (model likely needs refresh)

Usage
    python monitoring.py
=============================================================================
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend (works in Docker / Airflow workers)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def calculate_psi(expected_scores, actual_scores, n_bins=10):
    """
    Population Stability Index between a baseline score distribution
    (expected) and a current month's score distribution (actual).

    Bins are quantile edges of the baseline so each baseline bin holds ~10%.
    A small epsilon avoids division-by-zero / log(0).
    """
    # Build bin edges from the baseline distribution
    edges = np.quantile(expected_scores, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    edges[0], edges[-1] = -np.inf, np.inf  # capture the tails

    exp_counts, _ = np.histogram(expected_scores, bins=edges)
    act_counts, _ = np.histogram(actual_scores, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), eps, None)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), eps, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def safe_auc(y_true, y_score):
    """AUC only makes sense if both classes are present in the month."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
def load_predictions(pred_dir):
    """Read every monthly Gold prediction partition into one DataFrame."""
    parts = glob.glob(os.path.join(pred_dir, "*.parquet"))
    if not parts:
        raise FileNotFoundError(f"[CRITICAL] No prediction parquet found in {pred_dir}. "
                                f"Run inference.py first.")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.sort_values("snapshot_date")


def main():
    print("\n=====================================================================")
    print("          MODEL MONITORING  -  PERFORMANCE & STABILITY              ")
    print("=====================================================================\n")

    pred_dir = "datamart/gold/model_predictions/"
    monitor_dir = "datamart/gold/model_monitoring/"
    os.makedirs(monitor_dir, exist_ok=True)

    # 1. Load all scored predictions
    df = load_predictions(pred_dir)
    df["month"] = df["snapshot_date"].dt.to_period("M").dt.to_timestamp()
    print(f"[INFO] Loaded {len(df)} predictions across {df['month'].nunique()} months.")

    # 2. Choose a STABILITY BASELINE = the earliest month with enough volume.
    #    (Ideally this would be the training-period score distribution; the
    #    earliest scored month is a reasonable, reproducible proxy.)
    month_counts = df.groupby("month").size()
    baseline_month = month_counts[month_counts >= 50].index.min()
    if pd.isna(baseline_month):
        baseline_month = month_counts.index.min()
    baseline_scores = df.loc[df["month"] == baseline_month, "model_predict_proba"].values
    print(f"[INFO] PSI baseline month: {baseline_month.date()} (n={len(baseline_scores)})")

    # 3. Compute per-month performance + stability metrics
    records = []
    for month, g in df.groupby("month"):
        auc = safe_auc(g["label"].values, g["model_predict_proba"].values)
        psi = calculate_psi(baseline_scores, g["model_predict_proba"].values)
        records.append({
            "month": month,
            "n_records": len(g),
            "actual_bad_rate": round(g["label"].mean(), 4),
            "avg_predicted_proba": round(g["model_predict_proba"].mean(), 4),
            "auc": round(auc, 4) if auc == auc else np.nan,   # NaN-safe
            "gini": round(2 * auc - 1, 4) if auc == auc else np.nan,
            "psi_vs_baseline": round(psi, 4),
        })
    monitor_df = pd.DataFrame(records).sort_values("month")

    # 4. Persist the monitoring results as a Gold table
    out_table = os.path.join(monitor_dir, "gold_model_monitoring.parquet")
    monitor_df.to_parquet(out_table, index=False)
    monitor_df.to_csv(os.path.join(monitor_dir, "gold_model_monitoring.csv"), index=False)
    print(f"[SUCCESS] Monitoring table written to {out_table}")
    print("\n--- Monitoring summary ---")
    print(monitor_df.to_string(index=False))

    # 5. Visualise performance & stability across time
    plot_path = os.path.join(monitor_dir, "model_monitoring_plot.png")
    _plot_monitoring(monitor_df, baseline_month, plot_path)
    print(f"\n[SUCCESS] Monitoring chart saved to {plot_path}")

    # 6. Simple governance signal based on latest PSI / Gini
    latest = monitor_df.dropna(subset=["psi_vs_baseline"]).iloc[-1]
    psi_val = latest["psi_vs_baseline"]
    if psi_val >= 0.25:
        verdict = "MAJOR DRIFT -> recommend model refresh"
    elif psi_val >= 0.10:
        verdict = "MODERATE DRIFT -> monitor closely"
    else:
        verdict = "STABLE -> no action needed"
    print(f"\n[GOVERNANCE] Latest month {latest['month'].date()} "
          f"PSI={psi_val} -> {verdict}")

    print("\n=====================================================================")
    print("                 MONITORING STAGE COMPLETED CLEANLY                 ")
    print("=====================================================================\n")


def _plot_monitoring(monitor_df, baseline_month, plot_path):
    """Three stacked panels: AUC/Gini, PSI (with thresholds), volume & bad-rate."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    x = monitor_df["month"]

    # Panel 1: Performance (AUC & Gini)
    axes[0].plot(x, monitor_df["auc"], marker="o", label="AUC", color="#1f77b4")
    axes[0].plot(x, monitor_df["gini"], marker="s", label="Gini", color="#ff7f0e")
    axes[0].axhline(0.5, ls="--", color="grey", lw=1)
    axes[0].set_title("Model Performance over Time (AUC / Gini)")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # Panel 2: Stability (PSI) with governance thresholds
    axes[1].plot(x, monitor_df["psi_vs_baseline"], marker="o", color="#2ca02c", label="PSI")
    axes[1].axhline(0.10, ls="--", color="orange", lw=1, label="0.10 moderate")
    axes[1].axhline(0.25, ls="--", color="red", lw=1, label="0.25 major")
    axes[1].set_title(f"Score Stability over Time (PSI vs baseline {baseline_month.date()})")
    axes[1].set_ylabel("PSI")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    # Panel 3: Volume & actual vs predicted bad-rate
    ax3 = axes[2]
    ax3.bar(x, monitor_df["n_records"], width=20, alpha=0.3, color="grey", label="N records")
    ax3.set_ylabel("N records")
    ax3b = ax3.twinx()
    ax3b.plot(x, monitor_df["actual_bad_rate"], marker="o", color="#d62728", label="Actual bad-rate")
    ax3b.plot(x, monitor_df["avg_predicted_proba"], marker="^", color="#9467bd", label="Avg predicted proba")
    ax3b.set_ylabel("Rate")
    ax3.set_title("Volume & Actual vs Predicted Bad-rate")
    ax3.set_xlabel("Month")
    # merge legends from both twin axes
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="best")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
