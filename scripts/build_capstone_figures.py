"""Render public-safe capstone figures from aggregate notebook results."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_figures(repo):
    outputs = repo / "work" / "outputs"
    receipt = json.loads((outputs / "capstone_visuals.json").read_text(encoding="utf-8"))
    folder = outputs / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    colors = {"model": "#4267B2", "rule": "#087F8C", "random": "#89949F"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#253746",
        "text.color": "#253746", "xtick.color": "#253746", "ytick.color": "#253746",
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
    })

    def finish(fig, name, note):
        fig.text(0.09, 0.055, note, fontsize=9, color="#536575", va="bottom")
        for extension in ["png", "svg"]:
            fig.savefig(folder / f"{name}.{extension}", dpi=170, facecolor="white")
        plt.close(fig)

    # Identical seven-client evaluation population for both rankings.
    rows = receipt["test_comparison"]
    values = [rows[0]["precision_at_20_pct"], rows[1]["precision_at_20_pct"],
              receipt["expected_random_precision_at_20_pct"]]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.25)
    fig.suptitle("The model did not improve top-20 ranking over the rule",
                 x=0.10, y=0.95, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.10, 0.88, "Observed later decline among each client's first 20 ranked pages",
             fontsize=11, color="#536575")
    bars = ax.bar(["Logistic regression", "Frozen rule", "Expected random"], values,
                  color=[colors["model"], colors["rule"], colors["random"]], width=0.60)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean client Precision@20 (%)")
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%",
                ha="center", fontweight="bold", fontsize=13)
    finish(fig, "test_precision_at_20",
           "7 eligible test clients · 140 top-20 slots · Model 85 / rule 86 declining pages\n"
           "March 2026 · Same evaluation rows · One-slot difference does not establish superiority")

    # Anonymous labels; clients receive equal weight in the headline metric.
    clients = receipt["test_client_results"]
    x = np.arange(len(clients))
    fig, ax = plt.subplots(figsize=(10, 6.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.27)
    fig.suptitle("The overall average hides differences between clients",
                 x=0.09, y=0.95, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.09, 0.88, "Client-specific top-20 precision versus that client's random expectation",
             fontsize=11, color="#536575")
    ax.bar(x - 0.18, [r["model_precision_at_20_pct"] for r in clients], 0.34,
           label="Logistic regression", color=colors["model"])
    ax.bar(x + 0.18, [r["rule_precision_at_20_pct"] for r in clients], 0.34,
           label="Frozen rule", color=colors["rule"])
    ax.scatter(x, [r["random_expected_precision_at_20_pct"] for r in clients],
               marker="D", s=40, color="#253746", label="Expected random", zorder=3)
    ax.set_xticks(x, [f"{r['client_alias']}\nn={r['pages']:,}" for r in clients], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Precision@20 (%)")
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3, frameon=False, fontsize=10)
    finish(fig, "precision_by_client",
           "n = labeled pages per client; each ranking selects 20 · Client labels are anonymous\n"
           "1 additional test client has fewer than 20 pages and is excluded from this metric")

    # Signed coefficients describe conditional associations, not causal effects.
    names = {
        "log_impressions": "Search exposure (log)", "log_clicks": "Clicks (log)",
        "ctr_pct": "CTR (%)", "weighted_position": "Average search position",
        "prior_week_change_pct": "Past weekly impression change",
    }
    weights = sorted(receipt["standardized_feature_weights"].items(), key=lambda item: abs(item[1]))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.subplots_adjust(left=0.35, right=0.94, top=0.81, bottom=0.25)
    fig.suptitle("What the first model learned from its five inputs",
                 x=0.08, y=0.95, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.88, "Signed weights after standardizing the training features",
             fontsize=11, color="#536575")
    positions = np.arange(len(weights))
    ax.barh(positions, [value for _, value in weights], height=0.58,
            color=[colors["rule"] if value >= 0 else colors["model"] for _, value in weights])
    ax.set_yticks(positions, [names[name] for name, _ in weights], fontsize=10)
    ax.axvline(0, color="#536575", linewidth=1)
    ax.set_xlim(-0.95, 0.80)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    for position, (_, value) in zip(positions, weights):
        ax.text(value + (0.035 if value >= 0 else -0.035), position, f"{value:+.3f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=10)
    finish(fig, "model_feature_weights",
           "14,070 training pages · 24 clients · Positive weights raise the fitted decline score\n"
           "Features are correlated; weights are conditional associations, not causal effects")
    print("Saved three figures as PNG and SVG in work/outputs/figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    build_figures(arguments.repo.resolve())
