#!/usr/bin/env python3
"""
Generate high-quality adversarial ELR figure for paper.

Data from paper.tex tikz figure:
- GPT-4o: Benign 23.3%, Adversarial 34.5% (+11.2pp)
- Claude-3.5: Benign 26.7%, Adversarial 38.0% (+11.3pp)
- GPT-4o-mini: Benign 30.0%, Adversarial 40.0% (+10.0pp)
- Qwen-72B: Benign 40.0%, Adversarial 55.0% (+15.0pp)
- Claude-3-Haiku: Benign 40.0%, Adversarial 60.0% (+20.0pp)
- Qwen-7B: Benign 80% (n=30), Adversarial 50% (n=10, different scenarios)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    }
)

# Data from paper
models = ["GPT-4o", "Claude-3.5", "GPT-4o-mini", "Qwen-72B", "Claude-Haiku", "Qwen-7B"]
benign = [23.3, 26.7, 30.0, 40.0, 40.0, 80.0]
adversarial = [34.5, 38.0, 40.0, 55.0, 60.0, 50.0]

# Create figure with better spacing
fig, ax = plt.subplots(figsize=(10, 5.5))

# Set positions for bars
x = np.arange(len(models))
width = 0.35

# Define colors (more modern palette)
color_benign = "#2E86AB"  # Deep blue
color_adversarial = "#A23B72"  # Deep magenta

# Create bars
bars1 = ax.bar(
    x - width / 2,
    benign,
    width,
    label="Benign (n=30)",
    color=color_benign,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    adversarial,
    width,
    label="Adversarial (n=10)",
    color=color_adversarial,
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8,
)

# Add value labels on bars
for i, (b, a) in enumerate(zip(benign, adversarial)):
    ax.text(
        i - width / 2,
        b + 1.5,
        f"{b:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color_benign,
    )
    ax.text(
        i + width / 2,
        a + 1.5,
        f"{a:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color_adversarial,
    )

    # Add delta annotation
    delta = a - b
    ax.text(
        i,
        max(b, a) + 3.5,
        f"+{delta:.1f}pp",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        color="#555555",
    )

# Add special marker for Qwen-7B (different scenarios)
ax.plot(
    [5 - width / 2 - 0.15, 5 - width / 2 + 0.15],
    [benign[5] + 0.8, benign[5] + 0.8],
    "r-",
    linewidth=2,
    marker="_",
)
ax.text(
    5 - width / 2, benign[5] + 2.5, "†", fontsize=14, ha="center", color="red", fontweight="bold"
)

# Customize axes
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("Exact Leakage Rate (%)", fontsize=12, fontweight="bold")
ax.set_title(
    "Benign vs Adversarial ELR Comparison\nFrontier models show +10pp increase under attack",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 75)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Add legend
ax.legend(loc="upper left", framealpha=0.95, edgecolor="black", fontsize=10)

# Add footer note
fig.text(
    0.12,
    0.02,
    "†Qwen-7B benign (80%) from 30-scenario eval; adversarial (50%) from different 10 scenarios (not directly comparable)",
    fontsize=8,
    style="italic",
    color="#666666",
)

# Tight layout with space for footer
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save figure
output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "Fig_Adversarial_ELR.pdf")
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_path}")

# Also save PNG version
output_path_png = os.path.join(output_dir, "Fig_Adversarial_ELR.png")
plt.savefig(output_path_png, format="png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_path_png}")

plt.close()

print("\nFigure generation complete!")
