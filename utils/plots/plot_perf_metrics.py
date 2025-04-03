import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONTSIZE = 15
FONT = "Helvetica"
LINEWIDTH = 3


plt.rcParams["font.family"] = FONT


data = {
    "negative_ratio": [0.5, 0.5, 0.8, 0.8],
    "loss_type": ["multi", "pos.only", "multi", "pos.only"],
    "AUC": [0.91, 0.8476, 0.8173, 0.80],
    "balanced_accuracy": [0.85, 0.77, 0.75, 0.73],
    "f1_score": [0.9, 0.86, 0.841, 0.837],
}

df = pd.DataFrame(data)


fig, ax = plt.subplots(figsize=(10, 6))


metrics = ["AUC", "balanced_accuracy"]
bar_width = 0.2
x = np.arange(len(metrics))


hatches = {"multi": None, "pos.only": "///"}
ratio_colors = {"0.5": "blue", "0.8": "red"}

for i, (_, row) in enumerate(df.iterrows()):
    offset = i * bar_width

    hatch = hatches[row["loss_type"]]

    bars = ax.bar(
        x + offset,
        [row["AUC"], row["balanced_accuracy"]],
        width=bar_width,
        color=ratio_colors[str(row["negative_ratio"])],
        hatch=hatch,
        edgecolor="black",
        linewidth=LINEWIDTH - 1.5,
    )


ax.set_ylabel("Score", fontsize=FONTSIZE)
ax.set_ylim(0.7, 1.0)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(metrics, fontsize=FONTSIZE)
ax.set_title("Performance Comparison", fontsize=FONTSIZE)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.tick_params(axis="both", labelsize=FONTSIZE)


neg_ratio_patches = [
    mpatches.Patch(color=ratio_colors["0.5"], label="Neg=0.5"),
    mpatches.Patch(color=ratio_colors["0.8"], label="Neg=0.8"),
]

loss_type_patches = [
    mpatches.Patch(color="lightgray", label="multi"),
    mpatches.Patch(color="lightgray", hatch="///", label="pos.only"),
]


leg1 = ax.legend(
    handles=neg_ratio_patches,
    loc="upper left",
    title="Negative Ratio",
    fontsize=FONTSIZE,
)
ax.add_artist(leg1)
leg2 = ax.legend(
    handles=loss_type_patches, loc="upper right", title="Loss Type", fontsize=FONTSIZE
)


plt.setp(leg1.get_title(), fontsize=FONTSIZE)
plt.setp(leg2.get_title(), fontsize=FONTSIZE)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("output/performance_comparison.png", dpi=300)
