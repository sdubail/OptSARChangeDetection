import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

path = Path("output/nofreeze-p0.03-lr0.000001-targetneg0.8.txt")

logs = open(path).read()


train_pattern = r"Train:.*?loss=(\d+\.\d+)"
val_pattern = r"Val:.*?loss=(\d+\.\d+)"

# Extract loss values
train_loss_values = re.findall(train_pattern, logs)
val_loss_values = re.findall(val_pattern, logs)

# Convert strings to floats
train_loss_values = [float(val) for val in train_loss_values]
val_loss_values = [float(val) for val in val_loss_values]

# Convert your loss values to pandas Series
s_train = pd.Series(train_loss_values)
s_val = pd.Series(val_loss_values) if val_loss_values else None

# Calculate rolling mean and std with a window of 20
window = 20

# For training loss
rolling_mean_train = s_train.rolling(window, center=True).mean()
rolling_std_train = s_train.rolling(window, center=True).std()

# For the first and last few points where rolling calculation gives NaN
rolling_mean_train = rolling_mean_train.fillna(s_train)
rolling_std_train = rolling_std_train.fillna(s_train.std())

# Create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot training loss on first subplot
axes[0].plot(rolling_mean_train, linewidth=2, color="blue", label="Rolling Mean")
axes[0].fill_between(
    range(len(s_train)),
    rolling_mean_train - rolling_std_train,
    rolling_mean_train + rolling_std_train,
    color="blue",
    alpha=0.2,
    label="±1 std dev",
)
axes[0].set_title("Training Loss")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Plot validation loss on second subplot if it exists
if s_val is not None and len(s_val) > 0:
    rolling_mean_val = s_val.rolling(window, center=True).mean()
    rolling_std_val = s_val.rolling(window, center=True).std()

    rolling_mean_val = rolling_mean_val.fillna(s_val)
    rolling_std_val = rolling_std_val.fillna(s_val.std())

    axes[1].plot(rolling_mean_val, linewidth=2, color="red", label="Rolling Mean")
    axes[1].fill_between(
        range(len(s_val)),
        rolling_mean_val - rolling_std_val,
        rolling_mean_val + rolling_std_val,
        color="red",
        alpha=0.2,
        label="±1 std dev",
    )
    axes[1].set_title("Validation Loss")
else:
    axes[1].text(
        0.5,
        0.5,
        "No validation data found",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
    )
    axes[1].set_title("Validation Loss (No Data)")

axes[1].set_xlabel("Steps")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# Add space between subplots
plt.tight_layout()

# Save the figure
plt.savefig(path.with_suffix(".png"), dpi=300)
