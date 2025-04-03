import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

FONTSIZE = 20


def load_loss_data(base_dir, model_dirs):
    loss_types = ["train_losses", "val_losses", "train_losses_pos", "train_losses_neg"]

    data = {}

    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        model_data = {}

        for loss_type in loss_types:
            loss_file = os.path.join(model_path, f"{loss_type}.npy")
            if os.path.exists(loss_file):
                model_data[loss_type] = np.load(loss_file)
            else:
                print(f"Note: {loss_file} does not exist for {model_dir}")

        if model_data:
            data[model_dir] = model_data

    return data


def plot_posloss_comparison(
    data, smoothing_window=None, output_dir=None, log_scale=False, ylim=None
):
    plt.figure(figsize=(12, 8))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    plt.rcParams["font.size"] = 15

    ratio_colors = {"0.5": "blue", "0.8": "red"}

    data_types = {
        "train_losses": {"marker": "o", "linestyle": "-", "label": "Train"},
        "val_losses": {"marker": "s", "linestyle": "--", "label": "Validation"},
    }

    for model_name, model_data in data.items():
        if "posloss" not in model_name:
            continue

        if "negrat0.5" in model_name:
            ratio = "0.5"
        elif "negrat0.8" in model_name:
            ratio = "0.8"
        else:
            continue

        color = ratio_colors[ratio]

        for loss_type, style in data_types.items():
            if loss_type in model_data:
                losses = model_data[loss_type]

                x = range(1, len(losses) + 1)

                plt.plot(
                    x,
                    losses,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=color,
                    linewidth=3,
                    markevery=max(1, len(x) // 15),
                    label=f"{ratio} - {style['label']}",
                )

    plt.title("Positive Loss Models Comparison", fontsize=FONTSIZE)
    plt.xlabel("Epochs", fontsize=FONTSIZE)
    plt.ylabel("Loss Value", fontsize=FONTSIZE)
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    if ylim:
        plt.ylim(ylim)

    ratio_legend_elements = [
        Line2D([0], [0], color="blue", lw=3, label="Ratio 0.5"),
        Line2D([0], [0], color="red", lw=3, label="Ratio 0.8"),
    ]

    datatype_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="-",
            linewidth=3,
            markersize=8,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="gray",
            linestyle="--",
            linewidth=3,
            markersize=8,
            label="Validation",
        ),
    ]

    first_legend = plt.legend(
        handles=ratio_legend_elements,
        loc="upper right",
        title="Negative Ratio",
        fontsize=FONTSIZE,
    )
    plt.setp(first_legend.get_title(), fontsize=FONTSIZE)
    plt.gca().add_artist(first_legend)

    second_legend = plt.legend(
        handles=datatype_legend_elements,
        loc="upper center",
        title="Data Type",
        fontsize=FONTSIZE,
    )
    plt.setp(second_legend.get_title(), fontsize=FONTSIZE)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if smoothing_window and smoothing_window > 1:
        plt.annotate(
            f"Smoothing: moving average over {smoothing_window} epochs",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        smoothing_suffix = f"_smooth{smoothing_window}" if smoothing_window else ""
        log_suffix = "_log" if log_scale else ""
        output_path = os.path.join(
            output_dir, f"posloss_comparison{smoothing_suffix}{log_suffix}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"Posloss plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multiloss_comparison(
    data, smoothing_window=None, output_dir=None, log_scale=False, ylim=None
):
    plt.figure(figsize=(12, 8))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    plt.rcParams["font.size"] = 15

    ratio_colors = {"0.5": "blue", "0.8": "red"}

    loss_styles = {
        "train_losses": {"marker": "o", "linestyle": "-", "label": "Train (Total)"},
        "val_losses": {"marker": "s", "linestyle": "--", "label": "Validation (Total)"},
        "train_losses_pos": {
            "marker": "^",
            "linestyle": "-.",
            "label": "Train (Pos Term)",
        },
        "train_losses_neg": {
            "marker": "x",
            "linestyle": ":",
            "label": "Train (Neg Term)",
        },
    }

    negrat05_length = 0
    for model_name, model_data in data.items():
        if "multiloss_negrat0.5" in model_name and "train_losses" in model_data:
            negrat05_length = len(model_data["train_losses"])
            break

    def extend_array(arr, target_length):
        if len(arr) >= target_length:
            return arr

        slope = arr[-1] - arr[-2]

        extension = [arr[-1] - slope * (i + 1) for i in range(target_length - len(arr))]

        return np.append(arr, extension)

    for model_name, model_data in data.items():
        if "multiloss" not in model_name:
            continue

        if "negrat0.5" in model_name:
            ratio = "0.5"
        elif "negrat0.8" in model_name:
            ratio = "0.8"
        else:
            continue

        color = ratio_colors[ratio]

        for loss_type, style in loss_styles.items():
            if loss_type in model_data:
                losses = model_data[loss_type]

                x = range(1, len(losses) + 1)

                plt.plot(
                    x,
                    losses,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=color,
                    linewidth=3,
                    markevery=max(1, len(x) // 15),
                    label=f"{ratio} - {style['label']}",
                )

    plt.title("Multi Loss Models Comparison", fontsize=FONTSIZE)
    plt.xlabel("Epochs", fontsize=FONTSIZE)
    plt.ylabel("Loss Value", fontsize=FONTSIZE)
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale("log")

    if ylim:
        plt.ylim(ylim)

    ratio_legend_elements = [
        Line2D([0], [0], color="blue", lw=3, label="Ratio 0.5"),
        Line2D([0], [0], color="red", lw=3, label="Ratio 0.8"),
    ]

    loss_type_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            linestyle="-",
            linewidth=3,
            markersize=8,
            label="Train (Total)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="gray",
            linestyle="--",
            linewidth=3,
            markersize=8,
            label="Validation (Total)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="-.",
            linewidth=3,
            markersize=8,
            label="Train (Pos Term)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            linestyle=":",
            linewidth=3,
            markersize=8,
            label="Train (Neg Term)",
        ),
    ]

    first_legend = plt.legend(
        handles=ratio_legend_elements,
        loc="upper right",
        title="Negative Ratio",
        fontsize=FONTSIZE,
    )
    plt.setp(first_legend.get_title(), fontsize=FONTSIZE)
    plt.gca().add_artist(first_legend)

    second_legend = plt.legend(
        handles=loss_type_legend_elements,
        loc="upper center",
        title="Loss Type",
        fontsize=FONTSIZE,
    )
    plt.setp(second_legend.get_title(), fontsize=FONTSIZE)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if smoothing_window and smoothing_window > 1:
        plt.annotate(
            f"Smoothing: moving average over {smoothing_window} epochs",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        smoothing_suffix = f"_smooth{smoothing_window}" if smoothing_window else ""
        log_suffix = "_log" if log_scale else ""
        output_path = os.path.join(
            output_dir, f"multiloss_comparison{smoothing_suffix}{log_suffix}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"Multiloss plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Outil de comparaison des courbes de perte pour modèles de détection de changements"
    )
    parser.add_argument(
        "--base_dir",
        default="output",
        help="Répertoire de base contenant les dossiers des modèles",
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=[
            "posloss_negrat0.5",
            "posloss_negrat0.8",
            "multiloss_negrat0.5",
            "multiloss_negrat0.8",
        ],
        help="Liste des dossiers de modèles à comparer",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=None,
        help="Taille de la fenêtre de lissage (moyenne mobile)",
    )
    parser.add_argument(
        "--output_dir",
        default="output/loss_comparison",
        help="Répertoire pour sauvegarder les graphiques",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Utiliser l'échelle logarithmique pour l'axe Y",
    )
    parser.add_argument(
        "--ylim", nargs=2, type=float, default=None, help="Limites de l'axe Y (min max)"
    )

    args = parser.parse_args()

    data = load_loss_data(args.base_dir, args.model_dirs)

    plot_posloss_comparison(
        data,
        smoothing_window=args.smoothing,
        output_dir=args.output_dir,
        log_scale=args.log_scale,
        ylim=tuple(args.ylim) if args.ylim else None,
    )

    plot_multiloss_comparison(
        data,
        smoothing_window=args.smoothing,
        output_dir=args.output_dir,
        log_scale=args.log_scale,
        ylim=tuple(args.ylim) if args.ylim else None,
    )


if __name__ == "__main__":
    main()
