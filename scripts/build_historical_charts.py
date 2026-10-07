"""Render the documented historical notebook aggregates; no dataset required."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "docs/historical_metrics.json").read_text())
    destination = root / "assets"
    destination.mkdir(exist_ok=True)
    models = source["models"]
    for name, report in models.items():
        matrix = np.array(report["confusion_matrix"], dtype=int)
        if matrix.shape != (10, 10) or matrix.sum() != report["tested"] or np.trace(matrix) != report["correct"]:
            raise ValueError(f"Inconsistent historical matrix for {name}")
    ink, burgundy, rose, pale = "#24303B", "#672744", "#BD7898", "#EDF0F2"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": ink, "axes.labelcolor": ink,
                         "svg.fonttype": "none", "svg.hashsalt": "svhn-historical"}):
        names = ["Dense network", "Convolutional network"]
        values = [models["ann_deep"]["accuracy"], models["cnn_regularized"]["accuracy"]]
        counts = [models["ann_deep"]["correct"], models["cnn_regularized"]["correct"]]
        fig, ax = plt.subplots(figsize=(8.6, 4.7))
        bars = ax.barh(names, values, color=[rose, burgundy], height=.55)
        for bar, count, value in zip(bars, counts, values):
            ax.text(value+.01, bar.get_y()+bar.get_height()/2,
                    f"{value:.2%}  ({count:,}/18,000)", va="center", weight="bold")
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0,.25,.5,.75,1], labels=["0%","25%","50%","75%","100%"])
        ax.set_xlabel("Accuracy on the course subset's test split")
        ax.set_title("CNN outperformed the dense model in the notebook", loc="left", weight="bold", pad=16)
        ax.grid(axis="x", color=pale)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(destination / "model-comparison.svg", format="svg", metadata={"Date":None})
        plt.close(fig)

        matrix = np.array(models["cnn_regularized"]["confusion_matrix"], dtype=int)
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        ax.imshow(normalized, cmap="RdPu", vmin=0, vmax=1)
        ax.set_xticks(range(10), labels=range(10))
        ax.set_yticks(range(10), labels=range(10))
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("Actual digit")
        ax.set_title("CNN predictions by digit · 18,000 test images", loc="left", weight="bold", pad=16)
        for row in range(10):
            for col in range(10):
                value = int(matrix[row,col])
                ax.text(col,row,str(value) if value else "·", ha="center",va="center",
                        fontsize=8.5, color="white" if normalized[row,col]>.55 else ink,
                        weight="bold" if row==col else "normal")
        ax.set_xticks(np.arange(-.5,10,1), minor=True)
        ax.set_yticks(np.arange(-.5,10,1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.tight_layout()
        fig.savefig(destination / "cnn-confusion-matrix.svg", format="svg", metadata={"Date":None})
        plt.close(fig)


if __name__ == "__main__":
    main()
