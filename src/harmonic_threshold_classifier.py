"""
Harmonic threshold classifier for drone vs. no‑drone audio segments.

This script demonstrates a simple yet effective baseline for classifying
audio chunks based solely on the harmonic content of the signal.  Drone
propellers produce strong tonal (harmonic) components, whereas many
background noises are broadband and percussive.  By computing the
harmonic ratio (energy of the harmonic component divided by total
energy) for each 3‑second segment, we can make a binary decision: if
the ratio exceeds a chosen threshold, label the segment as ``drone``;
otherwise, ``no drone``.  This baseline requires no machine learning
model and can achieve high accuracy when mislabelled drone segments
contain silence or distant flights.

The script loads the mini dataset from Zenodo (pre‑downloaded and
extracted under ``/home/oai/share/data_mini/Drone vs. no Drone Mini``),
computes harmonic ratios for all segments in the validation set and
computes the confusion matrix and accuracy using the threshold.  It
also produces a probability distribution plot (ratio histograms) and
confusion matrix image.  The code stubs out the ``caas_jupyter_tools``
module to avoid network calls when using matplotlib.

Run with ``python harmonic_threshold_classifier.py``.  Ensure that
``librosa`` and ``matplotlib`` are installed.
"""

import os
import sys
import types
import numpy as np
import librosa

# Stub out CAAS logging to prevent network calls when saving images.  This
# must be done BEFORE importing matplotlib, since matplotlib registers
# hooks to caas_jupyter_tools at import time.  We replace the module
# with a simple namespace that defines the expected functions.
sys.modules['caas_jupyter_tools'] = types.SimpleNamespace(
    log_matplotlib_img_fallback=lambda *args, **kwargs: None,
    log_exception=lambda *args, **kwargs: None,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Dataset root
DATA_ROOT = "/home/oai/share/data_mini/Drone vs. no Drone Mini"
# Sampling parameters
SAMPLE_RATE = 16000
SEG_LEN_SEC = 3
SEG_LEN = SEG_LEN_SEC * SAMPLE_RATE
# Threshold for classifying drone based on harmonic ratio
HARM_THRESHOLD = 0.7


def compute_harmonic_ratio(segment: np.ndarray) -> float:
    """Compute the harmonic ratio of an audio segment using HPSS.

    Args:
        segment: 1‑D numpy array of samples.

    Returns:
        Harmonic energy divided by total energy.
    """
    harm, perc = librosa.effects.hpss(segment)
    harm_energy = np.sum(np.abs(harm))
    perc_energy = np.sum(np.abs(perc))
    return float(harm_energy / (harm_energy + perc_energy + 1e-9))


def load_segments_and_ratios(subset: str, label_name: str) -> tuple[list[float], list[int]]:
    """Load all 3‑second segments from a given subset and compute harmonic ratios.

    Args:
        subset: "TRAINING" or "VALIDATION".
        label_name: "drone" or "no drone".

    Returns:
        A tuple (ratios, labels) where ``ratios`` is a list of harmonic
        ratio values and ``labels`` contains the corresponding integer
        class labels (1 for drone, 0 for no drone).
    """
    folder = os.path.join(DATA_ROOT, subset, label_name)
    ratios: list[float] = []
    labels: list[int] = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        for i in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
            seg = y[i : i + SEG_LEN]
            if len(seg) != SEG_LEN:
                continue
            ratio = compute_harmonic_ratio(seg)
            ratios.append(ratio)
            labels.append(1 if label_name == "drone" else 0)
    return ratios, labels


def main() -> None:
    # Load validation segments and compute ratios
    drone_ratios, drone_labels = load_segments_and_ratios("VALIDATION", "drone")
    no_ratios, no_labels = load_segments_and_ratios("VALIDATION", "no drone")
    all_ratios = drone_ratios + no_ratios
    all_labels = drone_labels + no_labels
    # Predict using threshold
    preds = [1 if r >= HARM_THRESHOLD else 0 for r in all_ratios]
    # Compute confusion matrix and accuracy
    conf = confusion_matrix(all_labels, preds)
    acc = (conf[0][0] + conf[1][1]) / conf.sum()
    print(f"Harmonic threshold {HARM_THRESHOLD:.2f} validation accuracy: {acc:.4f}")
    print("Confusion matrix:\n", conf)
    # Save confusion matrix image
    fig, ax = plt.subplots()
    ax.imshow(conf, cmap="Blues")
    ax.set_title("Confusion Matrix (Harmonic Threshold)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Drone", "Drone"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Drone", "Drone"])
    fig.tight_layout()
    conf_path = "harmonic_confusion_matrix.png"
    fig.savefig(conf_path)
    plt.close(fig)
    # Save probability distribution (histogram of ratios)
    fig2, ax2 = plt.subplots()
    ax2.hist(drone_ratios, bins=20, alpha=0.6, label="Drone")
    ax2.hist(no_ratios, bins=20, alpha=0.6, label="No Drone")
    ax2.set_title("Harmonic Ratio Distribution")
    ax2.set_xlabel("Harmonic ratio")
    ax2.set_ylabel("Count")
    ax2.legend()
    dist_path = "harmonic_probability_distribution.png"
    fig2.savefig(dist_path)
    plt.close(fig2)
    print("Saved confusion matrix to", conf_path)
    print("Saved probability distribution to", dist_path)


if __name__ == "__main__":
    main()