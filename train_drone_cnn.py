"""
Train a drone vs. no‑drone classifier using real room impulse responses (RIRs),
augmented audio and harmonic pre‑classification.

This script builds on the previous training pipeline but addresses three
shortcomings:

1. **Real RIRs:** Instead of synthesising random RIRs, it loads a pool of
   authentic RIR recordings from a ``RIRs`` directory (e.g. extracted from
   the droneClassifier_v2 repository). Each drone segment is convolved
   with a randomly chosen RIR to simulate realistic reverberation.

2. **Shorter segments:** The segment length has been reduced from 3 to 2
   seconds. Shorter chunks increase the number of training examples and
   reduce the likelihood of including silent or distant drone phases in
   a labelled "drone" segment.

3. **Improved network:** A deeper convolutional neural network
   with four convolutional blocks and dropout is used to extract
   richer spectro‑temporal features from the log‑Mel spectrograms.

As before, drone segments are filtered by harmonic content before being
added to the training set, and various augmentations such as echo,
background noise, pitch shifting and time stretching are applied. The
script prints the epoch‑wise training loss and validation accuracy, and
saves the confusion matrix and a histogram of predicted probabilities.
"""

import os
import sys
import types
import random
from typing import List, Tuple

import numpy as np
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import scipy.signal

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Additional imports for exporting the model to TFLite.  These imports
# are done lazily inside the conversion function to avoid import
# failures on systems where ONNX or TensorFlow are not available.


# Prevent matplotlib from contacting a local Jupyter server when saving
# figures.  Matplotlib registers fallback hooks via caas_jupyter_tools
# that attempt to POST images to localhost:8080; inserting a dummy
# module disables these hooks.
sys.modules['caas_jupyter_tools'] = types.SimpleNamespace(
    log_matplotlib_img_fallback=lambda *args, **kwargs: None,
    log_exception=lambda *args, **kwargs: None,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths to dataset and RIRs.  Adjust ``DATA_ROOT`` to point to the
# extracted mini dataset on your machine.  ``RIR_ROOT`` should point to
# the folder containing RIR wav files.
DATA_ROOT = "d:/Dropbox/03 H2 Think/AuDroK mFund/Auswertungen/Datensätze/Drone vs. No Drone/"
RIR_ROOT = "d:/Dropbox/03 H2 Think/AuDroK mFund/Auswertungen/25-03 Drone Classifier/RIRs/"

# Audio parameters
SAMPLE_RATE = 16000
SEG_LEN_SEC = 1  # 2‑second segments
SEG_LEN = SEG_LEN_SEC * SAMPLE_RATE

# Pre‑classification threshold: discard drone segments with low harmonic
# content.  The threshold 0.7 was chosen empirically.
HARM_THRESHOLD = 0.7

# Augmentation probabilities
P_APPLY_RIR = 0.7
P_APPLY_ECHO = 0.5
P_APPLY_NOISE = 0.8
P_PITCH_SHIFT = 0.2
P_TIME_STRETCH = 0.6


def compute_harmonic_ratio(signal: np.ndarray) -> float:
    """Compute the harmonic ratio of an audio signal using HPSS."""
    harm, perc = librosa.effects.hpss(signal)
    harm_energy = np.sum(np.abs(harm))
    perc_energy = np.sum(np.abs(perc))
    return float(harm_energy / (harm_energy + perc_energy + 1e-9))


def load_rir_pool() -> List[np.ndarray]:
    """Load real room impulse responses from ``RIR_ROOT``."""
    rir_files: List[np.ndarray] = []
    if os.path.isdir(RIR_ROOT):
        for fname in os.listdir(RIR_ROOT):
            if not fname.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                continue
            path = os.path.join(RIR_ROOT, fname)
            try:
                rir, sr = librosa.load(path, sr=SAMPLE_RATE)
                rir = rir.astype(np.float32)
                if np.max(np.abs(rir)) > 1e-6:
                    rir = rir / np.max(np.abs(rir))
                rir_files.append(rir)
            except Exception:
                continue
    return rir_files


def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve an audio signal with a RIR and adjust length."""
    conv = np.convolve(signal, rir, mode="full")
    if len(conv) >= len(signal):
        conv = conv[: len(signal)]
    else:
        conv = np.pad(conv, (0, len(signal) - len(conv)))
    return conv


def add_ground_reflection(signal, delay_ms, amplitude):
    delay_samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
    echo = np.zeros_like(signal)
    if delay_samples < len(signal):
        echo[delay_samples:] = signal[:-delay_samples]
    # leichte Tiefpassfilterung auf Reflexion
    echo = scipy.signal.lfilter([0.7, 0.3], [1.0], echo)
    return signal + amplitude * echo


def mix_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(clean)]
    sig_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / (noise_power + 1e-9))
    return clean + noise_scaled


def pitch_shift(signal: np.ndarray, steps: float) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(signal, sr=SAMPLE_RATE, n_steps=steps)
    if len(shifted) > len(signal):
        shifted = shifted[: len(signal)]
    elif len(shifted) < len(signal):
        shifted = np.pad(shifted, (0, len(signal) - len(shifted)))
    return shifted.astype(signal.dtype)


def time_stretch(signal: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(signal, rate=rate)
    if len(stretched) > len(signal):
        stretched = stretched[: len(signal)]
    else:
        stretched = np.pad(stretched, (0, len(signal) - len(stretched)))
    return stretched.astype(signal.dtype)


def load_noise_pool(no_folder: str) -> List[np.ndarray]:
    pool: List[np.ndarray] = []
    for fname in os.listdir(no_folder):
        path = os.path.join(no_folder, fname)
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE)
        except Exception:
            continue
        hop = SEG_LEN // 2  # 50% overlap
        for i in range(0, len(y) - SEG_LEN + 1, hop):
            seg = y[i:i+SEG_LEN]
            if len(seg) == SEG_LEN:
                pool.append(seg)
    return pool


def augment_signal(signal: np.ndarray, noise_pool: List[np.ndarray], rir_pool: List[np.ndarray]) -> np.ndarray:
    augmented = signal.copy()
    if rir_pool and random.random() < P_APPLY_RIR:
        rir = random.choice(rir_pool)
        augmented = convolve_rir(augmented, rir)
    if random.random() < P_APPLY_ECHO:
        delay = random.randint(5, 30)
        amp = random.uniform(0.02, 0.08)
        augmented = add_ground_reflection(augmented, delay_ms=delay, amplitude=amp)
    if noise_pool and random.random() < P_APPLY_NOISE:
        noise_seg = random.choice(noise_pool)
        snr = random.uniform(0, 10)
        augmented = mix_noise(augmented, noise_seg, snr_db=snr)
    if random.random() < P_PITCH_SHIFT:
        steps = random.uniform(-2, 2)
        augmented = pitch_shift(augmented, steps)
    if random.random() < P_TIME_STRETCH:
        rate = random.uniform(0.9, 1.1)
        augmented = time_stretch(augmented, rate)
    max_abs = np.max(np.abs(augmented))
    if max_abs > 1e-3:
        augmented = augmented / max_abs
    return augmented


def load_segments(folder: str, label: int, pre_classify: bool, noise_pool: List[np.ndarray],
                  rir_pool: List[np.ndarray], augment: bool = False, num_augment: int = 1) -> Tuple[List[np.ndarray], List[int]]:
    segments: List[np.ndarray] = []
    labels: List[int] = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE)
        except Exception:
            continue
        for i in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
            seg = y[i : i + SEG_LEN]
            if len(seg) < SEG_LEN:
                continue
            if label == 1 and pre_classify:
                ratio = compute_harmonic_ratio(seg)
                if ratio < HARM_THRESHOLD:
                    continue
            segments.append(seg)
            labels.append(label)
            if label == 1 and augment:
                for _ in range(num_augment):
                    aug = augment_signal(seg, noise_pool, rir_pool)
                    segments.append(aug)
                    labels.append(label)
    return segments, labels


# ---------------------------------------------------------------------------
# Evaluation and plotting utilities
#
# The following helper functions encapsulate the logic for computing
# predictions on a data loader, plotting a confusion matrix with both
# absolute and relative values, plotting a probability distribution of
# predicted probabilities, and exporting the trained model to a TFLite
# format.  Keeping these operations in separate functions makes the
# main training loop cleaner and the code easier to understand.

def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model over all batches in a DataLoader and collect the
    ground‑truth labels and predicted probabilities.

    Parameters
    ----------
    model : nn.Module
        The trained neural network in evaluation mode.
    loader : DataLoader
        DataLoader providing batches of (feature, label) pairs.
    device : torch.device
        The device (CPU or GPU) on which the model resides.

    Returns
    -------
    labels : np.ndarray
        1‑D array of true labels (1 for Drone, 0 for No Drone).
    probs : np.ndarray
        1‑D array of predicted probabilities for the Drone class.
    """
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            # forward pass produces probabilities due to final sigmoid
            batch_probs = model(X_batch).cpu().numpy()
            all_probs.extend(batch_probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    return labels, probs


def plot_confusion(labels: np.ndarray, probs: np.ndarray, suffix: str) -> str:
    """
    Generate and save a confusion matrix plot showing both absolute and
    relative values.  The orientation is defined such that the top row
    and left column correspond to the "Drone" class.  The resulting
    figure is saved to disk and the filename is returned.

    Parameters
    ----------
    labels : np.ndarray
        True binary labels (1 = Drone, 0 = No Drone).
    probs : np.ndarray
        Predicted probabilities for the Drone class.
    suffix : str
        Suffix added to the filename to distinguish different plots.

    Returns
    -------
    filename : str
        Path to the saved PNG image containing the confusion matrix.
    """
    # Convert probabilities to hard predictions with a 0.5 threshold
    preds = (probs >= 0.5).astype(int)
    # Compute confusion matrix with the Drone class (label 1) as the first index.
    cm = confusion_matrix(labels, preds, labels=[1, 0])
    # Normalize to obtain percentages relative to all samples
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # Prepare labels for axes
    classes = ["Drone", "No Drone"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
    # Axis ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix ({suffix})")
    # Write absolute counts and percentages into each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = cm_percent[i, j] * 100.0
            ax.text(j, i, f"{count}\n({perc:.1f}% )", ha="center", va="center",
                    color="white" if cm_percent[i, j] > 0.5 else "black")
    fig.tight_layout()
    filename = f"confusion_matrix_{suffix}.png"
    fig.savefig(filename)
    plt.close(fig)
    return filename


def plot_probability_distribution(labels: np.ndarray, probs: np.ndarray, suffix: str) -> str:
    """
    Create a histogram showing the distribution of predicted probabilities
    separately for the Drone and No‑Drone classes.

    Parameters
    ----------
    labels : np.ndarray
        True binary labels (1 = Drone, 0 = No Drone).
    probs : np.ndarray
        Predicted probabilities for the Drone class.
    suffix : str
        Suffix added to the filename to distinguish different plots.

    Returns
    -------
    filename : str
        Path to the saved PNG file containing the histogram.
    """
    drone_probs = probs[labels == 1]
    no_probs = probs[labels == 0]
    fig, ax = plt.subplots()
    # Histogram bins across [0,1]
    bins = np.linspace(0.0, 1.0, 40)
    ax.hist(drone_probs, bins=bins, alpha=0.6, label="Drone", density=True)
    ax.hist(no_probs, bins=bins, alpha=0.6, label="No Drone", density=True)
    ax.set_xlabel("Predicted probability for class 'Drone'")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution ({suffix})")
    ax.legend()
    fig.tight_layout()
    filename = f"probability_distribution_{suffix}.png"
    fig.savefig(filename)
    plt.close(fig)
    return filename


def export_to_tflite(model: nn.Module, device: torch.device, input_shape: Tuple[int, int, int, int],
                     filename: str = "model.tflite") -> None:
    """
    Export a PyTorch model to the TensorFlow Lite format.  The model is
    first exported to ONNX and then converted to TFLite using the
    onnx‑tf backend and TensorFlow Lite converter.  If the required
    packages are not available, a message is printed and the function
    returns without creating a file.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model.
    device : torch.device
        The device on which the model resides.
    input_shape : Tuple[int, int, int, int]
        Shape of a dummy input tensor (batch_size, channels, height, width).
    filename : str, optional
        Output filename for the TFLite model.  Defaults to 'model.tflite'.
    """
    # Create a dummy input tensor with the appropriate shape
    dummy_input = torch.randn(*input_shape, device=device)
    # Export to ONNX
    onnx_path = "model.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11,
        )
    except Exception as e:
        print(f"Failed to export to ONNX: {e}")
        return
    # Attempt to convert the ONNX model to TFLite
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        concrete_func = tf_rep.export_graph()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        with open(filename, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model to {filename}")
    except Exception as e:
        print(f"Failed to convert to TFLite: {e}")
        # If conversion fails, leave the partially generated ONNX file in place



def extract_logmel(segment: np.ndarray, n_mels: int = 64, hop_length: int = 512) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE,
                                         n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


class DroneDataset(Dataset):
    def __init__(self, segments: List[np.ndarray], labels: List[int]):
        self.features = [extract_logmel(seg) for seg in segments]
        self.labels = labels
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int):
        x = self.features[idx][np.newaxis, :, :]
        y = self.labels[idx]
        return x, y


class ImprovedCNN(nn.Module):
    """A deeper CNN with four convolutional blocks and dropout."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            # Determine flatten size dynamically by passing a dummy tensor
            dummy = torch.zeros(1, 1, 64, int(np.ceil((SEG_LEN / 512))))
            out = self.pool(self.bn1(torch.relu(self.conv1(dummy))))
            out = self.pool(self.bn2(torch.relu(self.conv2(out))))
            out = self.pool(self.bn3(torch.relu(self.conv3(out))))
            out = self.pool(self.bn4(torch.relu(self.conv4(out))))
            self.flatten_dim = out.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)


def train_model() -> Tuple[str, str]:
    """Load data, train the CNN and generate evaluation plots."""
    # Load RIRs and noise pool
    rir_pool = load_rir_pool()
    noise_pool = load_noise_pool(os.path.join(DATA_ROOT, "TRAINING", "no drone"))
    # Load training data
    train_drone_segments, train_drone_labels = load_segments(
        os.path.join(DATA_ROOT, "TRAINING", "drone"), label=1, pre_classify=True,
        noise_pool=noise_pool, rir_pool=rir_pool, augment=True, num_augment=1
    )
    train_no_segments, train_no_labels = load_segments(
        os.path.join(DATA_ROOT, "TRAINING", "no drone"), label=0, pre_classify=False,
        noise_pool=None, rir_pool=None, augment=False
    )
    train_segments = train_drone_segments + train_no_segments
    train_labels = train_drone_labels + train_no_labels
    combined = list(zip(train_segments, train_labels))
    random.shuffle(combined)
    train_segments, train_labels = zip(*combined)
    # Load validation data without augmentation
    val_drone_segments, val_drone_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "drone"), label=1, pre_classify=True,
        noise_pool=None, rir_pool=None, augment=False
    )
    val_no_segments, val_no_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "no drone"), label=0, pre_classify=False,
        noise_pool=None, rir_pool=None, augment=False
    )
    val_segments = val_drone_segments + val_no_segments
    val_labels = val_drone_labels + val_no_labels
    # Datasets and loaders
    train_dataset = DroneDataset(list(train_segments), list(train_labels))
    val_dataset = DroneDataset(val_segments, val_labels)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().to(device)
                outputs = model(X_batch)
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")
    # Final evaluation on the pre‑classified validation set
    # Collect labels and probabilities for the harmonically filtered validation
    val_labels_array, val_probs_array = evaluate_loader(model, val_loader, device)
    # Plot confusion matrix and probability distribution for the pre‑classified validation set
    plot_confusion(val_labels_array, val_probs_array, suffix="pre_classified")
    plot_probability_distribution(val_labels_array, val_probs_array, suffix="pre_classified")
    # ---------------------------------------------------------------------
    # Evaluation on the full validation set (without harmonic filtering)
    # Reload drone segments with pre_classify=False to include all segments
    full_drone_segments, full_drone_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "drone"), label=1, pre_classify=False,
        noise_pool=None, rir_pool=None, augment=False
    )
    full_no_segments, full_no_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "no drone"), label=0, pre_classify=False,
        noise_pool=None, rir_pool=None, augment=False
    )
    full_segments = full_drone_segments + full_no_segments
    full_labels = full_drone_labels + full_no_labels
    full_dataset = DroneDataset(full_segments, full_labels)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    full_labels_array, full_probs_array = evaluate_loader(model, full_loader, device)
    plot_confusion(full_labels_array, full_probs_array, suffix="full")
    plot_probability_distribution(full_labels_array, full_probs_array, suffix="full")
    # ---------------------------------------------------------------------
    # Convert the trained model to TFLite.  The dummy input shape is
    # (batch_size=1, channels=1, n_mels, time_steps).  The Mel dimension
    # (n_mels) is fixed at 64; the time dimension depends on SEG_LEN and the
    # hop length used in the log‑Mel extraction.  Here we approximate
    # time_steps by passing a dummy tensor through the model during
    # instantiation (see ImprovedCNN.__init__).  We reuse that value as
    # model.flatten_dim to infer the feature map size.
    # Calculate the input shape for a single example:
    time_steps = int(np.ceil(SEG_LEN / 512))  # hop length 512 matches extract_logmel
    export_to_tflite(model, device, input_shape=(1, 1, 64, time_steps), filename="trained_model.tflite")
    # Return paths to the confusion matrix of the full dataset and the histogram
    return


if __name__ == "__main__":
    train_model()