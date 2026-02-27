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
SEG_LEN_SEC = 2  # 2‑second segments
SEG_LEN = SEG_LEN_SEC * SAMPLE_RATE

# Pre‑classification threshold: discard drone segments with low harmonic
# content.  The threshold 0.7 was chosen empirically.
HARM_THRESHOLD = 0.7

# Augmentation probabilities
P_APPLY_RIR = 0.7
P_APPLY_ECHO = 0.5
P_APPLY_NOISE = 0.8
P_PITCH_SHIFT = 0.5
P_TIME_STRETCH = 0.5


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


def add_echo(signal: np.ndarray, delay_ms: int, amplitude: float) -> np.ndarray:
    delay_samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
    echo_signal = np.zeros_like(signal)
    if delay_samples < len(signal):
        echo_signal[delay_samples:] = signal[:-delay_samples]
    return signal + amplitude * echo_signal


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
        for i in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
            seg = y[i : i + SEG_LEN]
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
        augmented = add_echo(augmented, delay_ms=delay, amplitude=amp)
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
    # Final evaluation
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = model(X_batch).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    conf = confusion_matrix(all_labels, preds)
    acc_final = (conf[0][0] + conf[1][1]) / conf.sum()
    print("Final Validation Accuracy:", acc_final)
    print("Confusion Matrix:\n", conf)
    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.imshow(conf, cmap="Blues")
    ax.set_title("Confusion Matrix (Improved CNN)")
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
    conf_path = "improved_confusion_matrix.png"
    fig.savefig(conf_path)
    plt.close(fig)
    # Plot probability distribution
    fig2, ax2 = plt.subplots()
    drone_probs = [p for p, lab in zip(all_probs, all_labels) if lab == 1]
    no_probs = [p for p, lab in zip(all_probs, all_labels) if lab == 0]
    ax2.hist(drone_probs, bins=20, alpha=0.6, label="Drone")
    ax2.hist(no_probs, bins=20, alpha=0.6, label="No Drone")
    ax2.set_title("Probability Distribution (Improved CNN)")
    ax2.set_xlabel("Predicted probability for class 'Drone'")
    ax2.set_ylabel("Count")
    ax2.legend()
    dist_path = "improved_probability_distribution.png"
    fig2.savefig(dist_path)
    plt.close(fig2)
    return conf_path, dist_path


if __name__ == "__main__":
    train_model()