"""
Train a drone vs. no‑drone classifier using real audio data with
pre‑classification based on harmonic content and heavy data augmentation.

This script expects the mini dataset downloaded from Zenodo to be placed under
``/home/oai/share/data_mini/Drone vs. no Drone Mini``. It first loads all
recordings, segments them into 3‑second chunks and filters drone chunks
without a clear harmonic signature. Augmentations such as synthetic room
impulse responses, floor reflections (echo), background noise mixing,
pitch‑shifting and time‑stretching are applied to the drone segments to
simulate diverse real‑world conditions. A simple convolutional neural
network is then trained on log‑Mel spectrograms extracted from each
segment. After training, the model is evaluated on the validation set
(also pre‑classified to discard silent drone segments) and confusion
matrix / probability distributions are saved as PNG images.

Run this script with Python 3 and ensure that ``librosa``, ``numpy`` and
``torch`` are available in the environment.
"""

import os
import sys
import types
# Disable CAAS/Jupyter fallback logging which may attempt to contact
# an external server when saving figures. Setting this environment
# variable prevents matplotlib fallback logging from initiating
# HTTP requests in environments where a GUI backend is not
# available (e.g., CAAS sandbox).  See internal CAAS docs.
os.environ['CAAS_JUPYTER_TOOL_DISABLE_LOGGING'] = '1'
# Stub out the caas_jupyter_tools module to prevent matplotlib from
# attempting to contact a local Jupyter server when saving images.  In
# the CAAS environment, matplotlib's monkeypatch triggers a fallback
# logger that tries to POST data to localhost:8080.  By inserting a
# dummy module with the required functions, we avoid these network
# calls entirely.
sys.modules['caas_jupyter_tools'] = types.SimpleNamespace(
    log_matplotlib_img_fallback=lambda *args, **kwargs: None,
    log_exception=lambda *args, **kwargs: None
)
import random
from typing import List, Tuple

import numpy as np
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
# Use non‑interactive backend to avoid attempts to contact a GUI or external server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dataset paths
DATA_ROOT = "d:/Dropbox/03 H2 Think/AuDroK mFund/Auswertungen/Datensätze/Drone vs. No Drone/"

# Audio parameters
SAMPLE_RATE = 16000
SEG_LEN_SEC = 3
SEG_LEN = SEG_LEN_SEC * SAMPLE_RATE

# Pre‑classification threshold for drone harmonic content. Drone segments
# with harmonic ratios below this threshold will be discarded.  The value 0.7
# was chosen based on exploring harmonic ratios of training and validation
# segments: about 14 % of training chunks and 11 % of validation chunks
# fall below this threshold, which likely correspond to silent or far‑distant
# drone recordings.  Raising the threshold beyond 0.8 would discard
# over 90 % of validation chunks, leading to underfitting.  Feel free to
# adjust this constant if you analyse other datasets.
HARM_THRESHOLD = 0.75

# Augmentation probabilities
P_APPLY_RIR = 0.7
P_APPLY_ECHO = 0.5
P_APPLY_NOISE = 0.8
P_PITCH_SHIFT = 0.5
P_TIME_STRETCH = 0.5

def compute_harmonic_ratio(signal: np.ndarray) -> float:
    """Compute the harmonic ratio of an audio signal using HPSS.

    Args:
        signal: 1‑D numpy array of audio samples.

    Returns:
        Ratio of harmonic energy to total energy.
    """
    harm, perc = librosa.effects.hpss(signal)
    harm_energy = np.sum(np.abs(harm))
    perc_energy = np.sum(np.abs(perc))
    return float(harm_energy / (harm_energy + perc_energy + 1e-9))

def generate_random_rir(length: int = 4096) -> np.ndarray:
    """Generate a synthetic room impulse response.

    A simple exponentially decaying noise is used to simulate a random
    reverberation. The length of the RIR is drawn uniformly between 512
    samples and ``length``.
    """
    rir_len = random.randint(512, length)
    rir = np.random.randn(rir_len).astype(np.float32)
    decay = np.exp(-np.linspace(0, 5, rir_len))
    rir *= decay
    rir /= np.maximum(np.abs(rir).max(), 1e-6)
    return rir

def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
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

def augment_signal(signal: np.ndarray, noise_pool: List[np.ndarray]) -> np.ndarray:
    augmented = signal.copy()
    if random.random() < P_APPLY_RIR:
        rir = generate_random_rir()
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

def load_noise_pool(no_folder: str) -> List[np.ndarray]:
    pool = []
    for fname in os.listdir(no_folder):
        path = os.path.join(no_folder, fname)
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        for i in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
            seg = y[i : i + SEG_LEN]
            if len(seg) == SEG_LEN:
                pool.append(seg)
    return pool

def load_segments(folder: str, label: int, pre_classify: bool, noise_pool: List[np.ndarray] = None,
                  augment: bool = False, num_augment: int = 2) -> Tuple[List[np.ndarray], List[int]]:
    segments: List[np.ndarray] = []
    labels: List[int] = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
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
            if label == 1 and augment and noise_pool is not None:
                for _ in range(num_augment):
                    aug = augment_signal(seg, noise_pool)
                    segments.append(aug)
                    labels.append(label)
    return segments, labels

def extract_logmel(segment: np.ndarray, n_mels: int = 64, hop_length: int = 512) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_mels=n_mels, hop_length=hop_length)
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

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 94)
            out = self.pool(self.bn1(torch.relu(self.conv1(dummy))))
            out = self.pool(self.bn2(torch.relu(self.conv2(out))))
            out = self.pool(self.bn3(torch.relu(self.conv3(out))))
            self.flatten_dim = out.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(-1)

def train_model():
    noise_pool = load_noise_pool(os.path.join(DATA_ROOT, "TRAINING", "no drone"))
    # Load and augment training data
    train_drone_segments, train_drone_labels = load_segments(
        os.path.join(DATA_ROOT, "TRAINING", "drone"), label=1, pre_classify=True,
        noise_pool=noise_pool, augment=True, num_augment=1
    )
    train_no_segments, train_no_labels = load_segments(
        os.path.join(DATA_ROOT, "TRAINING", "no drone"), label=0, pre_classify=False,
        noise_pool=None, augment=False
    )
    train_segments = train_drone_segments + train_no_segments
    train_labels = train_drone_labels + train_no_labels
    # Shuffle training data
    combined = list(zip(train_segments, train_labels))
    random.shuffle(combined)
    train_segments, train_labels = zip(*combined)
    # Load validation data (no augmentation)
    val_drone_segments, val_drone_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "drone"), label=1, pre_classify=True,
        noise_pool=None, augment=False
    )
    val_no_segments, val_no_labels = load_segments(
        os.path.join(DATA_ROOT, "VALIDATION", "no drone"), label=0, pre_classify=False,
        noise_pool=None, augment=False
    )
    val_segments = val_drone_segments + val_no_segments
    val_labels = val_drone_labels + val_no_labels
    # Create dataset and dataloader
    train_dataset = DroneDataset(list(train_segments), list(train_labels))
    val_dataset = DroneDataset(val_segments, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
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
        # Evaluate on validation
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
    all_probs = []
    all_labels = []
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
    # Save confusion matrix plot
    fig, ax = plt.subplots()
    im = ax.imshow(conf, cmap="Blues")
    ax.set_title("Confusion Matrix (CNN)")
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
    conf_path = "pytorch_confusion_matrix.png"
    fig.savefig(conf_path)
    plt.close(fig)
    # Save probability distribution plot
    fig2, ax2 = plt.subplots()
    drone_probs = [p for p, lab in zip(all_probs, all_labels) if lab == 1]
    no_probs = [p for p, lab in zip(all_probs, all_labels) if lab == 0]
    ax2.hist(drone_probs, bins=20, alpha=0.6, label="Drone")
    ax2.hist(no_probs, bins=20, alpha=0.6, label="No Drone")
    ax2.set_title("Probability Distribution (CNN)")
    ax2.set_xlabel("Predicted probability for class 'Drone'")
    ax2.set_ylabel("Count")
    ax2.legend()
    dist_path = "pytorch_probability_distribution.png"
    fig2.savefig(dist_path)
    plt.close(fig2)
    return conf_path, dist_path

if __name__ == "__main__":
    train_model()