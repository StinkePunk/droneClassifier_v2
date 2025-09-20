import numpy as np
import random
from scipy import signal
import glob, os
from tqdm import tqdm  # Fortschrittsbalken-Bibliothek
import time

random.seed(42)

def simulate_ground_reflection(signal, sample_rate, src_pos, mic_pos, attenuation_range=(0.01, 1.0), speed_of_sound=343):
    """
    Simuliert eine Bodenreflexion basierend auf Drohnen- und Mikrofonposition.

    Parameters:
        signal (np.array): Originalsignal (1D).
        sample_rate (int): Abtastrate in Hz.
        src_pos (tuple): Position der Drohne (x, y, z).
        mic_pos (tuple): Position des Mikrofons (x, y, z).
        attenuation_range (tuple): Reflexionsdämpfung (z. B. (0.01, 1.0)).
        speed_of_sound (float): Schallgeschwindigkeit in m/s.

    Returns:
        np.array: Signal mit reflektiertem Anteil.
    """
    # Spiegelung der Quelle an Boden (z = -z)
    src_reflected = (src_pos[0], src_pos[1], -src_pos[2])

    # Distanz berechnen
    direct_distance = np.linalg.norm(np.array(src_pos) - np.array(mic_pos))
    reflected_distance = np.linalg.norm(np.array(src_reflected) - np.array(mic_pos))

    # Laufzeitdifferenz berechnen (in Sekunden → in Samples)
    delay_time = (reflected_distance - direct_distance) / speed_of_sound
    delay_samples = int(np.round(delay_time * sample_rate))

    # Dämpfung zufällig wählen
    attenuation = np.random.uniform(*attenuation_range)

    # Reflektiertes Signal verschieben
    reflected = np.zeros_like(signal)
    if delay_samples < len(signal):
        reflected[delay_samples:] = signal[:-delay_samples] * attenuation

    return signal + reflected

def apply_ground_reflection_to_dataset(dataset, sample_rate, ranges=None):
    """
    Applies ground reflection augmentation to all AudioData objects in a dataset.
    Each signal gets randomized parameters within the given range.
    """
    from audio_data import AudioData, AudioDataSet  # vermeiden von zirkulären Importen

    # Standardbereiche
    default_ranges = {
        "src_x": (0, 10),
        "src_y": (0, 10),
        "src_z": (1, 5),
        "mic_z": (1, 2),
        "attenuation": (0.01, 1.0),
    }
    if ranges:
        default_ranges.update(ranges)

    augmented = []
    for audio in dataset.audio:
        src_pos = (
            np.random.uniform(*default_ranges["src_x"]),
            np.random.uniform(*default_ranges["src_y"]),
            np.random.uniform(*default_ranges["src_z"])
        )
        mic_pos = (0.0, 0.0, np.random.uniform(*default_ranges["mic_z"]))
        attenuation_range = default_ranges["attenuation"]

        new_signal = simulate_ground_reflection(
            signal=audio.signal,
            sample_rate=sample_rate,
            src_pos=src_pos,
            mic_pos=mic_pos,
            attenuation_range=attenuation_range
        )

        augmented.append(AudioData(
            signal=new_signal,
            sample_rate=audio.sample_rate,
            label=audio.label,
            source_file=audio.source_file
        ))

    return AudioDataSet(augmented)


def add_background_noise(signal, noise, max_snr_db):
    """
    Mischt ein Hintergrundgeräusch unter Berücksichtigung eines maximalen SNR-Werts zum Signal.

    Parameter:
    - signal: Drohnensignal (1D np.array)
    - noise: Hintergrundgeräusch (1D np.array, gleiche Länge wie signal)
    - max_snr_db: maximaler SNR in dB (z. B. -6 bedeutet SNR zwischen -∞ und -6 dB)

    Rückgabe:
    - gemischtes Signal mit Hintergrundgeräusch
    """
    if len(noise) != len(signal):
        noise = noise[:len(signal)] if len(noise) > len(signal) else np.pad(noise, (0, len(signal) - len(noise)))

    signal_rms = np.sqrt(np.mean(signal**2))
    noise_rms = np.sqrt(np.mean(noise**2))

    if noise_rms == 0 or signal_rms == 0:
        return signal  # nichts mischen, falls eins still ist

    max_snr_linear = 10 ** (max_snr_db / 20)
    actual_snr = np.random.uniform(0, max_snr_linear)
    desired_noise_rms = signal_rms * actual_snr
    scaled_noise = noise * (desired_noise_rms / noise_rms)

    mixed = signal + scaled_noise

    # Clipping vermeiden: auf maximale Amplitude normieren
    max_orig = np.max(np.abs(signal))
    max_new = np.max(np.abs(mixed))
    if max_new > max_orig:
        mixed = mixed * (max_orig / max_new)

    return mixed

def apply_noise_augmentation(drone_dataset, noise_dataset, sample_rate, max_snr_db=-6):
    """
    Augmentiert alle Drohnensignale mit Zufallsgeräuschen aus NoDrone.

    Parameter:
    - drone_dataset: AudioDataSet mit Drohnen-Signalen
    - noise_dataset: AudioDataSet mit Hintergrundgeräuschen (z. B. NoDrone)
    - sample_rate: Abtastrate
    - max_snr_db: maximaler SNR-Wert in dB (negativer Wert)

    Rückgabe:
    - AudioDataSet mit augmentierten Signalen
    """
    from audio_data import AudioData, AudioDataSet
    import random

    augmented = []
    noise_signals = [n.signal for n in noise_dataset.audio]

    for drone_audio in tqdm(drone_dataset.audio, desc="Adding Background Noise"):
        noise_signal = random.choice(noise_signals)
        mixed_signal = add_background_noise(drone_audio.signal, noise_signal, max_snr_db)

        augmented.append(AudioData(
            signal=mixed_signal,
            sample_rate=drone_audio.sample_rate,
            label=drone_audio.label,
            source_file=drone_audio.source_file
        ))

    return AudioDataSet(augmented)


def add_echo(chunk, sample_rate, delay_ms=20, echo_amplitude=0.1):
    """
    Fügt den Audiodaten ein Echo hinzu.

    Parameter:
    - chunk: Eingabedaten (z. B. 1D-Array für eine Audiodatei).
    - sample_rate: Abtastrate der Audiodaten (in Hz).
    - delay_ms: Verzögerung des Echos (in Millisekunden).
    - echo_amplitude: Amplitude des Echos (relativ zur Originaldatenamplitude).

    Rückgabe:
    - Augmentierte Audiodaten mit Echo.
    """
    delay_samples = int(sample_rate * delay_ms / 1000)  # Verzögerung in Samples
    echo = np.zeros_like(chunk)
    if delay_samples < len(chunk):  # Sicherstellen, dass Delay nicht größer als die Länge des Chunks ist
        echo[delay_samples:] = chunk[:-delay_samples] * echo_amplitude
    return chunk + echo

def _mix_noise(signal, noise, snr_db):
    """Mixes noise at target SNR (dB)."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal)/len(noise)))
        noise = np.tile(noise, reps)[:len(signal)]
    else:
        start = np.random.randint(0, len(noise)-len(signal)+1)
        noise = noise[start:start+len(signal)]
    ps = np.mean(signal**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    target_pn = ps / (10**(snr_db/10.0))
    noise = noise * np.sqrt(target_pn/pn)
    y = signal + noise
    return y / (np.max(np.abs(y)) + 1e-9)


def apply_augmentations(chunk, sample_rate, noise_pool=None, snr_range=(0, 20), p_noise=0.9):
    """
    Wendet nur das Hinzufügen von Echos auf die Audiodaten an.
    Augmentiert Chunks: optional Noise-Mix (real-world), immer leichtes Echo.

    Parameter:
    - chunk: Eingabedaten (z. B. 1D-Array für eine Audiodatei).
    - sample_rate: Abtastrate der Audiodaten (in Hz).

    Rückgabe:
    - Augmentierte Audiodaten (nur mit Echo).
    """

    chunk = np.array(chunk, dtype=np.float32)  # Liste in NumPy-Array umwandeln
    num_signals, signal_length = chunk.shape  # Anzahl der Signale und Länge der Zeitreihe
    augmented_chunk = np.zeros_like(chunk, dtype=np.float32)  # Speicher für augmentierte Daten

    for i in tqdm(range(num_signals), desc="Augmenting Audio", unit="signals"):
        sig = chunk[i]
        # 1) optionaler Noise-Mix (macht RAR realistischer)
        if noise_pool and random.random() < p_noise:
            noise = random.choice(noise_pool)
            snr = random.uniform(*snr_range)
            sig = _mix_noise(sig, noise, snr_db=snr)
        # 2) mildes Echo (Reflexionsanmutung)
        delay_ms = random.randint(5, 30)
        echo_amplitude = random.uniform(0.02, 0.2)
        augmented_chunk[i] = add_echo(sig, sample_rate, delay_ms, echo_amplitude)
    return augmented_chunk if np.any(augmented_chunk) else chunk

def _load_rirs(rir_dir):
    paths = glob.glob(os.path.join(rir_dir, "*.wav"))
    return [librosa.load(p, sr=None)[0] for p in paths] if paths else []

def apply_rir(x, rir):
    if rir is None or len(rir) == 0:
        return x
    y = signal.fftconvolve(x, rir, mode="full")[: len(x)]
    m = np.max(np.abs(y)) + 1e-9
    return (y / m).astype(x.dtype)

def mix_noise(x, noise, snr_db):
    if noise is None or len(noise) == 0:
        return x
    if len(noise) < len(x):
        reps = int(np.ceil(len(x)/len(noise)))
        noise = np.tile(noise, reps)[:len(x)]
    else:
        start = np.random.randint(0, len(noise) - len(x) + 1)
        noise = noise[start:start+len(x)]
    # SNR einstellen
    px = np.mean(x**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    target_pn = px / (10**(snr_db/10.0))
    noise = noise * np.sqrt(target_pn / pn)
    y = x + noise
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

class RealWorldAug:
    def __init__(self, rir_dir=None, noise_list=None, snr_range=(0, 20)):
        self.rirs = _load_rirs(rir_dir) if rir_dir else []
        self.noises = noise_list or []
        self.snr_range = snr_range

    def __call__(self, chunk, sr):
        y = chunk.copy()
        # 1) RIR-Faltung (mit p=0.7)
        if self.rirs and random() < 0.7:
            rir = np.random.choice(self.rirs)
            y = apply_rir(y, rir)
        # 2) Noisemix (mit p=0.9)
        if self.noises and random() < 0.9:
            noise = np.random.choice(self.noises)
            snr = np.random.uniform(*self.snr_range)  # 0–20 dB
            y = mix_noise(y, noise, snr_db=snr)
        return y

def augment_audio_data(audio_files, sample_rate):
    """
    Führt die Augmentierung (nur Echo) auf eine Liste von Audiodateien durch.

    Parameter:
    - audio_files: Liste von Audiodateien (z. B. 2D-Array mit mehreren Chunks).
    - sample_rate: Abtastrate der Audiodateien (in Hz).

    Rückgabe:
    - Liste augmentierter Audiodateien.
    """
    augmented_files = []
    total_files = len(audio_files)  # Anzahl der Dateien einmalig ermitteln
    for i, audio in enumerate(audio_files, start=1):
        print(f"Processing Augmentation Audio {i} of {total_files}")
        # Sicherstellen, dass das Audioformat korrekt ist
        audio = np.array(audio, dtype=np.float32)
        # Augmentierung anwenden
        augmented_audio = apply_augmentations(audio, sample_rate)
        augmented_files.append(augmented_audio)
    return augmented_files
