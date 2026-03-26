import queue
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import sounddevice as sd


# =========================
# Konfiguration
# =========================
MODEL_PATH = "model.onnx"   # Pfad zu deinem ONNX-Modell
SAMPLE_RATE = 16000
CHUNK_SEC = 1
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC

# äquivalent zum Training
N_MELS = 64
HOP_LENGTH = 512

# Schwelle für binäre Klassifikation
THRESHOLD = 0.5


# =========================
# Feature-Extraktion
# =========================
def extract_logmel(segment: np.ndarray) -> np.ndarray:
    """
    Reproduziert die Trainings-Features:
    librosa.feature.melspectrogram(..., n_mels=64, hop_length=512)
    -> power_to_db(..., ref=np.max)
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel


def prepare_input_for_onnx(log_mel: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """
    Formt das Feature passend für ONNX.
    Erwartet i.d.R. [1, 1, 64, T].
    Falls T statisch ist, wird gepadded/gecroppt.
    """
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape  # z.B. [1, 1, 64, 32] oder [None, 1, 64, None]

    x = log_mel[np.newaxis, np.newaxis, :, :]  # [1, 1, 64, T]
    x = x.astype(np.float32)

    # Falls Zeitachse statisch vorgegeben ist
    if len(input_shape) == 4 and isinstance(input_shape[3], int):
        target_t = input_shape[3]
        current_t = x.shape[3]

        if current_t < target_t:
            pad_width = target_t - current_t
            x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode="constant")
        elif current_t > target_t:
            x = x[:, :, :, :target_t]

    return x


# =========================
# Audioaufnahme
# =========================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio-Status: {status}", file=sys.stderr)
    # mono
    audio_queue.put(indata[:, 0].copy())


def collect_exact_chunk(num_samples: int) -> np.ndarray:
    """
    Sammelt exakt num_samples Samples aus dem Callback-Stream.
    """
    collected = []
    total = 0

    while total < num_samples:
        block = audio_queue.get()
        collected.append(block)
        total += len(block)

    chunk = np.concatenate(collected, axis=0)

    if len(chunk) > num_samples:
        chunk = chunk[:num_samples]

    return chunk.astype(np.float32)


# =========================
# ONNX-Inferenz
# =========================
def build_session(model_path: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def predict_chunk(session: ort.InferenceSession, chunk: np.ndarray) -> tuple[int, float]:
    # Optional: Pegelbegrenzung wie oft bei Audio üblich
    max_abs = np.max(np.abs(chunk))
    if max_abs > 1e-6:
        chunk = chunk / max_abs

    feat = extract_logmel(chunk)
    x = prepare_input_for_onnx(feat, session)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x})[0]

    # robust gegen [1], [1,1], scalar-artige Ausgaben
    score = float(np.asarray(output).reshape(-1)[0])
    pred = int(score >= THRESHOLD)

    return pred, score


# =========================
# Hauptprogramm
# =========================
def main():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"ONNX-Modell nicht gefunden: {model_file.resolve()}")

    session = build_session(str(model_file))

    print("Modell geladen.")
    print("Starte Mikrofonaufnahme ...")
    print("Abbruch mit Ctrl+C.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=4096,
        ):
            while True:
                t0 = time.time()
                chunk = collect_exact_chunk(CHUNK_SAMPLES)
                pred, score = predict_chunk(session, chunk)

                if pred == 1:
                    print(f"Drohne (1) | Score: {score:.4f}")
                else:
                    print(f"Keine Drohne (0) | Score: {score:.4f}")

                # nur zur sauberen 1s-Taktung in der Ausgabe
                elapsed = time.time() - t0
                if elapsed < 0.01:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    main()