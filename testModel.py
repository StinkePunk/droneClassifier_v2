import os
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

MODEL_PATH = r"model_set0_seed7.onnx"
DATA_ROOT = r"d:\Dropbox\03 H2 Think\AuDroK mFund\Auswertungen\Datensätze\Drone vs. No Drone"

SAMPLE_RATE = 16000
SEG_LEN_SEC = 1
SEG_LEN = SAMPLE_RATE * SEG_LEN_SEC

# Diese Parameter müssen zum trainierten Modell passen.
N_MELS = 128
HOP_LENGTH = 256
IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

VALIDATION_DRONE_DIR = os.path.join(DATA_ROOT, "VALIDATION", "drone")
VALIDATION_NO_DRONE_DIR = os.path.join(DATA_ROOT, "VALIDATION", "no drone")


# =========================
# PREPROCESSING
# =========================

def compute_logmel_for_plot(segment: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel


def preprocess_segment(segment: np.ndarray) -> np.ndarray:
    """
    Erzeugt genau den Model-Input:
    [1, 3, IMG_SIZE, IMG_SIZE]
    """
    log_mel = compute_logmel_for_plot(segment)

    x = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    x = np.expand_dims(x, axis=0)              # [1, H, W]
    x = np.repeat(x, 3, axis=0)                # [3, H, W]

    # Resize auf 224x224
    # librosa/np haben kein einfaches bilinear für 3D, daher pro Kanal separat
    resized = []
    for c in range(3):
        chan = x[c]
        chan_resized = librosa.util.fix_length(chan, size=chan.shape[1], axis=1)
        # Resize mit scipy wäre eleganter; hier nutzen wir librosa + matplotlib-freie Lösung über np.interp
        # zuerst Zeilen, dann Spalten
        y_old = np.linspace(0, 1, chan.shape[0], dtype=np.float32)
        y_new = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)
        tmp = np.empty((IMG_SIZE, chan.shape[1]), dtype=np.float32)
        for j in range(chan.shape[1]):
            tmp[:, j] = np.interp(y_new, y_old, chan[:, j])

        x_old = np.linspace(0, 1, chan.shape[1], dtype=np.float32)
        x_new = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)
        out = np.empty((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i in range(tmp.shape[0]):
            out[i, :] = np.interp(x_new, x_old, tmp[i, :])

        resized.append(out)

    x = np.stack(resized, axis=0).astype(np.float32)   # [3, 224, 224]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.expand_dims(x, axis=0).astype(np.float32)   # [1, 3, 224, 224]
    return x


# =========================
# DATA LOADING
# =========================

def iter_validation_segments(folder: str, label: int):
    """
    Liefert:
    {
        'segment': np.ndarray,
        'label': int,
        'file_path': str,
        'file_name': str,
        'segment_index': int
    }
    """
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        seg_idx = 0
        for start in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
            seg = y[start:start + SEG_LEN]
            if len(seg) < SEG_LEN:
                continue
            seg_idx += 1
            seg_idx = 0
            total_segments = max(0, (len(y) - SEG_LEN) // SEG_LEN + 1)
            total_duration_sec = len(y) / SAMPLE_RATE

            for start in range(0, len(y) - SEG_LEN + 1, SEG_LEN):
                seg = y[start:start + SEG_LEN]
                if len(seg) < SEG_LEN:
                    continue
                seg_idx += 1

                start_sec = start / SAMPLE_RATE
                end_sec = (start + SEG_LEN) / SAMPLE_RATE
                center_ratio = ((start + SEG_LEN / 2) / len(y)) if len(y) > 0 else 0.0

                if center_ratio < 1/3:
                    position_label = "Anfang"
                elif center_ratio < 2/3:
                    position_label = "Mitte"
                else:
                    position_label = "Ende"

                yield {
                    "segment": seg,
                    "label": label,
                    "file_path": path,
                    "file_name": fname,
                    "segment_index": seg_idx,
                    "total_segments": total_segments,
                    "start_sample": start,
                    "end_sample": start + SEG_LEN,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "total_duration_sec": total_duration_sec,
                    "position_label": position_label,
                    "center_ratio": center_ratio,
                }


# =========================
# ONNX INFERENCE
# =========================

def create_session(model_path: str):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_model(sess, input_name: str, output_name: str, x: np.ndarray) -> float:
    out = sess.run([output_name], {input_name: x})[0]
    out = np.asarray(out).reshape(-1)[0]

    # Falls das ONNX schon Sigmoid enthält, liegt out oft bereits in [0,1].
    # Falls nicht, wenden wir Sigmoid an.
    if 0.0 <= out <= 1.0:
        prob = float(out)
    else:
        prob = float(sigmoid(out))
    return prob


# =========================
# DEBUG DISPLAY
# =========================

def show_case(case_type: str, item: dict, prob: float):
    log_mel = compute_logmel_for_plot(item["segment"])

    print("\n" + "=" * 80)
    print(f"{case_type}")
    print(f"Datei          : {item['file_name']}")
    print(f"Pfad           : {item['file_path']}")
    print(f"Segmentindex   : {item['segment_index']} / {item['total_segments']}")
    print(f"Position       : {item['position_label']} ({item['center_ratio']*100:.1f}% der Dateilänge)")
    print(f"Zeitfenster    : {item['start_sec']:.2f}s bis {item['end_sec']:.2f}s "
          f"von insgesamt {item['total_duration_sec']:.2f}s")
    print(f"Samples        : {item['start_sample']} bis {item['end_sample']}")
    print(f"True label     : {item['label']} ({'Drone' if item['label'] == 1 else 'No Drone'})")
    print(f"Pred prob      : {prob:.4f}")
    print("=" * 80)

    plt.figure(figsize=(10, 5))
    plt.imshow(log_mel, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(f"{case_type} | {item['file_name']} | seg={item['segment_index']} | p={prob:.4f}")
    plt.xlabel("Frames")
    plt.ylabel("Mel bin")
    plt.tight_layout()
    plt.show(block=False)

    input("Enter drücken für nächstes Fehlsegment ... ")
    plt.close("all")


# =========================
# MAIN
# =========================

def main():
    sess, input_name, output_name = create_session(MODEL_PATH)

    all_items = []
    all_items.extend(iter_validation_segments(VALIDATION_DRONE_DIR, label=1))
    all_items.extend(iter_validation_segments(VALIDATION_NO_DRONE_DIR, label=0))

    fp = 0
    fn = 0
    total = 0

    for item in all_items:
        total += 1
        x = preprocess_segment(item["segment"])
        prob = run_model(sess, input_name, output_name, x)
        pred = 1 if prob >= 0.5 else 0
        true = item["label"]

        if true == 0 and pred == 1:
            fp += 1
            show_case("FALSE POSITIVE", item, prob)

        elif true == 1 and pred == 0:
            fn += 1
            show_case("FALSE NEGATIVE", item, prob)

    print("\nFertig.")
    print(f"Gesamtsegmente : {total}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")


if __name__ == "__main__":
    main()