# Standardbibliotheken (in Python bereits eingebaut, kein Install nötig)
import collections   # für deque: eine Liste mit maximaler Länge, alte Einträge fallen automatisch raus
import queue         # für eine Thread-sichere Warteschlange zwischen Audioaufnahme und Klassifikation
import sys           # für Fehlerausgabe auf stderr
import time          # für Zeitstempel und Pausen
from pathlib import Path  # komfortabler Umgang mit Dateipfaden

# Externe Pakete (müssen per pip installiert sein)
import librosa       # Audioverarbeitung: Mel-Spektrogramm berechnen
import numpy as np   # numerische Arrays (Herzstück für Audio- und Bilddaten)
import onnxruntime as ort  # lädt und führt ONNX-Modelle aus (plattformunabhängiges KI-Format)
import sounddevice as sd   # Mikrofon aufnehmen und Töne abspielen


# ==============================================================
# Konfiguration – hier kannst du die wichtigsten Parameter ändern
# ==============================================================

MODEL_PATH = "model_set0_seed1.onnx"   # Dateiname des ONNX-Modells (muss im selben Ordner liegen)
DEVICE = None       # Mikrofon-Auswahl: None = Windows-Standardmikrofon, oder z.B. 1 für Gerät Nr. 1
SAMPLE_RATE = 16000 # Abtastrate in Hz – muss identisch mit dem Training sein (16.000 Samples pro Sekunde)
CHUNK_SEC = 1       # Länge eines Auswertungsfensters in Sekunden
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC  # daraus ergibt sich die Anzahl Samples pro Fenster (hier: 16.000)

# Mel-Spektrogramm-Parameter – müssen exakt mit dem Training übereinstimmen
N_MELS = 64         # Anzahl der Mel-Frequenzbänder (Zeilen im Spektrogramm)
HOP_LENGTH = 512    # Schrittweite in Samples zwischen zwei Spektrogramm-Spalten

# Entscheidungsschwelle: Score >= 0.5 → Drohne erkannt
THRESHOLD = 0.5


# ==============================================================
# Lautstärken für die fünf Alarmstufen (0 = kein Ton, 1.0 = max)
# Reihenfolge: [Stufe0, Stufe1, Stufe2, Stufe3, Stufe4]
# ==============================================================
ALARM_VOLUMES = [0.0, 0.15, 0.35, 0.60, 1.0]


# ==============================================================
# Feature-Extraktion: Audio → Log-Mel-Spektrogramm
# ==============================================================
def extract_logmel(segment: np.ndarray) -> np.ndarray:
    """
    Wandelt einen Audio-Abschnitt (als numpy-Array mit Floats) in ein
    Log-Mel-Spektrogramm um – das ist die Darstellung, mit der das CNN trainiert wurde.

    Ein Mel-Spektrogramm zeigt, wie stark welche Frequenzen zu welchem Zeitpunkt
    vorhanden sind – ähnlich wie ein Musiknoten-Bild, aber für Maschinen optimiert.
    Die logarithmische Skalierung (power_to_db) entspricht der menschlichen Lautstärkewahrnehmung.

    Rückgabe: 2D-Array der Form [N_MELS, Zeitschritte], also z.B. [64, 32]
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    # Umrechnung von linearer Leistung in Dezibel (logarithmisch), relativ zum Maximum
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel


# ==============================================================
# Eingabe vorbereiten: Spektrogramm in die Form bringen, die das Modell erwartet
# ==============================================================
def prepare_input_for_onnx(log_mel: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """
    Das ONNX-Modell erwartet eine ganz bestimmte Array-Form (Shape).
    Diese Funktion fragt das Modell, welche Form es erwartet, und passt
    das Spektrogramm entsprechend an.

    Es gibt zwei unterstützte Modelltypen:
    - Eigenes CNN: erwartet [1, 1, 64, T]  (1 Bild, 1 Kanal, 64 Mel-Bänder, T Zeitschritte)
    - Vortrainiertes Bildmodell (z.B. ResNet): erwartet [1, 3, 224, 224]  (1 Bild, 3 Farbkanäle, 224×224 Pixel)
    """
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape  # liest die erwartete Form direkt aus dem Modell aus

    # --- Fall 1: Bildklassifikationsmodell (3 Kanäle wie RGB) ---
    if len(input_shape) == 4 and input_shape[1] == 3:
        # Zielgröße aus dem Modell auslesen (meist 224×224)
        h = input_shape[2] if isinstance(input_shape[2], int) else 224
        w = input_shape[3] if isinstance(input_shape[3], int) else 224

        # Spektrogramm auf den Wertebereich [0, 1] normalisieren
        img = log_mel - log_mel.min()
        if img.max() > 0:
            img = img / img.max()
        img = img.astype(np.float32)

        # Größe auf 224×224 skalieren (bilineare Interpolation wie beim Zoomen eines Bildes)
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

        # ImageNet-Normalisierung: diese Mittelwerte und Standardabweichungen wurden beim
        # Vortrainieren des Modells auf Millionen von Fotos verwendet – Modell erwartet sie
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Spektrogramm dreimal stapeln, damit es wie ein RGB-Bild aussieht (alle 3 Kanäle gleich)
        img3 = np.stack([img, img, img], axis=0)          # → Form: [3, H, W]
        img3 = (img3 - mean[:, None, None]) / std[:, None, None]  # normalisieren
        return img3[np.newaxis].astype(np.float32)        # → Form: [1, 3, H, W]  (Batch-Dimension hinzufügen)

    # --- Fall 2: Eigenes CNN mit 1 Kanal ---
    # Dimensionen hinzufügen: [64, T] → [1, 1, 64, T]  (Batch + Kanal-Dimension)
    x = log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)

    # Falls das Modell eine feste Zeitlänge erwartet: auffüllen oder kürzen
    if len(input_shape) == 4 and isinstance(input_shape[3], int):
        target_t = input_shape[3]
        current_t = x.shape[3]
        if current_t < target_t:
            # Am Ende mit Nullen auffüllen (Stille)
            x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, target_t - current_t)), mode="constant")
        elif current_t > target_t:
            # Überschüssige Zeitschritte abschneiden
            x = x[:, :, :, :target_t]
    return x


# ==============================================================
# Audioaufnahme im Hintergrund
# ==============================================================

# Diese Warteschlange verbindet die Audioaufnahme (läuft im Hintergrund)
# mit der Klassifikation (läuft im Hauptprogramm).
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """
    Diese Funktion wird automatisch von sounddevice aufgerufen, sobald neue
    Audiodaten vom Mikrofon ankommen (ca. alle 4096 Samples = ~0,25 Sekunden).

    indata hat die Form [Samples, Kanäle]. Da wir Mono wollen, nehmen wir nur
    den ersten Kanal (Spalte 0) und legen ihn in die Warteschlange.
    """
    if status:
        # Gibt Warnungen aus, z.B. wenn der Puffer überläuft
        print(f"Audio-Status: {status}", file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())  # nur Kanal 0 (Mono), Kopie um Speicherfehler zu vermeiden


def collect_exact_chunk(num_samples: int) -> np.ndarray:
    """
    Holt so viele Audio-Blöcke aus der Warteschlange, bis genau num_samples
    Samples gesammelt wurden (hier: 16.000 = 1 Sekunde Audio).

    Da sounddevice immer Blöcke fester Größe liefert (4096 Samples), kann es
    sein, dass wir etwas zu viele bekommen – der Überschuss wird abgeschnitten.
    """
    collected = []
    total = 0

    while total < num_samples:
        block = audio_queue.get()   # wartet, bis ein neuer Block verfügbar ist
        collected.append(block)
        total += len(block)

    # Alle Blöcke zu einem langen Array zusammenfügen
    chunk = np.concatenate(collected, axis=0)

    # Genau num_samples zurückgeben (eventuellen Überschuss abschneiden)
    if len(chunk) > num_samples:
        chunk = chunk[:num_samples]

    return chunk.astype(np.float32)


# ==============================================================
# Modell laden und Klassifikation durchführen
# ==============================================================

def build_session(model_path: str) -> ort.InferenceSession:
    """
    Lädt das ONNX-Modell und bereitet es für die Inferenz vor.

    Wichtig: Der Pfad muss absolut sein, damit onnxruntime bei großen Modellen
    die zugehörige *.data-Datei (externe Gewichte) im selben Ordner findet.
    Mit einem relativen Pfad wie "model.onnx" schlägt das fehl.
    """
    providers = ["CPUExecutionProvider"]  # Berechnung auf der CPU (keine GPU nötig)
    abs_path = str(Path(model_path).resolve())  # relativen Pfad in absoluten umwandeln
    return ort.InferenceSession(abs_path, providers=providers)


def predict_chunk(session: ort.InferenceSession, chunk: np.ndarray) -> tuple[int, float]:
    """
    Klassifiziert einen Audio-Abschnitt und gibt zurück:
    - pred:  0 = keine Drohne,  1 = Drohne erkannt
    - score: Wahrscheinlichkeit zwischen 0.0 und 1.0
    """
    # Lautstärke normalisieren: Audio auf den Bereich [-1, 1] skalieren.
    # Das verhindert, dass leises oder lautes Umgebungsgeräusch das Modell verwirrt.
    max_abs = np.max(np.abs(chunk))
    if max_abs > 1e-6:   # Schutz vor Division durch 0 bei Stille
        chunk = chunk / max_abs

    # Audio → Spektrogramm → in Modell-Form bringen
    feat = extract_logmel(chunk)
    x = prepare_input_for_onnx(feat, session)

    # Modell ausführen: Input-Name aus dem Modell auslesen und Daten übergeben
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x})[0]

    # Modellausgabe kann verschiedene Formen haben → alles zu einer einzigen Zahl machen
    logit = float(np.asarray(output).reshape(-1)[0])

    # Manche Modelle geben rohe "Logits" aus (Werte außerhalb [0,1]).
    # Die Sigmoid-Funktion rechnet diese in eine Wahrscheinlichkeit zwischen 0 und 1 um.
    # Sigmoid(0) = 0.5, Sigmoid(+∞) → 1, Sigmoid(-∞) → 0
    score = float(1.0 / (1.0 + np.exp(-logit))) if abs(logit) > 1 or logit < 0 else logit

    pred = int(score >= THRESHOLD)  # 1 wenn Drohne wahrscheinlich, sonst 0
    return pred, score


# ==============================================================
# Alarmlogik: Wie oft wurde die Drohne zuletzt erkannt?
# ==============================================================

def alarm_level(detection_times: collections.deque) -> int:
    """
    Berechnet die aktuelle Alarmstufe (0–4) anhand der gespeicherten
    Erkennungszeitpunkte der letzten Sekunden.

    Stufe 0: kein Piep         (≤1 Detektion in 5s)
    Stufe 1: leises Piepen     (≥2 Detektionen in 5s)
    Stufe 2: mittleres Piepen  (≥3 Detektionen in 8s)
    Stufe 3: lautes Piepen     (≥4 Detektionen in 8s)
    Stufe 4: lautestes Piepen  (≥5 Detektionen in 8s)
    """
    now = time.time()
    # Zählen wie viele Detektionen in den letzten 5 bzw. 8 Sekunden liegen
    count_5s = sum(1 for t in detection_times if now - t <= 5)
    count_8s = sum(1 for t in detection_times if now - t <= 8)

    # Prüfung von oben (schlimmste Stufe) nach unten
    if count_8s >= 5:
        return 4
    if count_8s >= 4:
        return 3
    if count_8s >= 3:
        return 2
    if count_5s >= 2:
        return 1
    return 0


def beep(volume: float, duration_ms: int = 300, freq_hz: float = 880.0):
    """
    Erzeugt einen Piepton als Sinuswelle und spielt ihn sofort ab.

    volume:      Lautstärke zwischen 0.0 (still) und 1.0 (maximal)
    duration_ms: Länge des Tons in Millisekunden (Standard: 300 ms)
    freq_hz:     Tonhöhe in Hertz (Standard: 880 Hz = hohes A)
    """
    # Zeitachse erzeugen: z.B. 4800 gleichmäßige Punkte für 300 ms bei 16000 Hz
    t = np.linspace(0, duration_ms / 1000, int(SAMPLE_RATE * duration_ms / 1000), endpoint=False)
    # Sinuswelle berechnen und mit Lautstärke skalieren
    tone = (volume * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    # Ton abspielen und warten bis er fertig ist (blocking=True)
    sd.play(tone, samplerate=SAMPLE_RATE, blocking=True)


# ==============================================================
# Hauptprogramm
# ==============================================================

def main():
    # Prüfen ob das Modell überhaupt existiert
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"ONNX-Modell nicht gefunden: {model_file.resolve()}")

    # Modell laden
    session = build_session(str(model_file))

    # Alle verfügbaren Mikrofone auflisten – hilfreich um DEVICE richtig zu setzen
    print("Verfügbare Aufnahmegeräte:")

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " <-- aktiv" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print(f"Verwendetes Gerät: DEVICE={DEVICE!r} (None = Systemstandard)\n")

    print("Modell geladen.")
    print("Starte Mikrofonaufnahme ...")
    print("Abbruch mit Ctrl+C.\n")

    try:
        # Mikrofon öffnen: sounddevice ruft audio_callback auf, sobald neue Daten kommen.
        # Das passiert im Hintergrund, während der Hauptloop klassifiziert.
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=4096,   # Größe eines Audio-Blocks in Samples (~0,25 Sekunden)
            device=DEVICE,
        ):
            # Speichert die Zeitpunkte der letzten Drohnen-Detektionen.
            # maxlen=10 bedeutet: maximal 10 Einträge, älteste fallen automatisch raus.
            detection_times: collections.deque = collections.deque(maxlen=10)

            while True:
                t0 = time.time()

                # 1 Sekunde Audio sammeln
                chunk = collect_exact_chunk(CHUNK_SAMPLES)

                # Klassifizieren: Drohne oder nicht?
                pred, score = predict_chunk(session, chunk)

                # Wenn Drohne erkannt: Zeitpunkt merken
                if pred == 1:
                    detection_times.append(time.time())

                # Alarmstufe aus der Detektionshistorie ableiten
                level = alarm_level(detection_times)
                vol = ALARM_VOLUMES[level]
                label = ["--", "!", "!!", "!!!", "!!!!"][level]
                print(f"Score: {score:.3f} | Alarmstufe {level} {label}")

                # Piepen falls Alarmstufe > 0
                if level > 0:
                    beep(vol)

                # Minimale Pause damit die Schleife nicht unkontrolliert dreht
                elapsed = time.time() - t0
                if elapsed < 0.01:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nBeendet.")


# Dieser Block stellt sicher, dass main() nur ausgeführt wird, wenn das Skript
# direkt gestartet wird – nicht wenn es von einem anderen Skript importiert wird.
if __name__ == "__main__":
    main()
