import os
import pickle
import numpy as np
from loadDataStore import load_audio_data
from splitAudioFiles import split_data_and_labels
from augmentations import apply_augmentations, load_rirs
from featureExtraction import extract_features
from tensorflow.keras.models import load_model
from training import train_classifier
from plotting import plot_training_history, plot_confusion_matrix, plot_probability_distribution
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

class AudioDataset:
    """
    Handles loading, chunking, and augmentation of audio data.

    Attributes:
        path (str): Path to the audio dataset.
        sample_rate (int): Target sampling rate for the audio.
        chunk_length (int): Length of audio chunks in samples.
        audio_files (list): Loaded audio data.
        labels (list): Corresponding labels for the audio data.
    """
    def __init__(self, path, sample_rate, chunk_length):
        self.path = path
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.audio_files, self.labels = load_audio_data(path, sample_rate)

    def chunk_and_label(self):
        """Split audio into uniform chunks and replicate labels accordingly."""
        return split_data_and_labels(self.audio_files, self.labels, self.chunk_length)

    def augment(self):
        """Apply augmentation to audio files (currently only echo)."""
        self.audio_files = apply_augmentations(self.audio_files, self.sample_rate)


class FeatureExtractor:
    """
    Extracts Mel-spectrogram-based features from audio data.

    Attributes:
        sample_rate (int): Sampling rate for feature extraction.
        max_length (int): Maximum time steps for the feature matrix.
        fmax (int): Maximum frequency for Mel-spectrogram.
    """
    def __init__(self, sample_rate, max_length=128, fmax=None):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.fmax = fmax or sample_rate // 2

    def extract(self, audio_data, is_training=False, apply_pre_emphasis=True, pre_emph_alpha=0.97):
        """Extract Mel-spectrogram features from a list of audio arrays."""
        return extract_features(
            audio_data, self.max_length, sr=self.sample_rate, fmax=self.fmax,
            is_training=is_training, apply_pre_emphasis=apply_pre_emphasis, pre_emph_alpha=pre_emph_alpha
        )


class DroneClassifier:
    """
    Manages training, loading, prediction, and evaluation of the classification model.

    Attributes:
        model_path (str): Path to save/load the trained model.
        model (Keras Model): Loaded or trained Keras model.
        history (History object): Training history returned from model.fit().
    """
    def __init__(self, model_path, trainable_layers=0, scaler_path=None):
        self.model_path = model_path
        self.trainable_layers = trainable_layers
        self.model = None
        self.history = None
        self.scaler = None
        self.scaler_path = scaler_path or (os.path.splitext(model_path)[0] + "_scaler.pkl")
        self.temperature = 1.0  # für Probabilitäts-Kalibrierung

    def train(self, X_train, y_train, X_val, y_val):
        """Train the classifier and save it to disk."""
        # Einheitliches Binär-Mapping: drone=1, no drone=0
        y_train_enc = (np.asarray(y_train, dtype=str) == 'drone').astype('int32')
        y_val_enc   = (np.asarray(y_val,   dtype=str) == 'drone').astype('int32')
        print("Label mapping (fixed): {'no drone': 0, 'drone': 1}")
        self.model, self.history, self.scaler = train_classifier(
            X_train, y_train_enc, X_val, y_val_enc,
            model_name="MobileNet",
            trainable_layers=self.trainable_layers,
            dropout_rate=0.3
        )
        self.model.save(self.model_path)
        # Kein externer Scaler mehr notwendig
        self.scaler = None
        # Nach Training: Temperatur auf Val-Daten kalibrieren
        yv = (np.asarray(y_val, dtype=str) == 'drone').astype('int32') if isinstance(y_val[0], str) else np.asarray(y_val, dtype='int32')
        self._calibrate_temperature(X_val, yv)

    def _calibrate_temperature(self, X_val, y_val):
        """Grid-Search über T zur Minimierung der NLL."""
        p = self.model.predict(self._prepare_inputs(X_val), verbose=0).ravel()
        p = np.clip(p, 1e-6, 1-1e-6)
        logit = np.log(p) - np.log(1-p)
        Ts = np.linspace(0.8, 3.0, 23)
        def nll(T):
            z = logit / T
            q = 1/(1+np.exp(-z))
            q = np.clip(q, 1e-6, 1-1e-6)
            return -np.mean(y_val*np.log(q) + (1-y_val)*np.log(1-q))
        self.temperature = float(Ts[int(np.argmin([nll(T) for T in Ts]))])
        print(f"[Calib] Temperature T={self.temperature:.2f}")

    def load(self):
        """Load a pre-trained model from disk."""
        self.model = load_model(self.model_path, compile=True)
        # Per-Sample-Standardisierung: kein externer Scaler nötig
        self.scaler = None

    def _prepare_inputs(self, X):
        """Ensure channel dim + per-sample standardization."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)  # (N,128,128,1)
        m = np.mean(X, axis=(1,2,3), keepdims=True)
        s = np.std(X,  axis=(1,2,3), keepdims=True) + 1e-8
        return (X - m) / s

    def predict(self, X):
        """Predict probabilities for given feature inputs."""
        Xp = self._prepare_inputs(X)
        p = self.model.predict(self._prepare_inputs(X), verbose=0).ravel()
        if getattr(self, "temperature", 1.0) != 1.0:
            p = np.clip(p, 1e-6, 1-1e-6)
            z = np.log(p) - np.log(1-p)
            p = 1/(1+np.exp(-z/self.temperature))
        return p

    def evaluate(self, X_val, y_val, min_precision_drone: float = 0.85, threshold_strategy: str = "macro_f1"):
        """
        Evaluate the model on validation data.

        Shows confusion matrix and prediction probability distribution.
        """
        # Einheitliche Label-Kodierung
        if isinstance(y_val[0], str):
            y_val = (np.asarray(y_val, dtype=str) == 'drone').astype('int32')

        print(X_val.shape)  # Erwartet (N,128,128); predict hebt auf (N,128,128,1)
        y_pred_prob = self.predict(X_val).ravel()

        # Schwellenwahl
        from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, brier_score_loss
        if threshold_strategy == "recall_at_precision":
            prec, rec, thr = precision_recall_curve(y_val, y_pred_prob)
            mask = prec[:-1] >= min_precision_drone
            best_t = thr[mask][np.argmax(rec[:-1][mask])] if np.any(mask) else 0.5
        elif threshold_strategy == "youden":
            fpr, tpr, thr = roc_curve(y_val, y_pred_prob)
            best_t = thr[np.argmax(tpr - fpr)]
        else:  # "macro_f1"
            ts = np.linspace(0.05, 0.95, 19)
            best_t, best_macro = 0.5, -1.0
            for t in ts:
                y_hat = (y_pred_prob >= t).astype(int)
                f1_pos = f1_score(y_val, y_hat, pos_label=1)
                f1_neg = f1_score(1 - y_val, 1 - y_hat, pos_label=1)
                macro = 0.5 * (f1_pos + f1_neg)
                if macro > best_macro:
                    best_macro, best_t = macro, t

        y_pred = (y_pred_prob >= best_t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        pos_rate = float(y_pred.mean())
        # Kalibrierungsmetriken
        brier = brier_score_loss(y_val, y_pred_prob)
        # sehr einfache ECE (10-Bins)
        bins = np.linspace(0,1,11); idx = np.digitize(y_pred_prob, bins)-1
        ece = 0.0
        for b in range(10):
            m = idx==b
            if np.any(m):
                conf = y_pred_prob[m].mean()
                acc_b = (y_val[m]==(y_pred_prob[m]>=best_t)).mean()
                ece += np.abs(acc_b - conf) * (m.mean())
        print(f"Threshold={best_t:.3f} | acc={acc:.3f} | pos_rate={pos_rate:.3f} | Brier={brier:.4f} | ECE@10={ece:.4f} | T={getattr(self,'temperature',1.0):.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred, labels=[1,0])
        plot_confusion_matrix(cm)
        plot_probability_distribution(y_pred_prob)
        print(classification_report(y_val, y_pred, target_names=['no drone (0)','drone (1)'], digits=3))
        return acc

class TrainerPipeline:
    """
    Orchestrates the full training pipeline from loading data to evaluation.

    Attributes:
        config (dict): Configuration parameters.
        chunk_length (int): Audio chunk length in samples.
        extractor (FeatureExtractor): Feature extractor instance.
        classifier (DroneClassifier): Classifier instance.
    """
    def __init__(self, config):
        self.config = config
        self.chunk_length = config['sample_rate'] * config['audio_length']
        self.extractor = FeatureExtractor(config['sample_rate'])
        self.classifier = DroneClassifier(config['model_file'])

    def load_or_generate_features(self):
        """
        Load pre-extracted features from disk or compute them from raw audio.

        Returns:
            train_features, y_train, val_features, y_val
        """
        if all(os.path.exists(f) for f in self.config['feature_files'].values()):
            with open(self.config['feature_files']['train_features'], "rb") as f:
                train_features = pickle.load(f)
            with open(self.config['feature_files']['val_features'], "rb") as f:
                val_features = pickle.load(f)
            with open(self.config['feature_files']['train_labels'], "rb") as f:
                y_train = pickle.load(f)
            with open(self.config['feature_files']['val_labels'], "rb") as f:
                y_val = pickle.load(f)
        else:
            train_data = AudioDataset(self.config['train_path'], self.config['sample_rate'], self.chunk_length)
            val_data = AudioDataset(self.config['val_path'], self.config['sample_rate'], self.chunk_length)
            
            X_train, y_train = train_data.chunk_and_label()
            X_val, y_val     = val_data.chunk_and_label()

            # Augmentiere nur das Train-Set:
            # Noise-Mischung NUR aus "no drone"-Chunks des TRAINING-SETS (keine Val-Daten in Training!)
            def _norm(lbl:str) -> str:
                return str(lbl).strip().lower().replace("-", " ").replace("_", " ")
            noise_pool = [x for x, lbl in zip(X_train, y_train) if "no drone" in _norm(lbl) or _norm(lbl) == "0"]
            # Optional: RIRs laden (wenn konfiguriert)
            rir_list = []
            rir_dir = self.config.get('rir_dir')
            if rir_dir:
                rir_list = load_rirs(
                    rir_dir,
                    self.config['sample_rate'],
                    trim_ms=self.config.get('rir_trim_ms', None)
                )
            # >>> Nur DRONE-Chunks augmentieren <<<
            drone_idx = [i for i, lbl in enumerate(y_train) if "drone" == _norm(lbl)]
            if drone_idx:
                X_drone = [X_train[i] for i in drone_idx]
                X_drone_aug = apply_augmentations(
                     X_drone, self.config['sample_rate'],
                     noise_pool=noise_pool, snr_range=(8,20), p_noise=0.9,
                    rir_list=rir_list, rir_skip_prob=0.5
                 )
                # Zurückschreiben, No-Drone bleibt unverändert
                X_train = list(X_train)
                for j, idx in enumerate(drone_idx):
                    X_train[idx] = X_drone_aug[j]
            # <<< Ende nur-Drone-Augmentierung

             # konsistent: Pre-Emphasis in Features, für Train & Val gleich
            pe = self.config.get("apply_pre_emphasis", True)
            alpha = self.config.get("pre_emph_alpha", 0.97)
            train_features = self.extractor.extract(X_train, is_training=True,  apply_pre_emphasis=pe, pre_emph_alpha=alpha)
            val_features   = self.extractor.extract(X_val,   is_training=False, apply_pre_emphasis=pe, pre_emph_alpha=alpha)

            with open(self.config['feature_files']['train_features'], "wb") as f:
                pickle.dump(train_features, f)
            with open(self.config['feature_files']['val_features'], "wb") as f:
                pickle.dump(val_features, f)
            with open(self.config['feature_files']['train_labels'], "wb") as f:
                pickle.dump(y_train, f)
            with open(self.config['feature_files']['val_labels'], "wb") as f:
                pickle.dump(y_val, f)

        return train_features, y_train, val_features, y_val

    def run(self):
        """Execute the full pipeline: feature loading/training/evaluation."""
        train_features, y_train, val_features, y_val = self.load_or_generate_features()

        if os.path.exists(self.config['model_file']):
            self.classifier.load()
        else:
            self.classifier.train(train_features, y_train, val_features, y_val)
            plot_training_history(self.classifier.history)

        self.classifier.evaluate(val_features, y_val)