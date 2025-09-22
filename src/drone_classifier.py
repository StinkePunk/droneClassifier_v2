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

    def extract(self, audio_data, is_training=False):
        """Extract Mel-spectrogram features from a list of audio arrays."""
        return extract_features(audio_data, self.max_length, sr=self.sample_rate, fmax=self.fmax, is_training=is_training)


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
        return self.model.predict(Xp, verbose=0)

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on validation data.

        Shows confusion matrix and prediction probability distribution.
        """
        # Einheitliche Label-Kodierung
        if isinstance(y_val[0], str):
            y_val = (np.asarray(y_val, dtype=str) == 'drone').astype('int32')

        print(X_val.shape)  # Erwartet (N,128,128); predict hebt auf (N,128,128,1)
        y_pred_prob = self.predict(X_val).ravel()
        # Threshold-Suche: F1(drone=1) maximieren
        from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
        ts = np.linspace(0.05, 0.95, 19)
        best_t, best_macro = 0.5, -1.0
        for t in ts:
            y_hat = (y_pred_prob >= t).astype(int)
            f1_pos = f1_score(y_val, y_hat, pos_label=1)
            f1_neg = f1_score(1 - y_val, 1 - y_hat, pos_label=1)  # F1 für no-drone
            macro = 0.5 * (f1_pos + f1_neg)
            if macro > best_macro:
                best_macro, best_t = macro, t
        y_pred = (y_pred_prob >= best_t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        print(f"Best threshold (macro-F1): {best_t:.3f} | macro-F1: {best_macro:.3f} | acc: {acc:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred, labels=[0,1])
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
            noise_pool = [x for x, lbl in zip(X_train, y_train) if str(lbl).lower() in ("no drone","no_drone","no-drone","0","nd")]
            # Optional: RIRs laden (wenn konfiguriert)
            rir_list = []
            if 'rir_dir' in self.config and self.config['rir_dir']:
                rir_list = load_rirs(self.config['rir_dir'], self.config['sample_rate'])
            X_train = apply_augmentations(
                X_train, self.config['sample_rate'],
                noise_pool=noise_pool, snr_range=(0,20), p_noise=0.9,
                rir_list=rir_list, rir_skip_prob=0.2
            )

            # SpecAugment nur im Training aktiv
            train_features = self.extractor.extract(X_train, is_training=True)
            val_features   = self.extractor.extract(X_val)

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