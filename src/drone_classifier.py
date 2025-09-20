import os
import pickle
import numpy as np
from loadDataStore import load_audio_data
from splitAudioFiles import split_data_and_labels
from augmentations import apply_augmentations
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

    def extract(self, audio_data):
        """Extract Mel-spectrogram features from a list of audio arrays."""
        return extract_features(audio_data, self.max_length, sr=self.sample_rate, fmax=self.fmax)


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
        # Einheitliches Label-Mapping über beide Splits
        self.label_encoder = LabelEncoder()
        all_y = np.concatenate([y_train, y_val])
        self.label_encoder.fit(all_y)
        y_train_enc = self.label_encoder.transform(y_train)
        y_val_enc   = self.label_encoder.transform(y_val)
        print("Label mapping:", dict(zip(self.label_encoder.classes_,
                                         self.label_encoder.transform(self.label_encoder.classes_))))

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
        if hasattr(self, "label_encoder") and isinstance(y_val[0], str):
            y_val = self.label_encoder.transform(y_val)

        print(X_val.shape)  # Erwartet (N,128,128); predict hebt auf (N,128,128,1)
        y_pred_prob = self.predict(X_val).ravel()
        # datengetriebener Threshold (max F1 für Klasse "no drone"==1)
        import numpy as np
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 33):
            pred = (y_pred_prob >= t).astype(int)
            tp = np.sum((pred==1)&(y_val==1)); fp = np.sum((pred==1)&(y_val==0))
            fn = np.sum((pred==0)&(y_val==1))
            f1 = 2*tp/(2*tp+fp+fn+1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        y_pred = (y_pred_prob >= best_t).astype(int)
        print(f"Best threshold: {best_t:.3f} | F1(no-drone): {best_f1:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(cm)
        plot_probability_distribution(y_pred_prob)


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
            X_val, y_val = val_data.chunk_and_label()

            # Augmentiere nur das train-Set
            X_train = apply_augmentations(X_train, self.config['sample_rate'])

            train_features = self.extractor.extract(X_train)
            val_features = self.extractor.extract(X_val)

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