import os
import pickle
from audio_data import AudioDataSet
from feature_set import FeatureSet
from classifier import DroneClassifier

class TrainerPipeline:
    """
    Full training and evaluation pipeline with persistent feature and model handling.
    """
    def __init__(self, config):
        self.config = config
        self.chunk_length = config['sample_rate'] * config['audio_length']
        self.model_path = config['model_file']
        self.feature_files = config['feature_files']

    def load_or_generate_features(self):
        """
        Load feature and label arrays from disk or compute them.
        """
        if all(os.path.exists(f) for f in self.feature_files.values()):
            with open(self.feature_files['train_features'], "rb") as f:
                X_train = pickle.load(f)
            with open(self.feature_files['val_features'], "rb") as f:
                X_val = pickle.load(f)
            with open(self.feature_files['train_labels'], "rb") as f:
                y_train = pickle.load(f)
            with open(self.feature_files['val_labels'], "rb") as f:
                y_val = pickle.load(f)
        else:
            train_raw = AudioDataSet.from_path(self.config['train_path'], self.config['sample_rate'])
            val_raw = AudioDataSet.from_path(self.config['val_path'], self.config['sample_rate'])

            train_chunks = train_raw.chunk_all(self.chunk_length).augment_all()
            val_chunks = val_raw.chunk_all(self.chunk_length)

            train_features = FeatureSet(train_chunks, sample_rate=self.config['sample_rate'])
            val_features = FeatureSet(val_chunks, sample_rate=self.config['sample_rate'])

            X_train, y_train = train_features.extract()
            X_val, y_val = val_features.extract()

            with open(self.feature_files['train_features'], "wb") as f:
                pickle.dump(X_train, f)
            with open(self.feature_files['val_features'], "wb") as f:
                pickle.dump(X_val, f)
            with open(self.feature_files['train_labels'], "wb") as f:
                pickle.dump(y_train, f)
            with open(self.feature_files['val_labels'], "wb") as f:
                pickle.dump(y_val, f)

        return X_train, y_train, X_val, y_val

    def run(self):
        """
        Execute training pipeline with full evaluation.
        """
        X_train, y_train, X_val, y_val = self.load_or_generate_features()
        clf = DroneClassifier(self.model_path)

        if os.path.exists(self.model_path):
            clf.load()
        else:
            clf.train(X_train, y_train, X_val, y_val)

        clf.evaluate(X_val, y_val)
        return clf