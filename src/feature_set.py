import numpy as np
from featureExtraction import extract_features

class FeatureSet:
    """
    Handles feature extraction from a list of AudioData objects.

    Attributes:
        audio_dataset (AudioDataSet): Collection of AudioData objects.
        sample_rate (int): Target sample rate for features.
        max_length (int): Max length of the time dimension.
        fmax (int): Max frequency for Mel-spectrogram.
    """
    def __init__(self, audio_dataset, sample_rate=16000, max_length=128, fmax=None):
        self.audio_dataset = audio_dataset
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.fmax = fmax or sample_rate // 2

    def extract(self):
        """
        Extract features and labels from the dataset.

        Returns:
            tuple: (features, labels) as NumPy arrays.
        """
        signals, labels = self.audio_dataset.get_signals_and_labels()
        features = extract_features(
            audio_files=signals,
            max_length=self.max_length,
            sr=self.sample_rate,
            fmax=self.fmax
        )
        return features, labels