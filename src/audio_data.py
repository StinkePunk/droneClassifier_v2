import numpy as np
import matplotlib.pyplot as plt
from loadDataStore import load_audio_data
from splitAudioFiles import split_into_chunks
from augmentations import apply_augmentations


class AudioData:
    """
    Represents a single audio signal with metadata.
    """
    def __init__(self, signal: np.ndarray, sample_rate: int, label: str = None, source_file: str = None):
        self.signal = signal
        self.sample_rate = sample_rate
        self.label = label
        self.source_file = source_file

    def plot(self, title=None):
        """Plot the waveform of the signal."""
        t = np.linspace(0, len(self.signal) / self.sample_rate, len(self.signal))
        plt.figure(figsize=(10, 2))
        plt.plot(t, self.signal)
        plt.title(title or f"Audio ({self.sample_rate} Hz)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def chunk(self, chunk_length: int) -> list:
        """Split this audio signal into multiple AudioData chunks."""
        chunks = split_into_chunks(self.signal, chunk_length)
        return [AudioData(c, self.sample_rate, self.label) for c in chunks]

    def augment_echo(self):
        """Apply echo augmentation and return new AudioData object."""
        augmented_signal = apply_augmentations([self.signal], self.sample_rate)[0]
        return AudioData(augmented_signal, self.sample_rate, self.label)
    
    def apply_pre_emphasis(self, alpha=0.97):
        """Apply pre-emphasis filter to the signal (in-place)."""
        emphasized = np.append(self.signal[0], self.signal[1:] - alpha * self.signal[:-1])
        self.signal = emphasized

    def apply_rms_normalization(self, target_rms=0.1):
        rms = np.sqrt(np.mean(self.signal**2))
        if rms > 0:
            self.signal = self.signal * (target_rms / rms)

    

class AudioDataSet:
    """
    Collection of multiple AudioData objects.
    """
    def __init__(self, audio_list: list[AudioData]):
        self.audio = audio_list

    @classmethod
    def from_path(cls, path: str, sample_rate: int):
        """Load and wrap audio files from a directory, including filenames."""
        signals, labels, filenames = load_audio_data(path, sample_rate, return_filenames=True)
        audio_objects = [AudioData(sig, sample_rate, lbl, fname) for sig, lbl, fname in zip(signals, labels, filenames)]
        return cls(audio_objects)


    def chunk_all(self, chunk_length: int):
        """Apply chunking to all contained AudioData objects."""
        chunks = []
        for audio in self.audio:
            chunks.extend(audio.chunk(chunk_length))
        return AudioDataSet(chunks)

    def augment_all(self):
        """Apply augmentation to all AudioData objects."""
        augmented = [a.augment_echo() for a in self.audio]
        return AudioDataSet(augmented)

    def get_signals_and_labels(self):
        """Return raw signal arrays and corresponding labels."""
        signals = [a.signal for a in self.audio]
        labels = [a.label for a in self.audio]
        return signals, labels

    def plot_random(self):
        """Plot a random AudioData signal from the set."""
        import random
        random.choice(self.audio).plot()