import librosa
import numpy as np
import sys

def extract_features(audio_files, max_length=128, message="Processing features",
                     sr=16000, n_mels=128, fmax=8000, is_training=False):
    features = []
    hop_length = int(sr / max_length)

    total_files = len(audio_files)
    print(f"{message} ({total_files} files to process):")

    for idx, audio in enumerate(audio_files, start=1):

        # Fortschrittsbalken anzeigen
        progress = (idx / total_files) * 100
        bar_length = 40
        block = int(bar_length * idx // total_files)
        progress_bar = f"[{'#' * block}{'.' * (bar_length - block)}] {progress:.2f}%"
        sys.stdout.write(f"\r{progress_bar}")
        sys.stdout.flush()

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, fmax=fmax, hop_length=hop_length
        )
        feature = librosa.power_to_db(mel_spectrogram, ref=np.max)
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

        # SpecAugment (nur im Training): kleine Time/Freq-Masks
        if is_training:
            n_mels_cur, n_frames = feature.shape
            # 1–2 Frequenzmasken
            for _ in range(np.random.randint(1,3)):
                F = np.random.randint(5, 11)
                f0 = np.random.randint(0, max(1, n_mels_cur - F))
                feature[f0:f0+F, :] = 0
            # 1–2 Zeitmasken
            for _ in range(np.random.randint(1,3)):
                T = np.random.randint(5, 13)
                t0 = np.random.randint(0, max(1, n_frames - T))
                feature[:, t0:t0+T] = 0

        # Debugging
        # print(f"Mel-Spektrogramm Shape (vor Padding): {mel_spectrogram.shape}")
        
        # Truncate or pad feature to max_length
        if feature.shape[1] > max_length:
            feature = feature[:, :max_length]
        elif feature.shape[1] < max_length:
            padding = max_length - feature.shape[1]
            feature = np.pad(feature, ((0, 0), (0, padding)), mode='constant')

            # Debugging
            # print(f"Padding erforderlich: {padding} Frames")
        
        features.append(feature)
    
    print()  # Neue Zeile nach Abschluss des Fortschrittsbalkens
    return np.array(features)
