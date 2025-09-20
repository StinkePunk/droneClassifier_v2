import librosa
import soundfile as sf
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

# Zur Bestimmung der Samplingrate aus den geladenen Dateien
def get_sampling_rate(file_path):
    # Nur die Metadaten des Audiofiles auslesen
    with sf.SoundFile(file_path) as f:
        return f.samplerate

# Laden der Audiodaten mit optionalem Resampling
def load_audio_data(train_path, target_sample_rate=48000, message="Processing audio files", return_filenames=False):
   
    audio_files = []
    labels = []
    filenames = []

    # Zähle alle `.wav`-Dateien für die Fortschrittsanzeige
    total_files = sum(len(files) for _, _, files in os.walk(train_path) if any(f.endswith('.wav') for f in files))
    processed_files = 1  # Zähler für bereits verarbeitete Dateien

    print(f"{message} ({total_files} files to process):")
    
    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            if file.endswith('.wav'):

                processed_files += 1  # Aktualisiere den Zähler
                
                # Fortschrittsbalken anzeigen
                progress = (processed_files / total_files) * 100
                bar_length = 40  # Länge des Fortschrittsbalkens
                block = int(bar_length * processed_files // total_files)
                progress_bar = f"[{'#' * block}{'.' * (bar_length - block)}] {progress:.2f}%"
                sys.stdout.write(f"\r{progress_bar}")  # \r überschreibt die vorherige Zeile
                sys.stdout.flush()

                file_path = os.path.join(subdir, file)
                
                # Aktuelle Samplingrate der Datei ermitteln
                current_sample_rate = get_sampling_rate(file_path)
                
                # Resampling nur durchführen, wenn die Abtastrate nicht übereinstimmt
                if current_sample_rate != target_sample_rate:
                    audio, _ = librosa.load(file_path, sr=target_sample_rate)
                else:
                    audio, _ = librosa.load(file_path, sr=None)  # Laden ohne Resampling
                
                label = os.path.basename(subdir)
                audio_files.append(audio)
                labels.append(label)
                if return_filenames:
                    filenames.append(file)
    print()                 
    
    if return_filenames:
        return audio_files, labels, filenames
    return audio_files, labels