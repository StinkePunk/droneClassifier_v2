import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal, fs, title="Signal Plot", domain="time"):
    """
    Plots a signal in time or frequency domain.

    Parameters:
    signal (array): The signal data.
    fs (int): The sampling frequency.
    title (str): The title of the plot.
    domain (str): The domain of the plot ('time' or 'frequency').
    """
    # Vorherige Figuren schließen
    plt.close('all')

    plt.figure()
    
    if domain == "time":
        # Plot in time domain
        t = np.arange(len(signal)) / fs
        plt.plot(t, signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(title + " (Time Domain)")
        plt.grid()
    elif domain == "frequency":
        # Plot in frequency domain
        n = len(signal)
        f = np.fft.fftfreq(n, d=1/fs)
        magnitude = np.abs(np.fft.fft(signal)) / n
        plt.semilogx(f[:n//2], 20 * np.log10(magnitude[:n//2]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.title(title + " (Frequency Domain)")
        plt.grid(which='both', linestyle='--', linewidth=0.5)
    else:
        raise ValueError("Invalid domain. Use 'time' or 'frequency'.")

    plt.show()

def plot_correlation(correlation, fs, title="Correlation Plot"):
    """
    Plots the cross-correlation signal in the time domain with adjusted x-axis.

    Parameters:
    correlation (array): The correlation data.
    fs (int): The sampling frequency.
    title (str): The title of the plot.
    """
    plt.figure()
    lags = np.arange(-len(correlation) // 2, len(correlation) // 2)
    t = lags / fs
    plt.plot(t, correlation)
    plt.xlabel('Time Lag [s]')
    plt.ylabel('Correlation')
    plt.title(title)
    plt.grid()
    plt.show()


# Beispielverwendung
# input_fs, input_signal = wavfile.read("sine_logarithmic_20Hz-20kHz_30sec_mono.wav")
# plot_signal(input_signal, input_fs, title="Input Signal", domain="time")
# plot_signal(input_signal, input_fs, title="Input Signal", domain="frequency")

def plot_audio_spectrum(audio, fs):
    """
    Plottet das Frequenzspektrum eines Audio-Files.
    
    Parameters:
    audio (numpy.ndarray): 1D-Array mit den Audiodaten.
    fs (int): Sampling-Frequenz des Audiosignals in Hz.
    """
    # Fourier-Transformation durchführen
    n = len(audio)
    fft_result = np.fft.fft(audio)
    freqs = np.fft.fftfreq(n, 1/fs)

    # Nur positive Frequenzen verwenden
    positive_freqs = freqs[:n // 2]
    magnitude = np.abs(fft_result[:n // 2])

    # Plot des Spektrums
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, magnitude)
    plt.title("Frequenzspektrum")
    plt.xlabel("Frequenz (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Beispielaufruf
# if __name__ == "__main__":
#    import librosa
#    
#    # Lade ein Beispiel-Audiofile
#    audio, fs = librosa.load("example.wav", sr=None)
#    
#    # Plotte das Spektrum
#    plot_audio_spectrum(audio, fs)

def plot_probability_distribution(y_pred, title="Wahrscheinlichkeitsverteilung von y_pred", filename="probability_distribution.png"):
    """
    Erstellt ein Histogramm zur Visualisierung der Wahrscheinlichkeitsverteilung in y_pred.
    
    Parameter:
    - y_pred: Array mit Wahrscheinlichkeiten (z. B. Ausgabe von model.predict()).
    - title: Titel des Plots (optional).
    """
    # Wahrscheinlichkeiten in eine flache Liste umwandeln
    y_pred_flat = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_flat, bins=50, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel("Wahrscheinlichkeit für Drone (1)")
    plt.ylabel("Anzahl der Samples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.xlim(0,1)

    # Speichern des Plots
    plt.savefig(filename)
    plt.close()  # optional, um den Plot direkt anzuzeigen

# Beispielaufruf: y_pred ist die Ausgabe von model.predict(val_features)
# y_pred = model.predict(val_features)  # Wahrscheinlichkeiten berechnen
# plot_probability_distribution(y_pred)

def plot_training_history(history, filename_loss='training_loss.png', filename_accuracy='training_accuracy.png'):
    """
    Plottet die Loss- und Accuracy-Kurven aus der Trainingshistorie und speichert sie als PNG.
    
    Parameter:
    - history: Das zurückgegebene Objekt von model.fit (enthält die Trainingshistorie).
    - filename_loss: Dateiname für den Loss-Plot (default: 'training_loss.png').
    - filename_accuracy: Dateiname für den Accuracy-Plot (default: 'training_accuracy.png').
    """
    # Loss-Kurven
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(filename_loss)
    print(f"Loss-Kurven gespeichert als: {filename_loss}")
    plt.close()

    # Accuracy-Kurven (nur plotten, wenn vorhanden)
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(filename_accuracy)
        print(f"Accuracy-Kurven gespeichert als: {filename_accuracy}")
        plt.close()

# Beispielaufruf
# history = model.fit(train_features, y_train, validation_data=(val_features, y_val), epochs=30, batch_size=128)
# plot_training_history(history)

def plot_confusion_matrix(cm):
    """
    Diese Funktion plottet die Konfusionsmatrix
    """

    print("Erstellen eines Plots für die Konfusionsmatrix")
    
    # Normalisierung zeilenweise in Prozent
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    print("Normalisierte Konfusionsmatrix (in Prozent):")
    print(cm_normalized)

    # Erstellen des Plots
    plt.figure(figsize=(8, 6))
    ax = plt.gca() 
    im = ax.matshow(cm_normalized, cmap=plt.cm.Blues) 
    plt.colorbar(im, ax=ax)

    # Dynamische Schriftfarbe und Zahlenformatierung
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            color = "white" if cm_normalized[i, j] > 50 else "black"  # Schriftfarbe je nach Wert
            plt.text(j, i, f"{cm_normalized[i, j]:.1f}%", ha="center", va="center", color=color)

    # Achsenbeschriftung
    print("Anpassen der Achsenbeschriftung der Konfusionsmatrix")
    plt.xticks(np.arange(2), ['Drone', 'No Drone'], rotation=45)
    plt.yticks(np.arange(2), ['Drone', 'No Drone'])
    plt.xlabel("Vorhergesagte Klassen")
    plt.ylabel("Wahre Klassen")
    plt.title("Normalisierte Konfusionsmatrix (in Prozent)")

    # Konfusionsmatrix speichern
    print("Speichern des Plots für die Konfusionsmatrix")
    plt.savefig('confusion_matrix.png')
    # plt.show()  # optional, um den Plot direkt anzuzeigen


def plot_spectrogram(audio, sample_rate, filename="spectrogram.png"):
    """
    Speichert das Spektrogramm eines Audiosignals als PNG-Datei.
    
    Parameter:
    - audio: 1D-Array des Audiosignals.
    - sample_rate: Abtastrate des Signals in Hz.
    - filename: Name der Datei, in der das Spektrogramm gespeichert wird.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.specgram(audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Intensity (dB)")
        
        # Speichern der Figure
        plt.savefig(filename)
        print(f"Spectrogram saved as {filename}")
    except Exception as e:
        print(f"Error while plotting spectrogram: {e}")
    finally:
        plt.close()  # Alle offenen Plots schließen
