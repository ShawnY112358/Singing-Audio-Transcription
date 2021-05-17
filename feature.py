import librosa
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt


def log_melspectrogram(path, sample_rate=44100, n_mels=128):
    y, sr = librosa.load(path, sr=sample_rate)
    feature = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    feature = librosa.power_to_db(feature)
    return feature

def test_log_mel():
    path = '../data_test/child1.wav'
    feature = log_melspectrogram(path)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, sr=44100, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log_Mel spectrogram')
    plt.tight_layout()
    plt.show()

def main():
    test_log_mel()

if __name__ == '__main__':
    main()