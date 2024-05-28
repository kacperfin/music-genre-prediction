import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=10, res_type='kaiser_fast') 

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        mfccs_p = np.mean(mfccs, axis=1)

        rms = librosa.feature.rms(y=audio)
        rms_p = np.mean(rms, axis=1)

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroids_p = np.mean(spectral_centroids, axis=1)

        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidths_p = np.mean(spectral_bandwidths, axis=1)

        zero_crossing_rates = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rates_p = np.mean(zero_crossing_rates, axis=1)

        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
        chroma_cens_p = np.mean(chroma_cens, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_p = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_p = np.mean(mel, axis=1)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_p = np.mean(contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_p = np.mean(tonnetz, axis=1)

        stack = np.hstack([tempo, mfccs_p, chroma_p, mel_p, contrast_p, tonnetz_p, rms_p, spectral_centroids_p, spectral_bandwidths_p, zero_crossing_rates_p, chroma_cens_p])

        return stack

    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        print("Exception message:", e)
        return None

