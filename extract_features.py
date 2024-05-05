import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=20, res_type='kaiser_fast',) 

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs, axis=1)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_processed = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_processed = np.mean(mel, axis=1)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_processed = np.mean(contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        tonnetz_processed = np.mean(tonnetz, axis=1)

        return np.hstack([mfccs_processed, chroma_processed, mel_processed, contrast_processed, tonnetz_processed])
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        print("Exception:", e)
        return None 
    return mfccs_processed
