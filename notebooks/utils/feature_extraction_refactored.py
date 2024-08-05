import librosa
import numpy as np
import pandas as pd

class Config:
    NUMBER_OF_MFCC = 13 
    N_FFT = 2048  
    HOP_LENGTH = 512  

def load_and_preprocess_audio(audio_file_path: str):
    y, sr = librosa.load(audio_file_path)
    signal, _ = librosa.effects.trim(y)
    return signal, sr

def extract_spectral_features(signal: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict:
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, _ = librosa.magphase(stft)
    features = {
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=signal)),
        "zero_crossings": np.sum(librosa.zero_crossings(y=signal, pad=False)),
        "spectrogram": np.mean(librosa.amplitude_to_db(magnitude, ref=np.max)),
        "mel_spectrogram": np.mean(librosa.amplitude_to_db(librosa.feature.melspectrogram(S=magnitude, sr=sr), ref=np.max)),
        "harmonics": np.mean(librosa.effects.hpss(signal)[0]),
        "perceptual_shock_wave": np.mean(librosa.effects.hpss(signal)[1]),
        "spectral_centroids": np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)),
        "spectral_centroids_delta": np.mean(librosa.feature.delta(librosa.feature.spectral_centroid(y=signal, sr=sr))),
        "spectral_centroids_accelerate": np.mean(librosa.feature.delta(librosa.feature.spectral_centroid(y=signal, sr=sr), order=2)),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)),
        "spectral_flux": np.mean(librosa.onset.onset_strength(y=signal, sr=sr)),
        "spectral_bandwidth_2": np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)),
        "spectral_bandwidth_3": np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=3)),
        "spectral_bandwidth_4": np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=4)),
        "rms": np.mean(librosa.feature.rms(y=signal)),
        "bpm": np.mean(librosa.feature.tempo(y=signal, sr=sr)),
        "poly_features": np.mean(librosa.feature.poly_features(S=magnitude, sr=sr)),
        "tonnetz": np.mean(librosa.feature.tonnetz(y=signal, sr=sr)),
        "tempogram": np.mean(librosa.feature.tempogram(y=signal, sr=sr, hop_length=hop_length))
    }
    return features

def extract_chroma_features(signal: np.ndarray, sr: int, n_fft: int, hop_length: int) -> dict:
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, _ = librosa.magphase(stft)
    chromagram = librosa.feature.chroma_stft(S=magnitude, sr=sr, hop_length=hop_length)
    chroma_features = {f'chroma{i+1}': np.mean(chromagram[i]) for i in range(12)}
    return chroma_features

def extract_mfcc_features(audio_file_name: str, signal: np.ndarray, sample_rate: int, number_of_mfcc: int) -> pd.DataFrame:
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc)
    accelerate = librosa.feature.delta(mfcc, order=2)

    mfcc_features = {"filename": audio_file_name}
    for i in range(number_of_mfcc):
        mfcc_features[f'mfcc{i}'] = np.mean(mfcc[i])
        mfcc_features[f'mfcc_delta_{i}'] = np.mean(delta[i])
        mfcc_features[f'mfcc_accelerate_{i}'] = np.mean(accelerate[i])

    return pd.DataFrame([mfcc_features])

def extract_pitch_shift_features(signal: np.ndarray, sr: int) -> dict:
    pitches, _ = librosa.core.piptrack(y=signal, sr=sr)
    pitch_shift_features = {
        "pitch_mean": np.mean(pitches[pitches > 0]),
        "pitch_std": np.std(pitches[pitches > 0]),
        "pitch_max": np.max(pitches[pitches > 0]),
        "pitch_min": np.min(pitches[pitches > 0])
    }
    return pitch_shift_features

def extract_feature_means(audio_file_path: str) -> pd.DataFrame:
    c = Config()
    signal, sr = load_and_preprocess_audio(audio_file_path)
    
    spectral_features = extract_spectral_features(signal, sr, c.N_FFT, c.HOP_LENGTH)
    chroma_features = extract_chroma_features(signal, sr, c.N_FFT, c.HOP_LENGTH)
    mfcc_df = extract_mfcc_features(audio_file_path, signal, sr, c.NUMBER_OF_MFCC)
    pitch_shift_features = extract_pitch_shift_features(signal, sr)
    
    features = {"filename": audio_file_path}
    features.update(spectral_features)
    features.update(chroma_features)
    features.update(pitch_shift_features)
    
    df = pd.DataFrame([features])
    df = pd.merge(df, mfcc_df, on='filename')
    
    return df