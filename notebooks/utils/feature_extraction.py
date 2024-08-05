import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import librosa.display

class Config:
    NUMBER_OF_MFCC = 13  # Example value, adjust according to your needs
    N_FFT = 2048  # Example FFT window size
    HOP_LENGTH = 512  # Example hop length

def extract_feature_means(audio_file_path: str, n_rates:int=660000) -> pd.DataFrame:
    c = Config()
    # config settings
    number_of_mfcc = c.NUMBER_OF_MFCC

    # 1. Importing 1 file
    y, sr = librosa.load(audio_file_path)

    # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
    signal, _ = librosa.effects.trim(y)

    # 2. Fourier Transform
    # Default FFT window size
    n_fft = c.N_FFT  # FFT window size
    hop_length = c.HOP_LENGTH  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(y=signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(y=signal)


    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(data=spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(data=spectral_centroids, order=2)

    # spectral_centroid_feats = np.stack((spectral_centroids, delta, accelerate))  # (3, 64, xx)

    # 8. Chroma Frequencies¶
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical octave.

    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = c.HOP_LENGTH

    # Chromogram
    chromagram = librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=hop_length)

    # 9. Tempo BPM (beats per minute)¶
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(y=signal, sr=sr)

    # 10. Spectral Rolloff
    # Note: Is a measure of the shape of the signal. It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=4)[0]

    audio_features = {
        "file_name": audio_file_path,
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": np.mean(db_audio[0]),
        "mel_spectrogram": np.mean(s_db_audio[0]),
        "harmonics": np.mean(y_harm),
        "perceptual_shock_wave": np.mean(y_perc),
        "spectral_centroids": np.mean(spectral_centroids),
        "spectral_centroids_delta": np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate": np.mean(spectral_centroids_accelerate),
        "chroma1": np.mean(chromagram[0]),
        "chroma2": np.mean(chromagram[1]),
        "chroma3": np.mean(chromagram[2]),
        "chroma4": np.mean(chromagram[3]),
        "chroma5": np.mean(chromagram[4]),
        "chroma6": np.mean(chromagram[5]),
        "chroma7": np.mean(chromagram[6]),
        "chroma8": np.mean(chromagram[7]),
        "chroma9": np.mean(chromagram[8]),
        "chroma10": np.mean(chromagram[9]),
        "chroma11": np.mean(chromagram[10]),
        "chroma12": np.mean(chromagram[11]),
        "tempo_bpm": tempo_y,
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(onset_env),
        "spectral_bandwidth_2": np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_3": np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_4": np.mean(spectral_bandwidth_4),
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_feature_means(audio_file_path,
                                    signal,
                                    sample_rate=sr,
                                    number_of_mfcc=number_of_mfcc)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='file_name')

    return df

    # librosa.feature.mfcc(signal)[0, 0]

def extract_mfcc_feature_means(audio_file_name: str,
                          signal: np.ndarray,
                          sample_rate: int,
                          number_of_mfcc: int) -> pd.DataFrame:

    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        # mfcc coefficient
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df