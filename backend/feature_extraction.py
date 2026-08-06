import librosa
import numpy as np

file_path = "../dataset/Actor_01/03-01-01-01-01-01-01.wav"

# Load audio
audio, sample_rate = librosa.load(file_path, sr=None)

# Extract MFCC (40 coefficients)
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print("MFCC Shape:", mfcc.shape)

# Take the average of each coefficient
mfcc_mean = np.mean(mfcc.T, axis=0)

print("Feature Vector Shape:", mfcc_mean.shape)

print("\nFirst 10 Features:")
print(mfcc_mean[:10])